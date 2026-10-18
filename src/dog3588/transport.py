"""RobotLink v1 transport, matching project/dog (CRC16, NOT CRC32).

Only the worker owns the serial device. Public methods exchange copied state
with it; neither construction nor the IMU facade opens another serial port.
See docs/stm32-protocol.md for firmware prerequisites and delivery-v1 status.
"""

from dataclasses import dataclass
import binascii
import math
import struct
import threading
import time

import numpy as np

from .config import positive
from .safety import SafetyFault
from .types import ImuState, JointState


MAGIC = b"\xa5\x5a"
VERSION = 1
MAX_PAYLOAD = 384
HEADER = struct.Struct("<2sBBHHI")
MAX_FRAME = MAX_PAYLOAD + 14
HEARTBEAT, JOINT_TARGET, STOP, ARM = 1, 2, 3, 4
JOINT_STATUS, IMU_STATUS, HEIGHT_SCAN, FAULT_STATUS, ACK = range(0x81, 0x86)
# Proposed application extension, disabled by default; absent from current C.
DELIVERY_REQUEST, DELIVERY_ACK = 0x10, 0x90
DELIVERY_EXTENSION_VERSION = 1
JOINT_RECORD = struct.Struct("<BBhIIffI")
IMU_RECORD = struct.Struct("<BII8f")
ACK_RECORD = struct.Struct("<BBHI")
ALL_JOINTS = 0x0FFF


class UnsupportedFeature(NotImplementedError):
    pass


@dataclass(frozen=True)
class Frame:
    type: int
    sequence: int
    timestamp_ms: int
    payload: bytes = b""


def crc16(data: bytes) -> int:
    """CRC16-CCITT-FALSE: poly 0x1021, init 0xffff, no reflection/xorout."""
    return binascii.crc_hqx(data, 0xFFFF)


def encode_frame(frame: Frame) -> bytes:
    if len(frame.payload) > MAX_PAYLOAD:
        raise ValueError("RobotLink payload exceeds 384 bytes")
    header = HEADER.pack(MAGIC, VERSION, frame.type, len(frame.payload),
                         frame.sequence, frame.timestamp_ms)
    data = header + frame.payload
    return data + struct.pack("<H", crc16(data[2:]))


def _newer(value, previous, bits):
    return 0 < ((value - previous) & ((1 << bits) - 1)) < (1 << (bits - 1))


class FrameParser:
    """Incremental, bounded parser with the C parser's 100 ms inter-byte timeout."""

    def __init__(self):
        self.buffer = bytearray()
        self.last_byte = None
        self.rejected = self.timeouts = self.frames = 0

    def feed(self, data: bytes, now=None) -> list[Frame]:
        now = time.monotonic() if now is None else now
        if self.buffer and self.last_byte is not None and now - self.last_byte > 0.1:
            self.buffer.clear()
            self.timeouts += 1
        result = []
        # Feed by byte so even a very large caller chunk cannot grow the buffer.
        for byte in data:
            self.last_byte = now
            if len(self.buffer) == MAX_FRAME:
                del self.buffer[0]
                self.rejected += 1
            self.buffer.append(byte)
            while len(self.buffer) >= 2:
                if self.buffer[:2] != MAGIC:
                    del self.buffer[0]
                    continue
                if len(self.buffer) < HEADER.size:
                    break
                _, version, kind, size, seq, stamp = HEADER.unpack_from(self.buffer)
                if version != VERSION or size > MAX_PAYLOAD:
                    del self.buffer[0]
                    self.rejected += 1
                    continue
                total = size + 14
                if len(self.buffer) < total:
                    break
                if crc16(self.buffer[2:total - 2]) != struct.unpack_from("<H", self.buffer, total - 2)[0]:
                    del self.buffer[0]
                    self.rejected += 1
                    continue
                result.append(Frame(kind, seq, stamp, bytes(self.buffer[12:total - 2])))
                del self.buffer[:total]
                self.frames += 1
        return result


class _ImuFacade:
    """Borrowed view: start is idempotent; close does not close its owner."""

    def __init__(self, bridge):
        self.bridge = bridge

    def start(self):
        self.bridge.start()

    def read(self):
        return self.bridge.read_imu()

    def close(self):
        pass


class STM32Bridge:
    """JointIO taking ONLY the bridge configuration; public times are seconds.

    command() queues the latest target. Its original monotonic timestamp is the
    watchdog lease, including while awaiting heartbeat/ARM/target ACKs. Faults
    cannot be reset in place. Explicit close and a new bridge are required.
    """

    def __init__(self, cfg, armed=False):
        c = dict(cfg)
        if type(armed) is not bool:
            raise ValueError("armed must be bool")
        self.armed = armed
        self.port = c["port"]
        if not isinstance(self.port, str) or not self.port:
            raise ValueError("bridge.port must name a serial device")
        self.baudrate = c.get("baudrate", 115200)
        if type(self.baudrate) is not int or self.baudrate <= 0:
            raise ValueError("invalid bridge.baudrate")
        if armed:
            for key in ("verified", "imu_mount_verified", "joint_mapping_verified"):
                if c.get(key) is not True:
                    raise ValueError(f"armed STM32 requires bridge.{key}=true after commissioning")
        mapping = c.get("joint_map", list(range(12)))
        if (not isinstance(mapping, (list, tuple)) or len(mapping) != 12
                or any(type(i) is not int for i in mapping) or sorted(mapping) != list(range(12))):
            raise ValueError("bridge.joint_map must be a permutation of 0..11")
        self.mapping = np.asarray(mapping)
        for name, default in (("heartbeat_period_s", .1), ("heartbeat_timeout_s", .3),
                              ("ack_timeout_s", c.get("arm_timeout_s", .08)),
                              ("arm_timeout_s", .08), ("write_timeout_s", .02),
                              ("read_timeout_s", .005), ("poll_period_s", .002),
                              ("delivery_timeout_s", 2.), ("target_timeout_s", .08),
                              ("startup_timeout_s", 3.), ("telemetry_timeout_s", .1)):
            setattr(self, name, positive(c.get(name, default), "bridge." + name))
        self.joint_timeout_s = self.imu_timeout_s = self.telemetry_timeout_s
        if not (self.target_timeout_s <= .1 and self.heartbeat_period_s < .3
                and self.ack_timeout_s <= self.target_timeout_s
                and self.ack_timeout_s <= self.arm_timeout_s
                and self.write_timeout_s < self.ack_timeout_s
                and self.read_timeout_s < self.ack_timeout_s
                and self.poll_period_s < self.ack_timeout_s
                and self.heartbeat_period_s < self.heartbeat_timeout_s):
            raise ValueError("timing must fit firmware 100 ms target / 300 ms heartbeat deadlines")
        self.duration_ms = c.get("duration_ms", 20)
        if type(self.duration_ms) is not int or not 1 <= self.duration_ms <= 100:
            raise ValueError("bridge.duration_ms must be 1..100")
        self.delivery_version = c.get("delivery_extension_version", 0)
        if type(self.delivery_version) is not int or self.delivery_version not in (0, 1):
            raise ValueError("delivery_extension_version must be 0 (off) or proposed extension 1")
        self.imu = _ImuFacade(self)
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._finish = threading.Event()
        self._thread = self._serial = None
        self._started = self._closed = False
        self._fault = None
        self._phase = "new"
        self._parser = FrameParser()
        self._tx_seq = 0
        self._rx_seq = None
        self._remote_tick = None
        self._remote_elapsed = 0
        self._anchor = None
        self._last_rx = self._last_heartbeat = 0.
        self._pending = None
        self._arm_began = None
        self._target = None
        self._generation = self._sent_generation = 0
        self._q, self._dq, self._stamps = np.zeros(12), np.zeros(12), np.zeros(12)
        self._raw_position = [None] * 12
        self._joint_flags = [0] * 12
        self._joint_samples = [None] * 12
        self._imu_sample = None
        self._omega, self._gravity, self._imu_stamp = np.zeros(3), np.zeros(3), 0.
        self._deliveries = {}

    @property
    def fault(self):
        with self._lock:
            return self._fault

    @property
    def startup_sync_ready(self) -> bool:
        """Whether the initial STOP ACK was matched; not sensor or ARM readiness."""
        with self._lock:
            return self._anchor is not None

    def status(self) -> dict:
        """Copied JSON-safe diagnostics, also available before start/after faults.

        received_frames counts CRC-valid parser frames, including frames later
        ignored by the session. None means an unavailable age/raw measurement.
        Raw counts and flags follow the configured policy joint order; they do
        not establish calibrated feedback validity or renew any watchdog.
        """
        with self._lock:
            synced = self._anchor is not None
            return {
                "phase": "fault" if self._fault else "closed" if self._closed else self._phase,
                "startup_sync_ready": synced,
                "received_frames": self._parser.frames,
                "last_rx_age_s": max(0., time.monotonic() - self._last_rx) if synced else None,
                "fault": self._fault,
                "raw_position": [self._raw_position[i] for i in self.mapping],
                "flags": [self._joint_flags[i] for i in self.mapping],
            }

    def _latch(self, reason):
        if self._fault is None:
            self._fault = str(reason)
        self._target = None
        self._finish.set()

    def _check(self):
        if self._fault is not None:
            raise SafetyFault(self._fault)
        if not self._started or self._closed:
            raise RuntimeError("STM32Bridge is not started or is closed")

    def start(self):
        """Open serial once with bounded I/O and start its dedicated worker."""
        with self._lock:
            if self._closed or self._fault:
                raise SafetyFault(self._fault or "closed bridge cannot restart")
            if self._started:
                return
            import serial
            try:
                self._serial = serial.Serial(self.port, self.baudrate, timeout=self.read_timeout_s,
                                             write_timeout=self.write_timeout_s)
                with self._io_lock:
                    self._serial.reset_input_buffer()
                self._began = time.monotonic()
                self._phase = "sync"
                self._started = True
                self._thread = threading.Thread(target=self._run, name="stm32-bridge", daemon=True)
                self._thread.start()
            except Exception as exc:
                self._latch(f"serial start failed: {exc}")
                if self._serial is not None:
                    self._serial.close()
                raise SafetyFault(self._fault) from exc

    def read(self) -> JointState:
        with self._lock:
            self._check()
            return JointState(self._q[self.mapping].copy(), self._dq[self.mapping].copy(),
                              self._stamps[self.mapping].copy())

    def read_imu(self) -> ImuState:
        with self._lock:
            self._check()
            return ImuState(self._omega.copy(), self._gravity.copy(), self._imu_stamp)

    def _sensors_ready(self, now):
        return (np.all(self._stamps > 0) and np.all(now - self._stamps >= 0)
                and np.all(now - self._stamps <= self.joint_timeout_s)
                and self._imu_stamp > 0 and 0 <= now - self._imu_stamp <= self.imu_timeout_s)

    def command(self, targets, now):
        """Publish radians and the ORIGINAL control-cycle monotonic time, never retime."""
        with self._lock:
            self._check()
            try:
                values = np.asarray(targets, dtype=float)
                if (values.shape != (12,) or not np.isfinite(values).all()
                        or np.any(np.abs(values) > np.finfo(np.float32).max)):
                    raise ValueError("expected twelve finite float32 motor targets")
                current = time.monotonic()
                if (isinstance(now, bool) or not math.isfinite(now)
                        or not 0 <= current - now < self.target_timeout_s):
                    raise ValueError("target timestamp is future or expired")
                if self._target is not None and now <= self._target[1]:
                    raise ValueError("duplicate/out-of-order target timestamp")
                if self._anchor is None:
                    raise ValueError("startup STOP has not been acknowledged")
                if self.armed and not self._sensors_ready(current):
                    raise ValueError("joint/IMU feedback missing, invalid or stale")
                wire = np.empty(12)
                wire[self.mapping] = values
                self._generation += 1
                self._target = (wire, float(now), self._generation)
            except (ValueError, TypeError, OverflowError) as exc:
                self._latch(str(exc))
                raise SafetyFault(self._fault) from exc

    def estop(self):
        """Latch a local fault and request STOP; worker is the sole serial writer."""
        with self._lock:
            self._latch("software emergency stop")

    def delivery(self, request_id):
        """Queue proposed delivery-v1 once; base firmware raises UnsupportedFeature."""
        with self._lock:
            self._require_delivery(request_id)
            self._check()
            if request_id in self._deliveries:
                return  # idempotent within this bridge's lifetime
            now = time.monotonic()
            if (not self.armed or self._phase != "running" or self._target is None
                    or now - self._target[1] >= self.target_timeout_s):
                raise SafetyFault("delivery requires an armed session and a live control loop")
            if any(not d["complete"] for d in self._deliveries.values()):
                raise RuntimeError("a delivery request is already pending")
            if len(self._deliveries) >= 256:
                raise RuntimeError("delivery ID history full; IDs cannot be evicted safely")
            self._deliveries[request_id] = dict(sent=False, accepted=False, complete=False, began=now)

    def delivery_ack(self, request_id) -> bool:
        """True only for a matching successful completion AND ordinary request ACK."""
        with self._lock:
            self._require_delivery(request_id)
            self._check()
            value = self._deliveries.get(request_id)
            return bool(value and value["accepted"] and value["complete"])

    def _require_delivery(self, request_id):
        if not self.delivery_version:
            raise UnsupportedFeature("current STM32 v1 has no delivery; delivery-v1 needs paired firmware")
        if type(request_id) is not int or not 0 <= request_id <= 0xFFFFFFFF:
            raise ValueError("request_id must be uint32")

    def _write(self, data, deadline=None):
        end = time.monotonic() + self.write_timeout_s
        if deadline is not None:
            end = min(end, deadline)
        with self._io_lock:
            offset = 0
            while offset < len(data):
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("serial write deadline expired")
                # Partial writes share one deadline, not N separate 20 ms waits.
                self._serial.write_timeout = min(self.write_timeout_s, remaining)
                n = self._serial.write(data[offset:])
                if not isinstance(n, int) or not 0 < n <= len(data) - offset:
                    raise OSError("serial write made no progress")
                offset += n
            if time.monotonic() >= end:
                raise TimeoutError("serial write exceeded deadline")

    def _send(self, kind, payload=b"", *, context=None):
        now = time.monotonic()
        deadline = None
        if kind in (HEARTBEAT, ARM, JOINT_TARGET, DELIVERY_REQUEST) and self._target is not None:
            deadline = self._target[1] + self.target_timeout_s
            if now >= deadline:
                raise SafetyFault("target expired before transmission")
        seq = self._tx_seq
        self._tx_seq = (seq + 1) & 0xFFFF
        # Firmware requires an echoed STM32 tick, not Linux uptime or wall time.
        tick = self._remote_tick if self._remote_tick is not None else 0
        self._write(encode_frame(Frame(kind, seq, tick, payload)), deadline)
        self._pending = (kind, seq, now, context)
        if kind == HEARTBEAT:
            self._last_heartbeat = now

    def _watchdog(self, now):
        if self._anchor is None:
            if now - self._began >= self.startup_timeout_s:
                raise SafetyFault("startup STOP ACK timeout (firmware host UART may be disabled)")
        elif now - self._last_rx >= self.heartbeat_timeout_s:
            raise SafetyFault("STM32 receive/heartbeat timeout")
        if self._pending is not None and now - self._pending[2] >= self.ack_timeout_s:
            raise SafetyFault("STM32 command ACK timeout")
        if self._target is not None:
            if now - self._target[1] >= self.target_timeout_s:
                raise SafetyFault("control target watchdog expired")
            if self.armed and not self._sensors_ready(now):
                raise SafetyFault("joint/IMU feedback watchdog expired")
        if (self._arm_began is not None and self._phase != "running"
                and now - self._arm_began >= self.arm_timeout_s):
            raise SafetyFault("first ARM handshake timeout")
        for value in self._deliveries.values():
            if not (value["accepted"] and value["complete"]) and now - value["began"] >= self.delivery_timeout_s:
                raise SafetyFault("delivery completion timeout; physical outcome unknown")

    def _run(self):
        try:
            with self._lock:
                if not self._finish.is_set():
                    self._send(STOP)
            while not self._finish.is_set():
                with self._lock:
                    self._watchdog(time.monotonic())
                with self._io_lock:
                    # A size of 1024 would wait until read_timeout even for a
                    # complete 22-byte ACK. There is only one serial reader.
                    available = min(1024, self._serial.in_waiting)
                    data = self._serial.read(available) if available else b""
                now = time.monotonic()
                with self._lock:
                    if self._finish.is_set():
                        break
                    for frame in self._parser.feed(data, now):
                        self._receive(frame, now)
                    self._watchdog(time.monotonic())
                    self._transmit()
                self._finish.wait(self.poll_period_s)
        except Exception as exc:
            with self._lock:
                self._latch(f"{type(exc).__name__}: {exc}")
        finally:
            # STOP is accepted even with duplicate sequence, and resyncs C RX.
            try:
                self._write(encode_frame(Frame(STOP, self._tx_seq, self._remote_tick or 0)))
            except Exception as exc:
                with self._lock:
                    self._fault = f"{self._fault or 'shutdown'}; STOP write failed: {exc}"
            finally:
                try:
                    with self._io_lock:
                        self._serial.close()
                except Exception as exc:
                    with self._lock:
                        self._fault = f"{self._fault or 'shutdown'}; serial close failed: {exc}"

    def _transmit(self):
        if self._pending is not None or self._finish.is_set() or self._anchor is None:
            return
        now = time.monotonic()
        if self.armed and self._target is not None:
            if self._phase == "idle":
                self._phase = "heartbeat"
                self._arm_began = now
                self._send(HEARTBEAT)
                return
            if self._phase == "arm_ready":
                self._phase = "arming"
                self._send(ARM, struct.pack("<H", ALL_JOINTS))
                return
            if self._phase == "running":
                if now - self._last_heartbeat >= self.heartbeat_period_s:
                    self._send(HEARTBEAT)
                    return
                values, _, generation = self._target
                if generation != self._sent_generation:
                    self._send(JOINT_TARGET, struct.pack("<HH12f", ALL_JOINTS, self.duration_ms, *values))
                    self._sent_generation = generation
                    return
                for request_id, value in self._deliveries.items():
                    if not value["sent"]:
                        self._send(DELIVERY_REQUEST, struct.pack("<BI", 1, request_id), context=request_id)
                        value["sent"] = True
                        return
        # Shadow/idle heartbeats never ARM and cannot substitute for command().
        if now - self._last_heartbeat >= self.heartbeat_period_s:
            self._send(HEARTBEAT)

    def _receive(self, frame, arrival):
        sizes = {JOINT_STATUS: 290, IMU_STATUS: 41, HEIGHT_SCAN: 46, FAULT_STATUS: 76, ACK: 8}
        if self.delivery_version:
            sizes[DELIVERY_ACK] = 6
        if frame.type not in sizes:
            return
        if len(frame.payload) != sizes[frame.type]:
            raise SafetyFault("invalid STM32 payload length")
        if self._anchor is None:
            if frame.type != ACK or self._pending is None:
                return
            kind, code, sequence, faults = ACK_RECORD.unpack(frame.payload)
            if (kind, sequence) != self._pending[:2] or kind != STOP:
                return
            if code or faults & ~1:
                raise SafetyFault(f"startup STOP rejected/faulted: ack={code}, faults=0x{faults:x}")
            # ACK was assembled after our STOP was sent. Using SEND time minus
            # one tick gives a conservative anchor, including the full RTT.
            self._anchor = self._pending[2] - .001
            self._remote_tick = frame.timestamp_ms
            self._rx_seq = frame.sequence
            self._last_rx = arrival
            self._pending = None
            self._phase = "idle"
            return
        if not _newer(frame.sequence, self._rx_seq, 16):
            return
        delta = (frame.timestamp_ms - self._remote_tick) & 0xFFFFFFFF
        if delta >= 0x80000000:
            raise SafetyFault("STM32 tick moved backwards/rebooted")
        self._remote_elapsed += delta
        self._remote_tick = frame.timestamp_ms
        self._rx_seq = frame.sequence
        mapped = self._anchor + self._remote_elapsed / 1000.
        if mapped > arrival:
            # Clock correction may only age samples; never rejuvenate them.
            self._anchor -= mapped - arrival + .001
            mapped = arrival - .001
        # A backlog of old telemetry must not keep receive health alive.
        if arrival - mapped <= self.heartbeat_timeout_s:
            self._last_rx = arrival
        if frame.type == ACK:
            self._ack(frame)
        elif frame.type == JOINT_STATUS:
            self._joints(frame, mapped)
        elif frame.type == IMU_STATUS:
            self._imu(frame, mapped)
        elif frame.type == FAULT_STATUS:
            faults, mask, _ = struct.unpack_from("<IHH", frame.payload)
            self._check_remote_faults(faults)
            if (self._phase == "running" and mask != ALL_JOINTS) or (self._phase == "idle" and mask):
                raise SafetyFault("unexpected STM32 armed mask")
        elif frame.type == DELIVERY_ACK:
            version, request_id, status = struct.unpack("<BIB", frame.payload)
            if version != 1 or status not in (0, 1, 2):
                raise SafetyFault("invalid delivery-v1 completion")
            value = self._deliveries.get(request_id)
            if value and value["sent"]:
                if status == 2:
                    raise SafetyFault("delivery mechanism reported failure")
                if status == 1:
                    value["complete"] = True
        # HEIGHT_SCAN is consumed for sequence continuity only. ToF is host-owned.

    def _check_remote_faults(self, faults):
        # RobotSession_Init/STOP sets bit 0; only the FIRST ARM may clear it.
        allowed = 1 if self._phase != "running" else 0
        if faults & ~allowed:
            raise SafetyFault(f"STM32 fault bitmap 0x{faults:08x}")

    def _ack(self, frame):
        kind, code, sequence, faults = ACK_RECORD.unpack(frame.payload)
        self._check_remote_faults(faults)
        if self._pending is None or (kind, sequence) != self._pending[:2]:
            return
        if code:
            raise SafetyFault(f"STM32 rejected type {kind}: ACK code {code}")
        if kind == ARM:
            if faults:
                raise SafetyFault("ARM acknowledged with active faults")
            self._phase = "running"
        elif kind == HEARTBEAT and self._phase == "heartbeat":
            self._phase = "arm_ready"
        elif kind == DELIVERY_REQUEST:
            self._deliveries[self._pending[3]]["accepted"] = True
        self._pending = None

    @staticmethod
    def _sample_time(frame, tick, mapped):
        age = (frame.timestamp_ms - tick) & 0xFFFFFFFF
        if age >= 0x80000000:
            raise SafetyFault("sensor acquisition tick lies in the future")
        return mapped - age / 1000.

    @staticmethod
    def _fresh_sample(previous, tick, count):
        if previous is None:
            return count != 0
        return _newer(tick, previous[0], 32) and _newer(count, previous[1], 32)

    def _joints(self, frame, mapped):
        polling, = struct.unpack_from("<H", frame.payload)
        for i in range(12):
            flags, bus, raw, tick, count, q, dq, errors = JOINT_RECORD.unpack_from(frame.payload, 2 + i * 24)
            if flags & ~7 or bus > 3 or polling & ~ALL_JOINTS:
                raise SafetyFault("invalid joint status flags/bus/mask")
            self._raw_position[i] = raw
            self._joint_flags[i] = flags
            if flags != 7 or not polling & (1 << i):
                self._stamps[i] = 0.
                continue
            if not math.isfinite(q) or not math.isfinite(dq):
                raise SafetyFault("nonfinite calibrated joint feedback")
            stamp = self._sample_time(frame, tick, mapped)
            if self._fresh_sample(self._joint_samples[i], tick, count):
                self._q[i], self._dq[i], self._stamps[i] = q, dq, stamp
                self._joint_samples[i] = (tick, count)

    def _imu(self, frame, mapped):
        valid, tick, count, *values = IMU_RECORD.unpack(frame.payload)
        if valid not in (0, 1):
            raise SafetyFault("invalid IMU valid flag")
        if not valid:
            self._imu_stamp = 0.
            return
        if not np.isfinite(values).all():
            raise SafetyFault("nonfinite IMU feedback")
        roll, pitch = values[6:]
        if abs(roll) > math.pi + 1e-6 or abs(pitch) > math.pi / 2 + 1e-6:
            raise SafetyFault("invalid IMU roll/pitch")
        stamp = self._sample_time(frame, tick, mapped)
        if self._fresh_sample(self._imu_sample, tick, count):
            self._omega = np.asarray(values[3:6])
            self._gravity = np.array([math.sin(pitch), -math.sin(roll) * math.cos(pitch),
                                      -math.cos(roll) * math.cos(pitch)])
            self._imu_stamp = stamp
            self._imu_sample = (tick, count)

    def close(self):
        """Stop worker, attempt STOP, close device; never silently hide shutdown failure."""
        with self._lock:
            self._closed = True
            self._target = None
            self._finish.set()
            worker = self._thread
        if worker is not None:
            worker.join(timeout=max(1., self.write_timeout_s * 4))
            if worker.is_alive():
                with self._lock:
                    self._latch("serial worker did not stop; firmware watchdog/physical stop required")
                raise SafetyFault(self._fault)
        if self._fault:
            raise SafetyFault(self._fault)
