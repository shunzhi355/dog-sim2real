"""Wire compatibility and fail-closed lifecycle tests; no physical serial I/O."""

import math
import json
from pathlib import Path
import struct
import threading
import time

import numpy as np
import pytest
import yaml

from dog3588.safety import SafetyFault
from dog3588.transport import (
    ACK, ARM, DELIVERY_ACK, DELIVERY_REQUEST, FAULT_STATUS, HEARTBEAT,
    IMU_STATUS, JOINT_STATUS, JOINT_TARGET, STOP, Frame, FrameParser,
    STM32Bridge, UnsupportedFeature, crc16, encode_frame,
)


@pytest.fixture(autouse=True)
def high_resolution_monotonic(monkeypatch):
    # Python 3.11 on Windows may give monotonic() 15.6 ms resolution. The real
    # deployment is Linux; use the monotonic high-resolution clock for 50 Hz tests.
    monkeypatch.setattr(time, "monotonic", time.perf_counter)


def reference_packet(kind, seq, tick, payload=b""):
    """Literal port of RobotLink_Encode/Crc, independent of production helpers."""
    body = bytes([1, kind]) + len(payload).to_bytes(2, "little")
    body += seq.to_bytes(2, "little") + tick.to_bytes(4, "little") + payload
    crc = 0xFFFF
    for byte in body:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x1021 if crc & 0x8000 else 0)) & 0xFFFF
    return b"\xa5\x5a" + body + crc.to_bytes(2, "little")


def joint_payload(tick, count=1, flags=7, position=None):
    result = struct.pack("<H", 0xFFF)
    for i in range(12):
        # Includes a signed negative raw count: do not accidentally decode u16.
        result += struct.pack("<BBhIIffI", flags, i // 3, -10, tick, count,
                              i / 100 if position is None else position, .2, 0)
    return result


def imu_payload(tick, count=1, valid=1, roll=.2, pitch=.3):
    return struct.pack("<BII8f", valid, tick, count, 0, 0, 9.81, .1, .2, .3, roll, pitch)


def settings(**kwargs):
    return dict(port="FAKE", verified=True, imu_mount_verified=True, joint_mapping_verified=True,
                **kwargs)


class FakeSerial:
    """Partial-write serial + relevant RobotSession semantics and telemetry layouts."""

    def __init__(self):
        self.opened = self.closed = False
        self.rx = bytearray()
        self.frames = []
        self.parser = FrameParser()
        self.active = threading.Lock()
        self.data_lock = threading.RLock()
        self.io_threads = set()
        self.start_time = time.monotonic()
        self.next_status = 0.
        self.sequence = 0
        self.count = 0
        self.faults = 1
        self.armed = self.heartbeat = False
        self.host_sequence = None
        self.drop = set()
        self.reject = {}
        self.status_enabled = True
        self.freeze_samples = False
        self.joint_flags = 7
        self.last_sample = None
        self.write_chunk = 4096
        self.read_chunk = 1024
        self.break_read = self.break_write = False
        self.zero_write = False
        self.extension = False
        self.options = None
        self.read_sizes = []
        self.ack_delay_s = 0.
        self.delayed_acks = []

    def factory(self, port, baudrate, **kwargs):
        self.opened = True
        self.options = (port, baudrate, kwargs)
        return self

    def tick(self):
        return (1_000_000 + int((time.monotonic() - self.start_time) * 1000)) & 0xFFFFFFFF

    def reset_input_buffer(self):
        with self.data_lock:
            self.rx.clear()

    def emit(self, kind, payload, tick=None):
        with self.data_lock:
            self.rx.extend(reference_packet(kind, self.sequence, self.tick() if tick is None else tick, payload))
            self.sequence = (self.sequence + 1) & 0xFFFF

    def _enter(self):
        assert self.active.acquire(blocking=False), "serial read/write/close overlapped"
        self.io_threads.add(threading.get_ident())

    @property
    def in_waiting(self):
        self._enter()
        try:
            if self.break_read:
                raise OSError("unplugged")
            with self.data_lock:
                now = time.monotonic()
                while self.delayed_acks and self.delayed_acks[0][0] <= now:
                    _, payload = self.delayed_acks.pop(0)
                    self.emit(ACK, payload)
                if self.status_enabled and now >= self.next_status:
                    self.next_status = now + .008
                    if not self.freeze_samples or self.last_sample is None:
                        self.count += 1
                        self.last_sample = self.tick()
                    self.emit(JOINT_STATUS, joint_payload(self.last_sample, self.count, flags=self.joint_flags))
                    self.emit(IMU_STATUS, imu_payload(self.last_sample, self.count))
                    payload = struct.pack("<IHH", self.faults, 0xFFF if self.armed else 0, 0xFFF) + bytes(68)
                    self.emit(FAULT_STATUS, payload)
                return len(self.rx)
        finally:
            self.active.release()

    def read(self, size):
        self._enter()
        try:
            with self.data_lock:
                assert size <= len(self.rx), "would wait for unnecessary bytes"
                self.read_sizes.append(size)
                n = min(size, self.read_chunk, len(self.rx))
                result = bytes(self.rx[:n])
                del self.rx[:n]
                return result
        finally:
            self.active.release()

    def write(self, data):
        self._enter()
        try:
            if self.break_write:
                raise OSError("TX broken")
            if self.zero_write:
                return 0
            n = min(len(data), self.write_chunk)
            with self.data_lock:
                for frame in self.parser.feed(data[:n]):
                    self.frames.append(frame)
                    self._handle(frame)
            return n
        finally:
            self.active.release()

    def _handle(self, f):
        code = 0
        if f.type == STOP:
            self.armed, self.heartbeat, self.faults = False, False, self.faults | 1
            self.host_sequence = None
        elif f.type not in (HEARTBEAT, ARM, JOINT_TARGET, DELIVERY_REQUEST):
            code = 1
        elif (self.host_sequence is not None
              and not 0 < ((f.sequence - self.host_sequence) & 0xFFFF) < 0x8000):
            code = 2
        else:
            self.host_sequence = f.sequence
            if f.type == HEARTBEAT:
                self.heartbeat = True
            elif f.type in (ARM, JOINT_TARGET) and ((self.tick() - f.timestamp_ms) & 0xFFFFFFFF) > 100:
                code = 3
            elif f.type == ARM:
                assert f.payload == b"\xff\x0f"
                if self.armed or not self.heartbeat:
                    code = 5
                else:
                    self.armed, self.faults = True, 0
            elif f.type == JOINT_TARGET:
                assert len(f.payload) == 52
                if not self.armed:
                    code = 4
            elif f.type == DELIVERY_REQUEST and not self.extension:
                code = 1
        if f.type in self.reject:
            code = self.reject[f.type]
        if f.type not in self.drop:
            payload = struct.pack("<BBHI", f.type, code, f.sequence, self.faults)
            if self.ack_delay_s:
                self.delayed_acks.append((time.monotonic() + self.ack_delay_s, payload))
            else:
                self.emit(ACK, payload)

    def close(self):
        self._enter()
        try:
            self.closed = True
        finally:
            self.active.release()


def until(predicate, timeout=1.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.001)
    assert predicate(), "condition did not become true"


@pytest.fixture
def rig(monkeypatch):
    instances = []

    def create(armed=False, **kwargs):
        port = FakeSerial()
        monkeypatch.setattr("serial.Serial", port.factory)
        bridge = STM32Bridge(settings(**kwargs), armed=armed)
        instances.append((bridge, port))
        return bridge, port

    yield create
    for bridge, port in instances:
        try:
            bridge.close()
        except SafetyFault:
            pass
        if bridge._started:
            assert port.closed
            assert not bridge._thread.is_alive()


def ready(bridge):
    bridge.start()
    until(lambda: bridge._anchor is not None and np.all(bridge.read().stamps > 0)
          and bridge.read_imu().stamp > 0)


def types(port):
    return [f.type for f in port.frames]


def drive_until(bridge, predicate):
    last = 0.

    def tick():
        nonlocal last
        # Observe terminal outcomes before publishing another target. Holding
        # the state lock makes that check atomic with command(), so an expected
        # rejection cannot race into an unrelated command() exception.
        with bridge._lock:
            if predicate():
                return True
            now = time.monotonic()
            if now - last >= .02:
                bridge.command(np.zeros(12), now)
                last = now
            return False

    until(tick)


def bare_bridge():
    b = STM32Bridge(settings())
    b._started = True
    b._anchor, b._remote_tick, b._rx_seq = 100., 1000, 0
    b._phase = "idle"
    return b


def test_crc_known_vector_and_exact_c_wire_layout():
    assert crc16(b"123456789") == 0x29B1
    for payload in (b"", b"\xff\x0f", bytes(range(256)) + bytes(128)):
        f = Frame(ARM, 0x1234, 0x78563412, payload)
        wire = encode_frame(f)
        assert wire == reference_packet(f.type, f.sequence, f.timestamp_ms, payload)
        assert wire[:12] == b"\xa5\x5a\x01\x04" + struct.pack("<H", len(payload)) + bytes.fromhex("34 12 12 34 56 78")
        assert FrameParser().feed(wire) == [f]
    with pytest.raises(ValueError):
        encode_frame(Frame(1, 0, 0, bytes(385)))


def test_parser_fragment_noise_crc_version_length_and_timeout():
    p = FrameParser()
    f = Frame(IMU_STATUS, 9, 17, imu_payload(16))
    good = encode_frame(f)
    broken = good[:-1] + bytes([good[-1] ^ 0x80])
    wrong_version = b"\xa5\x5a\x02\x01" + bytes(10)
    oversized = b"\xa5\x5a\x01\x01\xff\xff" + bytes(8)
    assert p.feed(b"noise" * 1000 + broken + wrong_version + oversized, now=1.) == []
    result = []
    for value in good:
        result.extend(p.feed(bytes([value]), now=1.))
        assert len(p.buffer) <= 398
    assert result == [f] and p.rejected >= 3
    assert p.feed(good[:13], now=2.) == []
    assert p.feed(good, now=2.101) == [f]
    assert p.timeouts >= 1
    assert p.feed(good + good, now=2.102) == [f, f]


def test_current_deploy_config_bridge_only():
    path = Path(__file__).resolve().parents[1] / "config" / "robot-deploy.yaml"
    if not path.exists():
        pytest.skip("integration profile not present")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))["bridge"]
    b = STM32Bridge(cfg)
    assert b.target_timeout_s == .08 and b.write_timeout_s == .02
    assert b.joint_timeout_s == cfg["telemetry_timeout_s"]
    assert b._serial is None


def test_lifecycle_probe_facade_shadow_and_partial_writes(rig):
    b, port = rig()
    port.write_chunk = 3
    assert not port.opened
    with pytest.raises(RuntimeError):
        b.read()
    ready(b)
    b.start()
    b.imu.start()
    assert port.options[1] == 115200
    assert port.options[2] == {"timeout": .005, "write_timeout": .02, "exclusive": True}
    initial = b.read()
    initial.position[:] = 99
    assert b.read().position[0] != 99
    assert np.allclose(b.imu.read().omega, [.1, .2, .3])
    for _ in range(8):
        b.command(np.zeros(12), time.monotonic())
        time.sleep(.01)
    assert ARM not in types(port) and JOINT_TARGET not in types(port)
    b.imu.close()
    assert not port.closed
    b.close()
    b.close()
    assert port.closed and types(port)[0] == STOP and types(port)[-1] == STOP
    assert len(port.io_threads) == 1
    with pytest.raises(SafetyFault):
        b.start()


def test_first_command_arms_after_ack_maps_targets_and_original_lease_expires(rig):
    b, port = rig(armed=True, joint_map=list(reversed(range(12))))
    ready(b)
    assert ARM not in types(port)
    assert b.read().position[0] == pytest.approx(.11)
    b.command(np.arange(12) / 10, time.monotonic())
    until(lambda: JOINT_TARGET in types(port))
    sent = next(f for f in port.frames if f.type == JOINT_TARGET)
    mask, duration, *values = struct.unpack("<HH12f", sent.payload)
    assert (mask, duration) == (0xFFF, 20)
    assert values == pytest.approx(list(reversed(np.arange(12) / 10)))
    assert types(port).index(HEARTBEAT) < types(port).index(ARM) < types(port).index(JOINT_TARGET)
    assert 0 <= ((port.tick() - sent.timestamp_ms) & 0xFFFFFFFF) <= 100
    # Background telemetry/ACK/heartbeat must not renew this single command.
    until(lambda: port.closed)
    assert "target watchdog" in b.fault
    assert types(port).count(JOINT_TARGET) == 1 and types(port)[-1] == STOP
    with pytest.raises(SafetyFault):
        b.command(np.zeros(12), time.monotonic())


def test_probe_never_arms_or_demands_first_command(rig):
    b, port = rig(armed=True, startup_timeout_s=.025)
    ready(b)
    time.sleep(.06)
    assert b.fault is None and ARM not in types(port)


def test_status_reports_uncalibrated_link_without_making_feedback_armable(rig):
    b, port = rig(armed=True)
    port.joint_flags = 1  # Raw counts only; no calibrated angle/velocity validity.
    before = b.status()
    assert before["phase"] == "new" and before["received_frames"] == 0
    assert before["last_rx_age_s"] is None and not b.startup_sync_ready
    json.dumps(before, allow_nan=False)
    b.start()
    until(lambda: b.startup_sync_ready and b.status()["raw_position"][0] is not None)
    status = b.status()
    assert status["phase"] == "idle" and status["startup_sync_ready"]
    assert status["received_frames"] > 0 and status["fault"] is None
    assert math.isfinite(status["last_rx_age_s"]) and status["last_rx_age_s"] >= 0
    assert status["raw_position"] == [-10] * 12 and status["flags"] == [1] * 12
    json.dumps(status, allow_nan=False)
    assert not np.any(b.read().stamps) and ARM not in types(port)
    status["raw_position"][0] = 999
    status["flags"][0] = 7
    assert b.status()["raw_position"][0] == -10 and b.status()["flags"][0] == 1
    with pytest.raises(SafetyFault, match="feedback missing, invalid or stale"):
        b.command(np.zeros(12), time.monotonic())
    until(lambda: port.closed)
    assert ARM not in types(port) and JOINT_TARGET not in types(port)
    assert b.status()["phase"] == "fault" and b.status()["fault"] == b.fault
    json.dumps(b.status(), allow_nan=False)


@pytest.mark.parametrize("kind,code", [(ARM, 5), (JOINT_TARGET, 6)])
def test_negative_ack_latches_and_stops(rig, kind, code):
    b, port = rig(armed=True)
    port.reject[kind] = code
    ready(b)
    b.command(np.zeros(12), time.monotonic())
    until(lambda: port.closed)
    assert f"ACK code {code}" in b.fault and types(port)[-1] == STOP
    if kind == ARM:
        assert JOINT_TARGET not in types(port)


def test_missing_arm_ack_never_transmits_target(rig):
    b, port = rig(armed=True)
    port.drop.add(ARM)
    ready(b)
    b.command(np.zeros(12), time.monotonic())
    until(lambda: port.closed)
    assert JOINT_TARGET not in types(port) and types(port)[-1] == STOP
    assert "timeout" in b.fault or "watchdog" in b.fault


def test_disabled_uart_no_response_fails_closed(rig):
    b, port = rig()
    port.status_enabled = False
    port.drop.add(STOP)
    b.start()
    until(lambda: port.closed)
    assert "ACK timeout" in b.fault and types(port) == [STOP, STOP]


def test_delayed_startup_and_arm_acks_within_deadline(rig):
    b, port = rig(armed=True)
    port.ack_delay_s = .015
    ready(b)
    assert time.monotonic() - b.read_imu().stamp >= .015
    drive_until(b, lambda: JOINT_TARGET in types(port))
    assert b.fault is None
    assert port.read_sizes and max(port.read_sizes) < 1024


def test_startup_ack_after_deadline_is_not_retried_or_armed(rig):
    b, port = rig(armed=True)
    port.ack_delay_s = .12
    b.start()
    until(lambda: port.closed)
    assert "ACK timeout" in b.fault
    assert types(port) == [STOP, STOP]


def test_stale_sensor_watchdog_while_main_loop_and_link_remain_alive(rig):
    b, port = rig(armed=True, telemetry_timeout_s=.045)
    ready(b)
    drive_until(b, lambda: JOINT_TARGET in types(port))
    port.freeze_samples = True
    deadline = time.monotonic() + .5
    while b.fault is None and time.monotonic() < deadline:
        try:
            b.command(np.zeros(12), time.monotonic())
        except SafetyFault:
            break
        time.sleep(.015)
    until(lambda: port.closed)
    assert "feedback" in b.fault
    assert types(port)[-1] == STOP


def test_receive_watchdog_without_heartbeat_reply(rig):
    b, port = rig()
    ready(b)
    port.status_enabled = False
    # Exercise the independent RX watchdog, without a pending command ACK.
    with b._lock:
        b._last_rx = time.monotonic() - b.heartbeat_timeout_s - .01
    until(lambda: port.closed)
    assert "receive/heartbeat timeout" in b.fault


def test_unsolicited_or_wrong_ack_never_releases_pending_arm():
    b = bare_bridge()
    b._phase = "arming"
    b._pending = (ARM, 10, 100., None)
    b._receive(Frame(ACK, 1, 1000, struct.pack("<BBHI", ARM, 0, 9, 0)), 100.01)
    assert b._pending[1] == 10 and b._phase == "arming"
    b._receive(Frame(ACK, 2, 1000, struct.pack("<BBHI", ARM, 0, 10, 0)), 100.02)
    assert b._pending is None and b._phase == "running"


@pytest.mark.parametrize("failure", ["break_read", "break_write", "zero_write"])
def test_serial_errors_fail_closed_and_shutdown_failure_visible(rig, failure):
    b, port = rig()
    ready(b)
    setattr(port, failure, True)
    until(lambda: port.closed)
    assert b.fault
    if failure != "break_read":
        assert "STOP write failed" in b.fault
    with pytest.raises(SafetyFault):
        b.read_imu()


@pytest.mark.parametrize("bad", [np.zeros(11), np.full(12, np.nan), np.full(12, np.inf), np.full(12, 1e40)])
def test_invalid_targets_latch_even_in_shadow(rig, bad):
    b, port = rig()
    ready(b)
    with pytest.raises(SafetyFault):
        b.command(bad, time.monotonic())
    until(lambda: port.closed)
    assert JOINT_TARGET not in types(port)


@pytest.mark.parametrize("offset", [-1., 1., float("nan")])
def test_expired_future_nonfinite_target_time(rig, offset):
    b, port = rig()
    ready(b)
    with pytest.raises(SafetyFault):
        b.command(np.zeros(12), time.monotonic() + offset)


def test_duplicate_target_time_cannot_keep_control_alive(rig):
    b, port = rig()
    ready(b)
    stamp = time.monotonic()
    b.command(np.zeros(12), stamp)
    with pytest.raises(SafetyFault, match="duplicate"):
        b.command(np.zeros(12), stamp)


def test_sensor_sample_age_does_not_follow_packet_arrival_or_new_sequence():
    b = bare_bridge()
    b._receive(Frame(JOINT_STATUS, 1, 1000, joint_payload(950)), 100.02)
    b._receive(Frame(IMU_STATUS, 2, 1000, imu_payload(950)), 100.02)
    original, imu = b.read(), b.read_imu()
    assert original.stamps == pytest.approx(np.full(12, 99.95))
    assert imu.stamp == pytest.approx(99.95)
    b._receive(Frame(JOINT_STATUS, 3, 1100, joint_payload(950)), 100.8)
    b._receive(Frame(IMU_STATUS, 4, 1100, imu_payload(950)), 100.8)
    assert np.array_equal(b.read().stamps, original.stamps)
    assert b.read_imu().stamp == imu.stamp
    assert b._last_rx == 100.02  # delayed backlog doesn't refresh receive watchdog
    # Even incrementing the sample count without a new acquisition tick cannot refresh it.
    b._receive(Frame(IMU_STATUS, 5, 1101, imu_payload(950, count=2)), 100.8)
    assert b.read_imu().stamp == imu.stamp


def test_duplicate_out_of_order_rejected_wraps_and_reboot():
    b = bare_bridge()
    b._rx_seq, b._remote_tick = 65534, 0xFFFFFFFE
    b._receive(Frame(IMU_STATUS, 65535, 0xFFFFFFFF, imu_payload(0xFFFFFFFF)), 100.01)
    b._receive(Frame(IMU_STATUS, 0, 1, imu_payload(1, 2)), 100.02)
    stamp = b.read_imu().stamp
    assert stamp == pytest.approx(100.003)
    for seq in (0, 65535, 0x8000):
        b._receive(Frame(IMU_STATUS, seq, 2, imu_payload(2, 3)), 100.03)
        assert b.read_imu().stamp == stamp
    with pytest.raises(SafetyFault, match="backwards/rebooted"):
        b._receive(Frame(IMU_STATUS, 1, 0, imu_payload(0, 3)), 100.04)


def test_invalid_flags_do_not_fabricate_valid_samples_and_gravity_is_body_frame():
    b = bare_bridge()
    b._receive(Frame(JOINT_STATUS, 1, 1000, joint_payload(990, flags=3)), 100.02)
    assert not np.any(b.read().stamps)
    b._receive(Frame(IMU_STATUS, 2, 1000, imu_payload(990)), 100.02)
    g = b.read_imu().gravity
    assert g == pytest.approx([math.sin(.3), -math.sin(.2) * math.cos(.3), -math.cos(.2) * math.cos(.3)])
    assert np.linalg.norm(g) == pytest.approx(1.)
    b._receive(Frame(IMU_STATUS, 3, 1001, imu_payload(990, valid=0)), 100.02)
    assert b.read_imu().stamp == 0
    b._receive(Frame(IMU_STATUS, 4, 1002, imu_payload(990)), 100.02)
    assert b.read_imu().stamp == 0  # same previously invalidated sample cannot recover


@pytest.mark.parametrize("frame", [
    Frame(JOINT_STATUS, 1, 1000, joint_payload(990, position=float("nan"))),
    Frame(IMU_STATUS, 1, 1000, imu_payload(1001)),
    Frame(IMU_STATUS, 1, 1000, imu_payload(990, roll=float("inf"))),
    Frame(IMU_STATUS, 1, 1000, b"short"),
    Frame(FAULT_STATUS, 1, 1000, struct.pack("<I", 8) + bytes(72)),
])
def test_malformed_semantics_and_fault_bits_rejected(frame):
    with pytest.raises(SafetyFault):
        bare_bridge()._receive(frame, 100.02)


def test_estop_and_remote_fault_latch(rig):
    b, port = rig()
    ready(b)
    port.faults = 8
    until(lambda: port.closed)
    assert "fault bitmap" in b.fault and types(port)[-1] == STOP
    with pytest.raises(SafetyFault):
        b.read()


def test_local_estop_is_terminal(rig):
    b, port = rig(armed=True)
    ready(b)
    b.estop()
    until(lambda: port.closed)
    assert types(port)[-1] == STOP and ARM not in types(port)


def test_delivery_base_protocol_explicitly_unsupported(rig):
    b, port = rig()
    with pytest.raises(UnsupportedFeature):
        b.delivery(42)
    with pytest.raises(UnsupportedFeature):
        b.delivery_ack(42)
    assert not port.opened


def test_delivery_extension_ack_is_not_completion_and_ids_are_idempotent(rig):
    b, port = rig(armed=True, delivery_extension_version=1)
    port.extension = True
    ready(b)
    b.command(np.zeros(12), time.monotonic())
    drive_until(b, lambda: JOINT_TARGET in types(port))
    b.delivery(42)
    drive_until(b, lambda: b._deliveries[42]["accepted"])
    assert not b.delivery_ack(42)
    port.emit(DELIVERY_ACK, struct.pack("<BIB", 1, 43, 1))
    port.emit(DELIVERY_ACK, struct.pack("<BIB", 1, 42, 0))
    time.sleep(.005)
    assert not b.delivery_ack(42)
    port.emit(DELIVERY_ACK, struct.pack("<BIB", 1, 42, 1))
    drive_until(b, lambda: b.delivery_ack(42))
    b.delivery(42)
    assert types(port).count(DELIVERY_REQUEST) == 1


def test_proposed_extension_rejected_by_current_firmware(rig):
    b, port = rig(armed=True, delivery_extension_version=1)
    ready(b)
    drive_until(b, lambda: JOINT_TARGET in types(port))
    b.delivery(7)
    # This test isolates firmware FORMAT rejection, not the main-loop watchdog:
    # continue real command() calls until the worker reports its terminal fault.
    drive_until(b, lambda: b.fault is not None)
    until(lambda: port.closed)
    assert b.fault == f"SafetyFault: STM32 rejected type {DELIVERY_REQUEST}: ACK code 1"
    assert types(port).count(DELIVERY_REQUEST) == 1
    assert types(port)[-1] == STOP


def test_delivery_timeout_fault_even_when_main_loop_keeps_commanding(rig):
    b, port = rig(armed=True, delivery_extension_version=1, delivery_timeout_s=.035)
    port.extension = True
    ready(b)
    b.command(np.zeros(12), time.monotonic())
    until(lambda: JOINT_TARGET in types(port))
    b.delivery(7)
    deadline = time.monotonic() + .5
    while b.fault is None and time.monotonic() < deadline:
        try:
            b.command(np.zeros(12), time.monotonic())
        except SafetyFault:
            break
        time.sleep(.008)
    until(lambda: port.closed)
    assert "delivery completion timeout" in b.fault


@pytest.mark.parametrize("change", [
    {"target_timeout_s": .11}, {"read_timeout_s": float("inf")},
    {"heartbeat_period_s": .3}, {"joint_map": [0] * 12}, {"duration_ms": 101},
    {"delivery_extension_version": 2}, {"poll_period_s": 0},
])
def test_invalid_configuration(change):
    with pytest.raises(ValueError):
        STM32Bridge(settings(**change))


def test_arm_requires_all_commissioning_flags():
    for key in ("verified", "imu_mount_verified", "joint_mapping_verified"):
        cfg = settings()
        cfg[key] = False
        with pytest.raises(ValueError, match=key):
            STM32Bridge(cfg, armed=True)
