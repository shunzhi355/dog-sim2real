"""Query-only, nonblocking LX824 feedback; no LOAD/MOVE and no GPIO control.

One transaction per bus. Kernel output-queue empty plus a conservative wire-time
guard is a host estimate, NOT the STM32 UART TC interrupt. External hardware must
provide automatic half-duplex direction. Keep receiving during transmit so a fast
reply buffered by Linux is not discarded at a phase transition.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import signal
import time

from ..config import Calibration, positive
from .lx824 import FrameParser, POS_READ, packet


@dataclass(frozen=True)
class Timing:
    query_period_s: float = .006
    tx_timeout_s: float = .010
    rx_timeout_s: float = .010
    quiet_s: float = .002
    max_age_s: float = .100

    def __post_init__(self):
        for name, value in vars(self).items():
            positive(value, name)


class FeedbackPoller:
    """Single-owner poll(now) state machine; time is monotonic seconds.

    Transport exposes nonblocking read/write, in_waiting and out_waiting. Calling
    poll never sleeps, drains the UART, changes direction, or writes servo state.
    Snapshots mark stale/error data invalid while retaining the last good sample.
    """

    def __init__(self, transport, bus: str, joints: list[dict], timing=Timing()):
        ids = [j["id"] for j in joints]
        if (not ids or any(type(sid) is not int or not 0 <= sid <= 253 for sid in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("feedback requires unique unicast servo IDs")
        self.transport, self.bus, self.ids, self.timing = transport, bus, ids, timing
        self.calibrations = {j["id"]: Calibration.from_dict(j)
                             for j in joints if j.get("verified") is True}
        self.samples = {j["id"]: dict(name=j["name"], id=j["id"], raw_position=None,
                                     timestamp_s=None, interval_ms=None, sample_count=0,
                                     error_count=0, last_ok=False, position_rad=None,
                                     velocity_rad_s=None, velocity_valid=False)
                        for j in joints}
        self.phase = "IDLE"
        self.stats = dict(requested=0, successful=0, rx_timeouts=0, tx_timeouts=0,
                          transport_errors=0, rejected_frames=0, bad_frames=0,
                          tx_bytes=0, rx_bytes=0)
        self.cursor = 0
        self.next_start = 0.
        self.current_id = None
        self.last_error = None
        self.last_query_hex = None
        self.events = []

    def _start(self, now):
        self.current_id = self.ids[self.cursor % len(self.ids)]
        self.cursor += 1
        self.frame = packet(self.current_id, POS_READ)
        self.last_query_hex = self.frame.hex(" ")
        self.offset = 0
        self.started = now
        self.drained_at = None
        self.early_reply = None
        self.parser = FrameParser()
        self.phase = "TRANSMITTING"
        self.stats["requested"] += 1
        self.events.append(dict(event="query", bus=self.bus, id=self.current_id,
                                timestamp_s=now, tx_hex=self.last_query_hex))

    def _finish(self, now, error=None, reply=None):
        sample = self.samples[self.current_id]
        self.last_error = error
        event = dict(event="result", bus=self.bus, id=self.current_id,
                     timestamp_s=now, latency_ms=(now - self.started) * 1000)
        if error:
            self.stats[error] += 1
            sample["error_count"] += 1
            sample["last_ok"] = sample["velocity_valid"] = False
            sample["velocity_rad_s"] = None
            event["error"] = error
        else:
            raw, stamp = reply
            previous_stamp = sample["timestamp_s"]
            dt = None if previous_stamp is None else stamp - previous_stamp
            previous_position = sample["position_rad"]
            previous_ok = sample["last_ok"]
            calibration = self.calibrations.get(self.current_id)
            # Preserve signed raw feedback. Calibration.from_raw's command-range
            # restriction is intentionally not used to clip actual measurements.
            q = (calibration.zero_rad + calibration.direction *
                 (raw - calibration.raw_zero) / calibration.units_per_rad
                 if calibration is not None else None)
            valid_velocity = (q is not None and previous_position is not None
                              and previous_ok and dt is not None
                              and 0 < dt <= self.timing.max_age_s)
            sample.update(raw_position=raw, timestamp_s=stamp,
                          interval_ms=dt * 1000 if dt is not None else None,
                          sample_count=sample["sample_count"] + 1, last_ok=True,
                          position_rad=q, velocity_valid=valid_velocity,
                          velocity_rad_s=(q - previous_position) / dt if valid_velocity else None)
            self.stats["successful"] += 1
            event.update(raw_position=raw, rx_hex=packet(
                self.current_id, POS_READ, raw.to_bytes(2, "little", signed=True)).hex(" "))
        self.events.append(event)
        self.next_start = max(self.started + self.timing.query_period_s,
                              now + self.timing.quiet_s)
        self.phase = "QUIET"

    def poll(self, now: float):
        """Advance one bounded I/O step; return query/result events for logging."""
        self.events = []
        if self.phase == "QUIET" and now >= self.next_start:
            self.phase = "IDLE"
        if self.phase == "IDLE":
            # Discard unsolicited/late input before issuing a new transaction.
            self.stats["rx_bytes"] += len(self.transport.read(4096))
            if now >= self.next_start:
                self._start(now)
        if self.phase not in ("TRANSMITTING", "RECEIVING"):
            return self.events
        try:
            if self.phase == "TRANSMITTING" and self.offset < len(self.frame):
                n = self.transport.write(self.frame[self.offset:])
                if type(n) is not int or not 0 <= n <= len(self.frame) - self.offset:
                    raise IOError("invalid UART write count")
                self.offset += n
                self.stats["tx_bytes"] += n
            data = self.transport.read(4096)
            self.stats["rx_bytes"] += len(data)
            previous_bad = self.parser.bad_frames
            for sid, command, params in self.parser.feed(data):
                if sid == self.current_id and command == POS_READ and len(params) == 2:
                    if self.early_reply is None:
                        self.early_reply = (int.from_bytes(params, "little", signed=True), now)
                else:
                    self.stats["rejected_frames"] += 1
            self.stats["bad_frames"] += self.parser.bad_frames - previous_bad
            if self.phase == "TRANSMITTING":
                if self.offset == len(self.frame) and self.transport.out_waiting == 0:
                    if self.drained_at is None:
                        self.drained_at = now
                    wire_guard = len(self.frame) * 10 / 115200
                    if now - self.drained_at >= wire_guard:
                        self.phase = "RECEIVING"
                        self.receive_started = now
                else:
                    self.drained_at = None
                if self.phase == "TRANSMITTING" and now - self.started >= self.timing.tx_timeout_s:
                    # Cannot safely assume a timed-out transmitter released DATA.
                    # Caller closes the port and stops rather than overlapping queries.
                    self._finish(now, "tx_timeouts")
                    raise TimeoutError("UART transmit did not drain; feedback stopped")
            if self.phase == "RECEIVING":
                if self.early_reply is not None:
                    self._finish(now, reply=self.early_reply)
                elif now - self.receive_started >= self.timing.rx_timeout_s:
                    self._finish(now, "rx_timeouts")
        except TimeoutError:
            raise
        except Exception:
            self._finish(now, "transport_errors")
            raise
        return self.events

    def snapshot(self, now):
        joints = []
        for sid in self.ids:
            sample = dict(self.samples[sid])
            age = None if sample["timestamp_s"] is None else max(0, now - sample["timestamp_s"])
            raw_valid = sample["last_ok"] and age is not None and age <= self.timing.max_age_s
            sample.update(age_ms=age * 1000 if age is not None else None,
                          raw_valid=raw_valid,
                          feedback_valid=raw_valid and sid in self.calibrations,
                          velocity_valid=raw_valid and sample["velocity_valid"])
            if not sample["velocity_valid"]:
                sample["velocity_rad_s"] = None
            joints.append(sample)
        return dict(bus=self.bus, phase=self.phase, current_id=self.current_id,
                    last_query_hex=self.last_query_hex, last_error=self.last_error,
                    **self.stats, joints=joints)


def run_feedback(cfg, *, buses, seconds=10., log_path=None, report_s=1.):
    """Run query-only polling, Ctrl+C/SIGTERM closes ports; seconds=0 runs forever.

    Finite runs finish their current transaction/quiet period before returning.
    Log JSONL is rotated at 2 MiB; no unbounded full-rate terminal output.
    """
    import logging
    from logging.handlers import RotatingFileHandler
    import serial

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and >= 0")
    positive(report_s, "report_s")
    selected = [b for b in cfg["buses"] if buses == "all" or b["name"] == buses]
    if not selected or len({b["port"] for b in selected}) != len(selected):
        raise ValueError("select known buses with distinct ports")
    for b in selected:
        if b["direction"] != {"mode": "auto"} or b["baudrate"] != 115200:
            raise ValueError("feedback monitor requires 115200 and direction: {mode: auto}")
    options = cfg.get("feedback", {})
    timing = Timing(**options)
    transports, pollers = [], []
    handler = None
    logger = logging.getLogger("dog3588.feedback.events")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    running = True
    def stop(signum, frame):
        nonlocal running
        running = False
    old_handlers = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, stop)
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=3,
                                          encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        for b in selected:
            port = serial.Serial(b["port"], 115200, timeout=0, write_timeout=0, exclusive=True)
            transports.append(port)
            port.reset_input_buffer()
            pollers.append(FeedbackPoller(port, b["name"],
                           [j for j in cfg["joints"] if j["bus"] == b["name"]], timing))
        began = time.monotonic()
        last_report = began
        while running:
            now = time.monotonic()
            expired = seconds > 0 and now - began >= seconds
            for poller in pollers:
                # After the duration, complete pending work but start no new query.
                if expired and poller.phase in ("IDLE", "QUIET"):
                    continue
                try:
                    events = poller.poll(now)
                except Exception as exc:
                    if handler:
                        for event in poller.events:
                            logger.info(json.dumps(dict(time_unix=time.time(), **event), allow_nan=False))
                    print(json.dumps(dict(event="fault", bus=poller.bus,
                          error=str(exc), buses=[p.snapshot(now) for p in pollers]),
                          allow_nan=False), flush=True)
                    raise
                for event in events:
                    if handler:
                        logger.info(json.dumps(dict(time_unix=time.time(), **event), allow_nan=False))
            if now - last_report >= report_s:
                print(json.dumps(dict(event="snapshot", elapsed_s=now - began,
                      buses=[p.snapshot(now) for p in pollers]), allow_nan=False), flush=True)
                last_report = now
            if expired and all(p.phase in ("IDLE", "QUIET") for p in pollers):
                break
            time.sleep(.0005)  # Scheduling granularity, not a hard real-time promise.
        now = time.monotonic()
        result = dict(event="stopped", elapsed_s=now - began,
                      buses=[p.snapshot(now) for p in pollers])
        print(json.dumps(result, allow_nan=False), flush=True)
        return result
    finally:
        for port in transports:
            port.close()  # Never send STOP/LOAD/MOVE when ending a read-only monitor.
        if handler:
            logger.removeHandler(handler)
            handler.close()
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
