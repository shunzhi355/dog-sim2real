import copy
import json
import os
import struct

import pytest

from dog3588.hardware.feedback import FeedbackPoller, Timing, run_feedback
from dog3588.hardware.lx824 import POS_READ, packet


class Port:
    def __init__(self, chunk=6):
        self.chunk = chunk
        self.out_waiting = 0
        self.rx = bytearray()
        self.writes = bytearray()

    def read(self, count):
        data = bytes(self.rx[:count])
        del self.rx[:count]
        return data

    def write(self, data):
        count = min(len(data), self.chunk)
        self.writes.extend(data[:count])
        return count


def make_poller(cfg, port=None):
    port = Port() if port is None else port
    return FeedbackPoller(port, "fl", cfg["joints"][:3]), port


def answer(poller, port, now, raw, sid=None):
    sid = poller.current_id if sid is None else sid
    port.rx.extend(packet(sid, POS_READ, struct.pack("<h", raw)))
    return poller.poll(now)


def test_query_exact_bytes_partial_write_and_signed_early_response(cfg):
    p, port = make_poller(cfg, Port(chunk=2))
    p.poll(0.)
    p.poll(.0001)
    p.poll(.0002)
    assert bytes(port.writes) == bytes.fromhex("55 55 01 03 1c df")
    assert p.phase == "TRANSMITTING"
    answer(p, port, .0003, -50)
    assert p.phase == "TRANSMITTING"
    p.poll(.001)
    snap = p.snapshot(.001)
    assert snap["successful"] == 1
    j = snap["joints"][0]
    assert j["raw_position"] == -50 and j["raw_valid"]
    assert j["timestamp_s"] == .0003
    assert not j["feedback_valid"] and not j["velocity_valid"]
    assert j["position_rad"] is None


def test_one_pending_id_timeout_quiet_and_next_id(cfg):
    p, port = make_poller(cfg)
    p.poll(0.)
    p.poll(.001)
    p.poll(.006)
    assert p.stats["requested"] == 1 and p.current_id == 1
    assert p.phase == "RECEIVING"
    p.poll(.0111)
    assert p.phase == "QUIET" and p.stats["rx_timeouts"] == 1
    p.poll(.012)
    assert p.stats["requested"] == 1
    p.poll(.0132)
    assert p.current_id == 2
    assert bytes(port.writes) == packet(1, POS_READ) + packet(2, POS_READ)


def test_bad_frames_wrong_id_echo_and_fragmentation(cfg):
    p, port = make_poller(cfg)
    p.poll(0.)
    p.poll(.001)
    good = packet(1, POS_READ, b"\xf4\x01")
    bad = good[:-1] + bytes([good[-1] ^ 1])
    port.rx.extend(b"noise" + bad + packet(2, POS_READ, b"\x00\x00")
                   + packet(1, POS_READ) + good[:4])
    p.poll(.002)
    assert p.stats["successful"] == 0
    port.rx.extend(good[4:])
    p.poll(.003)
    assert p.stats["successful"] == 1
    assert p.stats["bad_frames"] >= 1
    assert p.stats["rejected_frames"] == 2
    assert p.snapshot(.003)["joints"][0]["raw_position"] == 500


def test_success_quiet_period_and_age_invalidates_without_destroying_value(cfg):
    p, port = make_poller(cfg)
    p.poll(0.)
    p.poll(.001)
    answer(p, port, .002, 500)
    p.poll(.0059)
    assert p.stats["requested"] == 1
    p.poll(.0061)
    assert p.stats["requested"] == 2 and p.current_id == 2
    j = p.snapshot(.103)["joints"][0]
    assert j["raw_position"] == 500 and j["sample_count"] == 1
    assert not j["raw_valid"] and j["age_ms"] > 100


def test_calibrated_velocity_actual_time_and_error_invalidation(cfg):
    joint = dict(cfg["joints"][0], verified=True, raw_zero=500,
                 direction=1, units_per_rad=200., zero_rad=0., min_rad=-1., max_rad=1.)
    port = Port()
    p = FeedbackPoller(port, "fl", [joint])
    p.poll(0.)
    p.poll(.001)
    answer(p, port, .002, 500)
    assert not p.snapshot(.002)["joints"][0]["velocity_valid"]
    p.poll(.0061)
    p.poll(.007)
    answer(p, port, .009, 502)
    j = p.snapshot(.009)["joints"][0]
    assert j["feedback_valid"] and j["velocity_valid"]
    assert j["interval_ms"] == pytest.approx(7.)
    assert j["velocity_rad_s"] == pytest.approx(.01 / .007)
    p.poll(.0122)
    p.poll(.013)
    p.poll(.024)
    j = p.snapshot(.024)["joints"][0]
    assert not j["raw_valid"] and not j["velocity_valid"]
    assert j["raw_position"] == 502 and j["error_count"] == 1
    p.poll(.027)
    p.poll(.028)
    answer(p, port, .029, 504)
    assert p.snapshot(.029)["joints"][0]["raw_valid"]
    assert not p.snapshot(.029)["joints"][0]["velocity_valid"]


def test_stuck_tx_is_fatal_and_does_not_start_next_query(cfg):
    p, port = make_poller(cfg)
    port.out_waiting = 6
    p.poll(0.)
    with pytest.raises(TimeoutError, match="transmit did not drain"):
        p.poll(.011)
    assert p.stats["tx_timeouts"] == 1
    assert bytes(port.writes) == packet(1, POS_READ)


def test_transport_failure_invalidates_sample_and_stops(cfg):
    p, port = make_poller(cfg)
    p.poll(0.)
    def broken_read(count):
        raise OSError("disconnected")
    port.read = broken_read
    with pytest.raises(OSError, match="disconnected"):
        p.poll(.001)
    assert p.stats["transport_errors"] == 1
    assert not p.snapshot(.001)["joints"][0]["raw_valid"]
    assert p.events[-1]["error"] == "transport_errors"


def test_independent_buses(cfg):
    a, pa = make_poller(cfg)
    pb = Port()
    b = FeedbackPoller(pb, "fr", cfg["joints"][3:6])
    a.poll(0.); b.poll(0.)
    a.poll(.001); b.poll(.001)
    answer(b, pb, .002, 700)
    a.poll(.0111)
    assert a.stats["rx_timeouts"] == 1 and b.stats["successful"] == 1
    assert b.snapshot(.002)["joints"][0]["id"] == 4


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_timing(value):
    with pytest.raises(ValueError):
        Timing(rx_timeout_s=value)


def test_cli_rejects_unknown_bus_and_nonfinite_duration(cfg):
    with pytest.raises(ValueError):
        run_feedback(cfg, buses="missing", seconds=1)
    with pytest.raises(ValueError):
        run_feedback(cfg, buses="fl", seconds=float("nan"))


@pytest.mark.skipif(os.name != "posix", reason="Linux real pyserial PTY integration")
def test_linux_pty_query_only_feedback(cfg, tmp_path, capsys):
    import pty
    import select
    import threading
    from dog3588.hardware.lx824 import FrameParser

    master, slave = pty.openpty()
    settings = copy.deepcopy(cfg)
    settings["buses"][0]["port"] = os.ttyname(slave)
    stop = threading.Event()
    seen = []
    def emulate():
        parser = FrameParser()
        while not stop.is_set():
            if not select.select([master], [], [], .01)[0]:
                continue
            for sid, cmd, params in parser.feed(os.read(master, 256)):
                seen.append((sid, cmd, params))
                os.write(master, packet(sid, cmd) + packet(20, POS_READ, b"\x00\x00")
                         + packet(sid, POS_READ, struct.pack("<h", 500 + sid)))
    thread = threading.Thread(target=emulate, daemon=True)
    thread.start()
    try:
        log = tmp_path / "feedback.jsonl"
        result = run_feedback(settings, buses="fl", seconds=.2, log_path=log, report_s=.05)
        bus = result["buses"][0]
        assert bus["successful"] >= 6 and bus["rx_timeouts"] == 0
        assert [sid for sid, _, _ in seen[:6]] == [1, 2, 3, 1, 2, 3]
        assert all(cmd == POS_READ and not params for _, cmd, params in seen)
        assert all(j["raw_valid"] and not j["feedback_valid"] for j in bus["joints"])
        events = [json.loads(line) for line in log.read_text().splitlines()]
        assert any(e["event"] == "result" and "raw_position" in e for e in events)
        assert '"event": "stopped"' in capsys.readouterr().out
    finally:
        stop.set()
        thread.join(timeout=1)
        os.close(master)
        os.close(slave)


@pytest.mark.skipif(os.name != "posix", reason="Linux SIGTERM/PTY cleanup integration")
def test_linux_continuous_monitor_sigterm_closes_port(cfg, tmp_path):
    import pty
    import signal
    import subprocess
    import sys
    import time
    import yaml
    import serial

    master, slave = pty.openpty()
    config = copy.deepcopy(cfg)
    config["buses"][0]["port"] = os.ttyname(slave)
    config_path, log = tmp_path / "local.yaml", tmp_path / "events.jsonl"
    config_path.write_text(yaml.safe_dump(config))
    process = subprocess.Popen([sys.executable, "-m", "dog3588", "--config", str(config_path),
                                "feedback-servos", "--bus", "fl", "--seconds", "0",
                                "--log", str(log)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if log.exists() and '"event": "result"' in log.read_text():
                break
            if process.poll() is not None:
                pytest.fail("monitor exited before SIGTERM")
            time.sleep(.01)
        else:
            pytest.fail("monitor did not log a transaction")
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=3)
        assert not stderr
        assert json.loads(stdout.splitlines()[-1])["event"] == "stopped"
        # No valid hardware feedback: CLI correctly returns nonzero despite clean stop.
        assert process.returncode == 1
        with serial.Serial(os.ttyname(slave), 115200, timeout=0, exclusive=True):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=3)
        os.close(master)
        os.close(slave)
