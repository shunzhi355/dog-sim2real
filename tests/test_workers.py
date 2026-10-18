import time

import pytest

from dog3588.hardware.joints import LegWorker, FourLegIO
from dog3588.hardware.lx824 import FrameParser, LOAD_WRITE, MOVE_TIME_WRITE


class FakeBus:
    def __init__(self, cfg):
        self.writes, self.stops, self.closed = [], [], False
    def send(self, data): self.writes.append(data)
    def read_position(self, sid): return 500, time.monotonic()
    def stop(self, ids): self.stops.append(ids)
    def close(self): self.closed = True


def test_worker_watchdog_stops_and_latches(cfg, calibrations):
    bus = FakeBus({})
    worker = LegWorker(cfg["buses"][0], calibrations[:3], armed=True,
                       target_timeout_s=.04, duration_ms=20, bus_factory=lambda _: bus)
    worker.submit([500] * 3, time.monotonic())
    worker.start()
    worker.thread.join(timeout=.8)
    assert not worker.thread.is_alive()
    assert "watchdog" in worker.error
    assert bus.stops == [[1, 2, 3]] and bus.closed
    commands = [cmd for data in bus.writes for _, cmd, _ in FrameParser().feed(data)]
    assert commands.count(LOAD_WRITE) == 3
    assert commands.count(MOVE_TIME_WRITE) == 3


def test_shadow_does_not_load_move_or_stop(cfg, calibrations):
    bus = FakeBus({})
    worker = LegWorker(cfg["buses"][0], calibrations[:3], armed=False,
                       target_timeout_s=.04, duration_ms=20, bus_factory=lambda _: bus)
    worker.submit([700] * 3, time.monotonic())
    worker.start()
    time.sleep(.03)
    worker.request_stop()
    worker.join()
    assert bus.writes == [] and bus.stops == [] and bus.closed


def test_partial_leg_start_failure_rolls_back_existing_worker(cfg, calibrations, monkeypatch):
    io = FourLegIO(cfg, calibrations, armed=False, control_hz=50)
    bus = FakeBus({})
    first = io.groups[0][1]
    first.bus_factory = lambda _: bus
    def fail_start():
        raise OSError('cannot start second worker')
    monkeypatch.setattr(io.groups[1][1], 'start', fail_start)
    with pytest.raises(OSError, match='second worker'):
        io.start()
    assert bus.closed and not first.thread.is_alive()
    assert bus.writes == [] and bus.stops == []
    io.close()  # unstarted workers do not cause join-before-start failures

