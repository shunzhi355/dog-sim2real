import json
import math
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dog3588.commands import CommandReceiver
from dog3588.hardware.imu import WitParser, GravityFilter, ThreadedImu
from dog3588.vision import LineDetector, line_velocity, detect_colours


def wit_frame(kind, values):
    frame = bytes([0x55, kind]) + struct.pack("<hhhh", *values, 0)
    return frame + bytes([sum(frame) & 255])


def test_wit_fragmentation_units_and_component_freshness():
    parser = WitParser()
    gyro = wit_frame(0x52, [16384, 0, 0])
    parser.feed(b"bad" + gyro[:6], 10)
    sample = parser.feed(gyro[6:], 10)
    assert sample.stamp == 0  # no angle data yet
    sample = parser.feed(wit_frame(0x53, [0, 8192, 0]), 11)
    assert sample.stamp == 10
    np.testing.assert_allclose(sample.omega, [1000 * math.pi / 180, 0, 0])
    np.testing.assert_allclose(sample.gravity, [math.sqrt(.5), 0, -math.sqrt(.5)], atol=1e-6)
    sample = parser.feed(gyro, 12)
    assert sample.stamp == 11
    corrupted = bytearray(wit_frame(0x53, [0, 0, 0]))
    corrupted[-1] ^= 1
    assert parser.feed(corrupted, 13).stamp == 11


def test_gravity_filter_stationary_and_body_rotation():
    f = GravityFilter()
    np.testing.assert_allclose(f.update(np.array([0., 0., 9.80665]), np.zeros(3), .005), [0, 0, -1])
    tilted = f.update(np.zeros(3), np.array([0., 1., 0.]), .01)
    assert tilted[0] > 0  # positive pitch: downward world gravity has positive body X


def test_mpu6050_registers_scaling_and_publish(cfg, monkeypatch):
    settings = dict(cfg["imu"], driver="mpu6050_i2c", gyro_bias_rad_s=[math.pi / 18, 0, 0])
    device = ThreadedImu(settings)
    writes, samples = [], []
    class Bus:
        def __init__(self, number): assert number == settings["i2c_bus"]
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read_byte_data(self, address, register): return 0x68 if register == 0x75 else 1
        def write_byte_data(self, address, register, value): writes.append((register, value))
        def read_i2c_block_data(self, address, register, count):
            assert register == 0x3B and count == 14
            return list(struct.pack(">7h", 0, 0, 8192, 0, 164, 0, 0))
    monkeypatch.setitem(sys.modules, "smbus2", SimpleNamespace(SMBus=Bus))
    monkeypatch.setattr(device.finish, "wait", lambda _: False)
    def publish(sample):
        samples.append(sample)
        device.finish.set()
    monkeypatch.setattr(device, "_publish", publish)
    device._i2c()
    assert writes == [(0x6B, 1), (0x1A, 3), (0x19, 4), (0x1B, 0x18), (0x1C, 0x08)]
    np.testing.assert_allclose(samples[0].omega, 0, atol=1e-12)
    np.testing.assert_allclose(samples[0].gravity, [0, 0, -1])
    assert samples[0].stamp > 0


def test_mpu6050_wrong_chip_is_rejected_before_configuration(cfg, monkeypatch):
    class WrongChip:
        def __init__(self, _): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read_byte_data(self, *_): return 0x71
        def write_byte_data(self, *_): pytest.fail("must not configure an unknown chip")
    monkeypatch.setitem(sys.modules, "smbus2", SimpleNamespace(SMBus=WrongChip))
    with pytest.raises(ValueError, match="WHO_AM_I"):
        ThreadedImu(dict(cfg["imu"], driver="mpu6050_i2c"))._i2c()


def test_command_expiry_nan_reordering_source_selection_estop(cfg, manifest):
    receiver = CommandReceiver(cfg["command"], manifest["velocity_ranges"])
    def push(stamp=10., source="manual", values=None, estop=False):
        receiver.accept(json.dumps({"version": 1, "source": source, "sent_monotonic": stamp,
                                    "velocity": [1., 0, 2.] if values is None else values, "estop": estop}).encode(), 10.1)
    push()
    np.testing.assert_allclose(receiver.read(10.1), [.3, 0, .4])
    np.testing.assert_allclose(receiver.read(10.4), 0)
    push(stamp=9.9, values=[.1, 0, 0])
    push(stamp=10.05, values=[float("nan"), 0, 0])
    assert receiver.invalid_packets == 2
    push(stamp=10.06, source="vision", values=[0, 0, 0])
    np.testing.assert_allclose(receiver.read(10.1), [.3, 0, .4])
    push(stamp=10.07, source="vision", estop=True)
    push(stamp=10.08)
    assert receiver.estop


def test_line_lost_turn_sign_and_colour(cfg):
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    detector = LineDetector(cfg["vision"]["line"])
    assert not detector.detect(frame)["found"]
    assert line_velocity(detector.detect(frame), cfg["vision"]["follow"]) == [0, 0, 0]
    frame[260:, 430:460] = 0
    result = detector.detect(frame)
    assert result["found"] and result["error"] > 0
    assert line_velocity(result, cfg["vision"]["follow"])[2] < 0
    frame[30:90, 30:90] = [0, 0, 255]
    assert "red" in [r["colour"] for r in detect_colours(frame, cfg["vision"]["colours"])]

