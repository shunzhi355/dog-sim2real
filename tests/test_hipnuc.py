import binascii
import math
import struct
import time

import numpy as np
import pytest

from dog3588.hardware.hipnuc import HipnucParser
from dog3588.types import ImuState


def frame(payload):
    header = b'\x5a\xa5' + struct.pack('<H', len(payload))
    crc = binascii.crc_hqx(payload, binascii.crc_hqx(header, 0))
    return header + struct.pack('<H', crc) + payload


def hi91(tick=100, status=0x1800, quat=(1., 0., 0., 0.)):
    p = bytearray(76)
    struct.pack_into('<BHBfI', p, 0, 0x91, status, 20, 101325., tick)
    struct.pack_into('<3f', p, 24, 180., 0., 0.)
    struct.pack_into('<4f', p, 60, *quat)
    return frame(p)


def hi83(tick=100000, bitmap=0x3b):
    fields = [struct.pack('<3f', 0., 0., 9.80665), struct.pack('<3f', math.pi, 0., 0.),
              bytes(12), bytes(12), struct.pack('<4f', 1., 0., 0., 0.), struct.pack('<Q', tick)]
    return frame(struct.pack('<BHBI', 0x83, 0x1800, 0, bitmap)
                 + b''.join(f for bit, f in enumerate(fields) if bitmap & (1 << bit)))


def test_mixed_protocol_units_fragmentation_and_crc():
    parser = HipnucParser()
    wire = b'noise' + hi91() + hi83()
    for part in range(0, len(wire), 7):
        sample = parser.feed(wire[part:part+7], 10.)
    assert parser.counts == {0x91: 1, 0x83: 1}
    assert sample.stamp == 10.
    np.testing.assert_allclose(sample.omega, [math.pi, 0., 0.])
    np.testing.assert_allclose(sample.gravity, [0., 0., -1.])
    bad = bytearray(hi91(110)); bad[-1] ^= 1
    assert parser.feed(bad, 11.).stamp == 10.
    assert parser.crc_errors == 1
    assert parser.feed(hi91(120), 12.).stamp == 12.


def test_quaternion_gravity_and_mount_rotate_consistently():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    parser = HipnucParser(rotation)
    sample = parser.feed(hi91(quat=(math.cos(math.pi/8), 0., math.sin(math.pi/8), 0.)), 10.)
    np.testing.assert_allclose(sample.omega, [0., math.pi, 0.], atol=1e-7)
    np.testing.assert_allclose(sample.gravity, [0., math.sqrt(.5), -math.sqrt(.5)], atol=1e-7)


@pytest.mark.parametrize('packet', [hi91(status=0x1808), hi91(status=0x1820),
    hi91(status=0x1840), hi91(status=0x1880), hi91(quat=(0., 0., 0., 0.)),
    hi91(quat=(float('nan'), 0., 0., 0.)), hi83(bitmap=0x2b), hi83(bitmap=0x1b),
    frame(bytes([0x83])), frame(struct.pack('<BHBI', 0x83, 0, 0, 0x1032))])
def test_invalid_ahrs_invalidates_previous_sample(packet):
    parser = HipnucParser()
    assert parser.feed(hi91(90), 1.).stamp == 1.
    assert parser.feed(packet, 2.).stamp == 0.
    assert parser.invalid == 1


@pytest.mark.parametrize('factory', [hi91, hi83])
def test_frozen_reordered_and_missing_data_cannot_refresh_timestamp(factory):
    parser = HipnucParser()
    assert parser.feed(factory(100), 1.).stamp == 1.
    assert parser.feed(b'', 2.).stamp == 1.
    assert parser.feed(factory(100), 3.).stamp == 0.
    assert parser.feed(factory(99), 4.).stamp == 0.
    assert parser.feed(factory(101), 5.).stamp == 5.


def test_hi91_day_rollover_and_hi83_optional_fields():
    parser = HipnucParser()
    parser.feed(hi91(86399999), 1.)
    assert parser.feed(hi91(1), 2.).stamp == 2.
    assert parser.feed(hi83(bitmap=0x32), 3.).stamp == 3.


def test_partial_timeout_oversize_resync_and_bounded_garbage():
    parser = HipnucParser()
    parser.feed(hi91()[:10], 1.)
    assert parser.feed(b'x' * 10000 + b'\x5a\xa5\xff\xff\x00\x00' + hi91(), 2.).stamp == 2.
    assert parser.timeouts == 1
    assert len(parser.buffer) <= parser.MAX_PAYLOAD + 6


def test_external_imu_watchdog_requires_live_data_without_mcu_imu():
    from dog3588.transport import STM32Bridge
    from dog3588.safety import SafetyFault
    clock = time.monotonic()
    samples = [ImuState(np.zeros(3), np.array([0., 0., -1.]), clock)]
    bridge = STM32Bridge(dict(port='FAKE'), imu_reader=lambda: samples[0])
    bridge.armed = True  # Unit test watchdog; no serial device or motion.
    bridge._anchor = clock
    bridge._last_rx = clock
    bridge._target = (np.zeros(12), clock, 1)
    bridge._stamps[:] = clock
    assert bridge._imu_stamp == 0.  # No MCU IMU telemetry ever arrived.
    assert bridge._sensors_ready(clock)
    samples[0] = ImuState(np.zeros(3), np.array([0., 0., -1.]), clock - 1.)
    with pytest.raises(SafetyFault, match='feedback watchdog'):
        bridge._watchdog(clock)
    samples[0] = ImuState(np.zeros(3), np.array([0., 0., -1.]), clock + 10.)
    assert not bridge._sensors_ready(clock)


def test_host_imu_errors_cannot_fall_back_to_mcu_sample():
    from dog3588.transport import STM32Bridge
    def failed():
        raise OSError('IMU unplugged')
    bridge = STM32Bridge(dict(port='FAKE'), imu_reader=failed)
    bridge._stamps[:] = bridge._imu_stamp = time.monotonic()
    assert not bridge._sensors_ready(time.monotonic())
