from pathlib import Path
from types import SimpleNamespace
import struct
import time

import numpy as np
import pytest

from dog3588.config import load_config, validate_bridge_hardware
from dog3588.policy import load_manifest
from dog3588.transport import (STM32Bridge, Frame, ACK, ARM, JOINT_TARGET,
    JOINT_STATUS, FAULT_STATUS, IMU_STATUS, STOP, POSE_CAPS, POSE_ARM, POSE_TARGET)
from dog3588.types import ImuState
from dog3588.safety import SafetyFault
from test_transport import FakeSerial, until

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def precise_clock(monkeypatch):
    monkeypatch.setattr(time, 'monotonic', time.perf_counter)


def profile():
    cfg = load_config(ROOT/'config/model4497-test.yaml')
    cfg['command']['port'] = 0
    cfg['bridge']['port'] = 'FAKE'
    return cfg, validate_bridge_hardware(cfg, load_manifest(cfg['policy_manifest'])['joint_names'], arm=True)


def orientation():
    return ImuState(np.zeros(3), np.array([0., 0., -1.]), time.monotonic())


class RawFirmware(FakeSerial):
    def __init__(self, calibrations):
        super().__init__()
        self.raw = [int(c.raw_zero) for c in calibrations]
        self.previous_target = self.raw.copy()
        self.supported = True

    def emit(self, kind, payload, tick=None):
        if kind == IMU_STATUS:
            return  # Existing board has no IMU telemetry.
        if kind == JOINT_STATUS:
            data = bytearray(payload)
            for i in range(12):
                struct.pack_into('<BBh', data, 2+24*i, 1, i//3, self.raw[i])
                struct.pack_into('<ff', data, 14+24*i, float('nan'), float('nan'))
            payload = bytes(data)
        elif kind == FAULT_STATUS:
            data = bytearray(payload)
            struct.pack_into('<H', data, 6, 0)  # No MCU angle calibration.
            payload = bytes(data)
        super().emit(kind, payload, tick)

    def _handle(self, f):
        if f.type not in (POSE_CAPS, POSE_ARM, POSE_TARGET):
            return super()._handle(f)
        code = 0
        if not self.supported:
            code = 1
        elif self.host_sequence is not None and not 0 < ((f.sequence-self.host_sequence)&65535) < 32768:
            code = 2
        elif f.type != POSE_CAPS and ((self.tick()-f.timestamp_ms)&0xffffffff) > 100:
            code = 3
        elif f.type == POSE_ARM:
            assert f.payload == b'\xff\x0f'
            if self.armed or not self.heartbeat:
                code = 5
            else:
                self.armed, self.faults = True, 0
                self.previous_target = self.raw.copy()
        elif f.type == POSE_TARGET:
            assert len(f.payload) == 28
            mask, duration, *raw = struct.unpack('<HH12H', f.payload)
            assert mask == 4095 and 1 <= duration <= 100
            assert self.armed and all(0 <= q <= 1000 for q in raw)
            assert max(abs(a-b) for a,b in zip(raw, self.previous_target)) <= 10
            assert max(abs(a-b) for a,b in zip(raw, self.raw)) <= 40
            self.raw = self.previous_target = raw
        self.host_sequence = f.sequence
        self.emit(ACK, struct.pack('<BBHI', f.type, code, f.sequence, self.faults))


def test_real_policy_runs_over_raw_counts_with_host_imu(monkeypatch, tmp_path):
    from dog3588.runtime import run
    cfg, calibrations = profile()
    fake = RawFirmware(calibrations)
    monkeypatch.setattr('serial.Serial', fake.factory)
    monkeypatch.setattr('dog3588.runtime.ThreadedImu', lambda _:
        SimpleNamespace(start=lambda: None, close=lambda: None, read=orientation))
    result = run(cfg, hardware=True, arm=True, seconds=.28, log_path=tmp_path/'policy.jsonl')
    kinds = [f.type for f in fake.frames]
    assert result['ticks'] >= 5 and fake.closed
    assert kinds[0] == kinds[-1] == STOP
    assert kinds.index(POSE_CAPS) < kinds.index(POSE_ARM) < kinds.index(POSE_TARGET)
    assert ARM not in kinds and JOINT_TARGET not in kinds
    assert any(a != int(c.raw_zero) for a,c in zip(fake.raw, calibrations))


def test_shadow_never_arms_raw_firmware(monkeypatch):
    from dog3588.runtime import run
    cfg, calibrations = profile()
    fake = RawFirmware(calibrations)
    monkeypatch.setattr('serial.Serial', fake.factory)
    monkeypatch.setattr('dog3588.runtime.ThreadedImu', lambda _:
        SimpleNamespace(start=lambda: None, close=lambda: None, read=orientation))
    result = run(cfg, hardware=True, arm=False, seconds=.14)
    assert result['ticks'] >= 3
    assert not any(f.type in (ARM, JOINT_TARGET, POSE_ARM, POSE_TARGET) for f in fake.frames)


def test_missing_caps_fails_before_enable(monkeypatch):
    cfg, calibrations = profile()
    fake = RawFirmware(calibrations); fake.supported = False
    monkeypatch.setattr('serial.Serial', fake.factory)
    bridge = STM32Bridge(dict(cfg['bridge'], imu_mount_verified=True), armed=True,
                         imu_reader=orientation, raw_calibrations=calibrations)
    bridge.start()
    until(lambda: bridge.fault is not None)
    with pytest.raises(SafetyFault): bridge.close()
    assert not any(f.type in (POSE_ARM, POSE_TARGET) for f in fake.frames)


def payload(raw, tick, count, mask=4095, flags=1):
    return struct.pack('<H', mask) + b''.join(struct.pack('<BBhIIffI', flags, i//3,
        int(r), tick, count, float('nan'), float('nan'), 0) for i,r in enumerate(raw))


def raw_bridge():
    cfg, calibrations = profile()
    return STM32Bridge(cfg['bridge'], imu_reader=orientation, raw_calibrations=calibrations), calibrations


def test_feedback_units_two_samples_counter_freeze_and_recovery():
    bridge, c = raw_bridge(); raw = [v.raw_zero for v in c]
    bridge._joints(Frame(JOINT_STATUS,1,100,payload(raw,100,1)),10.)
    assert np.all(bridge._stamps == 0)
    moved = [r+v.direction for r,v in zip(raw,c)]
    frame = Frame(JOINT_STATUS,2,120,payload(moved,120,2))
    bridge._joints(frame,10.02)
    np.testing.assert_allclose(bridge._q, 1/c[0].units_per_rad)
    np.testing.assert_allclose(bridge._dq, 1/c[0].units_per_rad/.02)
    bridge._joints(frame,11.)
    np.testing.assert_allclose(bridge._stamps,10.02)  # Duplicate cannot renew lease.
    bridge._joints(Frame(JOINT_STATUS,3,140,payload(moved,140,3,mask=0)),10.04)
    assert np.all(bridge._stamps==0)
    bridge._joints(Frame(JOINT_STATUS,4,160,payload(moved,160,4)),10.06)
    assert np.all(bridge._stamps==0)
    bridge._joints(Frame(JOINT_STATUS,5,180,payload(moved,180,5)),10.08)
    np.testing.assert_allclose(bridge._stamps,10.08)
    np.testing.assert_allclose(bridge._dq,0)


def test_target_conversion_bounds_steps_and_tracking(monkeypatch):
    bridge,c = raw_bridge();zero=np.array([v.raw_zero for v in c],dtype=int)
    bridge._raw_position=zero.tolist();bridge._last_sent_raw=zero.copy()
    frame=struct.unpack('<HH12H',bridge._target_payload(np.full(12,.02)))
    assert frame[:2]==(4095,20)
    assert list(frame[2:]) == [int(v.raw_zero)+5*v.direction for v in c]
    # Missed intermediate setpoints must slew safely, not jump to the newest.
    bridge._last_target_sent_at=10.
    monkeypatch.setattr(time, 'monotonic', lambda: 10.021)
    prior=bridge._last_sent_raw.copy()
    delayed=struct.unpack('<HH12H',bridge._target_payload(np.full(12,.2)))[2:]
    assert max(abs(a-b) for a,b in zip(delayed,prior)) <= 5
    with pytest.raises(ValueError,match='limits'): bridge._target_payload(np.ones(12))
    bridge._last_sent_raw=zero.copy();bridge._raw_position=(zero-36).tolist()
    with pytest.raises(SafetyFault,match='tracking'): bridge._target_payload(np.zeros(12))
