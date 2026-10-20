import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from dog3588.config import load_config
from dog3588.initialize_pose import initialize_pose, load_pose, NAMES, POSE_CAPS, POSE_ARM, POSE_TARGET
from dog3588.safety import SafetyFault
from dog3588.transport import Frame, FrameParser, encode_frame, STOP, HEARTBEAT, ACK, JOINT_STATUS


@pytest.fixture
def rig(tmp_path, monkeypatch):
    import dog3588.initialize_pose as module
    clock = SimpleNamespace(value=10.)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock.value))
    cfg = load_config(Path(__file__).resolve().parents[1] / 'config/model4497.yaml')
    commands = SimpleNamespace(estop=False, start=lambda: None, read=lambda now: np.zeros(3), close=lambda: None)
    monkeypatch.setattr(module, 'CommandReceiver', lambda *a: commands)
    pose = tmp_path / 'standing.json'
    pose.write_text(json.dumps(dict(servo_ids=list(range(1, 13)), joint_names=NAMES,
                                   units='servo_raw_count', raw_positions=list(range(500, 512)))))
    class Serial:
        def __init__(self):
            self.buffer = bytearray()
            self.sequence = 0
            self.sent = []
            self.q = np.full(12, 500)
            self.last_joint = 0
            self.support = self.follow = True
            self.closed = False
            self.invalid_feedback = False
            self.drop_ack = None
        @property
        def in_waiting(self): return len(self.buffer)
        def frame(self, kind, data):
            self.buffer.extend(encode_frame(Frame(kind, self.sequence, int(clock.value*1000), data)))
            self.sequence = (self.sequence+1) & 0xffff
        def write(self, data):
            f = FrameParser().feed(data)[0]
            self.sent.append(f)
            code = 1 if f.type == POSE_CAPS and not self.support else 0
            if f.type == POSE_TARGET and self.follow:
                self.q[:] = struct.unpack('<HH12H', f.payload)[2:]
            if f.type != self.drop_ack:
                self.frame(ACK, struct.pack('<BBHI', f.type, code, f.sequence, 0))
            return len(data)
        def read(self, n):
            clock.value += .001
            if clock.value-self.last_joint >= .02:
                payload = struct.pack('<H', 0xfff)
                for i in range(12):
                    payload += struct.pack('<BBhIIffI', 0 if self.invalid_feedback else 1, i//3,
                                           int(self.q[i]), int(clock.value*1000), 1, 0., 0., 0)
                self.frame(JOINT_STATUS, payload)
                self.last_joint = clock.value
            data = bytes(self.buffer[:n]); del self.buffer[:n]
            return data
        def close(self): self.closed = True
    serial = Serial()
    monkeypatch.setattr('serial.Serial', lambda *a, **kw: serial)
    return SimpleNamespace(cfg=cfg, pose=pose, serial=serial, clock=clock, commands=commands)


def test_initialization_uses_saved_counts_without_calibration_imu_or_policy(rig):
    s = rig
    assert not any(j['verified'] for j in s.cfg['joints'])
    result = initialize_pose(s.cfg, s.pose, arm=True, seconds=1)
    assert result['reached'] and s.serial.closed
    np.testing.assert_array_equal(s.serial.q, load_pose(s.pose))
    types = [f.type for f in s.serial.sent]
    assert types[0] == types[-1] == STOP
    assert types.index(POSE_CAPS) < types.index(POSE_ARM) < types.index(POSE_TARGET)
    assert set(types) <= {STOP, HEARTBEAT, POSE_CAPS, POSE_ARM, POSE_TARGET}
    targets = [struct.unpack('<HH12H', f.payload)[2:] for f in s.serial.sent if f.type == POSE_TARGET]
    assert np.max(np.abs(np.diff(np.array(targets), axis=0))) <= 10
    np.testing.assert_array_equal(targets[0], [500]*12)


def test_probe_does_not_enable_or_move(rig):
    result = initialize_pose(rig.cfg, rig.pose)
    assert not result['moved']
    assert [f.type for f in rig.serial.sent] == [STOP, POSE_CAPS, STOP]


def test_old_firmware_rejected_before_enabling(rig):
    rig.serial.support = False
    with pytest.raises(SafetyFault, match='does not support'):
        initialize_pose(rig.cfg, rig.pose, arm=True)
    assert all(f.type in (STOP, POSE_CAPS) for f in rig.serial.sent)
    assert rig.serial.closed


def test_missing_feedback_never_enables(rig):
    rig.serial.invalid_feedback = True
    with pytest.raises(SafetyFault, match='fresh feedback'):
        initialize_pose(rig.cfg, rig.pose, arm=True)
    assert POSE_ARM not in [f.type for f in rig.serial.sent]


def test_stuck_servo_fails_to_settle_and_stops(rig):
    rig.serial.follow = False
    with pytest.raises(SafetyFault, match='failed to reach'):
        initialize_pose(rig.cfg, rig.pose, arm=True, seconds=1)
    assert rig.serial.sent[-1].type == STOP and rig.serial.closed


def test_target_ack_timeout_stops_without_retrying_move(rig):
    rig.serial.drop_ack = POSE_TARGET
    with pytest.raises(SafetyFault, match='ACK timeout'):
        initialize_pose(rig.cfg, rig.pose, arm=True, seconds=1)
    assert sum(f.type == POSE_TARGET for f in rig.serial.sent) == 1
    assert rig.serial.sent[-1].type == STOP and rig.serial.closed


def test_emergency_stop_during_initialization(rig):
    def read(now):
        rig.commands.estop = now >= 10.6
        return np.zeros(3)
    rig.commands.read = read
    with pytest.raises(SafetyFault, match='emergency stop'):
        initialize_pose(rig.cfg, rig.pose, arm=True, seconds=1)
    assert rig.serial.sent[-1].type == STOP and rig.serial.closed


@pytest.mark.parametrize('value', [None, -1, 1001, True, 1.5])
def test_bad_position_is_rejected_before_serial_access(rig, value):
    p = json.loads(rig.pose.read_text());p['raw_positions'][-1] = value
    rig.pose.write_text(json.dumps(p))
    with pytest.raises(ValueError): initialize_pose(rig.cfg, rig.pose, arm=True)
    assert not rig.serial.sent
