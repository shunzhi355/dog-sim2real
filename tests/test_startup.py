"""Startup sequencing with synthetic calibration and a virtual clock; no hardware."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dog3588.commands import CommandReceiver, encode_command
from dog3588.config import load_config
from dog3588.safety import SafetyFault, SafetyGuard
from dog3588.startup import execute_startup, load_startup_plan
from dog3588.types import JointState, ImuState


@pytest.fixture
def setup(cfg, manifest, calibrations, tmp_path, monkeypatch):
    c = load_config(Path(__file__).resolve().parents[1] / 'config/model4497.yaml')['startup']
    c.update(trajectory_verified=True, initial_hold_s=.1, rise_duration_s=.4,
             max_speed_rad_s=1., settle_duration_s=.1, settle_timeout_s=.5)
    for name, q in (('prone', -.4), ('standing', 0.)):
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(dict(joint_names=manifest['joint_names'],
                                      model_positions_verified=True,
                                      model_joint_positions_rad=[q] * 12)))
        c[name + '_pose'] = str(path)
    cfg['startup'] = c
    plan = load_startup_plan(cfg, manifest, calibrations)
    clock = SimpleNamespace(value=10.)
    def sleep(dt):
        clock.value += max(dt, .000001)
    fake_time = SimpleNamespace(monotonic=lambda: clock.value, sleep=sleep)
    monkeypatch.setattr('dog3588.startup.time', fake_time)
    class Joints:
        def __init__(self):
            self.q = plan.prone.copy()
            self.velocity = np.zeros(12)
            self.calls = []
            self.closed = False
            self.follow = True
            self.stale = False
        def start(self): pass
        def read(self):
            return JointState(self.q.copy(), self.velocity.copy(), np.full(12, clock.value - (1 if self.stale else 0)))
        def command(self, q, stamp):
            self.calls.append((stamp, q.copy()))
            if self.follow:
                self.q = q.copy()
        def close(self): self.closed = True
    class Imu:
        gravity = np.array([0., 0., -1.])
        def start(self): pass
        def read(self): return ImuState(np.zeros(3), self.gravity.copy(), clock.value)
        def close(self): pass
    commands = CommandReceiver(cfg['command'], manifest['velocity_ranges'])
    return SimpleNamespace(cfg=cfg, manifest=manifest, limits=calibrations, plan=plan,
                           clock=clock, time=fake_time, joints=Joints(), imu=Imu(),
                           commands=commands, guard=SafetyGuard(cfg['safety'], manifest, calibrations))


def execute(s, **kwargs):
    return execute_startup(s.plan, s.joints, s.imu, s.guard, s.commands, period=.02,
                           stopping=lambda: False, **kwargs)


def test_hold_smooth_rise_settle_and_command_reset(setup):
    s = setup
    s.commands.velocity[:] = [.2, 0, 0]
    s.commands.stamp = s.clock.value
    events = []
    execute(s, emit=events.append)
    assert [e['phase'] for e in events if e['event'] == 'startup_phase'] == ['holding_prone', 'rising', 'settling']
    assert events[-1]['event'] == 'startup_complete'
    assert events[0]['rise_duration_s'] == pytest.approx(.75)  # automatic speed-based extension
    stamps = np.array([t for t, _ in s.joints.calls])
    targets = np.array([q for _, q in s.joints.calls])
    np.testing.assert_allclose(targets[stamps < 10.1], -.4)
    np.testing.assert_allclose(targets[-1], 0, atol=1e-10)
    assert np.all(np.diff(targets, axis=0) >= -1e-12)
    assert np.max(np.abs(np.diff(targets, axis=0)) / np.diff(stamps)[:, None]) <= 1. + 1e-9
    assert stamps[-1] >= 10.1 + .75 + .1
    assert not s.commands.read(s.clock.value).any()
    assert s.commands.velocity_not_before == s.clock.value


@pytest.mark.parametrize('fault, message', [
    ('wrong_pose', 'prone pose'), ('stale', 'stale'), ('tilt', 'tilt'),
    ('estop', 'emergency stop'), ('stuck', 'tracking'), ('unsettled', 'settle'),
])
def test_fault_prevents_handoff(setup, fault, message):
    s = setup
    if fault == 'wrong_pose': s.joints.q[:] = 0
    if fault == 'stale': s.joints.stale = True
    if fault == 'tilt': s.imu.gravity = np.array([1., 0, 0])
    if fault == 'estop': s.commands.estop = True
    if fault == 'stuck': s.joints.follow = False
    if fault == 'unsettled': s.joints.velocity[:] = .2
    events = []
    with pytest.raises(SafetyFault, match=message): execute(s, emit=events.append)
    assert not any(e['event'] == 'startup_complete' for e in events)
    if fault in ('wrong_pose', 'stale', 'tilt', 'estop'):
        assert not s.joints.calls


def test_estop_during_rise_stops_before_next_target(setup):
    s = setup
    def read(now):
        s.commands.estop = now > 10.3
        return np.zeros(3)
    s.commands.read = read
    with pytest.raises(SafetyFault, match='emergency stop'): execute(s)
    assert s.joints.calls[-1][0] <= 10.3


def test_computation_stall_does_not_publish_retimed_target(setup):
    s = setup
    def poll(orientation, now):
        s.clock.value += .06
        return s.clock.value
    perception = SimpleNamespace(poll=poll, require_ready=lambda now: None)
    with pytest.raises(SafetyFault, match='deadline'): execute(s, perception=perception)
    assert not s.joints.calls


@pytest.mark.parametrize('fault, message', [
    ('unverified_path', 'trajectory_verified'), ('raw_only', 'raw counts'),
    ('order', 'joint order'), ('nan', 'finite'), ('limit', 'limits'),
    ('handoff', 'startup tolerance'),
])
def test_bad_plan_rejected_before_io(setup, fault, message):
    s = setup
    path = Path(s.cfg['startup']['standing_pose'])
    pose = json.loads(path.read_text())
    if fault == 'unverified_path': s.cfg['startup']['trajectory_verified'] = False
    if fault == 'raw_only': pose['model_joint_positions_rad'] = None
    if fault == 'order': pose['joint_names'].reverse()
    if fault == 'nan': pose['model_joint_positions_rad'][0] = float('nan')
    if fault == 'limit': pose['model_joint_positions_rad'][0] = 2.
    if fault == 'handoff': pose['model_joint_positions_rad'][0] = .3
    path.write_text(json.dumps(pose))
    with pytest.raises(ValueError, match=message): load_startup_plan(s.cfg, s.manifest, s.limits)
    assert not s.joints.calls


def test_captured_raw_poses_cannot_be_mistaken_for_angles(setup):
    s = setup
    cfg = load_config(Path(__file__).resolve().parents[1] / 'config/model4497.yaml')
    with pytest.raises(ValueError, match='trajectory_verified'):
        load_startup_plan(cfg, s.manifest, s.limits)
    cfg['startup']['trajectory_verified'] = True
    with pytest.raises(ValueError, match='raw counts'):
        load_startup_plan(cfg, s.manifest, s.limits)


@pytest.mark.parametrize('stuck', [False, True])
def test_runtime_infers_only_after_standing_and_closes_on_failure(setup, monkeypatch, tmp_path, stuck):
    import dog3588.runtime as runtime
    s = setup
    s.joints.follow = not stuck
    # Replace only device/policy boundaries. Exercise real runtime and startup loop.
    monkeypatch.setattr(runtime, 'time', SimpleNamespace(**vars(s.time), perf_counter=lambda: s.clock.value))
    monkeypatch.setattr(runtime, 'validate_hardware', lambda *a, **kw: s.limits)
    monkeypatch.setattr(runtime, 'FourLegIO', lambda *a, **kw: s.joints)
    monkeypatch.setattr(runtime, 'ThreadedImu', lambda *a: s.imu)
    monkeypatch.setattr('dog3588.deployment.PerceptionSupervisor', lambda *a, **kw: None)
    s.commands.start = lambda: None
    monkeypatch.setattr(runtime, 'CommandReceiver', lambda *a: s.commands)
    inferred = []
    def infer(obs):
        inferred.append((s.clock.value, s.joints.q.copy(), obs.copy()))
        return np.zeros(12)
    monkeypatch.setattr(runtime, 'OnnxPolicy', lambda *a: SimpleNamespace(infer=infer))
    log = tmp_path / 'control.jsonl'
    if stuck:
        with pytest.raises(SafetyFault, match='tracking'):
            runtime.run(s.cfg, hardware=True, arm=True, seconds=.1, log_path=log)
        assert not inferred
    else:
        result = runtime.run(s.cfg, hardware=True, arm=True, seconds=.1, log_path=log)
        assert result['ticks'] >= 4
        assert inferred[0][0] >= 10.95
        np.testing.assert_allclose(inferred[0][1], 0, atol=1e-10)
        records = [json.loads(line) for line in log.read_text().splitlines()]
        complete = next(i for i, r in enumerate(records) if r['event'] == 'startup_complete')
        assert all(i > complete for i, r in enumerate(records) if r['event'] == 'tick')
    assert s.joints.closed


def test_pre_standing_command_cannot_arrive_late_and_start_walking(setup, monkeypatch):
    s = setup
    monkeypatch.setattr('dog3588.commands.time', s.time)
    old = encode_command([.2, 0, 0])
    s.clock.value += .05
    s.commands.discard_velocity(s.clock.value)
    s.commands.accept(old, s.clock.value)
    assert not s.commands.read(s.clock.value).any()
    s.clock.value += .02
    s.commands.accept(encode_command([.2, 0, 0]), s.clock.value)
    assert s.commands.read(s.clock.value)[0] == .2


def test_discarding_motion_does_not_discard_delayed_estop(setup, monkeypatch):
    s = setup
    monkeypatch.setattr('dog3588.commands.time', s.time)
    stop = encode_command([0, 0, 0], estop=True)
    s.clock.value += .05
    s.commands.discard_velocity(s.clock.value)
    s.commands.accept(stop, s.clock.value)
    assert s.commands.estop
