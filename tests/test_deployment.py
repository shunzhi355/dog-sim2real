from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dog3588.config import load_config, validate_bridge_hardware
from dog3588.deployment import PerceptionSupervisor, check_velocity
from dog3588.mission import DeliveryRequest, MissionOutput, MissionState
from dog3588.replay import replay_mission
from dog3588.runtime import run
from dog3588.safety import SafetyFault
from dog3588.tof import TerrainState
from dog3588.types import ImuState, JointState
from dog3588.vision import VisionState

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def host_cfg():
    cfg = load_config(ROOT / 'config/robot-deploy.yaml')
    cfg['command']['port'] = 0  # OS-chosen socket; never interfere with a running robot
    return cfg


def test_unverified_host_cannot_arm_or_claim_terrain(host_cfg, manifest):
    with pytest.raises(ValueError, match='verified'):
        validate_bridge_hardware(host_cfg, manifest['joint_names'], arm=True)
    with pytest.raises(ValueError, match='calibration'):
        PerceptionSupervisor(host_cfg, manifest)
    host_cfg['tof']['enabled'] = False
    with pytest.raises(ValueError, match='tof.enabled'):
        PerceptionSupervisor(host_cfg, manifest, autonomous=True)


def test_stop_is_allowed_but_untrained_slow_motion_is_not(manifest):
    np.testing.assert_array_equal(check_velocity([0., 0., 0.], manifest), 0.)
    with pytest.raises(SafetyFault, match='trained policy range'):
        check_velocity([.03, 0., 0.], manifest)  # baseline starts at .1, zero is separate


@pytest.mark.parametrize('armed', [False, True])
def test_real_runtime_onnx_through_fake_stm32(host_cfg, monkeypatch, armed, tmp_path):
    import test_transport as wire

    fake = wire.FakeSerial()
    monkeypatch.setattr('serial.Serial', fake.factory)
    original = wire.imu_payload
    monkeypatch.setattr(wire, 'imu_payload', lambda tick, count: original(tick, count, roll=0., pitch=0.))
    host_cfg['tof']['enabled'] = False
    host_cfg['bridge'].update(port='FAKE', verified=True, imu_mount_verified=True, joint_mapping_verified=True)
    host_cfg['imu']['driver'] = 'stm32'  # This fixture emits legacy MCU IMU telemetry.
    host_cfg['startup']['enabled'] = False  # This wire fixture starts in policy stance.
    for joint in host_cfg['joints']:
        joint.update(verified=True, min_rad=-2., max_rad=2.)
    summary = run(host_cfg, hardware=True, arm=armed, seconds=.22, log_path=tmp_path / 'control.jsonl')
    assert summary['ticks'] >= 5 and fake.closed
    types = [f.type for f in fake.frames]
    assert types[0] == types[-1] == wire.STOP
    assert (wire.ARM in types) == armed
    assert (wire.JOINT_TARGET in types) == armed
    assert wire.DELIVERY_REQUEST not in types
    assert '"target"' in (tmp_path / 'control.jsonl').read_text()


def test_policy_runtime_uses_host_imu_without_mcu_imu(host_cfg, monkeypatch):
    import time
    import test_transport as wire

    fake = wire.FakeSerial()
    emit = fake.emit
    monkeypatch.setattr(fake, 'emit', lambda kind, payload, tick=None:
                        None if kind == wire.IMU_STATUS else emit(kind, payload, tick))
    monkeypatch.setattr('serial.Serial', fake.factory)
    lifecycle = []
    class HostImu:
        def __init__(self, cfg):
            assert cfg['driver'] == 'hipnuc_serial'
        def start(self): lifecycle.append('start')
        def close(self): lifecycle.append('close')
        def read(self):
            assert lifecycle == ['start']
            return ImuState(np.zeros(3), np.array([0., 0., -1.]), time.monotonic())
    monkeypatch.setattr('dog3588.runtime.ThreadedImu', HostImu)
    host_cfg['tof']['enabled'] = False
    host_cfg['startup']['enabled'] = False
    host_cfg['imu']['mount_verified'] = True
    host_cfg['bridge'].update(port='FAKE', verified=True, joint_mapping_verified=True,
                              imu_mount_verified=False)
    for joint in host_cfg['joints']:
        joint.update(verified=True, min_rad=-2., max_rad=2.)
    summary = run(host_cfg, hardware=True, arm=True, seconds=.22)
    assert summary['ticks'] >= 5 and fake.closed and lifecycle == ['start', 'close']
    kinds = [f.type for f in fake.frames]
    assert kinds[0] == kinds[-1] == wire.STOP
    assert wire.JOINT_TARGET in kinds


def test_tof_and_imu_pairing_uses_nearest_acquisition(host_cfg, manifest, monkeypatch):
    from test_tof import scene

    geometry, frame, orientation = scene()
    host_cfg['tof'].update(geometry)
    supervisor = PerceptionSupervisor(host_cfg, manifest)
    supervisor.tof = SimpleNamespace(read=lambda: frame)
    supervisor.orientation_history.append(orientation)
    # Newest IMU differs by 60 ms; older history contains the exact frame time.
    latest = replace(orientation, stamp=10.06)
    monkeypatch.setattr('dog3588.deployment.time.monotonic', lambda: 10.061)
    now = supervisor.poll(latest, 10.06)
    supervisor.require_ready(now)
    np.testing.assert_allclose(supervisor.terrain.heights, -.3)
    assert supervisor.terrain.stamp == 10.
    with pytest.raises(SafetyFault, match='expired'):
        supervisor.require_ready(10.3)


def test_delayed_imu_matches_pending_tof_without_retiming_old_height(host_cfg, manifest, monkeypatch):
    from test_tof import scene

    geometry, frame, orientation = scene()
    host_cfg['tof'].update(geometry)
    supervisor = PerceptionSupervisor(host_cfg, manifest)
    samples = [frame]
    supervisor.tof = SimpleNamespace(read=lambda: samples[0])
    clock = [10.01]
    monkeypatch.setattr('dog3588.deployment.time.monotonic', lambda: clock[0])
    supervisor.poll(orientation, clock[0])
    assert supervisor.terrain.valid.all()
    samples[0] = replace(frame, stamp=10.067)
    clock[0] = 10.075
    supervisor.poll(orientation, clock[0])  # matching MCU packet has not arrived yet
    supervisor.require_ready(clock[0])
    assert supervisor.terrain.stamp == 10.  # reuse, never refresh age
    clock[0] = 10.09
    supervisor.poll(replace(orientation, stamp=10.067), clock[0])
    assert supervisor.terrain.stamp == 10.067 and supervisor.terrain.valid.all()
    # A later bad measured frame MUST invalidate the result, not keep good heights.
    samples[0] = replace(frame, stamp=10.10, target_status=np.zeros(64, dtype=int))
    clock[0] = 10.11
    with pytest.raises(SafetyFault, match='no valid raw zones'):
        supervisor.poll(replace(orientation, stamp=10.10), clock[0])


def test_bad_raw_tof_fails_before_delayed_imu_can_match(host_cfg, manifest, monkeypatch):
    from test_tof import scene

    geometry, frame, imu = scene()
    host_cfg['tof'].update(geometry)
    supervisor = PerceptionSupervisor(host_cfg, manifest)
    samples = [frame]
    supervisor.tof = SimpleNamespace(read=lambda: samples[0])
    clock = [10.01]
    monkeypatch.setattr('dog3588.deployment.time.monotonic', lambda: clock[0])
    supervisor.poll(imu, clock[0])
    samples[0] = replace(frame, stamp=10.067, target_status=np.zeros(64, dtype=int))
    clock[0] = 10.075
    with pytest.raises(SafetyFault, match='no valid raw zones'):
        supervisor.poll(imu, clock[0])


@pytest.mark.parametrize('name,rgb', [('red', (255, 0, 0)), ('brown', (165, 42, 42)),
                                     ('brown', (120, 31, 31)), ('blue', (0, 0, 255)),
                                     ('green', (0, 255, 0)), ('yellow', (255, 255, 0))])
def test_deploy_reference_colours_do_not_turn_brown_house_into_red_junction(host_cfg, name, rgb):
    from dog3588.vision import PerceptionDetector

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[300:350, 40:600] = rgb[::-1]
    state = PerceptionDetector(host_cfg['vision']).detect(frame, 10.)
    assert [item['colour'] for item in state.colours] == [name]


def supervisor_at_delivery(cfg, manifest, output):
    cfg = deepcopy(cfg)
    cfg['tof']['enabled'] = False
    supervisor = PerceptionSupervisor(cfg, manifest)
    supervisor.terrain = TerrainState(np.zeros(8), np.ones(8, bool), 10., np.ones(8), np.zeros(8), True)
    supervisor.vision = VisionState(10., {'found': True, 'error': 0., 'heading': 0., 'confidence': 1.})
    supervisor.mission = SimpleNamespace(step=lambda sample: output)
    return supervisor


def test_shadow_delivery_never_calls_transport(host_cfg, manifest):
    request = DeliveryRequest('round:delivery:1', 'yellow')
    output = MissionOutput((0., 0., 0.), None, 'delivery', request, MissionState.DELIVERY)
    supervisor = supervisor_at_delivery(host_cfg, manifest, output)
    bridge = SimpleNamespace(armed=False, delivery=lambda _: pytest.fail('shadow must never deliver'))
    joints = JointState(np.zeros(12), np.zeros(12), np.full(12, 10.))
    imu = ImuState(np.zeros(3), np.array([0., 0., -1.]), 10.)
    supervisor.control(joints, imu, 10., np.zeros(3), host_cfg['gait'], bridge)
    assert supervisor.delivery_request == request and supervisor.delivery_wire_id is None


def test_obstacle_gait_returns_to_base_when_fsm_gait_is_none(host_cfg, manifest):
    output = MissionOutput((0., 0., 0.), None, 'follow', None, MissionState.FOLLOW)
    supervisor = supervisor_at_delivery(host_cfg, manifest, output)
    joints = JointState(np.zeros(12), np.zeros(12), np.full(12, 10.))
    imu = ImuState(np.zeros(3), np.array([0., 0., -1.]), 10.)
    previous_obstacle_gait = np.array(host_cfg['gait'])
    previous_obstacle_gait[2] = .024
    _, gait, _ = supervisor.control(joints, imu, 10., np.zeros(3), previous_obstacle_gait)
    np.testing.assert_array_equal(gait, host_cfg['gait'])


def test_delivery_extension_preserves_request_and_only_uses_completed_ack(host_cfg, manifest):
    request = DeliveryRequest('round:delivery:1', 'yellow')
    output = MissionOutput((0., 0., 0.), None, 'delivery', request, MissionState.DELIVERY)
    supervisor = supervisor_at_delivery(host_cfg, manifest, output)
    wire_ids, inputs = [], []
    bridge = SimpleNamespace(armed=True, delivery=wire_ids.append, delivery_ack=lambda _: False)
    joints = JointState(np.zeros(12), np.zeros(12), np.full(12, 10.))
    imu = ImuState(np.zeros(3), np.array([0., 0., -1.]), 10.)
    supervisor.control(joints, imu, 10., np.zeros(3), host_cfg['gait'], bridge)
    assert len(wire_ids) == 1 and type(wire_ids[0]) is int
    def step(sample):
        inputs.append(sample)
        return replace(output, delivery_request=None)
    supervisor.mission.step = step
    supervisor.control(joints, imu, 10.01, np.zeros(3), host_cfg['gait'], bridge)
    assert inputs[-1].delivery_ack_status is None
    bridge.delivery_ack = lambda request_id: request_id == wire_ids[0]
    supervisor.control(joints, imu, 10.02, np.zeros(3), host_cfg['gait'], bridge)
    assert inputs[-1].delivery_ack_id == request.request_id
    assert inputs[-1].delivery_ack_status == 'completed'
    supervisor.control(joints, imu, 10.03, np.zeros(3), host_cfg['gait'], bridge)
    assert inputs[-1].delivery_ack_stamp == 10.02  # duplicates are not retimestamped
    assert len(wire_ids) == 1


@pytest.mark.parametrize('colour', ['yellow', 'brown'])
def test_generated_round_replays_to_completion(tmp_path, colour):
    import importlib.util

    spec = importlib.util.spec_from_file_location('mission_demo', ROOT / 'scripts/generate_mission_demo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config, replay = module.generate(tmp_path, colour)
    result = replay_mission(load_config(config), replay, mission_id='demo', require_complete=True)
    assert result['state'] == 'done' and result['delivery_completed']
    assert len(result['delivery_requests']) == 1
    assert result['delivery_requests'][0]['colour'] == colour
    with pytest.raises(ValueError, match='did not complete'):
        replay_mission(load_config(config), replay, mission_id='wrong-id', require_complete=True)
