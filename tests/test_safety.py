import copy
import math

import numpy as np
import pytest

from dog3588.config import Calibration, rotation_matrix, validate_hardware
from dog3588.hardware.joints import FourLegIO
from dog3588.runtime import run
from dog3588.safety import SafetyGuard, SafetyFault
from dog3588.types import JointState, ImuState


def states(now=10):
    return JointState(np.zeros(12), np.zeros(12), np.full(12, now)), ImuState(np.zeros(3), np.array([0, 0, -1.]), now)


def test_configuration_cannot_arm_with_unknown_calibration(cfg, manifest):
    with pytest.raises(ValueError, match="calibration"):
        validate_hardware(cfg, manifest["joint_names"], arm=True)
    with pytest.raises(ValueError, match="calibration"):
        run(cfg, hardware=True, arm=True, seconds=.01)


def test_calibration_inverse_negative_direction_and_bounds():
    cal = Calibration("test", "fl", 1, 500, -1, 200, .1, -.9, 1.1)
    assert cal.to_raw(.6) == 400
    assert cal.from_raw(400) == pytest.approx(.6)
    with pytest.raises(ValueError): cal.to_raw(float("nan"))
    with pytest.raises(ValueError): cal.to_raw(2)
    with pytest.raises(ValueError): cal.from_raw(-1)


def test_mount_rejects_reflection(cfg):
    imu = copy.deepcopy(cfg["imu"])
    imu["sensor_to_body"] = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    with pytest.raises(ValueError): rotation_matrix(imu)


@pytest.mark.parametrize("fault", ["joint_stale", "imu_stale", "tilt", "nan", "missing"])
def test_sensor_faults(cfg, manifest, fault):
    guard = SafetyGuard(cfg["safety"], manifest)
    q, imu = states()
    if fault == "joint_stale": q.stamps[7] = 9
    if fault == "imu_stale": imu = ImuState(imu.omega, imu.gravity, 9)
    if fault == "tilt": imu.gravity[:] = [math.sin(.5), 0, -math.cos(.5)]
    if fault == "nan": q.position[3] = np.nan
    if fault == "missing": q.stamps[1] = 0
    with pytest.raises(SafetyFault): guard.check_state(q, imu, 10)


def test_startup_pose_action_rejection_and_slew(cfg, manifest, calibrations):
    guard = SafetyGuard(cfg["safety"], manifest, calibrations)
    q, _ = states()
    q.position[0] = .5
    with pytest.raises(SafetyFault): guard.initialise(q, armed=True)
    q.position[:] = 0
    guard.initialise(q, armed=True)
    np.testing.assert_allclose(guard.target(np.ones(12), .02), .02)
    with pytest.raises(SafetyFault): guard.target(np.full(12, 5.), .02)  # 1.25 rad beyond limits
    with pytest.raises(SafetyFault): guard.target(np.full(12, np.nan), .02)


def test_all_targets_validated_before_any_leg_submission(cfg, calibrations):
    io = FourLegIO(cfg, calibrations, armed=True, control_hz=50)
    calls = []
    for _, worker in io.groups:
        worker.submit = lambda *args: calls.append(args)
    target = np.zeros(12)
    target[-1] = 2
    with pytest.raises(ValueError): io.command(target, 10)
    assert calls == []

