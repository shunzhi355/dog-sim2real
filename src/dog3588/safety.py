"""Pure state/target checks; transport-level watchdogs work independently."""

import math

import numpy as np


class SafetyFault(RuntimeError):
    pass


class SafetyGuard:
    def __init__(self, cfg, manifest, calibrations=None):
        self.cfg, self.manifest, self.calibrations = cfg, manifest, calibrations
        self.last_target = None

    def check_state(self, joints, imu, now):
        for value, shape in ((joints.position, (12,)), (joints.velocity, (12,)),
                             (joints.stamps, (12,)), (imu.omega, (3,)), (imu.gravity, (3,))):
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise SafetyFault("invalid sensor vector")
        ages = now - joints.stamps
        if np.any(joints.stamps <= 0) or np.any(ages < -0.01) or np.max(ages) > self.cfg["joint_timeout_s"]:
            raise SafetyFault("joint feedback missing/stale")
        if (not math.isfinite(imu.stamp) or imu.stamp <= 0
                or not -0.01 <= now - imu.stamp <= self.cfg["imu_timeout_s"]):
            raise SafetyFault("IMU missing/stale")
        gravity_norm = np.linalg.norm(imu.gravity)
        if not 0.95 <= gravity_norm <= 1.05:
            raise SafetyFault("invalid gravity unit vector")
        tilt = math.acos(float(np.clip(-imu.gravity[2] / gravity_norm, -1, 1)))
        if tilt > self.cfg["max_tilt_rad"]:
            raise SafetyFault(f"body tilt {tilt:.3f} rad exceeds limit")
        if self.calibrations:
            for j, q in zip(self.calibrations, joints.position):
                if not j.min_rad - 0.01 <= q <= j.max_rad + 0.01:
                    raise SafetyFault(f"{j.name}: measured position outside calibrated limits")

    def initialise(self, joints, *, armed):
        if armed and np.max(np.abs(joints.position - self.manifest["default_joint_pos"])) > self.cfg["startup_pose_tolerance_rad"]:
            raise SafetyFault("place supported robot near the trained initial pose before arming")
        self.last_target = joints.position.copy()

    def target(self, action, dt):
        action = np.asarray(action)
        if (action.shape != (12,) or not np.isfinite(action).all()
                or np.max(np.abs(action)) > self.cfg["max_action_abs"]):
            raise SafetyFault("policy returned invalid/excessive action")
        desired = np.asarray(self.manifest["default_joint_pos"]) + action * self.manifest["action_scale"]
        if self.calibrations:
            # Check desired, not just slew-limited target, so invalid policies cannot creep into a stop.
            try:
                for j, q in zip(self.calibrations, desired):
                    j.to_raw(float(q))
            except ValueError as exc:
                raise SafetyFault(str(exc)) from exc
        if self.last_target is None:
            raise SafetyFault("target guard was not initialised with measured positions")
        step = self.cfg["max_target_speed_rad_s"] * dt
        target = self.last_target + np.clip(desired - self.last_target, -step, step)
        self.last_target = target.copy()
        return target

