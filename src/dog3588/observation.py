"""Isaac Lab term-major history, and Robot12's conditioned gait clock."""

from collections import deque
import math

import numpy as np

from .types import ImuState, JointState


class ObservationBuilder:
    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.reset()

    def reset(self) -> None:
        self.history = {t["name"]: deque(maxlen=self.manifest["history_length"])
                        for t in self.manifest["terms"]}
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.phase = 0.0

    def build(self, joints: JointState, imu: ImuState, velocity: np.ndarray,
              gait: np.ndarray) -> np.ndarray:
        velocity, gait = np.asarray(velocity), np.asarray(gait)
        if (velocity.shape != (3,) or gait.shape != (7,)
                or not np.isfinite(velocity).all() or not np.isfinite(gait).all()
                or gait[0] <= 0 or not 0 < gait[1] < 1):
            raise ValueError("invalid velocity/gait command")
        moving = bool(np.any(velocity[:2] != 0) or
                      abs(velocity[2]) > self.manifest["gait"]["yaw_threshold_rad_s"])
        raw_phase = (self.phase + gait[3:]) % 1.0
        duty = gait[1]
        foot_phase = np.where(raw_phase < duty, 0.5 * raw_phase / duty,
                              0.5 + 0.5 * (raw_phase - duty) / (1 - duty))
        values = {
            "base_ang_vel": imu.omega, "projected_gravity": imu.gravity,
            "velocity_commands": velocity,
            "joint_pos": joints.position - self.manifest["default_joint_pos"],
            "joint_vel": joints.velocity - self.manifest["default_joint_vel"],
            "actions": self.previous_action,
            "gait_parameters": gait,
            "gait_clock": np.sin(2 * math.pi * foot_phase) * moving,
        }
        blocks = []
        for term in self.manifest["terms"]:
            value = np.asarray(values[term["name"]], dtype=np.float32).copy()
            if value.shape != (term["size"],) or not np.isfinite(value).all():
                raise ValueError(f"invalid observation: {term['name']}")
            if term["clip"] is not None:
                value = np.clip(value, *term["clip"])
            value = value * term["scale"]
            history = self.history[term["name"]]
            if not history:
                history.extend(value.copy() for _ in range(history.maxlen))
            else:
                history.append(value)
            blocks.append(np.stack(history).ravel())
        return np.concatenate(blocks).astype(np.float32)

    def advance(self, action: np.ndarray, velocity: np.ndarray, gait: np.ndarray) -> None:
        """Call once after each action; phase advances between policy observations."""
        self.previous_action = np.asarray(action, dtype=np.float32).copy()
        moving = np.any(velocity[:2] != 0) or abs(velocity[2]) > self.manifest["gait"]["yaw_threshold_rad_s"]
        self.phase = (self.phase + gait[0] / self.manifest["control_hz"] * moving) % 1.0

