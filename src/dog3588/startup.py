"""Measured prone -> standing transition, before policy inference or mission control.

Targets use the same calibrated radians and watchdog transport as the policy.
Raw pose captures alone never establish model angles or validate a linkage path.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .config import positive
from .safety import SafetyFault


@dataclass(frozen=True)
class StartupPlan:
    prone: np.ndarray
    standing: np.ndarray
    settings: dict
    sources: dict


def load_startup_plan(cfg, manifest, limits):
    section = cfg.get("startup", {})
    if type(section.get("enabled", False)) is not bool:
        raise ValueError("startup.enabled must be boolean")
    if not section.get("enabled", False):
        return None
    if section.get("trajectory_verified") is not True:
        raise ValueError("startup.trajectory_verified must be true after calibrated prone-to-standing path commissioning")
    settings = {key: positive(section.get(key), "startup." + key) for key in (
        "initial_hold_s", "rise_duration_s", "max_rise_duration_s", "max_speed_rad_s",
        "prone_tolerance_rad", "tracking_tolerance_rad", "standing_tolerance_rad",
        "settle_duration_s", "settle_timeout_s", "settle_speed_rad_s",
        "settle_omega_rad_s", "settle_tilt_rad",
    )}
    if settings["max_rise_duration_s"] < settings["rise_duration_s"]:
        raise ValueError("startup.max_rise_duration_s must cover rise_duration_s")
    if settings["settle_timeout_s"] <= settings["settle_duration_s"]:
        raise ValueError("startup.settle_timeout_s must exceed settle_duration_s")
    if settings["settle_tilt_rad"] > cfg["safety"]["max_tilt_rad"]:
        raise ValueError("startup.settle_tilt_rad must not exceed the safety tilt limit")
    settings["max_speed_rad_s"] = min(settings["max_speed_rad_s"], cfg["safety"]["max_target_speed_rad_s"])
    if limits is None or len(limits) != 12:
        raise ValueError("startup requires twelve verified joint limits")
    poses, sources = {}, {}
    for key in ("prone", "standing"):
        path = Path(section[key + "_pose"])
        data = path.read_bytes()
        pose = json.loads(data)
        if pose.get("joint_names") != manifest["joint_names"]:
            raise ValueError(f"startup {key}: joint order differs from policy manifest")
        if (pose.get("model_positions_verified") is not True
                or pose.get("model_joint_positions_rad") is None):
            raise ValueError(f"startup {key}: measured raw counts need verified model_joint_positions_rad; raw counts are not radians")
        q = np.asarray(pose["model_joint_positions_rad"], dtype=float)
        if q.shape != (12,) or not np.isfinite(q).all():
            raise ValueError(f"startup {key}: expected twelve finite model angles")
        for limit, value in zip(limits, q):
            limit.to_raw(float(value))
        poses[key] = q
        sources[key] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    # Check the handoff BEFORE arming, rather than reaching an incompatible pose.
    distance = np.max(np.abs(poses["standing"] - manifest["default_joint_pos"]))
    if distance + settings["standing_tolerance_rad"] > cfg["safety"]["startup_pose_tolerance_rad"]:
        raise ValueError("recorded standing pose is outside the trained policy startup tolerance")
    return StartupPlan(poses["prone"], poses["standing"], settings, sources)


def execute_startup(plan, joints, imu, guard, commands, *, period, stopping,
                    perception=None, emit=lambda record: None):
    """Run at control rate; caller's finally closes/stops actuators on every exit.

    No inference or locomotion commands run during the transition. A quintic
    blend has zero endpoint velocity/acceleration and peak derivative 1.875.
    Actual feedback, not elapsed time alone, decides whether standing is ready.
    """
    c = plan.settings
    state, orientation = joints.read(), imu.read()
    now = time.monotonic()
    guard.check_state(state, orientation, now)
    commands.read(now)
    if stopping() or commands.estop:
        raise SafetyFault("startup interrupted or emergency stop")
    start = state.position.copy()
    if np.max(np.abs(start - plan.prone)) > c["prone_tolerance_rad"]:
        raise SafetyFault("startup: measured pose is not near the recorded prone pose")
    delta = plan.standing - start
    duration = max(c["rise_duration_s"], 1.875 * float(np.max(np.abs(delta))) / c["max_speed_rad_s"])
    if duration > c["max_rise_duration_s"]:
        raise SafetyFault("startup: required rise duration exceeds max_rise_duration_s")
    # Endpoints and interpolation are within each joint's calibrated interval.
    for limit, value in zip(guard.calibrations, start):
        limit.to_raw(float(value))
    emit({"event": "startup_begin", "rise_duration_s": duration, "sources": plan.sources})
    began = next_tick = time.monotonic()
    rise_end = began + c["initial_hold_s"] + duration
    last_target = start.copy()
    stable_since = None
    previous_phase = None
    while True:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick - now)
            continue
        cycle_start = now
        if now - next_tick > guard.cfg["max_loop_lag_s"]:
            raise SafetyFault("startup control loop missed deadline")
        if stopping():
            raise SafetyFault("startup interrupted")
        state, orientation = joints.read(), imu.read()
        now = time.monotonic()
        guard.check_state(state, orientation, now)
        commands.read(now)  # Drain velocity requests, but always honour latched ESTOP.
        if commands.estop:
            raise SafetyFault("latched software emergency stop during startup")
        if perception:
            now = perception.poll(orientation, now)
            perception.require_ready(now)
        if np.max(np.abs(state.position - last_target)) > c["tracking_tolerance_rad"]:
            raise SafetyFault("startup joint tracking error exceeds tolerance")
        progress = np.clip((cycle_start - began - c["initial_hold_s"]) / duration, 0., 1.)
        blend = progress ** 3 * (10. + progress * (-15. + 6. * progress))
        target = start + blend * delta
        phase = "holding_prone" if cycle_start < began + c["initial_hold_s"] else "rising" if progress < 1 else "settling"
        if phase != previous_phase:
            emit({"event": "startup_phase", "phase": phase})
            previous_phase = phase
        ready = False
        if phase == "settling":
            tilt = float(np.arccos(np.clip(-orientation.gravity[2] / np.linalg.norm(orientation.gravity), -1., 1.)))
            stable = (np.max(np.abs(state.position - plan.standing)) <= c["standing_tolerance_rad"]
                      and np.max(np.abs(state.velocity)) <= c["settle_speed_rad_s"]
                      and np.linalg.norm(orientation.omega) <= c["settle_omega_rad_s"]
                      and tilt <= c["settle_tilt_rad"])
            stable_since = (cycle_start if stable_since is None else stable_since) if stable else None
            ready = stable_since is not None and cycle_start - stable_since >= c["settle_duration_s"]
            if cycle_start - rise_end > c["settle_timeout_s"]:
                raise SafetyFault("startup standing did not settle before timeout")
        publish_time = time.monotonic()
        if publish_time - cycle_start > guard.cfg["max_loop_lag_s"]:
            raise SafetyFault("startup computation exceeded deadline")
        guard.check_state(state, orientation, publish_time)
        if perception:
            perception.require_ready(publish_time)
        joints.command(target, cycle_start)
        last_target = target.copy()
        if ready:
            guard.initialise(state, armed=True)
            commands.discard_velocity(publish_time)
            emit({"event": "startup_complete", "elapsed_s": publish_time - began,
                  "q": state.position.tolist(), "target": target.tolist()})
            return state
        next_tick += period
        if time.monotonic() >= next_tick:
            # Never publish a burst of catch-up targets.
            next_tick = time.monotonic() + period
