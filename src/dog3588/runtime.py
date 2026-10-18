"""Composition root for policy, sensors and commands. Camera work stays in another process."""

import json
from pathlib import Path
import signal
import time

import numpy as np

from .commands import CommandReceiver
from .config import validate_hardware, validate_bridge_hardware
from .hardware.imu import MockImu, ThreadedImu
from .hardware.joints import FourLegIO, MockJoints
from .observation import ObservationBuilder
from .policy import OnnxPolicy, load_manifest
from .safety import SafetyFault, SafetyGuard


def validate_gait(values, manifest):
    gait = np.asarray(values, dtype=float)
    if gait.shape != (7,) or not np.isfinite(gait).all():
        raise ValueError("gait must contain seven finite values")
    for i, key in enumerate(("frequency_hz", "stance_fraction", "clearance_m")):
        lo, hi = manifest["gait"][key]
        if not lo <= gait[i] <= hi:
            raise ValueError(f"gait {key} outside training envelope")
    if not np.allclose(gait[3:], manifest["gait"]["phase_offsets"]):
        raise ValueError("gait phase offsets differ from training")
    return gait


def run(cfg, *, hardware=False, arm=False, autonomous=False, seconds=10., log_path=None):
    if arm and not hardware:
        raise ValueError("--arm requires --hardware")
    if autonomous and not hardware:
        raise ValueError("--autonomous requires --hardware; use mission-replay for offline scenarios")
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and nonnegative; 0 means until interrupted")
    manifest = load_manifest(cfg["policy_manifest"])
    gait = validate_gait(cfg["gait"], manifest)
    backend = cfg.get("backend", "direct")
    if backend not in ("direct", "stm32"):
        raise ValueError("backend must be stm32 or direct")
    validator = validate_bridge_hardware if backend == "stm32" else validate_hardware
    calibrations = validator(cfg, manifest["joint_names"], arm=arm) if hardware else None
    policy = OnnxPolicy(manifest)
    builder = ObservationBuilder(manifest)
    guard = SafetyGuard(cfg["safety"], manifest, calibrations)
    perception = None
    if hardware and backend == "stm32":
        from .transport import STM32Bridge
        from .deployment import BridgeImu
        joints = STM32Bridge(cfg['bridge'], armed=arm)
        imu = BridgeImu(joints)
    elif hardware:
        joints = FourLegIO(cfg, calibrations, armed=arm, control_hz=manifest["control_hz"])
        imu = ThreadedImu(cfg['imu'])
    else:
        joints, imu = MockJoints(manifest['default_joint_pos']), MockImu()
    if hardware:
        from .deployment import PerceptionSupervisor
        perception = PerceptionSupervisor(cfg, manifest, autonomous=autonomous)
    uses_height = any(t['name'] == 'height_scan' for t in manifest['terms'])
    commands = CommandReceiver(cfg["command"], manifest["velocity_ranges"])
    period = 1 / manifest["control_hz"]
    stopping = False
    previous_handlers = {}

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, request_stop)
    logger = None
    started_joints = started_imu = started_perception = False
    ticks, times, skipped_deadlines = 0, [], 0
    began = None
    try:
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            logger = path.open("a", encoding="utf-8", buffering=8192)
        commands.start()
        joints.start()
        started_joints = True
        imu.start()
        started_imu = True
        if perception:
            perception.start()
            started_perception = True
        deadline = time.monotonic() + cfg["safety"]["startup_timeout_s"]
        while True:
            if stopping:
                return {"ticks": 0, "stopped_during_startup": True}
            state, orientation = joints.read(), imu.read()
            now = time.monotonic()
            try:
                guard.check_state(state, orientation, now)
                if perception:
                    now = perception.poll(orientation, now)
                    perception.require_ready(now)
                break
            except SafetyFault:
                if now >= deadline:
                    raise
                time.sleep(period)
        guard.initialise(state, armed=arm)
        # All preflight checks complete before any motor receives LOAD or MOVE.
        began = next_tick = time.monotonic()
        last_status = began - 1
        print(json.dumps({"event": "started", "mode": "armed" if arm else "shadow" if hardware else "mock",
                          "backend": backend, "autonomous": autonomous,
                          "control_hz": manifest["control_hz"], "command_source": cfg["command"]["source"]}), flush=True)
        while not stopping and (seconds == 0 or time.monotonic() - began < seconds):
            now = time.monotonic()
            if now < next_tick:
                # A sleep may return before a coarse OS monotonic clock has
                # advanced. Recheck instead of publishing a duplicate stamp.
                time.sleep(next_tick - now)
                continue
            lag = now - next_tick
            if lag > cfg["safety"]["max_loop_lag_s"]:
                raise SafetyFault(f"control loop late by {lag * 1000:.1f} ms")
            cycle_start = now
            state, orientation = joints.read(), imu.read()
            now = time.monotonic()
            guard.check_state(state, orientation, now)
            velocity = commands.read(now)
            if commands.estop:
                raise SafetyFault("latched software emergency stop")
            # Synthetic heights exist only in the explicit no-device mock path.
            heights = np.zeros(8) if uses_height and not hardware else None
            if perception:
                from .deployment import check_velocity
                now = perception.poll(orientation, now)
                velocity, selected_gait, heights = perception.control(state, orientation, now, velocity, gait,
                                                                      joints if backend == 'stm32' else None)
                velocity = check_velocity(velocity, manifest)
                gait = validate_gait(selected_gait, manifest)
                if perception.done:
                    break
            observation = builder.build(state, orientation, velocity, gait, heights)
            inference_start = time.perf_counter()
            action = policy.infer(observation)
            infer_ms = (time.perf_counter() - inference_start) * 1000
            # An inference stall must never publish a fresh timestamp for an old observation.
            publish_time = time.monotonic()
            if publish_time - cycle_start > cfg["safety"]["max_loop_lag_s"]:
                raise SafetyFault("policy inference/control computation exceeded deadline")
            guard.check_state(state, orientation, publish_time)
            if perception:
                perception.require_ready(publish_time)
            target = guard.target(action, period)
            joints.command(target, cycle_start)
            builder.advance(action, velocity, gait)
            ticks += 1
            times.append(infer_ms)
            if len(times) > 10000:
                del times[:5000]
            record = {"event": "tick", "tick": ticks, "t": now, "inference_ms": infer_ms,
                      "joint_age_ms": float(np.max(now - state.stamps) * 1000),
                      "imu_age_ms": (now - orientation.stamp) * 1000,
                      "velocity": velocity.tolist(), "q": state.position.tolist(),
                      "target": target.tolist(), "action": action.tolist()}
            if perception:
                record.update(perception.status())
            if logger and ticks % 5 == 0:
                logger.write(json.dumps(record, allow_nan=False) + "\n")
            if now - last_status >= 1:
                print(json.dumps({k: record[k] for k in ("tick", "inference_ms", "joint_age_ms", "imu_age_ms", "velocity")}), flush=True)
                last_status = now
            next_tick += period
            delay = next_tick - time.monotonic()
            if -delay > cfg["safety"]["max_loop_lag_s"]:
                raise SafetyFault("control cycle exceeded deadline, including logging/status output")
            if delay <= 0:
                # Skip any expired slot, never send catch-up targets back to back.
                skipped_deadlines += 1
                next_tick = time.monotonic() + period
        elapsed = time.monotonic() - began
        return {"ticks": ticks, "elapsed_s": elapsed, "mean_hz": ticks / max(elapsed, 1e-6),
                "inference_p95_ms": float(np.percentile(times, 95)) if times else None,
                "skipped_deadlines": skipped_deadlines, "hardware": hardware, "armed": arm}
    finally:
        # Close actuators before IMU/log/socket cleanup; failure of one cleanup cannot skip the rest.
        errors = []
        for resource, started in ((joints, started_joints), (perception, started_perception),
                                  (imu, started_imu), (commands, True)):
            if started:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(str(exc))
        if logger:
            logger.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if errors:
            raise RuntimeError("cleanup faults: " + "; ".join(errors))

