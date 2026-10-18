"""Configuration and calibrated motor/servo conversion, independent of transports."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import yaml


def finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def positive(value, name: str) -> float:
    value = finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Calibration:
    name: str
    bus: str
    servo_id: int
    raw_zero: float
    direction: int
    units_per_rad: float
    zero_rad: float
    min_rad: float
    max_rad: float

    @classmethod
    def from_dict(cls, item: dict):
        if item.get("verified") is not True:
            raise ValueError(f"{item.get('name')}: calibration is not verified")
        values = {key: finite(item.get(key), key) for key in
                  ("raw_zero", "units_per_rad", "zero_rad", "min_rad", "max_rad")}
        if item.get("direction") not in (-1, 1) or isinstance(item.get("direction"), bool):
            raise ValueError("direction must be +1 or -1")
        sid = item.get("id")
        if type(sid) is not int or not 0 <= sid <= 253:
            raise ValueError("servo id must be 0..253")
        if not 0 <= values["raw_zero"] <= 1000 or values["units_per_rad"] <= 0:
            raise ValueError("invalid servo calibration scale/zero")
        if not values["min_rad"] < values["max_rad"]:
            raise ValueError("joint minimum must be below maximum")
        result = cls(item["name"], item["bus"], sid, direction=item["direction"], **values)
        result.to_raw(result.min_rad)
        result.to_raw(result.max_rad)
        return result

    def to_raw(self, q: float) -> int:
        q = finite(float(q), self.name)
        if not self.min_rad <= q <= self.max_rad:
            raise ValueError(f"{self.name}: {q:.4f} rad outside calibrated limits")
        raw = self.raw_zero + self.direction * (q - self.zero_rad) * self.units_per_rad
        if not 0 <= raw <= 1000:
            raise ValueError(f"{self.name}: raw position outside 0..1000")
        return int(math.floor(raw + 0.5))

    def from_raw(self, raw: int) -> float:
        if not 0 <= raw <= 1000:
            raise ValueError(f"{self.name}: unexpected raw feedback {raw}")
        return self.zero_rad + self.direction * (raw - self.raw_zero) / self.units_per_rad


def rotation_matrix(imu: dict) -> np.ndarray:
    rotation = np.asarray(imu["sensor_to_body"], dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("sensor_to_body must be a proper 3x3 rotation matrix")
    return rotation


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("version") != 1:
        raise ValueError("expected configuration version 1")
    cfg["policy_manifest"] = str((path.parent / cfg["policy_manifest"]).resolve())
    if cfg["command"]["host"] != "127.0.0.1":
        raise ValueError("command socket must bind to loopback; use SSH for remote control")
    if cfg["command"]["source"] not in ("manual", "vision"):
        raise ValueError("command source must be manual or vision")
    for section, keys in {
        "safety": ("joint_timeout_s", "imu_timeout_s", "target_timeout_s", "max_tilt_rad",
                   "max_target_speed_rad_s", "max_action_abs", "startup_pose_tolerance_rad",
                   "startup_timeout_s", "max_loop_lag_s"),
        "command": ("timeout_s",),
    }.items():
        for key in keys:
            positive(cfg[section][key], f"{section}.{key}")
    if not 0 < cfg["safety"]["max_tilt_rad"] < math.pi / 2:
        raise ValueError("max_tilt_rad must be between 0 and pi/2")
    rotation_matrix(cfg["imu"])
    return cfg


def validate_hardware(cfg: dict, joint_names: list[str], *, arm: bool) -> list[Calibration]:
    items = cfg["joints"]
    if [j["name"] for j in items] != joint_names:
        raise ValueError("joint order differs from the policy manifest")
    buses = cfg["buses"]
    if len(buses) != 4 or len({b["name"] for b in buses}) != 4:
        raise ValueError("four uniquely named leg buses are required")
    if len({b["port"] for b in buses}) != 4:
        raise ValueError("leg serial devices must be distinct")
    known = {b["name"] for b in buses}
    if any(j["bus"] not in known for j in items):
        raise ValueError("joint references an unknown bus")
    if len({(j["bus"], j["id"]) for j in items}) != len(items):
        raise ValueError("duplicate servo ID on a bus")
    if any(sum(j["bus"] == b for j in items) != 3 for b in known):
        raise ValueError("each leg bus must have three joints")
    calibrations = [Calibration.from_dict(j) for j in items]
    for bus in buses:
        if bus["direction"]["mode"] not in ("auto", "gpio"):
            raise ValueError("bus direction mode must be auto or gpio")
        if bus["baudrate"] != 115200:
            raise ValueError("LX824 bus requires 115200 baud")
        for key in ("read_timeout_s", "query_period_s", "quiet_after_timeout_s"):
            positive(bus[key], f"bus.{key}")
    if cfg["imu"]["driver"] not in ("wit_serial", "mpu6050_i2c"):
        raise ValueError("select a real IMU driver before using hardware")
    if arm:
        if not all(b.get("verified") is True for b in buses):
            raise ValueError("all four transports must pass readback commissioning")
        if cfg["imu"].get("mount_verified") is not True:
            raise ValueError("IMU axes and gyro calibration must be verified")
    return calibrations


@dataclass(frozen=True)
class JointLimits:
    """Host-side radians only. Raw LX824 conversion belongs to STM32."""

    name: str
    min_rad: float
    max_rad: float

    def to_raw(self, q: float) -> float:
        # SafetyGuard's existing calibration interface; no servo counts are generated.
        q = finite(float(q), self.name)
        if not self.min_rad <= q <= self.max_rad:
            raise ValueError(f"{self.name}: target outside calibrated limits")
        return q


def validate_bridge_hardware(cfg, joint_names, *, arm):
    bridge = cfg["bridge"]
    if not bridge.get("port") or type(bridge.get("baudrate")) is not int or bridge["baudrate"] <= 0:
        raise ValueError("configure an independent STM32 serial port and baudrate")
    if [j["name"] for j in cfg["joints"]] != joint_names:
        raise ValueError("STM32 joint order differs from policy manifest")
    if not arm:
        return None
    for key in ("verified", "imu_mount_verified", "joint_mapping_verified"):
        if bridge.get(key) is not True:
            raise ValueError(f"bridge.{key} must be verified before arming")
    limits = []
    for joint in cfg["joints"]:
        if joint.get("verified") is not True:
            raise ValueError(f"{joint['name']}: joint limits are not verified")
        lo, hi = finite(joint.get("min_rad"), "min_rad"), finite(joint.get("max_rad"), "max_rad")
        if lo >= hi:
            raise ValueError("joint minimum must be below maximum")
        limits.append(JointLimits(joint["name"], lo, hi))
    return limits

