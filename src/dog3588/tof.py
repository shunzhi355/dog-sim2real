"""VL53L5CX acquisition and calibrated, yaw-free terrain projection.

Hardware imports are lazy. Raw VL53L5CX distances are optical-axis depths,
NOT radial ranges. See docs/tof.md for the calibration and policy contract.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from .types import ImuState


@dataclass(frozen=True)
class ToFFrame:
    """64 optical-axis depths (m), vendor status/count, host monotonic stamp.

    stamp is data-ready polling time minus configured measurement latency;
    it is never the time a consumer calls read(). Zone order is driver order.
    """

    distance_m: np.ndarray
    target_status: np.ndarray
    nb_target_detected: np.ndarray
    stamp: float


@dataclass(frozen=True)
class TerrainState:
    """Eight raw leveled-body Z values, in training sample order.

    Invalid heights are NaN, never plausible zero observations. coverage counts
    valid measured points within each sample's XY tolerance (not area fraction).
    """

    heights: np.ndarray
    valid: np.ndarray
    stamp: float
    coverage: np.ndarray
    xy_error_m: np.ndarray
    calibrated: bool
    reason: str = ""


def _number(cfg, key, default, *, positive=False):
    value = cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"invalid {key}")
    return value


def _integer(cfg, key, default, low, high):
    value = cfg.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} must be an integer in {low}..{high}")
    return value


def _array(value, shape, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return result


def _empty_frame():
    return ToFFrame(np.full(64, np.nan), np.full(64, 255, dtype=np.uint8),
                    np.zeros(64, dtype=np.uint8), 0.0)


def _copy_frame(frame):
    return ToFFrame(frame.distance_m.copy(), frame.target_status.copy(),
                    frame.nb_target_detected.copy(), frame.stamp)


class ThreadedToF:
    """Single-owner Linux I2C worker using Pimoroni vl53l5cx-ctypes 0.0.3.

    start() is asynchronous (firmware upload can be slow); read() is nonblocking
    and returns an invalid stamp-zero frame until the first measurement. Worker
    failures surface as RuntimeError, and startup/stall deadlines as TimeoutError.
    Instances are single-use; repeated start/close are otherwise idempotent.
    """

    def __init__(self, cfg):
        self.cfg = deepcopy(cfg)
        if cfg.get("driver", "vl53l5cx_ctypes") != "vl53l5cx_ctypes":
            raise ValueError("ToF driver must be vl53l5cx_ctypes")
        self.bus_number = _integer(cfg, "i2c_bus", 2, 0, 255)
        self.address = _integer(cfg, "i2c_address", 0x29, 0x08, 0x77)
        self.frequency = _integer(cfg, "frequency_hz", 15, 1, 15)
        self.poll_s = _number(cfg, "poll_interval_s", 0.002, positive=True)
        if self.poll_s >= 1 / self.frequency:
            raise ValueError("poll_interval_s must be below one frame period")
        self.frame_timeout = _number(cfg, "frame_timeout_s", 0.1, positive=True)
        self.startup_timeout = _number(cfg, "startup_timeout_s", 30.0, positive=True)
        self.close_timeout = _number(cfg, "close_timeout_s", 2.0, positive=True)
        self.latency = _number(cfg, "measurement_latency_s", 0.0)
        if self.latency < 0:
            raise ValueError("measurement_latency_s must be nonnegative")
        self.finish = threading.Event()
        self.lock = threading.Lock()
        self.sample = _empty_frame()
        self.error = None
        self.started_at = None
        self.thread = threading.Thread(target=self._run, name="tof", daemon=True)

    def start(self):
        with self.lock:
            if self.finish.is_set():
                raise RuntimeError("ToF is closed; construct a new instance")
            if self.started_at is None:
                self.started_at = time.monotonic()
                self.thread.start()

    def read(self) -> ToFFrame:
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"ToF worker: {self.error}") from self.error
            if self.finish.is_set():
                raise RuntimeError("ToF is closed")
            now = time.monotonic()
            if self.started_at is not None:
                if self.sample.stamp == 0 and now - self.started_at > self.startup_timeout:
                    raise TimeoutError("ToF startup timeout")
                if self.sample.stamp > 0 and now - self.sample.stamp > self.frame_timeout:
                    raise TimeoutError("ToF frame timeout")
            return _copy_frame(self.sample)

    def _publish(self, frame):
        with self.lock:
            if frame.stamp <= self.sample.stamp:
                raise ValueError("non-increasing ToF acquisition timestamp")
            if not self.finish.is_set():
                self.sample = frame

    @staticmethod
    def _decode(data, stamp):
        # ctypes fields distance/status have shape (1, 64), count has (64,).
        # Never silently truncate a multi-target or a 4x4 driver build.
        arrays = []
        for name in ("distance_mm", "target_status", "nb_target_detected"):
            raw = np.asarray(getattr(data, name))
            if raw.shape not in ((64,), (1, 64)):
                raise ValueError(f"ToF {name}: expected 64 zones, single target")
            arrays.append(raw.reshape(64).copy())
        distance, status, count = arrays
        return ToFFrame(distance.astype(float) * 0.001, status, count, stamp)

    def _run(self):
        bus = sensor = None
        ranging = False
        try:
            import vl53l5cx_ctypes as driver
            from smbus2 import SMBus

            if driver.NB_TARGET_PER_ZONE != 1:
                raise ValueError("ToF requires the single-target driver build")
            bus = SMBus(self.bus_number)
            sensor = driver.VL53L5CX(i2c_addr=self.address, i2c_dev=bus)
            if self.finish.is_set():
                return
            sensor.set_resolution(64)
            sensor.set_ranging_frequency_hz(self.frequency)
            sensor.set_ranging_mode(driver.RANGING_MODE_CONTINUOUS)
            sensor.set_target_order(driver.TARGET_ORDER_CLOSEST)
            # These upstream setters and start/stop return None, not status bools.
            ranging = True
            sensor.start_ranging()
            last_ready = time.monotonic()
            while not self.finish.is_set():
                before = time.monotonic()
                # After a scheduler/transport stall a pending device buffer has
                # unknown age, even if data_ready now returns true.
                if before - last_ready > self.frame_timeout:
                    raise TimeoutError("ToF data-ready timeout")
                if sensor.data_ready():
                    frame = self._decode(sensor.get_data(), before - self.latency)
                    after = time.monotonic()
                    if after - frame.stamp > self.frame_timeout:
                        raise TimeoutError("ToF read/measurement latency exceeds frame timeout")
                    self._publish(frame)
                    last_ready = after
                self.finish.wait(self.poll_s)
        except Exception as exc:
            with self.lock:
                self.error = exc
        finally:
            # Cleanup stays in the I2C-owning worker, including on exceptions.
            try:
                if ranging:
                    sensor.stop_ranging()
            except Exception as exc:
                with self.lock:
                    if self.error is None:
                        self.error = exc
            finally:
                try:
                    if bus is not None:
                        bus.close()
                except Exception as exc:
                    with self.lock:
                        if self.error is None:
                            self.error = exc

    def close(self):
        with self.lock:
            self.finish.set()
            started = self.started_at is not None
        if started:
            self.thread.join(self.close_timeout)
            if self.thread.is_alive():
                raise TimeoutError("ToF I2C worker blocked during close")


class TerrainProjector:
    """Calibrated nearest samples in a gravity-leveled body frame.

    +X forward, +Y left, +Z up; yaw follows the body. Missing calibration is
    permitted for setup, but produces all-invalid output and cannot be armed.
    Bad supplied geometry or safety limits are rejected at construction.
    """

    _FLAGS = ("verified", "rays_verified", "mount_verified", "sampling_verified",
              "reference_verified")
    _GEOMETRY = {"ray": (64, 3), "sensor_to_body": (3, 3),
                 "origin_body_m": (3,), "sample_xy_m": (8, 2)}

    def __init__(self, cfg):
        self.cfg = deepcopy(cfg)
        self.frame_timeout = _number(cfg, "frame_timeout_s", 0.1, positive=True)
        self.imu_timeout = _number(cfg, "imu_timeout_s", 0.1, positive=True)
        self.max_skew = _number(cfg, "max_skew_s", 0.03, positive=True)
        self.max_rotation = _number(cfg, "max_rotation_rad", 0.1, positive=True)
        self.max_tilt = _number(cfg, "max_tilt_rad", 1.0, positive=True)
        if self.max_tilt >= math.pi / 2:
            raise ValueError("max_tilt_rad must be below pi/2")
        frequency = _integer(cfg, "frequency_hz", 15, 1, 15)
        self.exposure = _number(cfg, "frame_duration_s", 1 / frequency, positive=True)
        self.min_distance = _number(cfg, "min_distance_m", 0.02, positive=True)
        self.max_distance = _number(cfg, "max_distance_m", 4.0, positive=True)
        self.min_height = _number(cfg, "min_height_m", -2.0)
        self.max_height = _number(cfg, "max_height_m", 1.0)
        if self.min_distance >= self.max_distance or self.min_height >= self.max_height:
            raise ValueError("distance/height minimum must be below maximum")
        statuses = cfg.get("valid_statuses", [5])
        if (not isinstance(statuses, (list, tuple)) or not statuses
                or any(type(s) is not int or s not in (5, 9) for s in statuses)):
            raise ValueError("valid_statuses may only contain 5 and optionally 9")
        self.statuses = tuple(statuses)
        # Do not silently apply legacy offset/clip configuration to raw body Z.
        for key in ("reference_height_m", "nominal_height_m", "height_clip_m"):
            if key in cfg:
                raise ValueError(f"{key} is not supported; output is raw leveled body Z")
        if cfg.get("height_reference", "body_origin") != "body_origin":
            raise ValueError("height_reference must be body_origin")
        for name, shape in self._GEOMETRY.items():
            value = cfg.get(name)
            setattr(self, name, None if value is None else _array(value, shape, name))
        if self.ray is not None:
            if (not np.allclose(np.linalg.norm(self.ray, axis=1), 1, atol=1e-5, rtol=0)
                    or np.any(self.ray[:, 2] <= 1e-4)):
                raise ValueError("ray must contain 64 unit vectors with positive optical Z")
        if self.sensor_to_body is not None:
            r = self.sensor_to_body
            if (not np.allclose(r @ r.T, np.eye(3), atol=1e-5, rtol=0)
                    or not np.isclose(np.linalg.det(r), 1, atol=1e-5, rtol=0)):
                raise ValueError("sensor_to_body must be a proper rotation (det=+1)")
        if self.sample_xy_m is not None and len(np.unique(self.sample_xy_m, axis=0)) != 8:
            raise ValueError("sample_xy_m must contain eight distinct training samples")
        value = cfg.get("max_xy_error_m")
        self.max_xy_error = None if value is None else _number(
            cfg, "max_xy_error_m", None, positive=True)
        self.calibrated = (all(cfg.get(key) is True for key in self._FLAGS)
                           and all(getattr(self, key) is not None for key in self._GEOMETRY)
                           and self.max_xy_error is not None)
        self._last_frame_stamp = self._last_imu_stamp = 0.0

    def require_calibrated(self, contract=None):
        """Call before arming. Pass manifest['height_scan_contract'] to verify it."""
        if not self.calibrated:
            raise ValueError("ToF calibration/transport is not verified; cannot arm")
        if contract is not None:
            expected = {"frame": "leveled_body", "units": "m", "reference": "body_origin",
                        "invalid_policy": "stop"}
            if not isinstance(contract, dict) or any(contract.get(k) != v for k, v in expected.items()):
                raise ValueError("incompatible height_scan_contract")
            xy = _array(contract.get("sample_xy_m"), (8, 2), "height_scan_contract.sample_xy_m")
            if not np.allclose(xy, self.sample_xy_m, atol=1e-9, rtol=0):
                raise ValueError("height_scan_contract sample_xy_m/order differs from calibration")

    def _invalid(self, stamp, reason):
        return TerrainState(np.full(8, np.nan), np.zeros(8, dtype=bool), stamp,
                            np.zeros(8, dtype=int), np.full(8, np.inf),
                            self.calibrated, reason)

    def raw_valid_mask(self, frame):
        """Reject unusable raw frames before waiting for delayed IMU telemetry."""
        try:
            depth = np.asarray(frame.distance_m, dtype=float)
            status = np.asarray(frame.target_status, dtype=float)
            count = np.asarray(frame.nb_target_detected, dtype=float)
            if any(a.shape != (64,) for a in (depth, status, count)):
                return np.zeros(64, dtype=bool)
            return (np.isfinite(depth) & (depth >= self.min_distance)
                    & (depth <= self.max_distance) & np.isin(status, self.statuses)
                    & np.isfinite(count) & (count >= 1) & (count <= 4)
                    & (count == np.floor(count)))
        except (ValueError, TypeError):
            return np.zeros(64, dtype=bool)

    def project(self, frame: ToFFrame, imu: ImuState, now: float) -> TerrainState:
        """Project one frame; reuse is allowed until expiry, timestamp regression is not.

        Caller should supply the IMU sample nearest the ToF timestamp. This is a
        single-consumer projector, not an IMU interpolator or a world-height map.
        """
        stamp = frame.stamp
        if not self.calibrated:
            return self._invalid(stamp, "uncalibrated")
        times = (now, stamp, imu.stamp)
        if (not all(isinstance(t, (int, float)) and not isinstance(t, bool)
                    and math.isfinite(t) for t in times)
                or stamp <= 0 or imu.stamp <= 0 or stamp > now or imu.stamp > now):
            return self._invalid(stamp, "invalid_timestamp")
        if stamp < self._last_frame_stamp or imu.stamp < self._last_imu_stamp:
            return self._invalid(stamp, "out_of_order")
        self._last_frame_stamp, self._last_imu_stamp = stamp, imu.stamp
        if now - stamp > self.frame_timeout or now - imu.stamp > self.imu_timeout:
            return self._invalid(stamp, "stale")
        skew = abs(stamp - imu.stamp)
        if skew > self.max_skew:
            return self._invalid(stamp, "imu_skew")
        try:
            gravity = _array(imu.gravity, (3,), "gravity")
            omega = _array(imu.omega, (3,), "omega")
            depth = np.asarray(frame.distance_m, dtype=float)
            status = np.asarray(frame.target_status, dtype=float)
            count = np.asarray(frame.nb_target_detected, dtype=float)
        except (TypeError, ValueError):
            return self._invalid(stamp, "malformed_frame_or_imu")
        if any(a.shape != (64,) for a in (depth, status, count)):
            return self._invalid(stamp, "malformed_frame")
        norm = np.linalg.norm(gravity)
        if not np.isclose(norm, 1.0, atol=0.02, rtol=0):
            return self._invalid(stamp, "invalid_gravity")
        up = -gravity / norm
        if up[2] < math.cos(self.max_tilt):
            return self._invalid(stamp, "excessive_tilt")
        if np.linalg.norm(omega) * (skew + self.exposure) > self.max_rotation:
            return self._invalid(stamp, "excessive_rotation")
        # Equivalent to R_y(pitch) @ R_x(roll), with yaw removed. Projecting
        # body X onto the horizontal plane fixes the otherwise free yaw axis.
        forward = np.array([1., 0., 0.]) - up[0] * up
        forward /= np.linalg.norm(forward)
        left = np.cross(up, forward)
        body_to_level = np.stack((forward, left, up))
        valid_zone = self.raw_valid_mask(frame)
        # Filter before multiplication: NaN/negative/invalid depth cannot become
        # a plausible high obstacle after projection or clipping.
        rays = self.ray[valid_zone]
        points_sensor = rays * (depth[valid_zone] / rays[:, 2])[:, None]
        points_body = points_sensor @ self.sensor_to_body.T + self.origin_body_m
        points = points_body @ body_to_level.T
        keep = (np.isfinite(points).all(axis=1) & (points[:, 2] >= self.min_height)
                & (points[:, 2] <= self.max_height))
        points = points[keep]
        heights = np.full(8, np.nan)
        valid = np.zeros(8, dtype=bool)
        coverage = np.zeros(8, dtype=int)
        errors = np.full(8, np.inf)
        if len(points):
            errors_all = np.linalg.norm(points[:, None, :2] - self.sample_xy_m[None, :, :], axis=2)
            coverage = np.sum(errors_all <= self.max_xy_error, axis=0)
            nearest = np.argmin(errors_all, axis=0)
            errors = errors_all[nearest, np.arange(8)]
            valid = coverage > 0
            heights[valid] = points[nearest[valid], 2]
        return TerrainState(heights, valid, stamp, coverage, errors, True,
                            "" if valid.all() else "missing_coverage")
