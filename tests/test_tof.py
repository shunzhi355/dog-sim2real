"""Analytic scenes and fake drivers only; no I2C device is accessed."""

from copy import deepcopy
import ctypes
from dataclasses import replace
import math
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from dog3588.tof import ThreadedToF, TerrainProjector, ToFFrame
from dog3588.types import ImuState


def level_rotation(roll, pitch):
    sr, cr, sp, cp = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch)
    return np.array([[cp, sp * sr, sp * cr], [0, cr, -sr], [-sp, cp * sr, cp * cr]])


def scene(roll=0.0, pitch=0.0, heights=None):
    """Generate independent known level points, then invert optics/extrinsics."""
    xy = np.array([[x, y] for x in (.10, .18, .26, .34) for y in (.06, -.06)])
    heights = np.full(8, -.30) if heights is None else np.asarray(heights)
    points = np.array([[x + offset * .002, y, z]
                       for (x, y), z in zip(xy, heights) for offset in range(8)])
    body_to_level = level_rotation(roll, pitch)
    rotation = level_rotation(0, -.35) @ np.diag([1., -1., -1.])
    origin = np.array([.025, -.012, -.03])
    sensor_points = (points @ body_to_level - origin) @ rotation
    assert (sensor_points[:, 2] > 0).all()
    rays = sensor_points / np.linalg.norm(sensor_points, axis=1)[:, None]
    cfg = {
        "verified": True, "rays_verified": True, "mount_verified": True,
        "sampling_verified": True, "reference_verified": True,
        "ray": rays.tolist(), "sensor_to_body": rotation.tolist(),
        "origin_body_m": origin.tolist(), "sample_xy_m": xy.tolist(),
        "max_xy_error_m": .016,
    }
    frame = ToFFrame(sensor_points[:, 2].copy(), np.full(64, 5), np.ones(64), 10.)
    imu = ImuState(np.zeros(3), -body_to_level[2], 10.)
    return cfg, frame, imu


def contract(cfg):
    return {"frame": "leveled_body", "units": "m", "reference": "body_origin",
            "sample_xy_m": deepcopy(cfg["sample_xy_m"]), "invalid_policy": "stop"}


@pytest.mark.parametrize("roll,pitch", [(0., 0.), (.24, -.31), (-.29, .25)])
def test_flat_plane_optical_depth_extrinsics_gravity_and_exact_sampling(roll, pitch):
    cfg, frame, imu = scene(roll, pitch)
    state = TerrainProjector(cfg).project(frame, imu, 10.01)
    assert state.calibrated and state.valid.dtype == np.bool_ and state.valid.all()
    assert state.heights.shape == state.valid.shape == state.coverage.shape == (8,)
    np.testing.assert_allclose(state.heights, -.30, atol=1e-12)
    np.testing.assert_array_equal(state.coverage, 8)
    np.testing.assert_allclose(state.xy_error_m, 0, atol=1e-12)
    assert state.stamp == 10. and state.reason == ""
    # Slant range and positive depth are both different from the negative body Z.
    assert np.ptp(frame.distance_m) > .02
    assert not np.allclose(frame.distance_m, frame.distance_m / np.array(cfg["ray"])[:, 2])


def test_step_order_no_offset_no_clip_and_no_mean_substitution():
    expected = np.array([-.30, -.30, -.28, -.28, -.26, -.26, -.31, -.31])
    cfg, frame, imu = scene(.17, -.19, expected)
    # The first point is at exact training XY. Other points in its neighbourhood
    # deliberately have different heights: a bin average would corrupt output.
    rotation = np.asarray(cfg["sensor_to_body"])
    b = level_rotation(.17, -.19)
    pts = np.asarray(cfg["ray"]) * (frame.distance_m / np.asarray(cfg["ray"])[:, 2])[:, None]
    pts = (pts @ rotation.T + cfg["origin_body_m"]) @ b.T
    pts[np.arange(64) % 8 != 0, 2] += .015
    sensor = (pts @ b - cfg["origin_body_m"]) @ rotation
    cfg["ray"] = (sensor / np.linalg.norm(sensor, axis=1)[:, None]).tolist()
    frame = replace(frame, distance_m=sensor[:, 2])
    state = TerrainProjector(cfg).project(frame, imu, 10.01)
    np.testing.assert_allclose(state.heights, expected, atol=1e-12)
    assert state.valid.all()


@pytest.mark.parametrize("status", [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 255])
def test_invalid_target_status_never_becomes_high_ground(status):
    cfg, frame, imu = scene()
    frame.target_status[:8] = status
    state = TerrainProjector(cfg).project(frame, imu, 10.)
    assert not state.valid[0] and np.isnan(state.heights[0]) and state.coverage[0] == 0
    assert state.valid[1:].all()


def test_status9_only_by_explicit_opt_in_and_target_count_required():
    cfg, frame, imu = scene()
    frame.target_status[:] = 9
    cfg["valid_statuses"] = [5, 9]
    frame.nb_target_detected[:8] = 0
    state = TerrainProjector(cfg).project(frame, imu, 10.)
    assert not state.valid[0] and state.valid[1:].all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.01, 0, .001, 4.01])
def test_nonfinite_and_out_of_range_depth_is_rejected(bad):
    cfg, frame, imu = scene()
    frame.distance_m[:8] = bad
    state = TerrainProjector(cfg).project(frame, imu, 10.)
    assert not state.valid[0] and state.coverage[0] == 0 and np.isnan(state.heights[0])
    assert state.valid[1:].all()


def test_missing_coverage_not_filled_from_a_different_sample_or_old_frame():
    cfg, frame, imu = scene()
    p = TerrainProjector(cfg)
    assert p.project(frame, imu, 10.).valid.all()
    frame.target_status[:8] = 255
    result = p.project(replace(frame, stamp=10.02), replace(imu, stamp=10.02), 10.02)
    assert result.reason == "missing_coverage"
    assert not result.valid[0] and np.isnan(result.heights[0]) and result.coverage[0] == 0
    assert result.xy_error_m[0] > cfg["max_xy_error_m"]
    assert result.valid[1:].all()


def test_height_outliers_rejected_before_nearest_selection_not_clipped():
    cfg, frame, imu = scene(heights=[-.3, -.3, -.3, -.3, -.3, -.3, -.3, -.8])
    cfg["min_height_m"] = -.5
    state = TerrainProjector(cfg).project(frame, imu, 10.)
    assert state.valid[:7].all() and not state.valid[7]
    assert state.coverage[7] == 0 and np.isnan(state.heights[7])


@pytest.mark.parametrize("frame_time,imu_time,now,reason", [
    (0., 10., 10., "invalid_timestamp"), (10.1, 10., 10., "invalid_timestamp"),
    (10., 10.1, 10., "invalid_timestamp"), (float("nan"), 10., 10., "invalid_timestamp"),
    (10., 10., float("inf"), "invalid_timestamp"), (10., 10., 10.101, "stale"),
    (10., 9.89, 10., "stale"), (10., 9.96, 10., "imu_skew"),
])
def test_timestamp_failures(frame_time, imu_time, now, reason):
    cfg, frame, imu = scene()
    state = TerrainProjector(cfg).project(replace(frame, stamp=frame_time),
                                          replace(imu, stamp=imu_time), now)
    assert state.reason == reason and not state.valid.any()
    assert np.isnan(state.heights).all() and not state.coverage.any()


def test_repeated_read_does_not_refresh_stamp_and_old_frames_cannot_replace_new():
    cfg, frame, imu = scene()
    p = TerrainProjector(cfg)
    assert p.project(frame, imu, 10.).valid.all()
    assert p.project(frame, imu, 10.05).valid.all()
    expired = p.project(frame, imu, 10.2)
    assert expired.reason == "stale" and expired.stamp == 10.
    older = p.project(replace(frame, stamp=9.99), imu, 10.)
    assert older.reason == "out_of_order"


@pytest.mark.parametrize("gravity,reason", [
    ([0, 0, 0], "invalid_gravity"), ([0, 0, -9.81], "invalid_gravity"),
    ([0, 0, 1], "excessive_tilt"), ([1, 0, 0], "excessive_tilt"),
    ([float("nan"), 0, -1], "malformed_frame_or_imu"),
])
def test_invalid_gravity_and_tilt(gravity, reason):
    cfg, frame, imu = scene()
    state = TerrainProjector(cfg).project(frame, replace(imu, gravity=gravity), 10.)
    assert not state.valid.any() and state.reason == reason


def test_rotation_during_frame_exposure_is_rejected():
    cfg, frame, imu = scene()
    state = TerrainProjector(cfg).project(frame, replace(imu, omega=np.array([0., 0., 2.])), 10.)
    assert state.reason == "excessive_rotation" and not state.valid.any()


@pytest.mark.parametrize("rotation", [
    np.diag([1, 1, -1]), np.diag([1, 2, 1]), np.ones((2, 3)), np.full((3, 3), np.nan),
])
def test_bad_extrinsic_rotation_rejected(rotation):
    cfg, _, _ = scene()
    cfg["sensor_to_body"] = rotation
    with pytest.raises(ValueError, match="sensor_to_body"):
        TerrainProjector(cfg)


@pytest.mark.parametrize("key,value", [
    ("ray", np.zeros((64, 3))), ("ray", np.tile([0., 0., -1.], (64, 1))),
    ("ray", np.ones((63, 3))), ("origin_body_m", [0, 0, float("nan")]),
    ("sample_xy_m", np.zeros((8, 2))), ("max_xy_error_m", 0),
    ("valid_statuses", [6]), ("max_skew_s", -.1), ("frame_timeout_s", 0),
    ("max_tilt_rad", math.pi / 2), ("min_height_m", 2),
    ("reference_height_m", 0), ("nominal_height_m", .3), ("height_clip_m", [-.05, .05]),
])
def test_bad_calibration_or_implicit_offset_rejected(key, value):
    cfg, _, _ = scene()
    cfg[key] = value
    with pytest.raises(ValueError):
        TerrainProjector(cfg)


def test_default_and_each_unverified_calibration_cannot_arm():
    cfg, frame, imu = scene()
    for settings in [{}, {"ray": None, "sample_xy_m": None}] + [
            dict(cfg, **{key: False}) for key in TerrainProjector._FLAGS]:
        projector = TerrainProjector(settings)
        with pytest.raises(ValueError, match="cannot arm"):
            projector.require_calibrated()
        state = projector.project(frame, imu, 10.)
        assert state.reason == "uncalibrated" and not state.calibrated
        assert not state.valid.any()


def test_manifest_contract_and_configuration_are_snapshots():
    cfg, _, _ = scene()
    projector = TerrainProjector(cfg)
    expected = contract(cfg)
    projector.require_calibrated(expected)
    for key, value in [("frame", "world"), ("units", "mm"), ("reference", "ground"),
                       ("invalid_policy", "zero_fill"), ("sample_xy_m", cfg["sample_xy_m"][::-1])]:
        wrong = dict(expected, **{key: value})
        with pytest.raises(ValueError, match="height_scan_contract"):
            projector.require_calibrated(wrong)
    cfg["sample_xy_m"][0][0] = 100
    projector.require_calibrated(expected)


def test_malformed_frame_is_all_invalid():
    cfg, frame, imu = scene()
    state = TerrainProjector(cfg).project(replace(frame, distance_m=np.ones(16)), imu, 10.)
    assert state.reason == "malformed_frame" and not state.valid.any()


def fake_driver(monkeypatch, *, ready=True, fail=False):
    calls = []
    distance = ((ctypes.c_int16 * 64) * 1)()
    status = ((ctypes.c_uint8 * 64) * 1)()
    count = (ctypes.c_uint8 * 64)()
    for i in range(64):
        distance[0][i], status[0][i], count[i] = 300 + i, 5, 1
    data = SimpleNamespace(distance_mm=distance, target_status=status, nb_target_detected=count)

    class Bus:
        def __init__(self, number): calls.append(("bus", number))
        def close(self): calls.append(("close",))

    class Sensor:
        def __init__(self, *, i2c_addr, i2c_dev):
            calls.append(("init", i2c_addr, isinstance(i2c_dev, Bus)))
            self.available = ready
        def set_resolution(self, n): calls.append(("resolution", n))
        def set_ranging_frequency_hz(self, n): calls.append(("frequency", n))
        def set_ranging_mode(self, n): calls.append(("mode", n))
        def set_target_order(self, n): calls.append(("order", n))
        def start_ranging(self): calls.append(("start",))
        def stop_ranging(self): calls.append(("stop",))
        def data_ready(self): return self.available
        def get_data(self):
            self.available = False
            if fail:
                raise OSError("injected I2C error")
            return data

    monkeypatch.setitem(sys.modules, "vl53l5cx_ctypes", SimpleNamespace(
        VL53L5CX=Sensor, NB_TARGET_PER_ZONE=1, RANGING_MODE_CONTINUOUS=1, TARGET_ORDER_CLOSEST=1))
    monkeypatch.setitem(sys.modules, "smbus2", SimpleNamespace(SMBus=Bus))
    return calls, data


def test_threaded_driver_real_api_ctypes_units_lifecycle_and_copy(monkeypatch):
    calls, _ = fake_driver(monkeypatch)
    device = ThreadedToF({"frame_timeout_s": 2.0})
    assert device.read().stamp == 0
    published = threading.Event()
    original = device._publish

    def publish(frame):
        original(frame)
        published.set()

    monkeypatch.setattr(device, "_publish", publish)
    try:
        device.start()
        device.start()
        assert published.wait(1)
        sample = device.read()
        np.testing.assert_allclose(sample.distance_m, np.arange(300, 364) * .001)
        assert sample.stamp > 0 and sample.distance_m.shape == (64,)
        np.testing.assert_array_equal(sample.target_status, 5)
        sample.distance_m[:] = 0
        sample.target_status[:] = 255
        reread = device.read()
        assert reread.stamp == sample.stamp and reread.distance_m[0] == .3
        assert reread.target_status[0] == 5
    finally:
        device.close()
    device.close()
    assert calls == [("bus", 2), ("init", 0x29, True), ("resolution", 64),
                     ("frequency", 15), ("mode", 1), ("order", 1), ("start",),
                     ("stop",), ("close",)]
    with pytest.raises(RuntimeError, match="closed"):
        device.read()
    with pytest.raises(RuntimeError, match="closed"):
        device.start()


def test_worker_failure_is_visible_and_resources_close(monkeypatch):
    calls, _ = fake_driver(monkeypatch, fail=True)
    device = ThreadedToF({})
    device.start()
    device.thread.join(1)
    assert not device.thread.is_alive()
    with pytest.raises(RuntimeError, match="injected I2C error"):
        device.read()
    device.close()
    assert calls[-2:] == [("stop",), ("close",)]


def test_data_ready_timeout_cleanup_without_hardware_or_sleep(monkeypatch):
    calls, _ = fake_driver(monkeypatch, ready=False)
    device = ThreadedToF({})
    ticks = iter([10., 10.01, 10.2])
    monkeypatch.setattr("dog3588.tof.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr(device.finish, "wait", lambda _: False)
    device._run()
    with pytest.raises(RuntimeError, match="data-ready timeout"):
        device.read()
    assert calls[-2:] == [("stop",), ("close",)]


def test_slow_bus_transfer_is_not_retimestamped_as_fresh(monkeypatch):
    calls, _ = fake_driver(monkeypatch)
    device = ThreadedToF({})
    ticks = iter([10., 10.01, 10.2])
    monkeypatch.setattr("dog3588.tof.time.monotonic", lambda: next(ticks))
    device._run()
    with pytest.raises(RuntimeError, match="latency"):
        device.read()
    assert device.sample.stamp == 0 and calls[-1] == ("close",)


def test_ready_buffer_after_scheduler_stall_has_unknown_age_and_is_rejected(monkeypatch):
    calls, _ = fake_driver(monkeypatch)
    device = ThreadedToF({})
    ticks = iter([10., 10.2])
    monkeypatch.setattr("dog3588.tof.time.monotonic", lambda: next(ticks))
    device._run()
    assert device.sample.stamp == 0 and isinstance(device.error, TimeoutError)
    assert calls[-2:] == [("stop",), ("close",)]


def test_configured_latency_moves_acquisition_stamp_back(monkeypatch):
    fake_driver(monkeypatch)
    device = ThreadedToF({"measurement_latency_s": .02})
    ticks = iter([10., 10.01, 10.02])
    monkeypatch.setattr("dog3588.tof.time.monotonic", lambda: next(ticks))

    def published(frame):
        device.sample = frame
        device.finish.set()

    monkeypatch.setattr(device, "_publish", published)
    device._run()
    assert device.error is None
    assert device.sample.stamp == pytest.approx(9.99)


def test_initialization_failure_still_releases_owned_bus(monkeypatch):
    calls, _ = fake_driver(monkeypatch)

    def broken_init(**_):
        raise RuntimeError("firmware upload failed")

    monkeypatch.setattr(sys.modules["vl53l5cx_ctypes"], "VL53L5CX", broken_init)
    device = ThreadedToF({})
    device._run()
    assert calls == [("bus", 2), ("close",)]
    with pytest.raises(RuntimeError, match="firmware upload"):
        device.read()


def test_blocked_close_reports_failure_with_a_bounded_join():
    device = ThreadedToF({"close_timeout_s": .01})
    joined = []
    device.started_at = 10.
    device.thread = SimpleNamespace(join=lambda timeout: joined.append(timeout),
                                    is_alive=lambda: True)
    with pytest.raises(TimeoutError, match="blocked"):
        device.close()
    assert device.finish.is_set() and joined == [.01]


def test_read_startup_and_stalled_worker_deadlines(monkeypatch):
    device = ThreadedToF({"startup_timeout_s": 1.0})
    device.started_at = 10.
    monkeypatch.setattr("dog3588.tof.time.monotonic", lambda: 12.)
    with pytest.raises(TimeoutError, match="startup"):
        device.read()
    device.sample = replace(device.sample, stamp=11.)
    with pytest.raises(TimeoutError, match="frame timeout"):
        device.read()


def test_close_before_start_is_safe():
    device = ThreadedToF({})
    device.close()
    device.close()


def test_decode_rejects_wrong_target_layout():
    data = SimpleNamespace(distance_mm=np.ones((2, 64)), target_status=np.ones(64),
                           nb_target_detected=np.ones(64))
    with pytest.raises(ValueError, match="single target"):
        ThreadedToF._decode(data, 10.)


@pytest.mark.parametrize("settings", [
    {"frequency_hz": 16}, {"frequency_hz": True}, {"i2c_address": 0x80},
    {"i2c_bus": -1}, {"driver": "unknown"}, {"measurement_latency_s": -.1},
    {"poll_interval_s": 1}, {"frame_timeout_s": float("nan")},
])
def test_invalid_driver_config_rejected(settings):
    with pytest.raises(ValueError):
        ThreadedToF(settings)
