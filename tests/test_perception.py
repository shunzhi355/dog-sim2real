"""Synthetic perception and process-contract tests; no camera hardware required."""

from copy import deepcopy
import multiprocessing as mp
import os
from queue import Queue
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dog3588 import vision
from dog3588.vision import (LineDetector, PerceptionDetector, ProcessCamera,
                            VisionState, detect_colours, line_velocity)


@pytest.fixture
def camera_cfg():
    # Self-contained: robot.yaml intentionally retains legacy black-line HSV.
    return {
        "source": 0, "backend": "any", "max_frame_age_s": .15,
        "line": {"roi_top": .5, "hsv_low": [0, 0, 180],
                 "hsv_high": [179, 65, 255], "min_area_px": 30},
        "follow": {"lateral_gain": .5, "heading_gain": .4,
                   "max_yaw_rad_s": .4, "forward_m_s": .12},
        "colours": {
            "red": [[[0, 100, 191], [10, 255, 255]], [[170, 100, 191], [179, 255, 255]]],
            "green": [[[35, 80, 60], [85, 255, 255]]],
            "blue": [[[95, 100, 60], [130, 255, 255]]],
            "yellow": [[[20, 100, 100], [35, 255, 255]]],
            "brown": [[[0, 100, 70], [15, 255, 190]]],
        },
        "semantic_rois": {"track": [0, .5, 1, 1], "parcel": [0, 0, .3, .4]},
        "colour_filter": {"min_area_px": 100, "min_width_fraction": .4,
                          "min_aspect_ratio": 2.5, "min_confidence": .6},
        "parcel_filter": {"min_area_px": 80, "min_confidence": .5},
    }


def blank():
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.mark.parametrize("centre,sign", [(80, -1), (160, 0), (240, 1)])
def test_white_track_on_black_background_direction(camera_cfg, centre, sign):
    frame = blank()
    frame[120:, centre - 20:centre + 21] = 255
    result = LineDetector(camera_cfg["line"]).detect(frame)
    assert result["found"] and result["confidence"] > .5
    assert np.sign(result["error"]) == sign
    assert np.sign(line_velocity(result, camera_cfg["follow"])[2]) == -sign
    assert result["heading"] == pytest.approx(0)


@pytest.mark.parametrize("far_x,near_x,sign", [(225, 160, 1), (95, 160, -1)])
def test_heading_sign(camera_cfg, far_x, near_x, sign):
    frame = blank()
    cv2.line(frame, (far_x, 120), (near_x, 239), (255, 255, 255), 18)
    result = LineDetector(camera_cfg["line"]).detect(frame)
    assert result["found"] and np.sign(result["heading"]) == sign
    assert np.sign(line_velocity(result, camera_cfg["follow"])[2]) == -sign


def test_line_loss_and_legacy_black_line(camera_cfg):
    assert not LineDetector(camera_cfg["line"]).detect(blank())["found"]
    legacy = dict(camera_cfg["line"], hsv_low=[0, 0, 0], hsv_high=[179, 255, 70])
    frame = np.full_like(blank(), 255)
    frame[120:, 230:250] = 0
    assert LineDetector(legacy).detect(frame)["error"] > 0
    assert line_velocity(LineDetector(camera_cfg["line"]).detect(blank()),
                         camera_cfg["follow"]) == [0., 0., 0.]


def test_line_roi_keeps_full_image_error_coordinates(camera_cfg):
    frame = blank()
    frame[120:, 230:250] = 255
    line = LineDetector(dict(camera_cfg["line"], roi=[.5, .5, 1, 1])).detect(frame)
    assert line["error"] == pytest.approx((239.5 - 160) / 160)


@pytest.mark.parametrize("name,bgr", [
    ("red", (0, 0, 255)), ("green", (0, 255, 0)), ("blue", (255, 0, 0)),
    ("yellow", (0, 255, 255)), ("brown", (42, 42, 165)),
])
def test_colour_bands_all_five_colours(camera_cfg, name, bgr):
    frame = blank()
    frame[175:195, 25:295] = bgr
    state = PerceptionDetector(camera_cfg).detect(frame, 10.)
    assert name in [item["colour"] for item in state.colours]
    item = next(item for item in state.colours if item["colour"] == name)
    assert item["bbox"] == [25, 175, 270, 20]
    assert item["confidence"] > .9
    assert state.events == frozenset()  # never manufacture an arrival event
    assert state.parcel_colour is None


def test_wrapped_red_hue_and_legacy_detect_colours_signature(camera_cfg):
    hsv = np.full((30, 40, 3), (179, 255, 255), dtype=np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert detect_colours(frame, camera_cfg["colours"], 200)[0]["colour"] == "red"


def test_yellow_ball_does_not_become_door_band(camera_cfg):
    frame = blank()
    cv2.circle(frame, (160, 180), 35, (0, 255, 255), -1)
    state = PerceptionDetector(camera_cfg).detect(frame, 10.)
    assert not state.colours and not state.events and state.parcel_colour is None


def test_band_selected_even_when_larger_ball_exists(camera_cfg):
    frame = blank()
    cv2.circle(frame, (160, 150), 29, (0, 255, 255), -1)
    frame[210:218, 20:300] = (0, 255, 255)
    state = PerceptionDetector(camera_cfg).detect(frame, 10.)
    assert state.colours[0]["bbox"] == [20, 210, 280, 8]


def test_parcel_and_track_colours_are_independent(camera_cfg):
    frame = blank()
    cv2.circle(frame, (45, 45), 20, (0, 255, 255), -1)
    frame[180:200, 20:300] = (42, 42, 165)
    state = PerceptionDetector(camera_cfg).detect(frame, 10.)
    assert state.parcel_colour == "yellow"
    assert [item["colour"] for item in state.colours] == ["brown"]
    assert state.events == frozenset()


def test_overlapping_parcel_roi_excluded_from_track(camera_cfg):
    camera_cfg["semantic_rois"] = {"track": [0, 0, 1, 1], "parcel": [.05, .1, .95, .3]}
    frame = blank()
    frame[30:50, 20:300] = (0, 255, 255)
    state = PerceptionDetector(camera_cfg).detect(frame, 10.)
    assert state.parcel_colour == "yellow"
    assert state.colours == []


def test_unconfigured_or_ambiguous_parcel_returns_none(camera_cfg):
    frame = blank()
    frame[10:35, 10:35] = (0, 255, 255)
    frame[50:75, 50:75] = (42, 42, 165)
    assert PerceptionDetector(camera_cfg).detect(frame, 10.).parcel_colour is None
    frame[50:75, 50:75] = 0
    camera_cfg["semantic_rois"]["parcel"] = None
    assert PerceptionDetector(camera_cfg).detect(frame, 10.).parcel_colour is None


def test_filters_area_width_and_fill_confidence(camera_cfg):
    frame = blank()
    frame[180:190, 20:300] = (0, 255, 255)
    ranges = {"yellow": camera_cfg["colours"]["yellow"]}
    assert detect_colours(frame, ranges, min_area=4000) == []
    assert detect_colours(frame, ranges, min_width_fraction=.95) == []
    frame[:] = 0
    cv2.fillPoly(frame, [np.array([[20, 160], [300, 160], [160, 200]])], (0, 255, 255))
    assert detect_colours(frame, ranges, min_confidence=.7) == []
    assert len(detect_colours(frame, ranges, min_confidence=.4)) == 1


@pytest.mark.parametrize("roi", [[0, 0, 0, 1], [0, 0, 2, 1], [0, float("nan"), 1, 1]])
def test_bad_semantic_roi_rejected(camera_cfg, roi):
    camera_cfg["semantic_rois"]["track"] = roi
    with pytest.raises(ValueError, match="ROI"):
        PerceptionDetector(camera_cfg)


def test_events_only_enter_through_explicit_replay(camera_cfg):
    detector = PerceptionDetector(camera_cfg)
    frame = blank()
    assert detector.detect(frame, 10.).events == frozenset()
    replay = detector.detect(frame, 10., events={"home", "obstacle_passed", "turn_complete"})
    assert replay.events == frozenset({"home", "obstacle_passed", "turn_complete"})
    assert replay.stamp == 10.
    assert detector.detect(frame, 11.).events == frozenset()  # no event latching
    with pytest.raises(ValueError, match="events"):
        VisionState(events="home")


def test_latest_queue_discards_backlog():
    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=1)
    try:
        for stamp in range(1, 40):
            vision._put_latest(queue, VisionState(float(stamp)))
        assert queue.get(timeout=2).stamp == 39.
    finally:
        queue.close()
        queue.join_thread()


def test_read_freshness_never_restamps_and_clears_semantics(camera_cfg, monkeypatch):
    camera = ProcessCamera(camera_cfg)
    # Synchronous mailbox makes the boundary deterministic; real IPC is tested separately.
    camera._results.close()
    camera._results = Queue(maxsize=1)
    sample = VisionState(10., {"found": True, "error": 0., "heading": 0., "confidence": 1.},
                         [{"colour": "yellow"}], {"home"}, "brown")
    # Freeze only the camera clock. Linux multiprocessing also uses time.monotonic
    # to finish zero-timeout polls; freezing that global clock can hang get_nowait.
    monkeypatch.setattr(vision, "time", SimpleNamespace(monotonic=lambda: 10.05))
    camera._results.put_nowait(sample)
    first = camera.read()
    assert first.stamp == 10. and first.parcel_colour == "brown"
    first.line["found"] = False
    assert camera.read().line["found"]  # returned snapshot cannot mutate the cache
    monkeypatch.setattr(vision, "time", SimpleNamespace(monotonic=lambda: 10.3))
    stale = camera.read()
    assert stale.stamp == 10. and not stale.line["found"]
    assert stale.colours == [] and not stale.events and stale.parcel_colour is None
    # A late-delivered old frame does not become fresh at dequeue time.
    camera._results.put_nowait(sample)
    assert not camera.read().line["found"]
    camera._results = camera._context.Queue(maxsize=1)
    camera.close()


@pytest.mark.parametrize("read_delay,processing_delay,found", [
    (.5, 0., False), (.01, .5, False), (.01, .01, True),
])
def test_worker_stamps_before_read_and_processing(camera_cfg, monkeypatch,
                                                read_delay, processing_delay, found):
    finish = threading.Event()
    results, errors = Queue(maxsize=1), Queue(maxsize=1)
    results.cancel_join_thread = lambda: None
    clock = [10.]

    class Capture:
        released = False
        def isOpened(self): return True
        def set(self, *_): return True
        def read(self):
            clock[0] += read_delay
            frame = blank()
            frame[120:, 140:180] = 255
            return True, frame
        def release(self): self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: capture)
    monkeypatch.setattr(vision, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    real_detect = PerceptionDetector.detect

    def slow_detect(self, frame, stamp):
        result = real_detect(self, frame, stamp)
        clock[0] += processing_delay
        return result

    monkeypatch.setattr(PerceptionDetector, "detect", slow_detect)

    def publish(queue, state):
        queue.put_nowait(state)
        finish.set()

    monkeypatch.setattr(vision, "_put_latest", publish)
    vision._camera_worker(camera_cfg, results, errors, finish)
    state = results.get_nowait()
    assert state.stamp == 10. and state.fresh(clock[0], .15) == found
    assert state.line["found"] == found
    assert not state.colours and not state.events
    assert capture.released and errors.empty()


def test_worker_read_failure_is_reported_and_capture_released(camera_cfg, monkeypatch):
    results, errors = Queue(maxsize=1), Queue(maxsize=1)
    results.cancel_join_thread = lambda: None

    class Capture:
        released = False
        def isOpened(self): return True
        def set(self, *_): return True
        def read(self): return False, None
        def release(self): self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: capture)
    vision._camera_worker(camera_cfg, results, errors, threading.Event())
    assert "frame acquisition failed" in errors.get_nowait()
    assert capture.released and results.empty()


def _synthetic_child(cfg, results, errors, finish):
    """Picklable spawn target: flood IPC, then deliberately stop publishing."""
    for index in range(50):
        state = VisionState(time.monotonic(), {"found": True, "index": index, "pid": os.getpid()})
        vision._put_latest(results, state)
    finish.wait(10)


def test_spawn_latest_read_does_not_wait_for_capture(camera_cfg, monkeypatch):
    monkeypatch.setattr(vision, "_camera_worker", _synthetic_child)
    camera_cfg["max_frame_age_s"] = 3.
    camera = ProcessCamera({"vision": camera_cfg})
    try:
        camera.start()
        camera.start()  # lifecycle idempotence
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            state = camera.read()
            if state.line.get("index") == 49:
                break
            time.sleep(.005)
        else:
            pytest.fail("spawned camera never published the latest result")
        assert state.line["pid"] != os.getpid()
        began = time.monotonic()
        for _ in range(100):
            assert camera.read().stamp == state.stamp
        assert time.monotonic() - began < .5
    finally:
        camera.close()
        camera.close()


def _unexpected_exit_child(*_):
    os._exit(7)


def test_unexpected_child_exit_is_error(camera_cfg, monkeypatch):
    monkeypatch.setattr(vision, "_camera_worker", _unexpected_exit_child)
    camera = ProcessCamera(camera_cfg)
    try:
        camera.start()
        camera._process.join(8)
        with pytest.raises(RuntimeError, match="exitcode=7"):
            camera.read()
        assert camera.error
    finally:
        camera.close()


def test_camera_open_error_propagates_without_hardware(camera_cfg, tmp_path):
    camera_cfg["source"] = str(tmp_path / "nonexistent.avi")
    camera = ProcessCamera(camera_cfg)
    try:
        camera.start()
        camera._process.join(8)
        with pytest.raises(RuntimeError, match="cannot open camera"):
            camera.read()
    finally:
        camera.close()


def _stalled_child(*_):
    time.sleep(20)


def test_close_terminates_stalled_capture(camera_cfg, monkeypatch):
    monkeypatch.setattr(vision, "_camera_worker", _stalled_child)
    camera_cfg["close_timeout_s"] = .1
    camera = ProcessCamera(camera_cfg)
    camera.start()
    began = time.monotonic()
    camera.close()
    assert time.monotonic() - began < 3
    camera.close()
    with pytest.raises(RuntimeError, match="closed"):
        camera.read()


def test_initial_state_and_invalid_freshness(camera_cfg):
    camera = ProcessCamera(camera_cfg)
    try:
        assert camera.read() == VisionState()
        assert not VisionState(11.).fresh(10., .15)
        assert not VisionState(float("nan")).fresh(10., .15)
        assert not VisionState(0.).fresh(0., .15)
    finally:
        camera.close()
    bad = deepcopy(camera_cfg)
    bad["max_frame_age_s"] = float("nan")
    with pytest.raises(ValueError, match="max_frame_age_s"):
        ProcessCamera(bad)
