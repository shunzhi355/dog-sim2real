"""Camera perception and the backwards-compatible velocity commissioning command.

Runtime integration (camera work never runs in the control loop)::

    camera = ProcessCamera(cfg)
    camera.start()                       # once, before the control loop
    state = camera.read()                # each tick: non-waiting latest snapshot
    # Pass state.stamp, .line, .colours, .events, .parcel_colour to MissionInput.
    # Runtime must reject an expired stamp using vision.max_frame_age_s.
    camera.close()                       # in runtime's finally block

``vision.semantic_rois`` contains ``track`` and ``parcel`` rectangles, normalized
as [left, top, right, bottom]. Track defaults to the line's lower-image ROI;
parcel defaults to None (uncalibrated/disabled). Parcel pixels are excluded from
track colours, even if the ROIs overlap. ``vision.colour_filter`` accepts
min_area_px (200), min_width_fraction (.35, relative to track ROI width),
min_aspect_ratio (2.5), and min_confidence (.6). ``vision.parcel_filter`` accepts
the same keys with defaults 80, 0, 0, .5. Confidence is contour/bounding-box fill,
NOT a calibrated probability. ``vision.parcel_colours`` defaults to yellow/brown.
These thresholds require camera/lighting calibration; a colour candidate is not
proof of arrival, crossing a doorway, or successful delivery.
HSV intervals must distinguish red/brown (their hue ranges can overlap): this
module does not silently give one configured colour priority over another.

Events are UNCALIBRATED PERCEPTION HOOKS: live camera states have empty events.
There is no observable basis here for home, docked, obstacle_passed, or
turn_complete. An external, explicitly annotated replay can construct VisionState
or call PerceptionDetector.detect(..., events={...}) and pass those events into
MissionInput, preserving the acquisition stamp (rebase recorded time once at
replay startup, never on every read). No live event injection, extra field
markers, ArUco dependency, or inference from colour/elapsed time is enabled.
Runtime/FSM wiring is the caller's responsibility; this module never moves joints.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
import multiprocessing as mp
from queue import Empty, Full
import time

import numpy as np

from .commands import send_command


def _lost_line():
    return {"found": False, "error": 0., "heading": 0., "confidence": 0.}


@dataclass(frozen=True)
class VisionState:
    """Snapshot timestamped BEFORE acquisition, in the local monotonic clock.

    Repeated read() calls retain this stamp. Events are level observations for
    this frame, not latched notifications; consumers must deduplicate by stamp.
    Stamp zero is the startup/no-frame sentinel.
    """

    stamp: float = 0.
    line: dict = field(default_factory=_lost_line)
    colours: list = field(default_factory=list)
    events: frozenset = field(default_factory=frozenset)
    parcel_colour: str | None = None

    def __post_init__(self):
        if isinstance(self.events, str):
            raise ValueError("events must be a collection of nonempty event names")
        events = frozenset(self.events)
        if any(not isinstance(e, str) or not e for e in events):
            raise ValueError("events must be a collection of nonempty event names")
        object.__setattr__(self, "events", events)

    def fresh(self, now, max_age_s):
        return (math.isfinite(self.stamp) and self.stamp > 0
                and 0 <= now - self.stamp <= max_age_s)


def _check_frame(frame):
    if (frame is None or frame.ndim != 3 or frame.shape[2] != 3
            or frame.dtype != np.uint8 or min(frame.shape[:2]) < 2):
        raise ValueError("camera must supply a nonempty uint8 BGR image")


def _validate_roi(roi):
    values = np.asarray(roi, dtype=float)
    if (values.shape != (4,) or not np.isfinite(values).all()
            or not 0 <= values[0] < values[2] <= 1
            or not 0 <= values[1] < values[3] <= 1):
        raise ValueError("ROI must be normalized [left, top, right, bottom] within 0..1")
    return values


def _roi_bounds(frame, roi):
    left, top, right, bottom = _validate_roi(roi)
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = int(left * w), int(top * h), int(right * w), int(bottom * h)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("ROI has no pixels at this camera resolution")
    return x0, y0, x1, y1


class LineDetector:
    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, frame):
        import cv2

        _check_frame(frame)
        width = frame.shape[1]
        roi = self.cfg.get("roi", [0, self.cfg.get("roi_top", .55), 1, 1])
        x0, y0, x1, y1 = _roi_bounds(frame, roi)
        if y1 - y0 < 2:
            return _lost_line()
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        lo, hi = np.array(self.cfg["hsv_low"]), np.array(self.cfg["hsv_high"])
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        points, bands = [], []
        offset_x = x0
        for y0, y1 in ((0, mask.shape[0] // 2), (mask.shape[0] // 2, mask.shape[0])):
            contours, _ = cv2.findContours(mask[y0:y1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < self.cfg["min_area_px"]:
                continue
            m = cv2.moments(contour)
            if m["m00"] <= 0:
                continue
            points.append((m["m10"] / m["m00"] + offset_x, m["m01"] / m["m00"] + y0))
            bands.append(area)
        if len(points) != 2:
            return _lost_line()
        far, near = points
        return {"found": True, "error": (near[0] - width / 2) / (width / 2),
                "heading": math.atan2(far[0] - near[0], max(near[1] - far[1], 1)),
                "confidence": min(1., min(bands) / (self.cfg["min_area_px"] * 5))}


def detect_colours(frame, ranges, min_area=200, *, roi=None,
                   min_width_fraction=0., min_aspect_ratio=0., min_confidence=0.,
                   exclude_rois=()):
    """Colour candidates; old three-argument calls still search the full frame.

    Bounding boxes use full-frame pixels. Every contour is filtered before
    choosing the largest qualifying one, so a large ball cannot hide a band.
    """
    import cv2

    _check_frame(frame)
    if (not math.isfinite(min_area) or min_area < 0
            or not 0 <= min_width_fraction <= 1
            or not math.isfinite(min_aspect_ratio) or min_aspect_ratio < 0
            or not 0 <= min_confidence <= 1):
        raise ValueError("invalid colour area/width/aspect/confidence filter")
    x0, y0, x1, y1 = _roi_bounds(frame, [0, 0, 1, 1] if roi is None else roi)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    excluded = [_roi_bounds(frame, excluded_roi) for excluded_roi in exclude_rois]
    found = []
    for name, intervals in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in intervals:
            mask |= cv2.inRange(hsv, np.array(low), np.array(high))
        for ex0, ey0, ex1, ey1 in excluded:
            left, top = max(x0, ex0), max(y0, ey0)
            right, bottom = min(x1, ex1), min(y1, ey1)
            if left < right and top < bottom:
                mask[top - y0:bottom - y0, left - x0:right - x0] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            confidence = float(area / (w * h))
            width_fraction = w / (x1 - x0)
            if (area >= min_area and width_fraction >= min_width_fraction
                    and w / h >= min_aspect_ratio and confidence >= min_confidence):
                candidates.append({"colour": name, "area": float(area),
                                   "bbox": [x + x0, y + y0, w, h],
                                   "confidence": confidence, "width_fraction": width_fraction})
        if candidates:
            found.append(max(candidates, key=lambda candidate: candidate["area"]))
    return found


class PerceptionDetector:
    """Synchronous frame processing for the child process and annotated replay."""

    def __init__(self, cfg):
        self.cfg = cfg.get("vision", cfg)
        self.line = LineDetector(self.cfg["line"])
        rois = self.cfg.get("semantic_rois", {})
        self.track_roi = rois.get("track", [0, self.cfg["line"].get("roi_top", .55), 1, 1])
        self.parcel_roi = rois.get("parcel")
        _validate_roi(self.track_roi)
        if self.parcel_roi is not None:
            _validate_roi(self.parcel_roi)

    def _colours(self, frame, ranges, roi, settings, *, parcel=False):
        return detect_colours(
            frame, ranges, settings.get("min_area_px", 80 if parcel else 200), roi=roi,
            min_width_fraction=settings.get("min_width_fraction", 0. if parcel else .35),
            min_aspect_ratio=settings.get("min_aspect_ratio", 0. if parcel else 2.5),
            min_confidence=settings.get("min_confidence", .5 if parcel else .6),
            exclude_rois=() if parcel or self.parcel_roi is None else (self.parcel_roi,))

    def detect(self, frame, stamp, *, events=frozenset()):
        """events is an EXTERNAL REPLAY annotation, never inferred from this image."""
        colours = self._colours(frame, self.cfg["colours"], self.track_roi,
                                self.cfg.get("colour_filter", {}))
        parcel_colour = None
        if self.parcel_roi is not None:
            ranges = {name: intervals for name, intervals in self.cfg["colours"].items()
                      if name in self.cfg.get("parcel_colours", ("yellow", "brown"))}
            parcels = self._colours(frame, ranges, self.parcel_roi,
                                    self.cfg.get("parcel_filter", {}), parcel=True)
            # Ambiguous multi-colour objects must not choose a route arbitrarily.
            if len(parcels) == 1:
                parcel_colour = parcels[0]["colour"]
        return VisionState(stamp, self.line.detect(frame), colours, events, parcel_colour)


def _put_latest(queue, state):
    """Single producer, one-slot mailbox; waits only in the camera child.

    multiprocessing.Queue has a feeder thread: Full followed by Empty is possible.
    A bounded wait lets the feeder expose the old item before replacing it.
    """
    while True:
        try:
            queue.put_nowait(state)
            return
        except Full:
            try:
                queue.get(timeout=.01)
            except Empty:
                pass


def _camera_worker(vision, results, errors, finish):
    cap = None
    try:
        import cv2

        cv2.setNumThreads(1)
        detector = PerceptionDetector(vision)
        backend = cv2.CAP_V4L2 if vision.get("backend") == "v4l2" else cv2.CAP_ANY
        cap = cv2.VideoCapture(vision["source"], backend)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {vision['source']}")
        for key, prop in (("width", cv2.CAP_PROP_FRAME_WIDTH),
                          ("height", cv2.CAP_PROP_FRAME_HEIGHT), ("fps", cv2.CAP_PROP_FPS)):
            if key in vision:
                cap.set(prop, vision[key])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Buffer-size support is driver dependent. Acquisition-start timestamps
        # bound read/processing latency, not the unknown sensor exposure latency.
        while not finish.is_set():
            stamp = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera frame acquisition failed")
            if finish.is_set():
                break
            if time.monotonic() - stamp > vision.get("max_frame_age_s", .15):
                state = VisionState(stamp)
            else:
                state = detector.detect(frame, stamp)
                if time.monotonic() - stamp > vision.get("max_frame_age_s", .15):
                    state = VisionState(stamp)
            _put_latest(results, state)
    except BaseException as exc:
        errors.put_nowait(f"camera: {type(exc).__name__}: {exc}")
    finally:
        # Never let an unread data snapshot hold the child open during shutdown.
        results.cancel_join_thread()
        if cap is not None:
            cap.release()


class ProcessCamera:
    """Spawn-isolated capture with a one-slot result queue and non-waiting read.

    read() raises RuntimeError on a reported failure or an unexpected process
    exit (even with a cached frame). Expired snapshots retain their old stamp
    but clear line/colours/events/parcel. A stalled driver therefore cannot
    resurrect old detections. start() is idempotent; close() is bounded and
    idempotent. Use the normal __main__ guard on spawn-based platforms.
    """

    def __init__(self, cfg):
        self.cfg = deepcopy(cfg.get("vision", cfg))
        self.max_age = self.cfg.get("max_frame_age_s", .15)
        self.close_timeout = self.cfg.get("close_timeout_s", .5)
        if not math.isfinite(self.max_age) or self.max_age <= 0:
            raise ValueError("max_frame_age_s must be finite and positive")
        if not math.isfinite(self.close_timeout) or not 0 < self.close_timeout <= 5:
            raise ValueError("close_timeout_s must be within (0, 5]")
        self._context = mp.get_context("spawn")
        self._results = self._context.Queue(maxsize=1)
        self._errors = self._context.Queue(maxsize=1)
        self._finish = self._context.Event()
        self._process = None
        self._sample = VisionState()
        self._closed = False
        self.error = None

    def start(self):
        if self._closed:
            raise RuntimeError("camera is closed")
        if self._process is not None:
            return
        process = self._context.Process(target=_camera_worker,
                                        args=(self.cfg, self._results, self._errors, self._finish),
                                        name="camera", daemon=True)
        process.start()
        self._process = process

    def read(self):
        if self._closed:
            raise RuntimeError("camera is closed")
        try:
            self.error = self._errors.get_nowait()
        except Empty:
            pass
        if self._process is not None and self._process.exitcode is not None and not self.error:
            self.error = f"camera process exited unexpectedly (exitcode={self._process.exitcode})"
        if self.error:
            raise RuntimeError(self.error)
        # Exactly one non-waiting read: no drain loop that a fast producer can
        # keep alive. Pixels and OpenCV never cross into the runtime process.
        try:
            candidate = self._results.get_nowait()
            if candidate.stamp >= self._sample.stamp:
                self._sample = candidate
        except Empty:
            pass
        if not self._sample.fresh(time.monotonic(), self.max_age):
            return VisionState(self._sample.stamp)
        # Preserve snapshot ownership despite mutable dict/list compatibility.
        return deepcopy(self._sample)

    def close(self):
        if self._closed:
            return
        self._finish.set()
        process = self._process
        if process is not None:
            process.join(self.close_timeout)
            if process.is_alive():
                process.terminate()
                process.join(self.close_timeout)
            if process.is_alive():
                process.kill()
                process.join(self.close_timeout)
            if process.is_alive():
                raise RuntimeError("camera process could not be stopped")
            process.close()
        for queue in (self._results, self._errors):
            queue.cancel_join_thread()
            queue.close()
        self._closed = True


def line_velocity(result, cfg):
    if not result["found"]:
        return [0., 0., 0.]
    # A line to the image's right needs a negative (clockwise) body yaw command.
    yaw = -(cfg["lateral_gain"] * result["error"] + cfg["heading_gain"] * result["heading"])
    yaw = float(np.clip(yaw, -cfg["max_yaw_rad_s"], cfg["max_yaw_rad_s"]))
    speed = cfg["forward_m_s"] * max(0.3, 1 - abs(result["error"]))
    return [speed, 0., yaw]


def run_camera(cfg, *, seconds=0., publish=False, output=None):
    import cv2

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('camera seconds must be finite and nonnegative')
    cv2.setNumThreads(1)
    vision = cfg["vision"]
    source = vision["source"]
    backend = cv2.CAP_V4L2 if vision["backend"] == "v4l2" else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open camera {source}; check driver and /dev/videoN capture capability")
    began, last_report = time.monotonic(), 0.
    detector = PerceptionDetector(vision)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, vision["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vision["height"])
        cap.set(cv2.CAP_PROP_FPS, vision["fps"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while seconds == 0 or time.monotonic() - began < seconds:
            acquisition_start = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera frame acquisition failed")
            result = detector.detect(frame, acquisition_start)
            line, colours = result.line, result.colours
            velocity = line_velocity(line, vision["follow"])
            if time.monotonic() - acquisition_start > vision["max_frame_age_s"]:
                velocity = [0., 0., 0.]
            if publish:
                send_command(velocity, source="vision", port=cfg["command"]["port"])
            now = time.monotonic()
            if now - last_report >= 1:
                print(json.dumps({"line": line, "colours": colours, "velocity": velocity,
                                  "parcel_colour": result.parcel_colour,
                                  "frame_age_s": now - acquisition_start,
                                  "publishing": publish}), flush=True)
                last_report = now
            if output:
                # Commissioning snapshot, overwritten intentionally at the requested path.
                if not cv2.imwrite(str(output), frame):
                    raise RuntimeError("could not write camera snapshot")
                output = None
    finally:
        cap.release()
        if publish:
            send_command(source="vision", port=cfg["command"]["port"])

