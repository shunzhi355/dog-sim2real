"""Camera/colour/line following process. It publishes body velocity, never joint targets."""

import json
import math
import time

import numpy as np

from .commands import send_command


class LineDetector:
    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, frame):
        import cv2

        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("camera must supply a BGR image")
        height, width = frame.shape[:2]
        start = int(height * self.cfg["roi_top"])
        hsv = cv2.cvtColor(frame[start:], cv2.COLOR_BGR2HSV)
        lo, hi = np.array(self.cfg["hsv_low"]), np.array(self.cfg["hsv_high"])
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        points, bands = [], []
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
            points.append((m["m10"] / m["m00"], m["m01"] / m["m00"] + y0))
            bands.append(area)
        if len(points) != 2:
            return {"found": False, "error": 0., "heading": 0., "confidence": 0.}
        far, near = points
        return {"found": True, "error": (near[0] - width / 2) / (width / 2),
                "heading": math.atan2(far[0] - near[0], max(near[1] - far[1], 1)),
                "confidence": min(1., min(bands) / (self.cfg["min_area_px"] * 5))}


def detect_colours(frame, ranges, min_area=200):
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    found = []
    for name, intervals in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in intervals:
            mask |= cv2.inRange(hsv, np.array(low), np.array(high))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area >= min_area:
                found.append({"colour": name, "area": float(area),
                              "bbox": list(cv2.boundingRect(contour))})
    return found


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

    cv2.setNumThreads(1)
    vision = cfg["vision"]
    source = vision["source"]
    backend = cv2.CAP_V4L2 if vision["backend"] == "v4l2" else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open camera {source}; check driver and /dev/videoN capture capability")
    began, last_report = time.monotonic(), 0.
    detector = LineDetector(vision["line"])
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
            line = detector.detect(frame)
            colours = detect_colours(frame, vision["colours"])
            velocity = line_velocity(line, vision["follow"])
            if time.monotonic() - acquisition_start > vision["max_frame_age_s"]:
                velocity = [0., 0., 0.]
            if publish:
                send_command(velocity, source="vision", port=cfg["command"]["port"])
            now = time.monotonic()
            if now - last_report >= 1:
                print(json.dumps({"line": line, "colours": colours, "velocity": velocity,
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

