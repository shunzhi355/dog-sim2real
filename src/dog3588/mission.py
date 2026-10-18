"""Deterministic, non-blocking competition FSM; no camera, clock or actuator I/O.

See docs/mission.md for the perception/ACK contract. Sensor timestamps and ``now``
must share the runtime's monotonic clock. Timers can stop/fault, never invent
geometry, delivery completion or arrival.
"""

from dataclasses import dataclass, field, fields
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping

from .vision import line_velocity


class MissionState(str, Enum):
    START = "start"
    FOLLOW = "follow"
    APPROACH = "approach"
    CROSS = "cross"
    STABILIZE = "stabilize"
    TURN = "turn"
    DOCK = "dock"
    DELIVERY = "delivery"
    RETURN_TURN = "return_turn"
    RETURN = "return"
    HOME = "home"
    DONE = "done"
    FAULT = "fault"


EVENTS = frozenset({"obstacle_near", "obstacle_crossed", "turn_complete", "docked",
                    "delivery_confirmed", "return_aligned", "home_visible", "home_confirmed"})
COLOURS = frozenset({"blue", "red", "green", "yellow", "brown"})
ZERO = (0., 0., 0.)


@dataclass(frozen=True)
class MissionConfig:
    parcel_colour: str | None = None
    route_map: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    return_route_map: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    forward_m_s: float = .12
    approach_m_s: float = .06
    cross_m_s: float = .06
    dock_m_s: float = .03
    turn_rad_s: float = .3
    return_turn_direction: str = "left"
    lateral_gain: float = .6
    heading_gain: float = .8
    max_yaw_rad_s: float = .4
    min_line_confidence: float = .2
    max_vision_age_s: float = .25
    max_tof_age_s: float = .25
    max_ack_age_s: float = 1.
    sensor_fault_timeout_s: float = 2.
    lost_line_timeout_s: float = 2.
    startup_timeout_s: float = 15.
    path_timeout_s: float = 120.
    turn_timeout_s: float = 12.
    approach_timeout_s: float = 10.
    cross_timeout_s: float = 20.
    stabilize_timeout_s: float = 15.
    stable_duration_s: float = 3.
    dock_timeout_s: float = 15.
    delivery_timeout_s: float = 20.
    home_timeout_s: float = 15.
    debounce_frames: int = 2
    release_frames: int = 2
    reacquire_frames: int = 3
    gait: tuple[float, ...] | None = None
    obstacle_gait: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.parcel_colour not in (None, "yellow", "brown"):
            raise ValueError("parcel_colour must be yellow, brown or None")
        for name in ("route_map", "return_route_map"):
            routes = getattr(self, name)
            frozen = {}
            for colour, markers in routes.items():
                if colour not in ("yellow", "brown"):
                    raise ValueError(f"{name}: invalid parcel colour {colour}")
                for marker, direction in markers.items():
                    if marker not in ("red", "green", "red+green") or direction not in ("left", "right", "straight"):
                        raise ValueError(f"{name}: invalid marker/direction")
                frozen[colour] = MappingProxyType(dict(markers))
            object.__setattr__(self, name, MappingProxyType(frozen))
        if self.return_turn_direction not in ("left", "right"):
            raise ValueError("return_turn_direction must be left or right")
        for item in fields(self):
            name, value = item.name, getattr(self, item.name)
            if name.endswith(("_s", "_gain")):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.min_line_confidence <= 1:
            raise ValueError("min_line_confidence must be in [0, 1]")
        if self.stable_duration_s < 3:
            raise ValueError("competition stability duration must be at least 3 seconds")
        for name in ("debounce_frames", "release_frames", "reacquire_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("gait", "obstacle_gait"):
            values = getattr(self, name)
            if values is not None:
                if len(values) != 7 or not all(math.isfinite(v) for v in values):
                    raise ValueError(f"{name} must have seven finite values")
                object.__setattr__(self, name, tuple(values))

    @classmethod
    def from_mapping(cls, values: Mapping):
        """Accept the contents of cfg['mission']; unknown keys raise TypeError."""
        return cls(**dict(values))


@dataclass(frozen=True)
class MissionInput:
    now: float
    start: bool = False
    vision_stamp: float | None = None
    tof_stamp: float | None = None
    tof_valid: bool = False
    line: Mapping = field(default_factory=dict)
    colours: tuple[str, ...] = ()
    parcel_colour: str | None = None
    events: frozenset[str] = field(default_factory=frozenset)
    stable: bool = False
    delivery_ack_id: str | None = None
    delivery_ack_status: str | None = None
    delivery_ack_stamp: float | None = None
    fault: str | None = None


@dataclass(frozen=True)
class DeliveryRequest:
    request_id: str
    colour: str


@dataclass(frozen=True)
class MissionOutput:
    velocity: tuple[float, float, float]
    gait: tuple[float, ...] | None
    mode: str
    delivery_request: DeliveryRequest | None
    state: MissionState
    reason: str | None = None


class _Debounce:
    """Count distinct camera frames, then require confirmed absence to rearm."""

    def __init__(self, on, off):
        self.on, self.off = on, off
        self.counts = {}
        self.latched = set()

    def update(self, present):
        edges = set()
        for key in set(self.counts) | present | self.latched:
            count = self.counts.get(key, 0)
            count = max(0, count) + 1 if key in present else min(0, count) - 1
            self.counts[key] = max(-self.off, min(self.on, count))
            if count >= self.on and key not in self.latched:
                self.latched.add(key)
                edges.add(key)
            elif count <= -self.off:
                self.latched.discard(key)
        return edges

    def interrupt(self):
        # Retain consumed markers across sensor outages; reset partial evidence.
        self.counts.clear()


class MissionFSM:
    """One round per instance. Call step each control tick, including sensor loss.

    ``mission_id`` must be unique per physical round (persist it across transport
    retries). A new FSM is an explicit new round, never automatic fault recovery.
    """

    def __init__(self, config: MissionConfig | Mapping, *, mission_id: str):
        self.cfg = config if isinstance(config, MissionConfig) else MissionConfig.from_mapping(config)
        if not isinstance(mission_id, str) or not mission_id.strip():
            raise ValueError("mission_id must be a nonempty unique round identifier")
        self.mission_id = mission_id
        self.state = MissionState.START
        self.parcel_colour = self.cfg.parcel_colour
        self.reason = None
        self.obstacles_completed = 0
        self.delivery_completed = False
        self.delivery_request = None
        self._started = False
        self._now = None
        self._entered = None
        self._leg_since = None
        self._vision_stamp = None
        self._tof_stamp = None
        self._debounce = _Debounce(self.cfg.debounce_frames, self.cfg.release_frames)
        self._junction_used = False
        self._pause_since = None
        self._pause_reason = None
        self._reacquired = 0
        self._stable_since = None
        self._last_stable_stamp = None
        self._resume_state = MissionState.FOLLOW
        self._turn_direction = None
        self._turn_confirmed = False
        self._turn_line_frames = 0
        self._ack_completed = False
        self._delivery_confirmed = False
        self._request_stamp = None
        self._last_ack_stamp = None

    @staticmethod
    def _fresh(stamp, now, limit):
        return (type(stamp) in (int, float) and math.isfinite(stamp)
                and 0 <= now - stamp <= limit)

    def _line_ok(self, sample):
        line = sample.line
        if not isinstance(line, Mapping):
            return False
        try:
            return (line.get("found") is True
                    and all(math.isfinite(line[k]) for k in ("error", "heading", "confidence"))
                    and abs(line["error"]) <= 1
                    and self.cfg.min_line_confidence <= line["confidence"] <= 1)
        except (KeyError, TypeError, ValueError):
            return False

    def _enter(self, state, now):
        self.state, self._entered = state, now
        self._stable_since = self._last_stable_stamp = None
        self._turn_confirmed = False
        self._turn_line_frames = 0

    def _fault(self, reason):
        self.state, self.reason = MissionState.FAULT, reason
        return self._output()

    def _output(self, velocity=ZERO, request=None, *, stopped=None):
        gait = self.cfg.gait
        if self.state in (MissionState.APPROACH, MissionState.CROSS, MissionState.STABILIZE):
            gait = self.cfg.obstacle_gait or gait
        return MissionOutput(tuple(velocity), gait, "stop" if stopped else self.state.value,
                             request, self.state, stopped or self.reason)

    def _health(self, sample, new_frame):
        now, cfg = sample.now, self.cfg
        problem = None
        if not self._fresh(sample.vision_stamp, now, cfg.max_vision_age_s):
            problem = "stale_vision"
        elif self._vision_stamp is not None and sample.vision_stamp < self._vision_stamp:
            problem = "reordered_vision"
        elif not sample.tof_valid or not self._fresh(sample.tof_stamp, now, cfg.max_tof_age_s):
            problem = "invalid_or_stale_tof"
        elif self._tof_stamp is not None and sample.tof_stamp < self._tof_stamp:
            problem = "reordered_tof"
        elif (self.state in (MissionState.FOLLOW, MissionState.APPROACH, MissionState.CROSS,
                             MissionState.DOCK, MissionState.RETURN, MissionState.HOME)
              and not self._line_ok(sample)):
            problem = "lost_line"
        if problem:
            if self._pause_since is None:
                self._pause_since, self._pause_reason = now, problem
            self._reacquired = 0
            self._stable_since = self._last_stable_stamp = None
            self._debounce.interrupt()
        if self._pause_since is not None:
            limit = (cfg.lost_line_timeout_s if self._pause_reason == "lost_line"
                     else cfg.sensor_fault_timeout_s)
            if now >= self._pause_since + limit:
                return self._fault(self._pause_reason + "_timeout")
            if not problem and new_frame:
                self._reacquired += 1
            if problem or self._reacquired < cfg.reacquire_frames:
                return self._output(stopped=problem or "reacquiring_vision")
            self._pause_since = self._pause_reason = None
            self._reacquired = 0
        return None

    def _timeout(self, now):
        deadlines = {
            MissionState.START: self.cfg.startup_timeout_s,
            MissionState.APPROACH: self.cfg.approach_timeout_s,
            MissionState.CROSS: self.cfg.cross_timeout_s,
            MissionState.STABILIZE: self.cfg.stabilize_timeout_s,
            MissionState.TURN: self.cfg.turn_timeout_s,
            MissionState.DOCK: self.cfg.dock_timeout_s,
            MissionState.DELIVERY: self.cfg.delivery_timeout_s,
            MissionState.RETURN_TURN: self.cfg.turn_timeout_s,
            MissionState.HOME: self.cfg.home_timeout_s,
        }
        limit = deadlines.get(self.state)
        if limit is not None and now >= self._entered + limit:
            return self.state.value + "_timeout"
        if (self._leg_since is not None and self.state != MissionState.DELIVERY
                and now >= self._leg_since + self.cfg.path_timeout_s):
            return "path_timeout"
        return None

    def _drive(self, sample, speed):
        cfg = {"forward_m_s": speed, "lateral_gain": self.cfg.lateral_gain,
               "heading_gain": self.cfg.heading_gain, "max_yaw_rad_s": self.cfg.max_yaw_rad_s}
        return self._output(line_velocity(sample.line, cfg))

    def _navigation(self, sample, edges):
        # Mutually inconsistent scene markers stop rather than silently choose one.
        markers = set(sample.colours)
        if "blue" in markers and markers & {"red", "green", "yellow", "brown"}:
            return self._fault("ambiguous_scene_markers")
        if "blue" in edges:
            self._resume_state = self.state
            self._enter(MissionState.APPROACH, sample.now)
            return self._output()
        if "junction" in edges:
            signature = "red+green" if {"red", "green"} <= markers else ("red" if "red" in markers else "green")
            routes = self.cfg.return_route_map if self.state == MissionState.RETURN else self.cfg.route_map
            direction = routes.get(self.parcel_colour, {}).get(signature)
            if direction is None:
                return self._fault("unmapped_route:" + signature)
            self._turn_direction, self._resume_state = direction, self.state
            self._enter(MissionState.TURN, sample.now)
            return self._output()
        if self.state == MissionState.FOLLOW:
            if {"yellow", "brown"} <= markers:
                return self._fault("ambiguous_household")
            if self.parcel_colour in edges:
                self._enter(MissionState.DOCK, sample.now)
                return self._output()
        elif "home_visible" in edges:
            self._enter(MissionState.HOME, sample.now)
            return self._output()
        return self._drive(sample, self.cfg.forward_m_s)

    def _delivery_ack(self, sample):
        if sample.delivery_ack_id != self.delivery_request.request_id:
            return None
        stamp = sample.delivery_ack_stamp
        if (not self._fresh(stamp, sample.now, self.cfg.max_ack_age_s)
                or stamp <= self._request_stamp
                or (self._last_ack_stamp is not None and stamp <= self._last_ack_stamp)):
            return None
        self._last_ack_stamp = stamp
        status = sample.delivery_ack_status
        if status in ("failed", "unsupported"):
            return self._fault("delivery_" + status)
        if status == "completed":
            self._ack_completed = True
        elif status != "accepted":
            return self._fault("invalid_delivery_ack_status")
        return None

    def step(self, sample: MissionInput) -> MissionOutput:
        """Advance at most one phase; every output is a fresh command for this tick."""
        if self.state in (MissionState.DONE, MissionState.FAULT):
            return self._output()
        if (type(sample.now) not in (int, float) or not math.isfinite(sample.now)
                or sample.now < 0 or (self._now is not None and sample.now < self._now)):
            return self._fault("invalid_monotonic_clock")
        self._now = sample.now
        if sample.fault:
            return self._fault("external_fault:" + str(sample.fault))
        if any(type(value) is not bool for value in (sample.start, sample.stable, sample.tof_valid)):
            return self._fault("invalid_input_boolean")
        try:
            events, colours = set(sample.events), set(sample.colours)
        except TypeError:
            return self._fault("invalid_perception_input")
        if not events <= EVENTS or not colours <= COLOURS:
            return self._fault("unknown_perception_event_or_colour")
        if sample.parcel_colour not in (None, "yellow", "brown"):
            return self._fault("invalid_parcel_colour")
        if not self._started:
            if not sample.start:
                return self._output()
            self._started = True
            self._entered = sample.now
        if reason := self._timeout(sample.now):
            return self._fault(reason)
        # ACKs are independent of the camera and may arrive during a perception
        # stop. Remember completion, but never resume movement without recovery.
        if self.state == MissionState.DELIVERY:
            if result := self._delivery_ack(sample):
                return result
        new_frame = (self._fresh(sample.vision_stamp, sample.now, self.cfg.max_vision_age_s)
                     and (self._vision_stamp is None or sample.vision_stamp > self._vision_stamp))
        health = self._health(sample, new_frame)
        if new_frame:
            self._vision_stamp = sample.vision_stamp
        if self._fresh(sample.tof_stamp, sample.now, self.cfg.max_tof_age_s):
            self._tof_stamp = max(self._tof_stamp or sample.tof_stamp, sample.tof_stamp)
        if health is not None:
            return health
        edges = set()
        if new_frame:
            present = set(sample.colours) - {"red", "green"}
            if set(sample.colours) & {"red", "green"}:
                # One junction latch also covers flicker between red and green.
                present.add("junction")
                signature = "red+green" if {"red", "green"} <= set(sample.colours) else ("red" if "red" in sample.colours else "green")
                present.add("route:" + signature)
            present.update(sample.events)
            if sample.parcel_colour:
                present.add("parcel:" + sample.parcel_colour)
            edges = self._debounce.update(present)
            edges.discard("junction")
            if "junction" not in self._debounce.latched:
                self._junction_used = False
            if any(key.startswith("route:") for key in edges) and not self._junction_used:
                edges.add("junction")
                self._junction_used = True
            # Cached facts from before a phase entry cannot complete that phase.
            if sample.vision_stamp <= self._entered:
                edges -= EVENTS
        state = self.state
        if state == MissionState.START:
            if self.parcel_colour is None:
                for colour in ("yellow", "brown"):
                    if "parcel:" + colour in edges:
                        self.parcel_colour = colour
            if self.parcel_colour is not None:
                self._leg_since = sample.now
                self._enter(MissionState.FOLLOW, sample.now)
            return self._output()
        if state in (MissionState.FOLLOW, MissionState.RETURN):
            return self._navigation(sample, edges)
        if state == MissionState.APPROACH:
            if "obstacle_near" in edges:
                self._enter(MissionState.CROSS, sample.now)
                return self._output()
            return self._drive(sample, self.cfg.approach_m_s)
        if state == MissionState.CROSS:
            if "obstacle_crossed" in edges:
                self._enter(MissionState.STABILIZE, sample.now)
                return self._output()
            return self._drive(sample, self.cfg.cross_m_s)
        if state == MissionState.STABILIZE:
            if new_frame and sample.vision_stamp > self._entered:
                if not sample.stable:
                    self._stable_since = self._last_stable_stamp = None
                else:
                    stamp = sample.vision_stamp
                    if (self._last_stable_stamp is None
                            or stamp - self._last_stable_stamp > self.cfg.max_vision_age_s):
                        self._stable_since = stamp
                    self._last_stable_stamp = stamp
                    if stamp >= self._stable_since + self.cfg.stable_duration_s:
                        self.obstacles_completed += 1
                        self._enter(self._resume_state, sample.now)
            return self._output()
        if state in (MissionState.TURN, MissionState.RETURN_TURN):
            event = "turn_complete" if state == MissionState.TURN else "return_aligned"
            self._turn_confirmed |= event in edges
            if new_frame:
                self._turn_line_frames = (self._turn_line_frames + 1
                                         if self._turn_confirmed and self._line_ok(sample) else 0)
            if self._turn_confirmed:
                if self._turn_line_frames >= self.cfg.reacquire_frames:
                    self._enter(self._resume_state if state == MissionState.TURN else MissionState.RETURN, sample.now)
                return self._output()  # stop rotating while reacquiring the line
            direction = self._turn_direction if state == MissionState.TURN else self.cfg.return_turn_direction
            if direction == "straight":
                return self._drive(sample, self.cfg.approach_m_s) if self._line_ok(sample) else self._output()
            return self._output((0., 0., self.cfg.turn_rad_s * (1 if direction == "left" else -1)))
        if state == MissionState.DOCK:
            if "docked" in edges:
                self._enter(MissionState.DELIVERY, sample.now)
                self.delivery_request = DeliveryRequest(self.mission_id + ":delivery:1", self.parcel_colour)
                self._request_stamp = sample.now
                return self._output(request=self.delivery_request)
            return self._drive(sample, self.cfg.dock_m_s)
        if state == MissionState.DELIVERY:
            self._delivery_confirmed |= "delivery_confirmed" in edges
            if self._ack_completed and self._delivery_confirmed:
                self.delivery_completed = True
                self._leg_since = sample.now
                self._enter(MissionState.RETURN_TURN, sample.now)
            return self._output()
        if state == MissionState.HOME:
            if "home_confirmed" in edges:
                self._enter(MissionState.DONE, sample.now)
                return self._output()
            return self._drive(sample, self.cfg.dock_m_s)
        return self._fault("invalid_state")
