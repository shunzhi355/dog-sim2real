"""Host perception composition; all device acquisition stays off the policy loop."""

from collections import deque
import math
import time
import uuid

import numpy as np

from .mission import MissionFSM, MissionInput, MissionState
from .config import positive
from .safety import SafetyFault
from .tof import TerrainProjector, TerrainState, ThreadedToF
from .vision import ProcessCamera


class BridgeImu:
    """Share the bridge's serial owner without starting/closing it twice."""

    def __init__(self, bridge):
        self.bridge = bridge

    def start(self):
        pass

    def read(self):
        return self.bridge.read_imu()

    def close(self):
        pass


class PerceptionSupervisor:
    def __init__(self, cfg, manifest, *, autonomous=False):
        self.cfg, self.manifest = cfg, manifest
        self.base_gait = np.asarray(cfg['gait'], dtype=float).copy()
        self.uses_height = any(t['name'] == 'height_scan' for t in manifest['terms'])
        self.tof = self.projector = self.camera = self.mission = None
        self.orientation_history = deque(maxlen=100)
        self.pending_frames = deque(maxlen=8)
        self.last_tof_stamp = 0.
        self.terrain = self.vision = self.output = None
        self.started = []
        self.done = False
        self.delivery_request = None
        self.delivery_wire_id = None
        self.delivery_completed_stamp = None
        stability = cfg.get('autonomy', {}).get('stability', {})
        self.stable_tilt = positive(stability.get('max_tilt_rad', .1), 'stability.max_tilt_rad')
        self.stable_omega = positive(stability.get('max_omega_rad_s', .1), 'stability.max_omega_rad_s')
        self.stable_joint_speed = positive(stability.get('max_joint_speed_rad_s', .2), 'stability.max_joint_speed_rad_s')
        if self.uses_height or autonomous or cfg.get('tof', {}).get('enabled'):
            if not cfg.get('tof', {}).get('enabled'):
                raise ValueError('terrain actor/autonomous mode requires host tof.enabled')
            self.projector = TerrainProjector(cfg['tof'])
            self.projector.require_calibrated(manifest.get('height_scan_contract'))
            self.tof = ThreadedToF(cfg['tof'])
        if autonomous:
            if cfg['vision'].get('verified') is not True:
                raise ValueError('verify camera mounting, ROI and colour thresholds before autonomous mode')
            if cfg.get('autonomy', {}).get('require_terrain_policy', True) and not self.uses_height:
                raise ValueError('competition mode requires a trained terrain actor; current policy is blind')
            self.mission = MissionFSM(cfg['mission'], mission_id=uuid.uuid4().hex)
            self.camera = ProcessCamera(cfg['vision'])

    def start(self):
        try:
            for resource in (self.tof, self.camera):
                if resource is not None:
                    self.started.append(resource)
                    resource.start()
        except BaseException:
            self.close()
            raise

    def poll(self, imu, now):
        if imu.stamp > 0 and (not self.orientation_history or imu.stamp > self.orientation_history[-1].stamp):
            self.orientation_history.append(imu)
        if self.tof is not None:
            frame = self.tof.read()
            now = time.monotonic()
            if not np.isfinite(frame.stamp) or frame.stamp > now:
                raise SafetyFault('invalid ToF acquisition timestamp')
            if frame.stamp < self.last_tof_stamp:
                raise SafetyFault('ToF acquisition timestamp moved backwards')
            if frame.stamp > self.last_tof_stamp:
                raw_valid = self.projector.raw_valid_mask(frame)
                if not raw_valid.any():
                    raise SafetyFault('ToF frame has no valid raw zones')
                self.pending_frames.append(frame)
                self.last_tof_stamp = frame.stamp
            # An IMU sample can arrive AFTER its matching ToF frame because it
            # crosses the MCU serial link. Keep a bounded pending queue and use
            # the last projected result only within its ORIGINAL age limit.
            # Missing coverage/invalid geometry is never hidden by this queue.
            matched = None
            for candidate in reversed(self.pending_frames):
                nearest = min(self.orientation_history,
                              key=lambda s: abs(s.stamp - candidate.stamp), default=imu)
                if abs(nearest.stamp - candidate.stamp) <= self.projector.max_skew:
                    matched = candidate, nearest
                    break
            if matched is not None:
                candidate, nearest = matched
                self.terrain = self.projector.project(candidate, nearest, now)
                while self.pending_frames and self.pending_frames[0].stamp <= candidate.stamp:
                    self.pending_frames.popleft()
            elif self.terrain is None:
                self.terrain = TerrainState(np.full(8, np.nan), np.zeros(8, bool), 0.,
                                            np.zeros(8, int), np.full(8, np.inf),
                                            self.projector.calibrated, 'waiting_for_aligned_imu')
        if self.camera is not None:
            self.vision = self.camera.read()
        return time.monotonic()

    def require_ready(self, now):
        if self.tof is not None:
            if self.terrain is None or not self.terrain.valid.all():
                reason = self.terrain.reason if self.terrain is not None else 'missing'
                raise SafetyFault('host terrain unavailable: ' + reason)
            if not 0 <= now - self.terrain.stamp <= self.cfg['tof']['frame_timeout_s']:
                raise SafetyFault('host terrain expired')
        if self.camera is not None:
            if self.vision is None or not self.vision.fresh(now, self.cfg['vision']['max_frame_age_s']):
                raise SafetyFault('camera missing/stale')

    def control(self, joints, imu, now, velocity, gait, bridge=None):
        self.require_ready(now)
        if self.mission is not None:
            tilt = math.acos(float(np.clip(-imu.gravity[2], -1., 1.)))
            stable = (tilt <= self.stable_tilt and np.linalg.norm(imu.omega) <= self.stable_omega
                      and np.max(np.abs(joints.velocity)) <= self.stable_joint_speed)
            ack = {}
            if (self.delivery_request is not None and self.delivery_wire_id is not None
                    and bridge is not None and bridge.armed):
                if self.delivery_completed_stamp is None and bridge.delivery_ack(self.delivery_wire_id):
                    self.delivery_completed_stamp = now
                if self.delivery_completed_stamp is not None:
                    ack = dict(delivery_ack_id=self.delivery_request.request_id,
                               delivery_ack_status='completed', delivery_ack_stamp=self.delivery_completed_stamp)
            self.output = self.mission.step(MissionInput(
                now=now, start=True, vision_stamp=self.vision.stamp,
                tof_stamp=self.terrain.stamp, tof_valid=bool(self.terrain.valid.all()),
                line=self.vision.line, colours=tuple(c['colour'] for c in self.vision.colours),
                parcel_colour=self.vision.parcel_colour, events=self.vision.events, stable=stable, **ack))
            if self.output.state == MissionState.FAULT:
                raise SafetyFault('mission: ' + str(self.output.reason))
            if self.output.delivery_request is not None:
                # Current STM32 v1 has no delivery actuator command. The bridge
                # rejects it explicitly; never interpret a generic ACK as delivery.
                self.delivery_request = self.output.delivery_request
                if bridge is not None and bridge.armed:
                    # One host round has one delivery transaction; retain this
                    # wire ID for its entire lifetime (including any retry).
                    self.delivery_wire_id = uuid.uuid4().int & 0xFFFFFFFF
                    try:
                        bridge.delivery(self.delivery_wire_id)
                    except NotImplementedError as exc:
                        raise SafetyFault('delivery unsupported by paired STM32 firmware') from exc
                # Shadow mode logs the request and waits; it never operates an
                # auxiliary actuator or fabricates a completion response.
            velocity = np.asarray(self.output.velocity, dtype=float)
            gait = (np.asarray(self.output.gait, dtype=float) if self.output.gait is not None
                    else self.base_gait.copy())
            self.done = self.output.state == MissionState.DONE
        heights = self.terrain.heights if self.uses_height else None
        return velocity, gait, heights

    def status(self):
        return {'mission_state': self.output.state.value if self.output else None,
                'mission_reason': self.output.reason if self.output else None,
                'terrain_valid': self.terrain.valid.tolist() if self.terrain else None,
                'terrain_height_m': [float(h) if np.isfinite(h) else None for h in self.terrain.heights]
                if self.terrain else None,
                'vision_stamp': self.vision.stamp if self.vision else None}

    def close(self):
        errors = []
        for resource in reversed(self.started):
            try:
                resource.close()
            except Exception as exc:
                errors.append(str(exc))
        self.started.clear()
        if errors:
            raise RuntimeError('; '.join(errors))


def check_velocity(velocity, manifest):
    """Reject a task command outside the exported envelope instead of changing its route."""
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise SafetyFault('invalid mission velocity')
    for v, key in zip(velocity, ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')):
        lo, hi = manifest['velocity_ranges'][key]
        if v != 0 and not lo <= v <= hi:
            raise SafetyFault(f'mission {key} outside trained policy range')
    return velocity
