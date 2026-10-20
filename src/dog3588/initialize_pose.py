"""Standalone servo-position initialization. No policy, IMU or model calibration.

Requires RobotLink pose-v1 in firmware. Never reinterpret count targets as rad.
"""

import json
from pathlib import Path
import signal
import struct
import time

import numpy as np

from .commands import CommandReceiver
from .config import positive
from .safety import SafetyFault
from .transport import Frame, FrameParser, encode_frame, HEARTBEAT, STOP, ACK, JOINT_STATUS, FAULT_STATUS

POSE_CAPS, POSE_ARM, POSE_TARGET = 5, 6, 7
ALL = 0x0fff
NAMES = [f'{leg}_{joint}_joint' for leg in ('fl', 'fr', 'rl', 'rr') for joint in ('hip', 'upper', 'input')]


def load_pose(path):
    pose = json.loads(Path(path).read_text(encoding='utf-8'))
    raw = pose.get('raw_positions')
    if (pose.get('servo_ids') != list(range(1, 13)) or pose.get('joint_names') != NAMES
            or pose.get('units') != 'servo_raw_count' or not isinstance(raw, list)
            or len(raw) != 12 or any(type(q) is not int or not 0 <= q <= 1000 for q in raw)):
        raise ValueError('pose requires IDs 1..12 in FL/FR/RL/RR hip/upper/input order and twelve counts in 0..1000')
    return np.array(raw, dtype=int)


class PoseLink:
    def __init__(self, serial, interrupt):
        self.serial, self.interrupt = serial, interrupt
        self.parser = FrameParser()
        self.sequence = 0
        self.rx_sequence = None
        self.tick = 0
        self.acks = {}
        self.q = np.zeros(12, dtype=int)
        self.stamps = np.zeros(12)
        self.motion = False

    def poll(self):
        self.interrupt()
        data = self.serial.read(self.serial.in_waiting or 1)
        now = time.monotonic()
        for f in self.parser.feed(data):
            if self.rx_sequence is not None and not 0 < ((f.sequence - self.rx_sequence) & 0xffff) < 0x8000:
                continue
            if f.type not in (ACK, JOINT_STATUS, FAULT_STATUS):
                continue
            self.rx_sequence, self.tick = f.sequence, f.timestamp_ms
            if f.type == ACK and len(f.payload) == 8:
                kind, code, seq, faults = struct.unpack('<BBHI', f.payload)
                self.acks[(kind, seq)] = (code, faults)
                if len(self.acks) > 16:
                    self.acks.pop(next(iter(self.acks)))
            elif f.type == JOINT_STATUS and len(f.payload) == 290:
                for i in range(12):
                    flags, bus, raw, stamp, count, q, dq, errors = struct.unpack_from('<BBhIIffI', f.payload, 2 + 24*i)
                    age = (f.timestamp_ms - stamp) & 0xffffffff
                    self.q[i] = raw
                    self.stamps[i] = now - age/1000 if flags & 1 and count and age <= 100 and 0 <= raw <= 1000 and bus == i//3 else 0.
            elif f.type == FAULT_STATUS and len(f.payload) == 76:
                faults, armed = struct.unpack_from('<IH', f.payload)
                if self.motion and (faults or armed != ALL):
                    raise SafetyFault(f'initialization firmware stopped: faults=0x{faults:x}, armed=0x{armed:x}')

    def fresh(self):
        return bool(np.all(self.stamps > 0) and np.max(time.monotonic() - self.stamps) <= .10)

    def request(self, kind, payload=b'', *, cleanup=False):
        if not cleanup:
            self.interrupt()
        seq = self.sequence
        self.sequence = (seq + 1) & 0xffff
        packet = encode_frame(Frame(kind, seq, self.tick, payload))
        began = time.monotonic()
        if self.serial.write(packet) != len(packet):
            raise SafetyFault('incomplete initialization UART write')
        while time.monotonic() - began < .06:
            self.poll()
            result = self.acks.pop((kind, seq), None)
            if result is not None:
                code, faults = result
                if code:
                    if kind == POSE_CAPS:
                        raise SafetyFault('STM32 firmware does not support pose-v1 initialization; no enable or position command sent')
                    raise SafetyFault(f'initialization command {kind} rejected: ACK={code}, faults=0x{faults:x}')
                if kind in (POSE_ARM, POSE_TARGET) and faults:
                    raise SafetyFault(f'initialization command fault: 0x{faults:x}')
                return
        raise SafetyFault(f'initialization command {kind} ACK timeout')


def initialize_pose(cfg, path, *, arm=False, seconds=5., max_speed=40., log_path=None):
    target = load_pose(path)
    seconds, max_speed = positive(seconds, 'seconds'), positive(max_speed, 'max_speed')
    if seconds < 1 or not 1 <= max_speed <= 60:
        raise ValueError('initialization seconds >= 1 and max_speed in 1..60 counts/s required')
    if cfg.get('backend') != 'stm32':
        raise ValueError('initialize-pose requires the STM32 backend')
    commands = CommandReceiver(cfg['command'], {k: [0., 0.] for k in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')})
    interrupted = False
    previous = {}
    serial = link = logger = None
    def stop(*args):
        nonlocal interrupted
        interrupted = True
    def check_stop():
        commands.read(time.monotonic())
        if interrupted or commands.estop:
            raise SafetyFault('initialization interrupted or emergency stop')
    def emit(event, **values):
        line = json.dumps(dict(event=event, **values), allow_nan=False)
        print(line, flush=True)
        if logger:
            logger.write(line + '\n')
            logger.flush()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, stop)
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            logger = Path(log_path).open('a', encoding='utf-8')
        commands.start()
        import serial as pyserial
        serial = pyserial.Serial(cfg['bridge']['port'], cfg['bridge']['baudrate'], timeout=.002,
                                 write_timeout=.02, exclusive=True)
        link = PoseLink(serial, check_stop)
        link.request(STOP)
        link.request(POSE_CAPS)  # A capability check cannot enable motors.
        deadline = time.monotonic() + 3.
        while not link.fresh():
            if time.monotonic() >= deadline:
                raise SafetyFault('initialization requires fresh feedback from all twelve servos')
            link.poll()
        start = link.q.copy()
        duration = max(seconds, 1.875 * float(np.max(np.abs(target-start))) / max_speed)
        emit('pose_preflight', current=start.tolist(), target=target.tolist(), duration_s=duration, armed=arm)
        if not arm:
            return dict(moved=False, firmware_supported=True, current=start.tolist(), target=target.tolist())
        link.request(HEARTBEAT)
        link.request(POSE_ARM, struct.pack('<H', ALL))
        link.motion = True
        began = next_tick = time.monotonic()
        last_heartbeat = began
        last_target = start.copy()
        stable_since = None
        while True:
            link.poll()
            now = time.monotonic()
            if now < next_tick:
                continue
            if now - next_tick > .03:
                raise SafetyFault('initialization control deadline missed')
            if not link.fresh():
                raise SafetyFault('initialization servo feedback missing/stale')
            if np.max(np.abs(link.q-last_target)) > 35:
                raise SafetyFault('initialization servo tracking error exceeds 35 counts')
            if now - last_heartbeat >= .10:
                link.request(HEARTBEAT)
                last_heartbeat = time.monotonic()
            u = float(np.clip((now - began - .5) / duration, 0., 1.))
            blend = u**3 * (10. + u*(-15. + 6.*u))
            desired = np.rint(start + blend*(target-start)).astype(int)
            if np.max(np.abs(desired-last_target)) > 10:
                raise SafetyFault('initialization position increment exceeds 10 counts')
            # Short moves bound the residual travel even if STOP cannot reach the MCU.
            link.request(POSE_TARGET, struct.pack('<HH12H', ALL, 40, *desired))
            last_target = desired
            if u == 1.:
                stable = np.max(np.abs(link.q-target)) <= 5
                stable_since = (now if stable_since is None else stable_since) if stable else None
                if stable_since is not None and now-stable_since >= .5:
                    emit('pose_complete', target=target.tolist(), measured=link.q.tolist(), elapsed_s=time.monotonic()-began)
                    return dict(moved=True, reached=True, measured=link.q.tolist(), target=target.tolist())
                if now-began > .5+duration+5.:
                    raise SafetyFault('initialization failed to reach recorded standing positions')
            next_tick += .04
            if time.monotonic() >= next_tick:
                raise SafetyFault('initialization command/ACK exceeded the control period')
    finally:
        try:
            if link:
                link.motion = False
                link.interrupt = lambda: None
                link.request(STOP, cleanup=True)
        finally:
            if serial: serial.close()
            commands.close()
            if logger: logger.close()
            for sig, handler in previous.items(): signal.signal(sig, handler)
