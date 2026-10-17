"""Loopback-only, expiring velocity commands. Vision and CLI share this small contract."""

import json
import math
import socket
import time
import uuid

import numpy as np


def encode_command(velocity, *, source="manual", estop=False) -> bytes:
    return json.dumps({"version": 1, "source": source, "sent_monotonic": time.monotonic(),
                       "velocity": list(velocity), "estop": estop,
                       "message_id": uuid.uuid4().hex}, allow_nan=False).encode()


def send_command(velocity=(0., 0., 0.), *, source="manual", estop=False, port=9871):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(encode_command(velocity, source=source, estop=estop), ("127.0.0.1", port))


class CommandReceiver:
    def __init__(self, config, ranges):
        self.cfg, self.ranges = config, ranges
        self.velocity = np.zeros(3)
        self.stamp = 0.0
        self.estop = False
        self.invalid_packets = 0
        self.last_sender_stamp = {"manual": 0., "vision": 0.}
        self.sock = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((self.cfg["host"], self.cfg["port"]))
            self.sock.setblocking(False)
        except BaseException:
            self.sock.close()
            self.sock = None
            raise

    def accept(self, data: bytes, now: float):
        try:
            msg = json.loads(data)
            if not isinstance(msg, dict) or msg.get("version") != 1:
                raise ValueError("bad command schema")
            source = msg["source"]
            stamp = msg["sent_monotonic"]
            if (source not in self.last_sender_stamp or type(stamp) not in (int, float)
                    or not math.isfinite(stamp) or not 0 <= now - stamp <= self.cfg["timeout_s"]
                    or stamp <= self.last_sender_stamp[source]):
                raise ValueError("stale/reordered command")
            velocity = np.asarray(msg["velocity"], dtype=float)
            if velocity.shape != (3,) or not np.isfinite(velocity).all() or type(msg["estop"]) is not bool:
                raise ValueError("bad velocity/estop")
            self.last_sender_stamp[source] = stamp
            self.estop |= msg["estop"]
            if source == self.cfg["source"]:
                # Permit zero for standing, and the trained forward/yaw envelope.
                low = [min(0, self.ranges[k][0]) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")]
                high = [max(0, self.ranges[k][1]) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")]
                self.velocity = np.clip(velocity, low, high)
                self.stamp = stamp
        except (ValueError, KeyError, TypeError, OverflowError):
            self.invalid_packets += 1

    def read(self, now):
        if self.sock is not None:
            # Bound work per control cycle even if a producer floods the socket.
            for _ in range(64):
                try:
                    data, _ = self.sock.recvfrom(2048)
                    self.accept(data, now)
                except BlockingIOError:
                    break
        if now - self.stamp > self.cfg["timeout_s"]:
            return np.zeros(3)
        return self.velocity.copy()

    def close(self):
        if self.sock is not None:
            self.sock.close()

