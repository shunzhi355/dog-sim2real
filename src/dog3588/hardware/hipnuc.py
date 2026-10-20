"""HiPNUC HI91/HI83 stream reader; gyro rad/s and gravity in body axes.

Wire layout follows the HiPNUC IMU programming manual, sections 2.6 and 7.
HI91 gyro is degrees/s; HI83 gyro is already radians/s. Neither protocol
needs a configuration write. Only complete, CRC-checked AHRS samples publish.
"""

import binascii
import math
import struct

import numpy as np

from ..types import ImuState


class HipnucParser:
    MAX_PAYLOAD = 512
    FIELD_SIZES = (12, 12, 12, 12, 16, 8, 8, 4, 4, 12, 12, 12)
    # Bias/attitude not converged or accelerometer/gyro saturated. UTC sync
    # and magnetic heading are not required for angular rate + gravity.
    BAD_STATUS = (1 << 3) | (1 << 5) | (1 << 6) | (1 << 7)

    def __init__(self, rotation=None):
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        self.buffer = bytearray()
        self.last_byte = None
        self.frames = self.crc_errors = self.invalid = self.timeouts = 0
        self.counts = {0x83: 0, 0x91: 0}
        self.last_ticks = {}
        self.status = None
        self.sample = ImuState(np.zeros(3), np.array([0., 0., -1.]), 0.)

    def feed(self, data, now):
        if self.buffer and self.last_byte is not None and now - self.last_byte > .1:
            self.buffer.clear()
            self.timeouts += 1
        # Byte-wise ingestion bounds storage even for corrupt or oversized input.
        for byte in data:
            self.last_byte = now
            self.buffer.append(byte)
            while len(self.buffer) >= 2:
                if self.buffer[:2] != b'\x5a\xa5':
                    del self.buffer[0]
                    continue
                if len(self.buffer) < 6:
                    break
                size, expected_crc = struct.unpack_from('<HH', self.buffer, 2)
                if not 1 <= size <= self.MAX_PAYLOAD:
                    del self.buffer[0]
                    self.invalid += 1
                    continue
                if len(self.buffer) < size + 6:
                    break
                payload = bytes(self.buffer[6:6 + size])
                crc = binascii.crc_hqx(self.buffer[:4], 0)
                if binascii.crc_hqx(payload, crc) != expected_crc:
                    del self.buffer[0]
                    self.crc_errors += 1
                    continue
                del self.buffer[:size + 6]
                self.frames += 1
                if payload[0] not in self.counts:
                    continue
                self.counts[payload[0]] += 1
                try:
                    self._decode(payload, now)
                except (ValueError, struct.error):
                    self.invalid += 1
                    # A valid wire frame explicitly reports unusable data: do
                    # not continue presenting the preceding attitude as valid.
                    self.sample = ImuState(self.sample.omega, self.sample.gravity, 0.)
        return ImuState(self.sample.omega.copy(), self.sample.gravity.copy(), self.sample.stamp)

    def _decode(self, payload, now):
        kind = payload[0]
        if kind == 0x91:
            if len(payload) != 76:
                raise ValueError('HI91 payload length')
            tick = struct.unpack_from('<I', payload, 8)[0]
            omega = np.asarray(struct.unpack_from('<3f', payload, 24)) * math.pi / 180
            quat = np.asarray(struct.unpack_from('<4f', payload, 60))
        else:
            bitmap = struct.unpack_from('<I', payload, 4)[0]
            if bitmap & ~0xfff or bitmap & 0x32 != 0x32:
                raise ValueError('HI83 requires gyro, quaternion and timestamp')
            offset, fields = 8, {}
            for bit, size in enumerate(self.FIELD_SIZES):
                if bitmap & (1 << bit):
                    fields[bit] = offset
                    offset += size
            if offset != len(payload):
                raise ValueError('HI83 payload length disagrees with bitmap')
            tick = struct.unpack_from('<Q', payload, fields[5])[0]
            omega = np.asarray(struct.unpack_from('<3f', payload, fields[1]))
            quat = np.asarray(struct.unpack_from('<4f', payload, fields[4]))
        self.status = struct.unpack_from('<H', payload, 1)[0]
        if self.status & self.BAD_STATUS:
            raise ValueError('IMU convergence/saturation status')
        norm2 = float(quat @ quat)
        if not np.isfinite(omega).all() or not np.isfinite(norm2) or abs(norm2 - 1) > .02:
            raise ValueError('invalid angular rate/quaternion')
        previous = self.last_ticks.get(kind)
        if previous is not None:
            # HI91 time-of-day wraps every 24 hours; HI83 is uptime in us.
            delta = (tick - previous) % 86400000 if kind == 0x91 else tick - previous
            if delta <= 0 or (kind == 0x91 and delta >= 43200000):
                raise ValueError('duplicate or out-of-order IMU timestamp')
        self.last_ticks[kind] = tick
        w, x, y, z = quat / math.sqrt(norm2)
        gravity = -np.array([2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)])
        self.sample = ImuState(self.rotation @ omega, self.rotation @ gravity, now)
