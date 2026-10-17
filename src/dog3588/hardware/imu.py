"""Replaceable WIT serial and MPU6050 I2C providers; both output the same body-frame state."""

import math
import struct
import threading
import time

import numpy as np

from ..config import rotation_matrix
from ..types import ImuState


class WitParser:
    def __init__(self, rotation=None):
        self.rotation = np.eye(3) if rotation is None else rotation
        self.buffer = bytearray()
        self.omega, self.gravity = np.zeros(3), np.array([0., 0., -1.])
        self.gyro_stamp = self.angle_stamp = 0.0

    def feed(self, data: bytes, now: float) -> ImuState:
        self.buffer.extend(data)
        while len(self.buffer) >= 11:
            if self.buffer[0] != 0x55 or sum(self.buffer[:10]) & 255 != self.buffer[10]:
                del self.buffer[0]
                continue
            frame = self.buffer[:11]
            del self.buffer[:11]
            xyz = np.asarray(struct.unpack("<hhh", frame[2:8]), dtype=float)
            if frame[1] == 0x52:
                self.omega = self.rotation @ (xyz * (2000.0 / 32768) * math.pi / 180)
                self.gyro_stamp = now
            elif frame[1] == 0x53:
                roll, pitch, _ = xyz * math.pi / 32768
                self.gravity = self.rotation @ np.array([math.sin(pitch),
                    -math.sin(roll) * math.cos(pitch), -math.cos(roll) * math.cos(pitch)])
                self.angle_stamp = now
        # An arriving gyro frame must not make a stale orientation look fresh.
        return ImuState(self.omega.copy(), self.gravity.copy(), min(self.gyro_stamp, self.angle_stamp))


class GravityFilter:
    """Six-axis complementary filter. No yaw/translation estimate is claimed."""

    def __init__(self, tau_s=0.5):
        self.gravity = None
        self.tau = tau_s

    def update(self, specific_force, omega, dt):
        norm = np.linalg.norm(specific_force)
        if self.gravity is None:
            if not 0.8 * 9.80665 <= norm <= 1.2 * 9.80665:
                raise ValueError("MPU6050 must start stationary for gravity initialisation")
            self.gravity = -specific_force / norm
        predicted = self.gravity - np.cross(omega, self.gravity) * dt
        if 0.85 * 9.80665 <= norm <= 1.15 * 9.80665:
            alpha = dt / (self.tau + dt)
            predicted = (1 - alpha) * predicted - alpha * specific_force / norm
        self.gravity = predicted / np.linalg.norm(predicted)
        return self.gravity.copy()


class ThreadedImu:
    def __init__(self, config):
        self.cfg = config
        self.rotation = rotation_matrix(config)
        self.finish = threading.Event()
        self.lock = threading.Lock()
        self.sample = ImuState(np.zeros(3), np.array([0., 0., -1.]), 0.0)
        self.error = None
        self.thread = threading.Thread(target=self._run, name="imu", daemon=True)

    def start(self):
        self.thread.start()

    def read(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            return ImuState(self.sample.omega.copy(), self.sample.gravity.copy(), self.sample.stamp)

    def _publish(self, sample):
        with self.lock:
            self.sample = sample

    def _run(self):
        try:
            if self.cfg["driver"] == "wit_serial":
                self._serial()
            elif self.cfg["driver"] == "mpu6050_i2c":
                self._i2c()
            else:
                raise ValueError("unsupported IMU driver")
        except BaseException as exc:
            with self.lock:
                self.error = f"IMU: {type(exc).__name__}: {exc}"

    def _serial(self):
        import serial

        parser = WitParser(self.rotation)
        with serial.Serial(self.cfg["serial_port"], self.cfg["baudrate"], timeout=0.02,
                           exclusive=True) as port:
            port.reset_input_buffer()
            while not self.finish.is_set():
                data = port.read(min(max(port.in_waiting, 1), 256))
                if data:
                    self._publish(parser.feed(data, time.monotonic()))

    def _i2c(self):
        from smbus2 import SMBus

        address = self.cfg["i2c_address"]
        bias = np.asarray(self.cfg["gyro_bias_rad_s"], dtype=float)
        if bias.shape != (3,) or not np.isfinite(bias).all():
            raise ValueError("gyro_bias_rad_s must be three finite sensor-frame values")
        with SMBus(self.cfg["i2c_bus"]) as bus:
            if bus.read_byte_data(address, 0x75) != 0x68:
                raise ValueError("MPU6050 WHO_AM_I mismatch; this is not a generic I2C IMU driver")
            # PLL clock, 44 Hz DLPF, 200 Hz sampling, +/-2000 deg/s and +/-4g.
            for register, value in ((0x6B, 1), (0x1A, 3), (0x19, 4), (0x1B, 0x18), (0x1C, 0x08)):
                bus.write_byte_data(address, register, value)
            self.finish.wait(0.1)
            estimator, last = GravityFilter(), time.monotonic()
            while not self.finish.is_set():
                if not bus.read_byte_data(address, 0x3A) & 0x01:
                    if time.monotonic() - last > 0.1:
                        raise TimeoutError("MPU6050 data-ready stopped")
                    self.finish.wait(0.001)
                    continue
                raw = struct.unpack(">7h", bytes(bus.read_i2c_block_data(address, 0x3B, 14)))
                now = time.monotonic()
                dt = now - last
                if dt > 0.1:
                    raise TimeoutError("MPU6050 sample gap exceeds 100 ms")
                force = self.rotation @ (np.asarray(raw[:3]) / 8192.0 * 9.80665)
                omega = self.rotation @ (np.asarray(raw[4:]) / 16.4 * math.pi / 180 - bias)
                gravity = estimator.update(force, omega, max(dt, 1e-5))
                self._publish(ImuState(omega, gravity, now))
                last = now
                self.finish.wait(0.005)

    def close(self):
        self.finish.set()
        self.thread.join(timeout=1)
        if self.thread.is_alive():
            raise RuntimeError("IMU I/O thread blocked")


class MockImu:
    def start(self): pass
    def close(self): pass

    def read(self):
        return ImuState(np.zeros(3), np.array([0., 0., -1.]), time.monotonic())

