"""Four concurrent half-duplex buses; latest-target mailboxes prevent queued stale motion."""

import threading
import time

import numpy as np

from ..types import JointState
from .lx824 import SerialBus, move_packet, packet, LOAD_WRITE


class LegWorker:
    def __init__(self, config, calibrations, *, armed, target_timeout_s, duration_ms,
                 bus_factory=SerialBus):
        self.config, self.calibrations = config, calibrations
        self.armed, self.target_timeout = armed, target_timeout_s
        self.duration_ms, self.bus_factory = duration_ms, bus_factory
        self.lock = threading.Lock()
        self.finish = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"leg-{config['name']}", daemon=True)
        self.positions = np.zeros(3)
        self.velocities = np.zeros(3)
        self.stamps = np.zeros(3)
        self.target = None
        self.generation = 0
        self.error = None
        self.timeouts = 0

    def start(self):
        self.thread.start()

    def submit(self, raw, stamp):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            self.target = (list(raw), stamp)
            self.generation += 1

    def snapshot(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            return self.positions.copy(), self.velocities.copy(), self.stamps.copy()

    def _run(self):
        bus = None
        attempted_motion = False
        try:
            bus = self.bus_factory(self.config)
            index, sent = 0, 0
            while not self.finish.is_set():
                started = time.monotonic()
                with self.lock:
                    target, generation = self.target, self.generation
                if self.armed and target is not None:
                    raw, stamp = target
                    if started - stamp > self.target_timeout:
                        raise TimeoutError("policy target watchdog expired")
                    if generation != sent:
                        # Reject stale commands immediately before transmission as well.
                        if time.monotonic() - stamp > self.target_timeout:
                            raise TimeoutError("target expired before UART send")
                        if not attempted_motion:
                            attempted_motion = True
                            bus.send(b"".join(packet(j.servo_id, LOAD_WRITE, b"\x01")
                                              for j in self.calibrations))
                        bus.send(b"".join(move_packet(j.servo_id, value, self.duration_ms)
                                          for j, value in zip(self.calibrations, raw)))
                        sent = generation
                joint = self.calibrations[index]
                try:
                    raw, stamp = bus.read_position(joint.servo_id)
                    q = joint.from_raw(raw)
                    with self.lock:
                        dt = stamp - self.stamps[index]
                        # Do not derive a velocity from the initial/long-disconnected sample.
                        if self.stamps[index] > 0 and 0 < dt < 0.2:
                            measured = (q - self.positions[index]) / dt
                            alpha = dt / (0.02 + dt)
                            self.velocities[index] += alpha * (measured - self.velocities[index])
                        else:
                            self.velocities[index] = 0
                        self.positions[index], self.stamps[index] = q, stamp
                except TimeoutError:
                    self.timeouts += 1
                index = (index + 1) % 3
                self.finish.wait(max(0, self.config["query_period_s"] - (time.monotonic() - started)))
        except BaseException as exc:
            with self.lock:
                self.error = f"{self.config['name']}: {type(exc).__name__}: {exc}"
        finally:
            if bus is not None:
                try:
                    if attempted_motion:
                        bus.stop([j.servo_id for j in self.calibrations])
                except Exception as exc:
                    with self.lock:
                        self.error = f"{self.error or ''}; stop transmission failed: {exc}"
                finally:
                    bus.close()

    def request_stop(self):
        self.finish.set()

    def join(self):
        self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            raise RuntimeError(f"{self.config['name']}: I/O thread blocked; use physical stop")


class FourLegIO:
    def __init__(self, cfg, calibrations, *, armed, control_hz):
        self.calibrations = calibrations
        self.groups = []
        for bus in cfg["buses"]:
            indices = [i for i, j in enumerate(calibrations) if j.bus == bus["name"]]
            worker = LegWorker(bus, [calibrations[i] for i in indices], armed=armed,
                               target_timeout_s=cfg["safety"]["target_timeout_s"],
                               duration_ms=round(1000 / control_hz))
            self.groups.append((indices, worker))

    def start(self):
        for _, worker in self.groups:
            worker.start()

    def read(self):
        q, dq, stamps = np.zeros(12), np.zeros(12), np.zeros(12)
        for indices, worker in self.groups:
            q[indices], dq[indices], stamps[indices] = worker.snapshot()
        return JointState(q, dq, stamps)

    def command(self, targets, now):
        # Validate ALL twelve joints before any worker is allowed to transmit.
        if np.shape(targets) != (12,):
            raise ValueError("expected twelve motor targets")
        raw = [j.to_raw(float(q)) for j, q in zip(self.calibrations, targets)]
        for indices, worker in self.groups:
            worker.submit([raw[i] for i in indices], now)

    def close(self):
        # Signal every leg before joining any leg, so one slow UART cannot delay others.
        for _, worker in self.groups:
            worker.request_stop()
        errors = []
        for _, worker in self.groups:
            try:
                worker.join()
                if worker.error:
                    errors.append(worker.error)
            except RuntimeError as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))


class MockJoints:
    """Only an I/O test double. No rigid-body dynamics or sim2real validation."""

    def __init__(self, default):
        self.q = np.asarray(default, dtype=float).copy()
        self.dq = np.zeros(12)
        self.target = self.q.copy()
        self.last = time.monotonic()

    def start(self): pass

    def read(self):
        now = time.monotonic()
        dt = max(now - self.last, 1e-6)
        delta = np.clip(self.target - self.q, -dt * 2, dt * 2)
        self.q += delta
        self.dq = delta / dt
        self.last = now
        return JointState(self.q.copy(), self.dq.copy(), np.full(12, now))

    def command(self, targets, now):
        self.target = targets.copy()

    def close(self): pass

