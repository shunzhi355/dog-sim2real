"""LX824 115200/8N1 bus packets. Matches project/dog/Drivers/Servo/Src/lx824.c."""

import struct
import time

MOVE_TIME_WRITE = 1
MOVE_STOP = 12
POS_READ = 28
LOAD_WRITE = 31


def packet(servo_id: int, command: int, params: bytes = b"") -> bytes:
    if type(servo_id) is not int or not 0 <= servo_id <= 254:
        raise ValueError("invalid servo ID")
    if not 0 <= command <= 255 or len(params) > 4:
        raise ValueError("invalid command/parameters")
    body = bytes((servo_id, len(params) + 3, command)) + params
    return b"\x55\x55" + body + bytes((~sum(body) & 255,))


def move_packet(servo_id: int, raw: int, duration_ms: int) -> bytes:
    if type(raw) is not int or not 0 <= raw <= 1000:
        raise ValueError("servo raw target must be 0..1000")
    if type(duration_ms) is not int or not 0 <= duration_ms <= 30000:
        raise ValueError("servo duration must be 0..30000 ms")
    return packet(servo_id, MOVE_TIME_WRITE, struct.pack("<HH", raw, duration_ms))


class FrameParser:
    """Resynchronises noise, fragmented packets, bad lengths and bad checksums."""

    def __init__(self):
        self.buffer = bytearray()
        self.bad_frames = 0

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        self.buffer.extend(data)
        result = []
        while len(self.buffer) >= 2:
            start = self.buffer.find(b"\x55\x55")
            if start < 0:
                self.buffer[:] = self.buffer[-1:] if self.buffer[-1] == 0x55 else b""
                break
            del self.buffer[:start]
            if len(self.buffer) < 4:
                break
            length = self.buffer[3]
            if not 3 <= length <= 7:
                del self.buffer[0]
                self.bad_frames += 1
                continue
            size = length + 3
            if len(self.buffer) < size:
                break
            frame = self.buffer[:size]
            if sum(frame[2:]) & 255 != 255:
                del self.buffer[0]
                self.bad_frames += 1
                continue
            result.append((frame[2], frame[4], bytes(frame[5:-1])))
            del self.buffer[:size]
        return result


class AutoDirection:
    """External hardware must release the DATA bus after the last stop bit."""

    def tx(self): pass
    def rx(self): pass
    def idle(self): pass
    def close(self): pass


class GpioDirection:
    """Two active-high enables, using libgpiod v2 line offsets, NOT header numbers.

    Linux userspace turn-around is best effort. Validate response capture using
    a logic analyser; STM32 TC-interrupt timing is not reproduced by this driver.
    """

    def __init__(self, config: dict):
        import gpiod
        from gpiod.line import Direction, Value

        self.tx_line, self.rx_line = config["tx_enable"], config["rx_enable"]
        if (type(self.tx_line) is not int or type(self.rx_line) is not int
                or min(self.tx_line, self.rx_line) < 0 or self.tx_line == self.rx_line):
            raise ValueError("set distinct GPIO line offsets for TX_EN and RX_EN")
        self.Value = Value
        self.lines = gpiod.request_lines(config["chip"], consumer="dog3588-lx824",
                                        config={(self.tx_line, self.rx_line): gpiod.LineSettings(
                                            direction=Direction.OUTPUT, output_value=Value.INACTIVE)})

    def tx(self):
        self.lines.set_values({self.tx_line: self.Value.ACTIVE, self.rx_line: self.Value.INACTIVE})

    def rx(self):
        self.lines.set_values({self.tx_line: self.Value.INACTIVE, self.rx_line: self.Value.ACTIVE})

    def idle(self):
        self.lines.set_values({self.tx_line: self.Value.INACTIVE, self.rx_line: self.Value.INACTIVE})

    def close(self):
        self.idle()
        self.lines.release()


class SerialBus:
    """One owner thread per port. Reads never enable torque or change calibration."""

    def __init__(self, config: dict):
        import serial

        mode = config["direction"]["mode"]
        if mode not in ("gpio", "auto"):
            raise ValueError("unsupported direction mode; use gpio or verified auto adapter")
        if config["baudrate"] != 115200:
            raise ValueError("LX824 requires 115200 baud")
        for field in ("read_timeout_s", "quiet_after_timeout_s"):
            from ..config import positive
            positive(config[field], field)
        self.direction = GpioDirection(config["direction"]) if mode == "gpio" else AutoDirection()
        try:
            self.serial = serial.Serial(config["port"], config["baudrate"], timeout=0.001,
                                        write_timeout=0.02, exclusive=True)
        except BaseException:
            self.direction.close()
            raise
        self.timeout = config["read_timeout_s"]
        self.quiet = config["quiet_after_timeout_s"]
        self.not_before = 0.0

    def send(self, data: bytes, *, receive: bool = False) -> None:
        self.direction.tx()
        try:
            if self.serial.write(data) != len(data):
                raise IOError("partial UART write")
            # pyserial flush -> tcdrain on Linux. USB bridge buffering can add latency.
            self.serial.flush()
        finally:
            if receive:
                self.direction.rx()
            else:
                self.direction.idle()

    def read_position(self, servo_id: int) -> tuple[int, float]:
        if not 0 <= servo_id <= 253:
            raise ValueError("position queries require a unicast ID")
        remaining = self.not_before - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self.serial.reset_input_buffer()
        parser = FrameParser()
        self.send(packet(servo_id, POS_READ), receive=True)
        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                data = self.serial.read(min(max(self.serial.in_waiting, 1), 256))
                for sid, command, params in parser.feed(data):
                    if sid == servo_id and command == POS_READ and len(params) == 2:
                        return struct.unpack("<h", params)[0], time.monotonic()
            self.not_before = time.monotonic() + self.quiet
            raise TimeoutError(f"servo {servo_id}: position response timeout")
        finally:
            self.direction.idle()

    def stop(self, servo_ids: list[int]) -> None:
        # Stop interpolation and hold the current load state; never unload a standing robot.
        self.send(b"".join(packet(sid, MOVE_STOP) for sid in servo_ids))

    def close(self):
        try:
            self.direction.close()
        finally:
            self.serial.close()

