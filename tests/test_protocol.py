import struct

import pytest

from dog3588.hardware.lx824 import FrameParser, move_packet, packet, POS_READ, SerialBus


def test_default_buses_use_auto_direction_without_gpio(cfg, monkeypatch):
    import serial
    from dog3588.hardware import lx824

    def reject_gpio(*args, **kwargs):
        pytest.fail("default automatic-direction buses must not request GPIO")

    class Serial:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.writes = []

        def write(self, data):
            self.writes.append(data)
            return len(data)

        def flush(self): pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(lx824, "GpioDirection", reject_gpio)
    monkeypatch.setattr(serial, "Serial", Serial)
    assert len(cfg["buses"]) == 4
    for settings in cfg["buses"]:
        assert settings["direction"] == {"mode": "auto"}
        assert settings["verified"] is False
        bus = SerialBus(settings)
        try:
            assert isinstance(bus.direction, lx824.AutoDirection)
            bus.send(packet(1, POS_READ), receive=True)
            assert bus.serial.writes == [packet(1, POS_READ)]
        finally:
            bus.close()
        assert bus.serial.closed


def test_known_stm32_packet_bytes():
    assert packet(1, POS_READ).hex() == "555501031cdf"
    assert move_packet(1, 500, 20).hex() == "5555010701f4011400ed"


def test_fragment_noise_echo_bad_checksum_and_signed_position():
    parser = FrameParser()
    good = packet(2, POS_READ, struct.pack("<h", -5))
    bad = good[:-1] + bytes([good[-1] ^ 1])
    stream = b"noise\x55" + bad + packet(1, POS_READ) + good
    result = []
    for byte in stream:
        result += parser.feed(bytes([byte]))
    assert result[-1] == (2, POS_READ, struct.pack("<h", -5))
    assert struct.unpack("<h", result[-1][2])[0] == -5
    assert parser.bad_frames > 0
    assert len(parser.buffer) < 11


@pytest.mark.parametrize("raw,duration", [(-1, 20), (1001, 20), (500, -1), (500, 30001)])
def test_reject_bad_actuator_values(raw, duration):
    with pytest.raises(ValueError):
        move_packet(1, raw, duration)


def test_wrong_id_echo_and_wrong_command_do_not_satisfy_read():
    class Direction:
        def tx(self): pass
        def rx(self): pass
        def idle(self): pass
    class Serial:
        def __init__(self):
            self.buf = packet(1, POS_READ) + packet(2, POS_READ, b"\x11\x00") + packet(1, 27, b"\x22\x00") + packet(1, POS_READ, b"\xf4\x01")
        @property
        def in_waiting(self): return len(self.buf)
        def reset_input_buffer(self): pass
        def write(self, data): return len(data)
        def flush(self): pass
        def read(self, count):
            data, self.buf = self.buf[:count], self.buf[count:]
            return data
    bus = SerialBus.__new__(SerialBus)
    bus.serial, bus.direction = Serial(), Direction()
    bus.timeout, bus.quiet, bus.not_before = 0.01, 0.006, 0.
    assert bus.read_position(1)[0] == 500

