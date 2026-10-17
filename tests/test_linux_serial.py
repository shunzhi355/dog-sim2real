"""Exercise real Linux pyserial/termios against a virtual LX824 bus; no physical device."""

import os
import select
import struct
import threading

import pytest

from dog3588.hardware.lx824 import FrameParser, SerialBus, POS_READ, packet, move_packet


@pytest.mark.skipif(os.name != "posix", reason="Linux PTY integration")
def test_real_serial_transport_over_pty(cfg):
    import pty

    master, slave = pty.openpty()
    settings = dict(cfg["buses"][0], port=os.ttyname(slave), direction={"mode": "auto"}, read_timeout_s=.1)
    finish = threading.Event()
    received = []
    def emulate():
        parser = FrameParser()
        while not finish.is_set():
            if not select.select([master], [], [], .02)[0]:
                continue
            for sid, command, data in parser.feed(os.read(master, 256)):
                received.append((sid, command, data))
                if command == POS_READ:
                    # Noise, echo, wrong servo and then the expected response.
                    os.write(master, b"junk" + packet(sid, POS_READ) +
                             packet(20, POS_READ, struct.pack("<h", 800)) +
                             packet(sid, POS_READ, struct.pack("<h", 500 + sid)))
    bus = SerialBus(settings)
    worker = threading.Thread(target=emulate, daemon=True)
    worker.start()
    try:
        for sid in (1, 2, 3):
            assert bus.read_position(sid)[0] == 500 + sid
        bus.send(move_packet(1, 600, 20))
        assert bus.read_position(1)[0] == 501
        assert any(cmd == 1 and struct.unpack("<HH", params) == (600, 20)
                   for _, cmd, params in received)
    finally:
        finish.set()
        worker.join(timeout=1)
        bus.close()
        os.close(master)
        os.close(slave)

