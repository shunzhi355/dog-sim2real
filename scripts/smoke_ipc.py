"""End-to-end mock policy run with UDP commands, expiry and software stop. No hardware."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from dog3588.commands import send_command


def spawn(args):
    process = subprocess.Popen([sys.executable, "-m", "dog3588", "run", *args],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    first = process.stdout.readline()
    if '"event": "started"' not in first:
        process.terminate()
        _, error = process.communicate(timeout=5)
        raise RuntimeError(first + error)
    return process


def main():
    with tempfile.TemporaryDirectory(prefix="dog3588-ipc-") as folder:
        log = Path(folder) / "trace.jsonl"
        process = spawn(["--seconds", "2", "--log", str(log)])
        try:
            for _ in range(8):
                send_command((.12, 0., .2))
                time.sleep(.05)
            out, error = process.communicate(timeout=10)
            if process.returncode != 0:
                raise RuntimeError(out + error)
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            assert any(row["velocity"] == [.12, 0., .2] for row in rows)
            assert rows[-1]["velocity"] == [0., 0., 0.]
            print("PASS: velocity delivered, command expiry returns to zero")
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)
        process = spawn(["--seconds", "5"])
        try:
            send_command(estop=True)
            _, error = process.communicate(timeout=5)
            assert process.returncode == 1 and "latched software emergency stop" in error
            print("PASS: software stop latched and controller exited")
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)


if __name__ == "__main__":
    main()

