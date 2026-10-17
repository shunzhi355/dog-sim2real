"""Copy this project over SFTP, then set up its isolated Python environment.

Uses a password prompt (or named environment variable) or SSH agent. Never stores
credentials. New hosts require a separately supplied SHA256 host-key fingerprint.
Does not change UART overlays, start services, reboot, or arm motors.
"""

import argparse
import base64
import getpass
import hashlib
import os
from pathlib import Path, PurePosixPath
import shlex
import sys

import paramiko


class PinnedHostKey(paramiko.MissingHostKeyPolicy):
    def __init__(self, expected):
        self.expected = expected

    def missing_host_key(self, client, hostname, key):
        actual = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if actual != self.expected:
            raise paramiko.SSHException(f"unknown host key for {hostname}: {actual}; supply the verified fingerprint")


def mkdirs(sftp, path):
    parts = PurePosixPath(path)
    for parent in reversed([parts, *parts.parents]):
        try:
            sftp.stat(str(parent))
        except FileNotFoundError:
            sftp.mkdir(str(parent))


def upload(client, root, destination):
    count = 0
    with client.open_sftp() as sftp:
        mkdirs(sftp, destination)
        files = [root / name for name in ("pyproject.toml", "README.md", ".gitignore") if (root / name).is_file()]
        for directory in ("src", "config", "policies", "scripts", "tests", "docs", "deploy"):
            files.extend(p for p in (root / directory).rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
                         and not any(part.endswith(".egg-info") for part in p.parts)
                         and not p.name.startswith("local"))
        for path in files:
            target = str(PurePosixPath(destination) / path.relative_to(root).as_posix())
            mkdirs(sftp, str(PurePosixPath(target).parent))
            sftp.put(str(path), target)
            count += 1
    return count


def remote_run(client, command):
    _, stdout, stderr = client.exec_command(command, timeout=180)
    channel = stdout.channel
    # Drain both streams to avoid SSH flow-control deadlock during installation.
    import time
    while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
        if channel.recv_ready():
            print(channel.recv(8192).decode(errors="replace"), end="", flush=True)
        if channel.recv_stderr_ready():
            print(channel.recv_stderr(8192).decode(errors="replace"), end="", file=sys.stderr, flush=True)
        time.sleep(0.05)
    code = channel.recv_exit_status()
    if code:
        raise RuntimeError(f"remote command exited {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.102")
    parser.add_argument("--user", default="orangepi")
    parser.add_argument("--destination", default="/home/orangepi/dog-3588")
    parser.add_argument("--password-env")
    parser.add_argument("--key-auth", action="store_true")
    parser.add_argument("--host-key-sha256")
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = PurePosixPath(args.destination)
    if not dest.is_absolute() or ".." in dest.parts or len(dest.parts) < 4 or dest.name != "dog-3588":
        raise ValueError("destination must be an absolute, dedicated dog-3588 project directory")
    password = None if args.key_auth else os.environ[args.password_env] if args.password_env else getpass.getpass("SSH password: ")
    with paramiko.SSHClient() as client:
        # An explicitly supplied fingerprint is the trust anchor (e.g. after an
        # OS reinstall); leave the user's existing known_hosts file untouched.
        if not args.host_key_sha256:
            client.load_system_host_keys()
        client.set_missing_host_key_policy(PinnedHostKey(args.host_key_sha256))
        client.connect(args.host, username=args.user, password=password,
                       look_for_keys=args.key_auth, allow_agent=args.key_auth, timeout=10)
        print(f"Uploaded {upload(client, root, str(dest))} files to {args.host}:{dest}")
        if not args.sync_only:
            remote_run(client, f"cd {shlex.quote(str(dest))} && bash deploy/setup.sh")


if __name__ == "__main__":
    main()

