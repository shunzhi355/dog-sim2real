"""Install desktop launchers for the current Linux user; never starts motors."""

from pathlib import Path
import shutil
import subprocess
import sys


def desktop_quote(value):
    # Desktop Entry Exec uses its own quoting, not shell single quotes.
    value = str(value).replace('%', '%%')
    for char in ('\\', '"', '`', '$'):
        value = value.replace(char, '\\' + char)
    return '"' + value.replace('\\', '\\\\') + '"'


def main():
    if sys.platform != 'linux':
        raise SystemExit('Run this installer on the Orange Pi as its desktop user.')
    root = Path(__file__).resolve().parents[1]
    local = root / 'config/local-model4497.yaml'
    if not local.exists():
        shutil.copyfile(root / 'config/model4497.yaml', local)
    result = subprocess.run(['xdg-user-dir', 'DESKTOP'], capture_output=True, text=True, check=True)
    desktop = Path(result.stdout.strip())
    if not desktop.is_absolute() or desktop == Path.home():
        raise SystemExit('No dedicated desktop directory configured.')
    desktop.mkdir(parents=True, exist_ok=True)
    launcher = root / 'deploy/start-desktop.sh'
    launcher.chmod(0o755)
    python = root / '.venv/bin/python'
    entries = {
        'Robot12-model4497.desktop': (
            'Robot12 策略启动', '模型测试、真实反馈推理与实机控制',
            '/bin/bash ' + desktop_quote(launcher), 'utilities-terminal'),
        'Robot12-estop.desktop': (
            'Robot12 急停', '向本机 Robot12 控制进程发送锁存急停',
            ' '.join([desktop_quote(python), '-m dog3588 --config', desktop_quote(local), 'estop']),
            'process-stop'),
    }
    for filename, (name, comment, command, icon) in entries.items():
        path = desktop / filename
        path.write_text(
            '[Desktop Entry]\nType=Application\nVersion=1.0\n'
            f'Name={name}\nComment={comment}\nExec={command}\n'
            f'Icon={icon}\nTerminal=true\nStartupNotify=false\n', encoding='utf-8')
        path.chmod(0o755)
        if shutil.which('gio'):
            subprocess.run(['gio', 'set', str(path), 'metadata::trusted', 'true'],
                           capture_output=True, check=False)
        print(path)
    print(f'Local configuration: {local}')


if __name__ == '__main__':
    main()
