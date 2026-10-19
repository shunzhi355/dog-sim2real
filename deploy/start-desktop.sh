#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY="$PWD/.venv/bin/python"
CONFIG="$PWD/config/local-model4497.yaml"
if [[ ! -x "$PY" || ! -f "$CONFIG" ]]; then
    printf '缺少 Python 环境或本地配置，请先完成部署。\n'
    read -r -p '按回车关闭…' _
    exit 1
fi
mkdir -p logs
run_controller() {
    local mode="$1"
    shift
    # All desktop controller instances share this lock; command/estop do not.
    flock -n -E 75 "$PWD/logs/model4497.lock" "$PY" -m dog3588 \
        --config "$CONFIG" run "$@" \
        --log "$PWD/logs/model4497-${mode}-$(date +%Y%m%d-%H%M%S).jsonl"
    local result=$?
    if [[ "$result" == 75 ]]; then
        printf '\n已有桌面控制进程运行，请先关闭该进程。\n'
    elif [[ "$result" != 0 ]]; then
        printf '\n运行已退出，返回码 %s。请查看上方错误和日志。\n' "$result"
    fi
}
while true; do
    printf '\nRobot12 · model4497\n配置：%s\n' "$CONFIG"
    printf '1  模型测试（模拟反馈，10 秒）\n'
    printf '2  真实反馈推理（不驱动关节，10 秒）\n'
    printf '3  实机运行（会驱动关节，15 秒；需已完成标定）\n'
    printf '4  前进 0.1 m/s（向已运行的控制进程发送 3 秒命令）\n'
    printf '5  急停\n6  检查 STM32 反馈（10 秒）\n7  香橙派直读 ToF（原始 64 区距离，10 秒）\n0  退出\n'
    read -r -p '请选择 [默认 1]：' choice || exit 0
    case "${choice:-1}" in
        1) run_controller mock --seconds 10 ;;
        2) run_controller shadow --hardware --seconds 10 ;;
        3) run_controller armed --hardware --arm --seconds 15 ;;
        4) "$PY" -m dog3588 --config "$CONFIG" command --vx 0.1 --seconds 3 ;;
        5) "$PY" -m dog3588 --config "$CONFIG" estop ;;
        6) flock -n -E 75 "$PWD/logs/model4497.lock" "$PY" -m dog3588 --config "$CONFIG" probe-link --seconds 10 ;;
        7) flock -n -E 75 "$PWD/logs/model4497.lock" "$PY" -m dog3588 --config "$CONFIG" probe-tof --seconds 10 ;;
        0) exit 0 ;;
        *) printf '请选择 0–7。\n' ;;
    esac
done
