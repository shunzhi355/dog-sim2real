# 使用别人训练好的 model_4497

无需在香橙派安装 robot-lab、Isaac Lab 或重新训练。部署使用 ONNX Runtime CPU 推理。

已将 `model_4497_export(1)/policy.onnx` 和外置权重合并为
`policies/model4497/policy.onnx`，入口配置为 `config/model4497.yaml`。
原始导出目录保留；部署后的模型不再依赖外置 `.data` 或 `policy.pt`。

## 已采用的接口

提供者未附训练 YAML。按用户要求直接接入，当前沿用仓库已有 robot-lab
步态/前进策略约定；这些参数是部署假设，不是从新模型权重中恢复出的信息。
来源、模型校验值及 `contract_verified: false` 已记录在新 manifest。

- 输入 `obs: float32[1,280]`，输出 `actions: float32[1,12]`。
- 控制频率 50 Hz，每个观测项独立保留 5 帧，最旧在前。
- 项目顺序：角速度、重力投影、速度命令、相对关节角、相对关节速度、上次动作、步态参数、步态时钟。
- 关节速度缩放 0.05；动作目标为 `默认关节角 + 0.25 × action`（弧度）。
- 关节顺序：左前、右前、左后、右后；每腿 hip、upper、input。
- 采用的前进范围为 0.1–0.3 m/s，侧向为 0，转速为 ±0.4 rad/s，另允许零速度站立。

这是 280 维策略，不直接接收 ToF；本配置关闭 ToF 采集，用于手动运动调试。
相机和 ToF 仍由上位机管理，可单独使用诊断命令。本配置不启用自主比赛任务。
模型输入输出已验证，训练语义仍需与提供者核对，尤其是动作缩放和关节顺序。

## Windows 本机运行

在 dog-3588 项目根目录执行（运行真实策略，反馈为模拟数据，不控制实机）：

```powershell
.\.venv\Scripts\python.exe -m dog3588 --config config/model4497.yaml benchmark --count 1000
.\.venv\Scripts\python.exe -m dog3588 --config config/model4497.yaml run --seconds 10 --log logs/model4497-mock.jsonl
```

## Orange Pi 5 Max

将项目（包括 `policies/model4497`）复制到板上，不要复制 Windows 的 `.venv`。
在板上项目根目录创建 Linux 环境，先运行无硬件验证：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m dog3588 --config config/model4497.yaml benchmark --count 1000
.venv/bin/python -m dog3588 --config config/model4497.yaml run --seconds 10
cp config/model4497.yaml config/local-model4497.yaml
```

在 `config/local-model4497.yaml` 填入实际 `bridge.port`，波特率与 STM32 一致。
STM32 必须已启用 RobotLink，并连续上报关节角/速度和机身 IMU。

```bash
.venv/bin/python -m dog3588 --config config/local-model4497.yaml probe-link --seconds 10
.venv/bin/python -m dog3588 --config config/local-model4497.yaml run --hardware --seconds 10 --log logs/model4497-shadow.jsonl
```

第二条使用真实反馈进行影子推理，不下发关节动作。完成关节零位、方向、顺序、限位和
IMU 机身坐标核对后，填写各关节实际限位和配置中的对应 `verified` 标志。
不可仅为消除报错而将标志改为 true。

有限时长实机执行（会驱动关节；首轮在支撑条件下验证）：

```bash
.venv/bin/python -m dog3588 --config config/local-model4497.yaml run --hardware --arm --seconds 15 --log logs/model4497-armed.jsonl
```

同一台香橙派的另一个终端发送前进命令；默认无命令时目标速度为零：

```bash
.venv/bin/python -m dog3588 --config config/local-model4497.yaml command --vx 0.1 --seconds 3
.venv/bin/python -m dog3588 --config config/local-model4497.yaml estop
```

命令会过期；`estop` 锁存停机，需重新启动控制进程才能恢复。
相机和 ToF 的安装、诊断及完整任务接入参见 [部署说明](robot-deploy.md)。
## 已部署的香橙派与桌面入口

已部署到 `orangepi@192.168.0.102:/home/orangepi/dog-3588`，桌面目录为
`/home/orangepi/Desktop`：

- **Robot12 策略启动**：双击后显示菜单，回车或选择 1 运行 10 秒模型测试；
  2 为真实反馈推理，3 为 15 秒实机运行，4 为前进命令，5 为急停，6 为链路检查，
  7 为香橙派直接读取 ToF 原始距离（接线及 I²C 启用后使用，不依赖 STM32）。
- **Robot12 急停**：独立发送锁存急停命令。

入口使用 `config/local-model4497.yaml`；再次安装不会覆盖已存在的本地配置。
控制实例使用文件锁避免多个桌面实例同时打开链路。要在运行期间发送速度命令，
另开一次启动器选择 4；急停可直接双击独立图标。桌面入口安装命令为：

```bash
.venv/bin/python deploy/install_desktop.py
```

2026-10-18 板上验证：282 项测试通过；启动器模型测试 10 秒完成 500 周期，
约 50 Hz，漏周期 0；1000 次 CPU 推理基准 P95 约 0.043 ms。
旧部署文件已备份到 `/home/orangepi/deploy-backups/`。

尚未实机驱动关节。部署时未发现 USB 串口或相机节点；已有 `/dev/ttyS*`
不能据此判断哪个连接 STM32。`bridge.port` 仍需填写实际链路，关节和 IMU
标定标志保留未验证状态。桌面选项 1 可立即运行，选项 2/3/6 需要先完成对应连接配置。
