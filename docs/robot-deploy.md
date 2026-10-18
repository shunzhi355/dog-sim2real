# Robot12 上位机部署（Orange Pi 5 Max）

本次按用户明确要求，将相机和 ToF 都放在香橙派。原 AI 说明中“STM32 采集 ToF”的段落是旧设计；本配置不消费 STM32 的 HEIGHT_SCAN。相邻 `project/dog` 仅作为协议源读取，本次不修改其固件。

```text
USB UVC 相机 → 独立采集进程 → 白赛道/五色色带/快递 ROI → 任务 FSM
VL53L5CX → 香橙派 I²C 线程 → 深度/射线/外参 + 对时 IMU → 8 点局部高度
STM32 → 独立全双工串口 → 12 轴反馈 + IMU → 观测历史 → ONNX Actor
FSM → 速度/步态 → Actor → 目标限位/变化率检查 → RobotLink → STM32 → 腿部总线
```

## 安装和无设备验证

```bash
sudo apt install python3-venv python3-pip build-essential i2c-tools v4l-utils
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,vision,tof]'
.venv/bin/python -m pytest -q
.venv/bin/python -m dog3588 doctor
.venv/bin/python -m dog3588 run --seconds 5
.venv/bin/python scripts/generate_mission_demo.py --output tmp/mission-demo
.venv/bin/python -m dog3588 --config tmp/mission-demo/config.yaml mission-replay tmp/mission-demo/mission.jsonl --mission-id demo --require-complete
```

Windows 用 `.venv/Scripts/python.exe`。`run` 默认完全模拟 I/O，但运行仓库内真实 ONNX；没有打开硬件。生成的任务回放包含**合成的几何事件和完成响应**，用于验证状态转换，不是现场相机或投递成功记录。支持黄/棕两轮：生成器加 `--colour brown`。`mission-replay --output logs/replay.jsonl` 可保存每拍任务输出，输出目录需先存在。

## 板级连接

依据仓库内《OrangePi_5_Max_RK3588_用户手册_v1.4.pdf》201–202、218–223 页：

| 设备 | 主机连接/设置 | 核验 |
|---|---|---|
| USB 相机 | UVC，USB 插入香橙派 | `v4l2-ctl --list-devices`，实际 capture 节点未必是 video0；优先稳定 by-id 路径 |
| VL53L5CX 模块 | I2C2_M0，SDA 物理 3 脚、SCL 物理 5 脚，共地 | `sudo orangepi-config` → System → Hardware → i2c2-m0 → 保存重启；确认 `/dev/i2c-2` |
| STM32 主机链路 | 独立 USB-UART 或已确认的全双工头排 UART | 不能接到四条腿的 DATA 总线；主机、STM32 波特率一致 |

手册列出的其他 I²C 选项：I2C2_M4=11/13、I2C4_M3=15/16、I2C5_M2=37/12、I2C5_M3=27/28、I2C8_M2=36/32（SDA/SCL，物理脚号）。同一个控制器的 M 复用不能同时启用。UART1_M1=RX27/TX28、UART3_M1=RX33/TX31、UART4_M2=RX19/TX23、UART6_M1=RX11/TX13，注意与 I²C 的复用冲突。模块供电、电平转换、上拉和 LPn 接法按所买模块原理图确认，不把裸传感器供电要求当作模块供电要求。

本次未自动更改 overlay、重启板卡、安装服务或向设备发送动作。VL53L5CX 固件上传可能耗时，启动超时单独配置；8×8 频率最大 15 Hz，策略仍为 50 Hz，重复使用同一有效帧不会刷新采集时间戳。驱动依赖与光轴深度语义见 [ToF 说明](tof.md)。

## 分层调试

复制 `config/robot-deploy.yaml` 为被 git 忽略的 `config/local-deploy.yaml`，填写实际串口、相机节点。以下诊断不会 ARM 或下发关节目标：

```bash
dog3588 --config config/local-deploy.yaml probe-link --seconds 10
dog3588 --config config/local-deploy.yaml probe-tof --seconds 10
dog3588 --config config/local-deploy.yaml camera --seconds 10 --snapshot tmp/camera.jpg
```

`probe-link` 解码现有 C 固件的 290 字节关节状态、41 字节 IMU 状态及故障/ACK。相邻固件的主机 UART 目前默认关闭、波特率仍是 115200，所以默认配置采用 115200；921600 只是目标，必须两端同步修改并做压力测试。具体帧、时间戳、首次 ARM 流程见 [RobotLink 协议](stm32-protocol.md)。主机不会猜测或修改下位机 UART 引脚。

`probe-tof` 输出原始 64 区数据，不需要虚构几何标定；安装射线、旋转/平移、8 点采样坐标、容差和有效性标定完成后才能启用完整感知运行。`probe-link` 可单独验证链路；暂时只验证旧盲走 actor 的影子推理时，可在**本地配置**显式设 `tof.enabled: false`，这不代表带地形感知的部署完成。

```bash
# 完成相关传感器配置后，真实反馈 + 推理，串口不发送 ARM/TARGET：
dog3588 --config config/local-deploy.yaml run --hardware --seconds 10 --log logs/shadow.jsonl
# 完成现场视觉/ToF标定及320维策略导入后，增加任务监督：
dog3588 --config config/local-deploy.yaml run --hardware --autonomous --seconds 10
# 标定与联调完成后的前台有限时长执行入口（不会自动启用）：
dog3588 --config config/local-deploy.yaml run --hardware --autonomous --arm --seconds 10
```

不启动独立 `camera --publish` 与 autonomous 模式竞争相机设备。任务模式直接消费相机最新结果；UDP 9871 仍用于独立速度调试和急停。`dog3588 estop` 使用相同配置端口即可。感知/模型/反馈失效、循环卡顿和退出都会关闭执行通道；下位机仍必须执行自己的心跳/目标超时保护。

## 训练契约

280 维：8 个本体观测项，每项各自保留 5 帧（term-major，最旧在前）。320 维：在相同项之后追加 `height_scan` 的 8×5 个值。没有实现 288 维混合布局，也不会补零把旧模型当作地形策略。

新模型必须由训练端重新训练/导出。manifest 的 `height_scan_contract` 明确：`frame: leveled_body`、`units: m`、`reference: body_origin`、`invalid_policy: stop`、`sample_xy_m: 8×2`。部署端比较这 8 个位置及顺序，要求与标定一致。高度是相对于水平化机身原点的 Z；若训练定义的是相对地面高度、反向距离或带 offset 的 height scan，必须先统一定义，不能只改维度。裁剪/缩放在观测项中按训练契约执行。

`scripts/import_policy.py --height-contract path.json` 接受训练端导出的 JSON：其中包含上述 `height_scan_contract`，以及准确的 `observation_function` 和 `observation_params`。导入器逐项核对保存的训练 YAML、ONNX 的 320→12 签名及校验值，再写入策略目录；未提供此参数时仍只接受已知 280 维训练约定。

任务包含慢速停靠、转向和循线减速。新策略训练命令范围必须覆盖这些速度；零速度单独允许用于站立，但不会因此把训练最小前进速度以下的任意非零速度视为受支持。

## 任务能力边界

白赛道和红/绿/蓝/黄/棕色带可由 OpenCV 实时检测；背筐快递识别使用独立 ROI，未知或黄棕混淆不会随意选色。路线左右关系不由颜色名称猜测，必须填写 `mission.route_map` 和 `return_route_map`。当前空表会在遇到分岔时停机。

FSM 已覆盖出发、循线、障碍接近/跨越/连续 3 秒稳定、选路、停靠、投递确认、返程及到家，并对丢线/超时闭锁。**现场的越障完成、选中出口、停靠、快递越门线、足端接触中心等几何事件生产器尚无已标定实现**，当前 live 相机不凭空输出这些事件；完整回放使用显式注释事件。因此 live 模式会在相应阶段等待并超时，不能称为已完成自主配送。事件定义与扩展位置见 [任务文档](mission.md)。机身 IMU/关节静止检查不能替代足端在赛道内的比赛判定。

当前 STM32 v1 没有投递机构命令。主机不能将普通 ACK 解释为投递完成；请求必须明确 unsupported 或接入单独验证的协议扩展。影子模式不得操作投递机构。现场感知与机构确认缺口、地形策略训练、接线/限位/安装标定及板上性能验证，是实机验收前的剩余工作。
