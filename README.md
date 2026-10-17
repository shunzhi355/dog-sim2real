# dog-3588：Robot12 香橙派部署框架

面向香橙派 5 Max / Ubuntu 22.04 / 12 个 LX824。已导入 `robot-lab` 现有 actor；采用 **Python + ONNX Runtime CPU，50 Hz 控制，四条腿独立串口，视觉独立进程**。无需在香橙派安装 Isaac Lab、CUDA 或 ROS。

一台香橙派承担策略、摄像头和四路舵机通信在计算上可行。实际能否稳定运动，取决于半双工回读时序、机械标定、IMU 和仿真执行器与实物的差异。初版重点是把这些环节拆开调试。当前不代表机器狗已通过实物行走/越障验收。

## 已实现

| 模块 | 功能 | 文件 |
|---|---|---|
| 策略 | ONNX CPU 推理、维度/哈希检查、测速 | `src/dog3588/policy.py` |
| 训练接口 | 280 维观测、每项 5 帧历史、步态时钟、12 维动作 | `src/dog3588/observation.py` |
| 舵机 | LX824 帧校验、分片解析、位置回读、四路并行、目标邮箱/超时停止 | `src/dog3588/hardware/` |
| IMU | WIT 标准 11 字节串口协议；MPU6050 I2C 六轴互补滤波 | `src/dog3588/hardware/imu.py` |
| 视觉 | USB/V4L2 摄像头、HSV 颜色、黑线检测、循线速度指令 | `src/dog3588/vision.py` |
| 命令 | 本机 UDP 速度接口、来源选择、过期归零、软件急停 | `src/dog3588/commands.py` |
| 防护 | 标定检查、限位、变化率、姿态/数据年龄/推理超时 | `src/dog3588/safety.py` |
| 部署 | SSH/SFTP 同步、独立虚拟环境、只读服务模板、测试 | `scripts/`、`deploy/`、`tests/` |

相机与 IMU 型号尚未选定；驱动是明确协议的可选实现，不能认为任意 IMU 都兼容。四路通信和传感器尚未做实机接线验收。各关节零位/方向/限位故意留空；`--hardware --arm` 会拒绝未标定配置。

## 开始运行

已部署板端路径：`/home/orangepi/dog-3588`，地址 `192.168.1.102`，用户 `orangepi`。密码不写入项目。

登录香橙派后：

```bash
cd /home/orangepi/dog-3588
source .venv/bin/activate
dog3588 doctor
dog3588 benchmark --count 1000
dog3588 run --seconds 10 --log logs/mock.jsonl
```

最后一条使用真实 ONNX + 模拟传感器/舵机，只验证软件数据链与调度，不是物理仿真，不会打开串口。

本机 Windows：

```powershell
.\.venv\Scripts\python.exe -m dog3588 run --seconds 10
.\.venv\Scripts\python.exe -m pytest -q
```

新环境：Python 3.10+，创建虚拟环境后 `python -m pip install -e '.[dev,vision]'`。香橙派使用 `bash deploy/setup.sh`；若缺 venv，先安装 `python3-venv python3-pip`。

## VNC 远程桌面自启动

香橙派使用 `orangepi` 用户启动 TightVNC / XFCE 虚拟桌面，systemd 服务名为 `vncserver.service`。VNC 客户端填写 `192.168.1.102:1`，或在主机/端口分开的界面填写主机 `192.168.1.102`、端口 `5901`；分辨率为 `1024×768`、色深 24 位。连接密码已单独提供，不保存在仓库中。

服务文件位于板端 `/etc/systemd/system/vncserver.service`，桌面启动脚本位于 `/home/orangepi/.vnc/xstartup`；对应模板保存在 `deploy/vncserver.service` 和 `deploy/vnc-xstartup`。虚拟桌面使用显示号 `:1`，有独立的 DBus 会话、Xauthority 和运行目录。

```bash
systemctl status vncserver.service       # 查看运行状态
systemctl is-enabled vncserver.service   # 确认开机自启
sudo systemctl restart vncserver.service # 重启远程桌面，会断开当前 VNC 连接
vncpasswd                               # 以 orangepi 用户修改 VNC 密码
```

改密码后重启服务，使新连接使用更新后的密码。当前服务通过 systemd 管理，无需另外配置 `/etc/init.d/vncserver` 或 `update-rc.d`。

## 三种运行模式

| 命令 | 传感器 | 舵机 |
|---|---|---|
| `dog3588 run` | 模拟 | 模拟，无设备访问 |
| `dog3588 --config config/local.yaml run --hardware` | 真实 | 只发位置查询，计算目标但不下发运动 |
| 上一条加 `--arm` | 真实 | 标定和启动检查通过后允许 LOAD/MOVE |

`--seconds 0` 表示持续运行。故障退出，不自动重启主动控制。服务模板默认只读，未安装/启动任何开机运动服务。

先复制 `config/robot.yaml` 为 `config/local.yaml`，按 [接线与标定](docs/hardware.md) 完成一条腿回读、十二关节映射和 IMU 坐标核对，再进入实物只读推理。

## 硬件引脚与接口配置

硬件配置统一填写在 `config/local.yaml`。**排针物理脚号**用于接线，**UART/I2C 设备名**用于打开接口；两者不能混填。当前方案由外部转接电路自动切换半双工收发，四路均不使用方向使能 GPIO。

以下依据项目根目录的 [Orange Pi 5 Max 用户手册 v1.4](OrangePi_5_Max_RK3588_用户手册_v1.4.pdf)：第 16 页接口复用图、第 211–212 页 GPIO 对照图、第 219 页 I2C 表、第 221 页 UART 表。页码为手册印刷页码，也是此 PDF 的页序号。仅适用于 **Orange Pi 5 Max**，不是 5 / 5 Plus / 5 Ultra。

2026-10-17 已通过 SSH 只读核对当前板端 `rk3588-orangepi-5-max.dtb` 和对应 overlay：下列四路 UART 与 I2C2_M0 的 bank/offset 与手册一致。但当前仍只有 `/dev/ttyS7`，四路舵机 UART **尚未启用、尚未接线验收**；不能把“手册已核实”视为“硬件已可运行”。

### 40 针物理脚号与复用功能总表

按板上 **Pin 1 标记**定位，奇数脚为 1、3、…、39，另一列为 2、4、…、40；不要根据照片旋转后的左右方向猜脚号。结合手册第 16 页接口图与第 211–212 页 Physical 列核对。表中 TX/RX 均从香橙派视角命名；同一脚列出的功能是互斥选择，不能同时使用。

`GPIOx_Yn` 中 x 是 bank，A/B/C/D 的组内起点分别为 0/8/16/24；line offset = 组内起点 + n。手册的全局 GPIO 序号 = bank × 32 + offset，不是物理脚号，也不是 wiringOP 的 wPi 号。

| 物理脚号 | 电源/芯片引脚 | bank:offset（全局 GPIO 序号） | 手册列出的复用功能 |
|---:|---|---|---|
| 1 | 3.3 V | - | 电源，不是 GPIO |
| 2 | 5 V | - | 电源，不是 GPIO |
| 3 | GPIO0_C0 | 0:16（16） | I2C2_SDA_M0 / CAN0_RX_M0 / PWM1_M0 |
| 4 | 5 V | - | 电源，不是 GPIO |
| 5 | GPIO0_B7 | 0:15（15） | I2C2_SCL_M0 / CAN0_TX_M0 / PWM0_M0 |
| 6 | GND | - | 公共地 |
| 7 | GPIO1_A7 | 1:7（39） | PWM3_IR_M3 |
| 8 | GPIO0_B5 | 0:13（13） | UART2_TX_M0 / I2C1_SCL_M0；预留调试口 |
| 9 | GND | - | 公共地 |
| 10 | GPIO0_B6 | 0:14（14） | UART2_RX_M0 / I2C1_SDA_M0；预留调试口 |
| 11 | GPIO1_A0 | 1:0（32） | UART6_RX_M1 / I2C2_SDA_M4 / SPI4_MISO_M2 |
| 12 | GPIO4_A6 | 4:6（134） | I2C5_SCL_M2 |
| 13 | GPIO1_A1 | 1:1（33） | UART6_TX_M1 / I2C2_SCL_M4 / SPI4_MOSI_M2 |
| 14 | GND | - | 公共地 |
| 15 | GPIO1_A2 | 1:2（34） | I2C4_SDA_M3 / PWM0_M2 / SPI4_CLK_M2 |
| 16 | GPIO1_A3 | 1:3（35） | I2C4_SCL_M3 / PWM1_M2 / SPI4_CS0_M2 |
| 17 | 3.3 V | - | 电源，不是 GPIO |
| 18 | GPIO1_A4 | 1:4（36） | GPIO（手册未列其他复用） |
| 19 | GPIO1_B2 | 1:10（42） | UART4_RX_M2 / SPI0_MOSI_M2 |
| 20 | GND | - | 公共地 |
| 21 | GPIO1_B1 | 1:9（41） | SPI0_MISO_M2 |
| 22 | GPIO1_B0 | 1:8（40） | GPIO（手册未列其他复用） |
| 23 | GPIO1_B3 | 1:11（43） | UART4_TX_M2 / SPI0_CLK_M2 |
| 24 | GPIO1_B4 | 1:12（44） | SPI0_CS0_M2 |
| 25 | GND | - | 公共地 |
| 26 | GPIO1_B5 | 1:13（45） | SPI0_CS1_M2 |
| 27 | GPIO1_B7 | 1:15（47） | UART1_RX_M1 / I2C5_SDA_M3 / PWM13_M2 |
| 28 | GPIO1_B6 | 1:14（46） | UART1_TX_M1 / I2C5_SCL_M3 |
| 29 | GPIO3_C1 | 3:17（113） | SPI1_CLK_M1 |
| 30 | GND | - | 公共地 |
| 31 | GPIO3_B5 | 3:13（109） | UART3_TX_M1 / CAN1_RX_M0 / PWM12_M0 |
| 32 | GPIO1_D6 | 1:30（62） | I2C8_SCL_M2 / PWM14_M2 |
| 33 | GPIO3_B6 | 3:14（110） | UART3_RX_M1 / CAN1_TX_M0 / PWM13_M0 |
| 34 | GND | - | 公共地 |
| 35 | GPIO3_C2 | 3:18（114） | SPI1_CS0_M1 / PWM14_M0 |
| 36 | GPIO1_D7 | 1:31（63） | I2C8_SDA_M2 |
| 37 | GPIO4_A7 | 4:7（135） | I2C5_SDA_M2 |
| 38 | GPIO3_C0 | 3:16（112） | SPI1_MISO_M1 |
| 39 | GND | - | 公共地 |
| 40 | GPIO3_B7 | 3:15（111） | SPI1_MOSI_M1 |

所有 GPIO/UART/I2C 信号按 **3.3 V 逻辑**接入，不接受舵机电源电压或 5 V TTL 输出。PWM 是复用能力说明，LX824 本项目使用串口总线协议，不用这些 PWM 脚逐个控制舵机。表中只整理手册列出的板级复用，不等于 RK3588 芯片全部潜在功能。

### 四路舵机串口

每条腿三个 LX824 共用一条独立 DATA 总线，四条腿分别连接四路半双工转接电路，均使用 **115200、8N1**。

| 腿 / 配置总线名 | 舵机 ID：髋侧摆 / 上部驱动 / 连杆输入驱动 | UART 复用 / overlay 选项 | 预期 `port` | TX 物理脚（芯片名） | RX 物理脚（芯片名） |
|---|---|---|---|---|---|
| 左前 FL（LF） / `fl` | 1 / 2 / 3 | UART1_M1 / `uart1-m1` | `/dev/ttyS1` | **28**（GPIO1_B6） | **27**（GPIO1_B7） |
| 右前 FR（RF） / `fr` | 4 / 5 / 6 | UART3_M1 / `uart3-m1` | `/dev/ttyS3` | **31**（GPIO3_B5） | **33**（GPIO3_B6） |
| 左后 RL（LH） / `rl` | 7 / 8 / 9 | UART4_M2 / `uart4-m2` | `/dev/ttyS4` | **23**（GPIO1_B3） | **19**（GPIO1_B2） |
| 右后 RR（RH） / `rr` | 10 / 11 / 12 | UART6_M1 / `uart6-m1` | `/dev/ttyS6` | **13**（GPIO1_A1） | **11**（GPIO1_A0） |

上述 TX 接转接电路的主机发送输入，RX 接转接电路的主机接收输出；**不能将 TX/RX 直接并到舵机单线 DATA**。四块转接电路与香橙派公共地相连，可从 6/9/14/20/25/30/34/39 任一 GND 脚引出；舵机大电流回路不应经过香橙派细杜邦地线。

按手册运行 `sudo orangepi-config` → `System` → `Hardware`，选择 `uart1-m1`、`uart3-m1`、`uart4-m2`、`uart6-m1`；若采用下文 I2C IMU，再选 `i2c2-m0`。先备份 boot 配置，保存后在允许断开 SSH/VNC 的维护窗口重启，核对 `/dev/ttyS1`、`ttyS3`、`ttyS4`、`ttyS6` 和 `/dev/i2c-2`，再填写 `buses[].port`。**本次文档更新未执行启用或重启。**

不要选择同 UART 的其他 M 模式代替表中模式。此方案禁止同时启用 `i2c2-m4` / `spi4-m2`（冲突 UART6）、`i2c5-m3`（冲突 UART1）、`spi0-m2`（冲突 UART4）、`can1-m0`（冲突 UART3）。8/10 脚保留作手册定义的调试串口，不接舵机或 IMU；当前看到 `/dev/ttyS7` 并不能证明它就是 8/10 脚对应设备。USB 舵机转接器则可填写 `/dev/serial/by-id/...`。

### 无方向使能引脚的串口接线

四路默认均使用 `direction.mode: auto`，软件不申请 GPIO，也不输出 TX_EN/RX_EN。每路主机信号只需 **TX、RX、GND** 接自动方向转接电路，再由其单线 DATA 接该腿三个 LX824；转接器供电按其规格连接。

| 香橙派主机侧信号 | 接到转接电路的位置 | 配置字段 |
|---|---|---|
| UART TX | 主机侧发送数据输入，按转接板原理图核对 | `port` 对应 UART 的 TX 排针 |
| UART RX | 主机侧接收数据输出，按转接板原理图核对 | `port` 对应 UART 的 RX 排针 |
| GND | 转接板与舵机电源公共地 | 无软件字段 |

原计划的 **15、16、18、22、24、26、29、35** 脚不再分配给串口使能，可按总表另作 GPIO/I2C/SPI/PWM 用途，但新分配仍需检查复用冲突。四路串口的 TX/RX 物理脚号保持不变。

以下摘录左前腿配置，其余三路使用相同 `direction` 设置。已有 `config/local.yaml` 若仍为 `mode: gpio`，请把每路 `direction` 替换成 `mode: auto`，删除旧的 `chip`、`tx_enable`、`rx_enable` 字段。仅修改模式不等于完成实物通信验收，`verified` 仍保持 `false`：

```yaml
buses:
  - name: fl
    port: /dev/ttyS1       # UART1_M1，TX=物理28，RX=物理27
    baudrate: 115200
    verified: false       # 稳定回读验收通过后才能置 true
    read_timeout_s: 0.008
    query_period_s: 0.006
    quiet_after_timeout_s: 0.006
    direction:
      mode: auto
```

`auto` 表示收发方向由外部硬件完成，不是软件自动识别电路，也不是 UART 自动变成单线接口。应实测转接电路在发送末位后释放 DATA、完整接收应答，普通 USB-TTL 或 RS485 模块不能直接视为这种转接器；不能靠把旧使能脚悬空来实现自动切换。

### IMU 与摄像头

| 外设方案 | 接线 | 配置字段 | 核对内容 |
|---|---|---|---|
| WIT 标准串口 IMU | IMU TX → 主机独立 UART/USB 串口 RX，GND 共地 | `imu.driver: wit_serial`、`serial_port`、`baudrate` | 波特率一致，持续输出角速度与欧拉角帧；当前驱动只接收数据 |
| MPU6050 I2C IMU | 建议 SDA → **3 脚 GPIO0_C0**，SCL → **5 脚 GPIO0_B7**，GND → 6 脚；供电按模块规格 | `imu.driver: mpu6050_i2c`、**`i2c_bus: 2`**、`i2c_address: 0x68`（AD0=0） | 启用 **`i2c2-m0`** 后核对 `/dev/i2c-2`；上拉电压应为 3.3 V，不是 5 V |
| USB 摄像头 | 接 USB 接口 | `vision.source`、`backend: v4l2` | 填实际可采集的 `/dev/videoN`，当前默认 `/dev/video0` |
| MIPI 摄像头 | 接兼容的摄像头座与排线 | 系统驱动产生采集节点后再设置 `vision.source` | 传感器型号、排线方向、内核驱动与 ISP 支持 |

四路硬 UART 都分给舵机时，IMU 可使用 I2C 或独立 USB 串口。IMU 安装方向通过 `imu.sensor_to_body` 配置，核对角速度/重力方向并完成校准后再置 `mount_verified: true`。摄像头的 `video-dec0`、`video-enc0` 是编解码节点，不作为图像采集设备。

手册还给出以下 I2C 复用选择；同一控制器的不同 M 模式不是独立总线，不能同时启用。默认模板的 `i2c_bus: 1` **不适用于本建议接线**，需在本地配置改成 2。IMU 型号未确定，不能仅按接口相同就认定驱动兼容。

| I2C 复用 / overlay 选项 | SDA 物理脚 | SCL 物理脚 | 本项目四 UART、无使能 GPIO 方案下的状态 |
|---|---:|---:|---|
| I2C2_M0 / `i2c2-m0` | 3 | 5 | 推荐用于 IMU，预期 `/dev/i2c-2` |
| I2C2_M4 / `i2c2-m4` | 11 | 13 | 冲突 UART6，不启用 |
| I2C4_M3 / `i2c4-m3` | 15 | 16 | 使能脚取消后可备选，预期 `/dev/i2c-4`；需核对模块与复用 |
| I2C5_M2 / `i2c5-m2` | 37 | 12 | 引脚未分配，可备选；先确认设备树与模块 |
| I2C5_M3 / `i2c5-m3` | 27 | 28 | 冲突 UART1，不启用 |
| I2C8_M2 / `i2c8-m2` | 36 | 32 | 引脚未分配，可备选；先确认设备树与模块 |

USB 摄像头无需占用 40 针；MIPI 摄像头接专用 CSI 连接器，也不接 GPIO 排针。相机供电、排线和 IMU 模块电源应分别核对，不能仅凭“3.3 V/5 V 模块”标签推断其信号电平。

舵机使用满足负载电流需求的独立电源，并与主机/转接板共地；UART 与转接器须兼容香橙派 3.3 V 逻辑电平。接线时将舵机电源与主机信号脚分开。

配置后先执行 `dog3588 --config config/local.yaml doctor` 查看设备，再按下节分别测试单腿回读、IMU 和摄像头。完整接线验收与关节零位/方向/限位标定见 [接线与标定](docs/hardware.md)。

## 分开调试

```bash
# 无需关节弧度标定，设置 fl 真实串口，使用自动方向转接电路
dog3588 --config config/local.yaml probe-servos --bus fl --rounds 100

# IMU 不访问舵机；MPU6050 模式会配置该传感器寄存器
dog3588 --config config/local.yaml probe-imu --seconds 10

# 相机诊断不发送运动指令
dog3588 --config config/local.yaml camera --seconds 10 --snapshot /tmp/dog-camera.jpg

# 与控制进程同时运行：前进、左转；20 Hz 刷新，命令过期后归零
dog3588 command --vx 0.12 --wz 0.2 --seconds 3
dog3588 estop
```

视觉循线：把 `command.source` 改为 `vision`；另一个终端运行 `dog3588 --config config/local.yaml camera --publish`。丢线发送零速度，相机进程停止后命令超时归零。颜色识别目前只输出检测信息；没有默认的“看到颜色立即转弯”行为。先用 mock 模式验证图像左右与转向符号。

UDP 只监听 `127.0.0.1:9871`，远程通过 SSH 运行命令。`estop` 通过软件发送 MOVE_STOP 并保持已有负载状态；它不是断电急停，不能代替独立断电装置。

## 策略更新

```bash
python scripts/import_policy.py --run ../robot-lab/logs/rsl_rl/robot12/你的训练目录
```

导入器读取该次训练的 `params/env.yaml`，校验接口，把 `exported/policy.onnx` 与外置权重合并成单文件；保存模型 SHA256、训练参数和关节顺序。不会读取/执行 `.pt` pickle。新任务若改变观测或动作定义，导入器会明确拒绝，需要同步升级适配器。

当前模型来自 `2026-10-15_22-39-49_forward_baseline_continue`，输入 `obs[1,280]`、输出 `actions[1,12]`。原 ONNX 没有 checkpoint 元数据，不能仅凭目录内存在 `model_1498.pt` 断定它就是导出来源。

## 文档

- [架构与方案选择](docs/architecture.md)：为什么先做单板、借鉴 Microduck 的哪些部分。
- [接线、方向控制与标定](docs/hardware.md)：硬件还差什么以及逐步验收。
- [训练到实物的数据契约](docs/policy-contract.md)：观测切片、时钟、单位、动作。
- [部署、调试和扩展](docs/deployment.md)：同步、服务、常见故障与新增功能。
- [本次验证记录](docs/validation.md)：本机与香橙派实测结果和未验证边界。

