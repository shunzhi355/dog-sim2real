# STM32 通信协议、接口与测试

核对日期：2026-10-18。上位机文件：`src/dog3588/transport.py`；测试文件：`tests/test_transport.py`。

## 实现依据与当前边界

本模块按相邻 `../dog` 的以下源码实现，访问该目录仅为**只读核对**，没有修改固件：

- `AGENTS.md`：SI 单位、机体坐标系、不在固件周期路径中阻塞、明确未确认的硬件映射。
- `Protocol/Inc/robot_link.h`、`Protocol/Src/robot_link.c`：二进制帧、CRC、限长及重同步。
- `Protocol/Inc/robot_session.h`、`Protocol/Src/robot_session.c`：序号、ARM、目标、心跳、STOP 和 ACK。
- `Core/Src/lower_link.c`：实际消息打包偏移、遥测发送周期、UART 接入与发送队列。
- `Robot/Inc/robot_types.h`、`Robot/Inc/robot_config.h`、`Robot/robot_sensors` 对应的 `Inc` / `Src` 文件：关节顺序、总线数、IMU 单位及采样语义。
- `Core/Inc/bus_servo.h`、`Core/Src/bus_servo.c`：STOP 对运动队列的实际作用。

**这里实现的是现有 RobotLink v1 的 Python 对端，不是另外设计的 CRC32 协议。** 版本为 1，序号为 **uint16**，时间戳为 **uint32 毫秒**，校验为 **CRC16-CCITT-FALSE**。

源码当前仍有 `ROBOT_HOST_UART6_ENABLE=0` 和 `ROBOT_IMU_PROTOCOL_CONFIRMED=0`。实际连线、主机 UART 绑定、IMU 驱动、关节标定和映射未在本任务中验证；没有 Orange Pi / STM32 / 舵机硬件联调结果。默认 115200 baud 来自当前固件设置，不能把项目目标 921600 当成已实现。没有收到遥测时不会生成虚假有效样本。

USB 摄像头和 ToF 属于上位机。`HEIGHT_SCAN` 只消费帧顺序，不提供地形观测；当前相邻源码还包含 `TOF_RAW=0x86`，本模块忽略该消息，不把它连接到上位机 ToF 数据源。未知消息不会刷新接收健康时间。

**投递不属于现有固件 v1 能力。** 本文后半部分的 `delivery-v1` 是默认关闭的、尚无下位机配套实现的应用扩展。开启 Python 配置不等于固件支持，也没有硬件成功投递记录。

## 对外接口与集成

构造函数只接收 `bridge` 子配置：

```python
bridge = STM32Bridge(cfg["bridge"], armed=arm)
bridge.start()                       # 此时才打开串口；不 ARM
joints = bridge.read()               # JointState，三个独立复制的 12 维数组
imu = bridge.read_imu()              # ImuState，omega/gravity/stamp

# runtime 已先完成 guard/preflight，now 是原控制周期的 monotonic 时间。
bridge.command(targets, now)         # 12 个电机空间目标角，单位 rad

bridge.estop()                       # 闭锁；后台发送 STOP
bridge.close()                       # 停止后台、尝试 STOP、关闭串口
```

| 接口 | 行为 |
|---|---|
| `STM32Bridge(cfg, armed=False)` | 只初始化内存和校验配置，不打开设备、不发指令 |
| `start()` | 打开一个串口、清理旧输入、启动唯一 I/O 线程、发送初始 STOP；返回不表示握手或传感器已就绪；重复调用幂等 |
| `read()` | 返回策略顺序的关节位置 rad、速度 rad/s、采集 monotonic 秒；无有效反馈的 stamp 为 0 |
| `read_imu()` | 返回机体角速度 rad/s、单位向下重力和采集 monotonic 秒 |
| `command(targets, now)` | 复制并替换最新目标；严格检查形状、有限数、float32 可表示性及原时间戳；首次有效目标触发 ARM 握手 |
| `bridge.imu` | 借用同一个桥的 ImuIO facade；`read()` 调 `read_imu()`，`start()` 幂等启动桥，`close()` 不关闭桥 |
| `estop()` | 本地永久故障闭锁并唤醒后台停止，不在调用线程写串口 |
| `fault` | 线程安全查询故障字符串，健康时为 `None` |
| `startup_sync_ready` | bool；表示初始 STOP 已获得匹配 ACK，不表示传感器有效或可以 ARM |
| `status()` | 返回复制的 JSON 安全诊断字典，启动前及故障后也可查询；不续任何看门狗 |
| `close()` | 已打开时尝试发送 STOP 后关闭串口并 join；未打开时无设备操作；已有故障或停止失败抛 `SafetyFault` |
| `delivery(request_id)` | 仅显式开启配套扩展时可用，uint32 请求 ID，同一实例内幂等 |
| `delivery_ack(request_id)` | bool；只有匹配请求同时收到接收 ACK 和完成通知才为真 |

`deployment.BridgeImu` 调用 `read_imu()`，运行循环已经接入该适配器；不要求换成 `bridge.imu`。

`status()` 提供 `phase`、`startup_sync_ready`、`received_frames`、`last_rx_age_s`、`fault`、`raw_position`、`flags`。`received_frames` 统计通过 CRC 的解析帧，包括随后被会话忽略的帧；`last_rx_age_s` 是可信接收年龄，尚未同步时为 `None`，不是关节采集年龄。`raw_position` 是原始有符号计数，未收到时为 `None`；它与 `flags` 都按策略关节顺序排列。输出可用 `json.dumps(..., allow_nan=False)` 序列化，不输出 NaN/Infinity。这样 probe 可以识别“链路已同步、只收到未标定 raw 反馈”，而该反馈的 JointState stamp 仍为 0，不能绕过 ARM 的有效性检查。

首次 `command()` 的状态链为：初始 STOP 已匹配 ACK → 有效且新鲜的 12 关节和 IMU → HEARTBEAT 及匹配 ACK → ARM(全 12 关节)及匹配 ACK → 最新 TARGET。**无需另调 `arm()`**。每个 ACK 都必须匹配被发送的消息类型及主机序号。ARM 拒绝、ACK 缺失、反馈失效或目标过期时不发送 TARGET、不自动重 ARM。

`armed=False` 为 shadow：可以每 20 ms 调用 `command()` 做输入校验，但不发送 ARM、TARGET 或 DELIVERY_REQUEST。会发送初始/退出 STOP 和心跳。`probe-link` 只调用 start/read/read_imu 时持续 disarmed；首次 command 前没有目标看门狗期限，因此等待上位机 ToF 初始化不会触发“首目标超时”。

故障后 `read/read_imu/command` 抛出 `SafetyFault`；不能靠新目标或重新 `start()` 清故障。需要显式关闭、排查原因并新建桥。目标软限位及策略前置检查仍由现有 `SafetyGuard` 负责，固件还会检查标定和执行器限位；本模块不臆造机械范围。

## 配置

`config/robot-deploy.yaml` 的 `bridge` 键可直接使用。无需传入根配置或 `safety` 子配置；因此修改根 `safety` 的阈值不会自动改变桥的阈值。

| `bridge` 键 | 默认值 | 语义 |
|---|---|---|
| `port` | 必填 | 独立 STM32 主机链路设备路径，不是舵机串口 |
| `baudrate` | 115200 | 两端必须一致 |
| `verified` | false | `armed=True` 必须为 true，代表链路实测确认 |
| `imu_mount_verified` | false | `armed=True` 必须为 true |
| `joint_mapping_verified` | false | `armed=True` 必须为 true |
| `joint_map` | `[0,1,...,11]` | `joint_map[i]` 为策略第 i 个关节对应的固件索引，必须是完整排列 |
| `target_timeout_s` | 0.08 | 自原 `command(..., now)` 时间起的有效期，不得大于固件 0.10 s |
| `telemetry_timeout_s` | 0.10 | ARM/运行期间关节和 IMU 的最大采集年龄 |
| `heartbeat_period_s` | 0.10 | 心跳发送间隔，须小于固件 0.30 s |
| `heartbeat_timeout_s` | 0.30 | 上位机有效接收超时；固件没有独立的下行 HEARTBEAT，用可信 ACK/遥测检测 |
| `arm_timeout_s` | 0.08 | 首次 HEARTBEAT → ARM 总握手时限 |
| `ack_timeout_s` | `arm_timeout_s` | 每条已发送命令的匹配 ACK 时限，也适用于初始 STOP |
| `read_timeout_s` | 0.005 | pyserial 读取的上限；正常路径按 `in_waiting` 读取已有字节 |
| `write_timeout_s` | 0.02 | 整帧写入总预算，分段写共享期限，也受原目标期限约束 |
| `poll_period_s` | 0.002 | 后台循环等待时间 |
| `duration_ms` | 20 | 固件目标动作时间，整数 1..100 ms |
| `startup_timeout_s` | 3.0 | 初始 STOP 同步兜底时限；不能延长更短的 ACK 超时，不限制等待首目标 |
| `delivery_extension_version` | 0 | 0 禁用；1 为本文提议的 delivery-v1，下位机尚未实装 |
| `delivery_timeout_s` | 2.0 | 投递请求到明确完成的总时限 |

所有时间配置必须为有限正数；`ack_timeout_s <= target_timeout_s <= 0.10` 且 `ack_timeout_s <= arm_timeout_s`；读、写和轮询时间必须小于 ACK 时限。三个 verified 标志不能由本模块自动置真。

固件关节索引按 `LF/RF/LH/RH`，每腿 `HIP_ROLL/HIP_PITCH/KNEE`。默认 identity map 只表达数组排列，不证明机械命名、方向和零位与策略 `FL/FR/RL/RR: hip/upper/input` 一致。`joint_map` 只重排，不调整符号或零位，须使用固件经过标定的电机空间弧度。

## v1 帧结构

所有多字节量小端，float 为 IEEE754 binary32，无 C 结构体 padding。

| 偏移 | 长度 | 字段 |
|---|---|---|
| 0 | 2 | magic：`A5 5A` |
| 2 | 1 | version：`01` |
| 3 | 1 | type |
| 4 | 2 | payload length，0..384 |
| 6 | 2 | sequence，uint16 |
| 8 | 4 | timestamp_ms，uint32 |
| 12 | length | payload |
| 12+length | 2 | CRC16，小端 |

总长 `length + 14`，最大 398 字节。CRC 覆盖从 version（偏移 2）到 payload 末尾，不包含 magic 或 CRC 自身；poly=`0x1021`、init=`0xffff`、refin/refout=false、xorout=0。检查向量：ASCII `123456789` → `0x29B1`。

增量解析支持碎片、粘包、噪声和 CRC 失败后的逐字节重同步；内部缓存最多 398 字节，字节间隔超过 100 ms 清理半帧。错误版本、超长帧、CRC 错误不进入会话。CRC 正确但已知消息长度或有效字段错误会闭锁。合法长度的残缺帧可能等到后续字节或 100 ms 超时才恢复；此时安全看门狗照常运行。

## v1 命令与 ACK

| type | 消息 | payload |
|---|---|---|
| `0x01` | HEARTBEAT | 空 |
| `0x02` | JOINT_TARGET | `mask:u16, duration_ms:u16, target_rad:float[12]`，52 字节 |
| `0x03` | STOP | 空 |
| `0x04` | ARM | `mask:u16`，2 字节 |
| `0x85` | ACK，下行 | `request_type:u8, code:u8, request_sequence:u16, faults:u32`，8 字节 |

本实现 ARM/TARGET 的 mask 固定 `0x0fff`。ACK code：0 OK、1 FORMAT、2 SEQUENCE、3 STALE、4 DISARMED、5 NOT_READY、6 TARGET。非零匹配 ACK 闭锁；普通 TARGET ACK 表示固件接受/排队，不代表舵机已到位。

ARM/TARGET 帧头时间戳回显**最近接受的 STM32 tick**，不发送 Linux uptime 或 wall clock；固件按无符号差检查 `now_ms - timestamp_ms <= 100`。主机同时独立检查目标原始 `now`，不会因重新打包、ACK 或心跳而更新目标年龄。

固件接收命令和主机接收遥测各自使用一条全局 uint16 序列；前向差必须满足 `0 < delta < 0x8000`，支持 `65535→0`，拒绝重复、倒序和半空间歧义。主机只接受可信消息推进接收状态。固件的合法空 STOP 是例外：即使序号重复也停机并清除命令序列基准，因此适合主机重启时显式同步。

固件心跳超时为 300 ms、目标超时为 100 ms。后台只发送尚未发过的新目标，不循环重发缓存目标。目标看门狗基于主循环提供的原始 monotonic `now`；再次提交相同或更早的 `now` 被拒绝，读反馈和查询投递完成均不续期。握手过程中也使用同一目标有效期。

## v1 遥测布局

`JOINT_STATUS=0x81`：290 字节，开头 `polling_mask:u16`，随后 12 个 24 字节记录，记录内偏移如下：

| 偏移 | 字段 |
|---|---|
| 0 | flags:u8；bit0 raw_valid，bit1 calibrated feedback_valid，bit2 velocity_valid |
| 1 | bus:u8，当前 0..3 |
| 2 | raw_position:**int16** |
| 4 | acquisition_timestamp_ms:u32 |
| 8 | sample_count:u32 |
| 12 | position_rad:float |
| 16 | velocity_rad_s:float |
| 20 | error_count:u32 |

只有轮询位有效且 flags 三位全部有效时，才可产生有效 JointState stamp。没有速度有效性时不以“速度零”伪装完整有效反馈。上位机不从目标反推测量值。

`IMU_STATUS=0x82`：41 字节：`valid:u8`（0），`sample_timestamp_ms:u32`（1），`sample_count:u32`（5），`accel_m_s2:float[3]`（9），`gyro_rad_s:float[3]`（21），`roll_rad:float`（33），`pitch_rad:float`（37）。括号是字节偏移。数据已在机体坐标系：+X 前、+Y 左、+Z 上，不再次应用传感器安装变换。

`omega=gyro_rad_s`；投影向下重力由固件使用的 `R_y(pitch) R_x(roll)` 约定计算：

```text
gravity = [sin(pitch), -sin(roll)*cos(pitch), -cos(roll)*cos(pitch)]
```

不直接归一化含运动加速度的 accelerometer 作为重力。非有限数、越界 roll/pitch、无效标志不会成为有效 IMU 样本。

`HEIGHT_SCAN=0x83`：46 字节，当前仅消费帧，不输出地形。

`FAULT_STATUS=0x84`：**76 字节**，对应 C 的 `28 + 12 * SERVO_BUS_COUNT`，当前 `SERVO_BUS_COUNT=4`：

- 0：faults:u32；4：armed_mask:u16；6：configured_mask:u16。
- 8/12/16/20/24：rx_overflow / uart_errors / parser.rejected / session.rejected / tx_dropped，均 u32。
- 28 开始每总线 12 字节：successful / timeouts / transport_errors，均 u32。

fault 位：`0x01 STOP`、`0x02 HEARTBEAT`、`0x04 TARGET_TIMEOUT`、`0x08 FEEDBACK`、`0x10 TARGET_REJECTED`、`0x20 RX`。初始 STOP 产生的单独 bit0 在首次 ARM 前是预期状态；其它位立即闭锁。ARM ACK 必须 faults=0；进入运行后任何 fault 都闭锁，不会自动清除。运行状态的 armed_mask 必须为 `0x0fff`。

## 采集时间、串口并发与停止语义

启动时匹配初始 STOP ACK 的主机发送时间为 `t_send`，ACK 的 STM32 tick 为 `tick0`。将 `tick0` 映射为 `t_send - 1 ms`；ACK 是 STOP 发送之后形成，因此把整个往返等待计入保守年龄，避免直接使用接收时间把旧样本变新。后续展开 uint32 tick 的前向增量（支持回绕），并从帧头 tick 减去每个传感器的采集年龄。

同一传感器只有采集 tick 和 sample_count 都前进时才更新状态；重复、倒序或只换遥测帧序号不刷新采集 stamp。无效标志立即使 stamp=0；已经失效的同一旧样本不能仅靠重新设置 valid 恢复有效。接收积压中已经过旧的遥测也不会刷新链路存活时间。

初始同步不是专用时钟同步协议，没有硬件时钟漂移估计。映射超前于主机时只向更老方向修正基准，绝不按新接收时间抬高旧样本。长时间运行的晶振漂移、排队和 ms 量化仍须实测；保守映射可能提前触发超时。无单独 session nonce，CRC 也不提供身份认证。检测到前向序号下的 STM32 tick 倒退时闭锁，不把重启自动当作恢复。

唯一后台线程拥有 read/write/close，另有 I/O 互斥；public 方法只读写复制的状态。正常接收按 `min(in_waiting, 1024)` 取已有数据，空缓冲不调用阻塞读取，不会为一个 22 字节 ACK 等待 1024 字节。状态锁在发送期间仍持有，主循环方法可能等待一次整帧写入预算；分段写会调整 pyserial write_timeout，共享同一总期限，截止时间同时受目标剩余有效期限制。Linux 调度和驱动不是硬实时保证。

初始/退出 STOP 与运动报文使用同一线程串行发送。关闭和故障后不再发送新目标或续心跳；最终 STOP 是尽力写入，**不等待停止 ACK，也不把写入成功当作物理停稳确认**。发送失败、串口断开或线程无法退出会报告故障。当前固件 `BusServoInhibitMotion(true)` 禁止新的运动写入并丢弃未发送目标；已经发出的动作仍可能完成，且不改变力矩。紧急停止机构和电气保护需由实际系统提供。

## 115200 与超时核验

115200/8N1 每字节约 86.8 μs。304 字节关节帧约 26.4 ms，55 字节 IMU 帧约 4.8 ms，22 字节 ACK 约 1.9 ms。`lower_link.c` 的 TX 队列深度为 4，遥测最多占用 3 槽后为 ACK 留槽，**留槽不等于 ACK 抢占排队**。因此默认 80 ms ACK/ARM 超时是待实测的配置，不是已证明的最坏时延上界；超时会安全闭锁，不能通过无限重试掩盖。

当前源码常规发送周期为关节/IMU 50 ms、height 100 ms、fault 200 ms；复核时还出现了 `TOF_RAW` 273 字节载荷、200 ms 周期。若这些都按周期发送，叠加 50 Hz TARGET ACK 与 10 Hz HEARTBEAT ACK，下行理论占用约 109,850 bit/s，已接近 115200 baud 的容量；实际会因队列限制及本端单条待确认命令而降频。50 Hz 调 `command()` 不保证线上每秒发送 50 个目标：邮箱合并未发送目标，只发送最新值。

应实测 ACK 分位数/最大延迟、采集年龄、TX dropped 和反馈有效率；有排队问题时须协调固件遥测调度或两端波特率，不应只放宽目标安全期限。本任务没有启用 UART6、修改波特率或进行真实串口时延测量。fake serial 的延迟注入测试只能验证时限逻辑，不能证明 USB/UART 硬件性能。

## delivery-v1：显式、未配套的应用扩展

默认 `delivery_extension_version=0`：`delivery()` 与 `delivery_ack()` 都抛 `UnsupportedFeature`，不发送扩展报文。设为 1 只启用 Python 对端；**当前 `RobotSession_Handle` 会以 FORMAT 拒绝此请求，本模块随后闭锁**。必须另外实现并验证下位机配套后才能启用。

为避免改变基础 v1 CRC/头格式，扩展沿用 RobotLink version=1 的帧，另设 payload 中的扩展版本：

| type | 名称 | payload |
|---|---|---|
| `0x10` | DELIVERY_REQUEST | `extension_version:u8=1, request_id:u32`，5 字节 |
| `0x90` | DELIVERY_ACK | `extension_version:u8=1, request_id:u32, status:u8`，6 字节 |

这些 type 是此扩展的提议值，**不在现有 C 消息枚举中**；没有自动协商。配套固件需先返回普通 `0x85 ACK` 确认请求接收，再用 `0x90` 报告状态：0 接收/执行中、1 确认完成、2 失败。普通 ACK 不是完成；不匹配 ID、未发送请求的通知不能构成成功。只接收请求和完成通知都匹配时 `delivery_ack(id)` 才返回 true。

仅已 ARM 且主循环目标仍有效时允许请求。投递接口不续目标期限，不后台重试投递。每次仅一个未完成请求；请求从排队开始计时，超时或失败闭锁并报告“物理结果未知”，不能猜测已投出。一次桥生命周期保存最多 256 个 ID，重复 ID 不再次执行，不驱逐旧 ID 后重用。跨主机/固件重启的幂等性需要配套固件及任务层持久化保证，当前 Python 缓存不提供该保证。

## 测试与验收

运行：

```powershell
python -m pytest tests/test_transport.py -q
```

测试使用 fake serial，不打开物理串口。独立的逐位 CRC/编码参考函数逐字节对照 C 算法，覆盖最大载荷、端序和已知 CRC 向量。其余覆盖：

- 分片、粘包、噪声、错误版本/CRC/长度、半帧超时及限长。
- 构造无 I/O、bridge 子配置兼容、幂等 start/close、IMU facade、数组隔离和串口互斥。
- STOP 同步、ACK 类型/序号关联、首次 ARM 顺序、目标映射、shadow 禁止运动报文。
- 延迟 ACK、无回应/ACK 超时、负 ACK、断线/短写/零进度写、关闭失败暴露。
- 目标时间过期/未来/重复、非有限目标、反馈失效、主循环存活但传感器冻结、主循环停止但遥测仍活跃。
- uint16 序号和 uint32 tick 回绕、重复/乱序丢弃、重启检测、采集年龄不刷新、IMU 重力转换。
- 远端 fault 与本地 estop 闭锁；投递默认拒绝、现有固件拒绝扩展、请求幂等、接收与完成区分、未知 ID 和完成超时。

Windows Python 3.11 的 `monotonic()` 可能只有约 15.6 ms 分辨率；测试在该模块范围内以同样单调的 `perf_counter` 提供细粒度时钟，以检查 50 Hz 调用，不改生产代码的 Linux monotonic 时间域。

硬件验收还需实际启用并绑定独立主机 UART、确认 115200/8N1 和地线/电平、完成关节映射/标定及 IMU 驱动。先运行只读 probe/shadow 核验时间戳、valid/fault 与 ACK，再在机械约束和停机措施下验证首次 ARM、目标超时及停止行为；投递另须配套固件和机构的完成反馈测试。本任务的单元测试通过不表示上述硬件步骤已经完成。
