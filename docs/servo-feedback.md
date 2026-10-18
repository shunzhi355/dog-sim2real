# 香橙派 LX824 非阻塞位置反馈

实现位于 `src/dog3588/hardware/feedback.py`；命令为 `dog3588 feedback-servos`。
只发送 `POS_READ=0x1C` 单播位置查询，不发 LOAD、MOVE、STOP，不改标定和舵机 ID。
当前使用四路独立 UART 和外部自动方向电路，无方向 GPIO；不是把 Linux 串口
配置成单线半双工，也不要求重新接 STM32 的使能脚。

## 与 STM32 的对应关系

| STM32 环节 | 香橙派实现 |
|---|---|
| `BusServoTask()` 周期推进 | 主循环逐个调用各总线 `FeedbackPoller.poll(now)` |
| 每路一次一个单播事务 | 每路独立 `current_id` 和游标，按配置 ID 顺序轮询 |
| `TRANSMITTING` / TC 回调 | 非阻塞写入并检查内核输出队列，加保守帧线时长等待；并非硬件 TC |
| RXNE 中断、环形缓冲 | Linux UART 驱动接收缓冲，应用每次最多读 4096 字节 |
| 分片解析、ID/指令/校验检查 | 复用 `FrameParser`；只有当前 ID 的 `0x1C` 双字节应答成功 |
| 超时、清理、静默 | 发送超时停止该监控；接收超时使该关节无效，静默后查询下一个 |
| `g_servo_feedback_snapshot` | `snapshot()` 内存快照、终端每秒一条 JSON，事件另写 JSONL |

状态为 `IDLE → TRANSMITTING → RECEIVING → QUIET → IDLE`。诊断命令没有
位置写入队列，所以不需要 STM32 的 PENDING 等待写队列；没有 HAL 异步取消，
也不照搬 ABORTING 回调。发生发送超时或设备异常时关闭串口并退出，防止串口
尚未释放就开始下一次查询。成功和接收超时后均遵守静默间隔。

发送阶段仍持续收集输入：可能很快到达、先被 Linux 缓冲的应答，不会因应用
从 TRANSMITTING 切换到 RECEIVING 而被清空。请求回显、其他 ID、错误命令、
错误长度、校验失败不会更新位置。空闲期间的迟到输入被丢弃；但协议没有事务
序号，无法绝对区分后续相同 ID 查询期间到达的旧应答，静默不是数学上的保证。

## 配置和频率

```yaml
feedback:
  query_period_s: 0.006
  tx_timeout_s: 0.010
  rx_timeout_s: 0.010
  quiet_s: 0.002
  max_age_s: 0.100
```

6 ms 是相邻事务起点的最小间隔，不是必须周期；成功时理想三个舵机一轮约
18 ms，超时和 Linux 调度会拉长。程序不会在某个 ID 的接收窗口中查询另一个 ID。
参数仅用于新反馈命令，不会悄悄改变现有策略运行器或旧 `probe-servos` 的时序。

外部转接电路须自主释放 DATA 并接收回传。Linux `out_waiting==0` 只表示内核
输出队列空，不证明硬件移位寄存器的最后停止位已完成；本实现再等待六字节
8N1 帧的估计线时长（约 0.521 ms）。这仍是应用层估计，不是硬实时 TC 回调，
也不是物理 TX 波形证据。使用无 GPIO 自动方向电路时，方向切换由硬件完成；
逻辑分析仪验证仍不可省略。应用观测时间也不是硬件逐字节到达时间。

## 快照与日志

每路包含 `requested`、`successful`、`rx_timeouts`、`tx_timeouts`、
`transport_errors`、`rejected_frames`、`bad_frames`、`tx_bytes`、`rx_bytes`、
`phase`、`current_id`、`last_query_hex`、`last_error`。
`tx_bytes` 是被串口写入 API 接受的字节数，不表示已实测引脚输出。

每个关节保留：

| 字段 | 含义 |
|---|---|
| `raw_position` | 实际应答的有符号 16 位数，不截断成 0..1000 |
| `timestamp_s` | 应用观察到完整有效帧的单调时钟时间，不是舵机测量时刻 |
| `interval_ms` / `age_ms` | 实际有效采样间隔 / 最近样本年龄；从未收到为 null |
| `sample_count` / `error_count` | 成功和事务失败累计次数 |
| `raw_valid` | 最近事务成功且年龄不超过 100 ms |
| `feedback_valid` / `position_rad` | 标定已验证且原始反馈有效 / 换算关节角 |
| `velocity_valid` / `velocity_rad_s` | 按实际采样间隔计算；首次、失败后首帧、长间隔时无效 |

失败和过期不会覆盖旧位置或伪造新时间戳。使用者必须检查有效标志。
没有标定时角度、速度均为 null/无效。标定后换算使用当前项目 `Calibration`
的 `zero_rad + direction * (raw - raw_zero) / units_per_rad` 契约，不盲目复制
STM32 不同名称的零偏字段。速度无额外低通滤波。

终端输出单行 JSON，使用 `--report-s` 设置快照间隔；最后一条为 `event=stopped`。
`--log logs/feedback-fl.jsonl` 记录每次查询的精确报文及成功/失败结果；成功事件
包含校验通过的响应帧。日志单文件最多 2 MiB，保留三个轮转备份。日志不含密码。
默认运行 10 秒；`--seconds 0` 运行到 Ctrl+C/SIGTERM。结束只关闭串口，不发卸载、
停止运动等额外指令，也不改变舵机已有保持状态。

Linux UART 帧错误/溢出不等价于 Python 普通异常，`transport_errors` 只统计
应用遇到的传输异常，不能宣称覆盖 STM32 的所有 SR/FE/ORE 错误诊断。
不能把软件/虚拟串口测试通过当作实物链路验收通过。该模块是独立调试入口，
不与策略运行器争用串口；后续整合应统一总线所有权，而不是同时运行两个进程。

## 2026-10-18 实物验证记录

板端地址 `192.168.0.102`，左前腿 UART1_M1（TX=物理28、RX=物理27），
115200/8N1，自动方向模式。运行新状态机约 10.011 秒，共查询 700 次，
其中 ID1=234 次、ID2=233 次、ID3=233 次；700 次全部接收超时，
成功=0、RX=0 字节、应用写入=4200 字节、发送超时=0、应用传输错误=0。
平均查询起点间隔约 14.305 ms（每次等待接收超时及静默，所以不是 6 ms）。
CLI 正确返回 1，没有伪造原始位置，结束后无进程占用 `/dev/ttyS1`。

事件日志位于板端 `logs/feedback-state-machine-20261018-145845.jsonl`，
快照为同名前缀的 `.snapshots.jsonl`。这份记录说明状态机可运行，但实物回读
仍未验收成功；下一步需测量主机 TX、转接电路 DATA、主机 RX 的实际波形，
并沿原 STM32 已成功的接线逐段对照。不能单凭驱动 TX 计数认定电气输出正确。
