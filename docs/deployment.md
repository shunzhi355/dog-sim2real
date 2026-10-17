# 部署与开发

## 向香橙派同步

```powershell
.\.venv\Scripts\python.exe scripts/deploy.py --host 192.168.1.102 --user orangepi --host-key-sha256 'SHA256:ljKVjI5v+O/Tfn/aH98EVAiFWW+5vZORwrT2xVsfTSs'
```

密码通过终端提示输入，或 `--password-env` 从调用者环境读取；`--key-auth` 使用 SSH agent/密钥。该指纹是本次部署连接看到的主机密钥，换板/重装系统后应通过可信渠道重新确认。传入指纹时精确比对，未传时读取系统已知主机；不修改你的 `known_hosts`。

当前 Windows 的 `known_hosts` 为该 IP 保留了另一个公钥，所以普通 SSH 可能提示密钥变更。部署脚本使用上面指定的本次目标公钥。若需修复普通 SSH 登录，先在板端核对 SSH 主机公钥指纹，再更新这一条记录；本次没有删除你的已知主机记录。

同步目标为 `/home/orangepi/dog-3588`，只上传项目源码、默认配置、模型、文档、测试、部署脚本，不同步 `.venv`、日志、凭据或 `config/local*.yaml`。不会删除目标上额外文件；默认配置可被新版本覆盖，所以实际标定请放在 `local.yaml`。

`--sync-only` 仅同步；默认再执行 `deploy/setup.sh` 创建隔离环境、安装依赖、运行测试/测速和 5 秒 mock。没有主动控制服务、重启、boot 修改或串口运动。

## systemd

提供 `deploy/dog3588-shadow.service` 和 `dog3588-vision.service`，本次没有安装或启用。硬件/标定完成后，可检查路径并手动安装模板。控制模板只读且 `Restart=no`；不要直接把 `--arm` 加进自动重启服务，初次实机使用前台有限时长命令。

## 命令协议

本机 UDP，默认端口 9871。JSON 形状：

```json
{"version":1,"source":"vision","sent_monotonic":12345.67,"velocity":[0.12,0,0.1],"estop":false,"message_id":"diagnostic-id"}
```

`sent_monotonic` 必须取同一台机器的 `time.monotonic()`，允许来源 `manual/vision`；配置决定哪一个来源能驱动速度。初版不设计多来源抢占或远程直接 UDP。超过 0.3 秒、未来时间、倒序时间、非有限数字/维度错误的命令不生效。零速度会让策略进入训练的停车条件；与停止串口输出不同。`estop=true` 从任一已验证消息来源锁存，控制进程退出，通过停止帧中止当前运动；需要重新运行才能清除。

可使用 `dog3588.commands.send_command()` 集成任务程序，每 50–100 ms 刷新一次即可，不要从感知模块直接调用舵机驱动。

## 日志与故障

`run --log logs/session.jsonl` 每五个控制周期记录一次：单调时间、角度、动作、目标、速度指令、IMU/关节数据年龄与推理耗时。终端每秒一条状态，结束打印周期与频率统计。生产使用时由外部日志轮转管理容量；本版本没有无限制常驻日志管理服务。

| 现象 | 优先检查 |
|---|---|
| `calibration is not verified` | 硬件标定未完成，不可只为绕过错误而置 true |
| `cannot open camera` | 型号驱动、实际 capture 节点、video 组权限 |
| `position response timeout` | 供电/共地、ID、门控方向、引脚复用、丢失应答首字节 |
| `joint feedback missing/stale` | 任一电机未更新，读取轮询/错误率，不能改成反馈=目标 |
| `IMU missing/stale` | WIT 两类帧是否都有，波特率/校验/轴变换 |
| `body tilt ... exceeds limit` | 初始姿态或 IMU 坐标方向不正确 |
| `outside calibrated limits` | 标定、关节顺序、动作尺度，不能随意扩大机械限位 |
| `policy target watchdog expired` | 主循环卡顿或被中断，工作线程已尝试发停止帧 |
| `.onnx` 换文件后 checksum 不符 | 用导入器更新 manifest，不要手工忽略哈希 |

进程退出只发送 MOVE_STOP，不发送卸载；避免站立时突然失去支撑。串口断线、用户态死锁、内核挂起或断电都可能使停止帧无法送达。LX824 会完成最后一个有限时长目标并可能继续保持力矩，软件看门狗不等于硬件急停。

## 扩展接口

新增关节总线后端实现 `JointIO`：`start/read/command/close`，读出 12 个电机位置、速度、采集时间。控制代码不依赖其是直接 UART 还是将来的 STM32 桥。

新增 IMU 驱动实现 `ImuIO`：统一输出机身坐标系角速度与单位重力向量、单调采样时间。不要把原始加速度直接当成动态姿态，不能用旧姿态配新时间戳掩盖失联。

新增视觉/任务程序只依赖速度命令接口。更复杂的颜色路线、障碍决策和任务状态机应放在这里，再选择已经训练、验证的策略与步态。

若换模型结构、历史布局、关节数或循环网络，不是更换模型文件即可运行；更新 manifest schema、观测适配器、输入输出校验和回归测试。

导入器先在临时目录完成 ONNX 解析、接口契约验证和推理加载，再替换工作模型；导入前应停止运行中的控制程序，当前不提供模型热切换。

