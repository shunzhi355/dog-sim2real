# Robot12 策略数据契约

> 部署适配器现支持 280 和 320 维两种显式布局。下文的模型数值是当前随仓库携带的 280 维基线。320 维在原本体项后追加 `height_scan`（单帧 8 维、5 帧历史，切片 `[280:320]`），必须附带 `height_scan_contract` 且 ONNX 签名匹配。主机只在八点全部有效时推理，不把缺失数据补零；288 维布局拒绝。地形采样参考、导入方法见 [上位机部署指南](robot-deploy.md#训练契约) 与 [ToF 文档](tof.md)。

配置源：`policies/robot12/training_env.yaml` 和 `training_agent.yaml`，部署机器可直接检查。`manifest.json` 是提炼后的运行契约。

## 张量与时间

- 控制周期 0.02 s（仿真 `dt=0.002`，`decimation=10`）。
- ONNX 输入 `obs`：float32 `[1,280]`；输出 `actions`：float32 `[1,12]`。
- actor 没有机体线速度真值、足端接触真值或地形扫描；这些不能在实物中伪造为观测输入。
- 每项观测独立存 5 帧，按最旧到最新排列，随后拼接各项。不是把整个 56 维单帧连续堆五次。
- 首次采样将同一帧填满该项历史，复位同时清零上一动作、相位。
- 不添加训练用观测噪声，不在 ONNX 外重复加归一化。当前保存的 actor 设置为不使用 observation normalization。

| 观测项 | 单帧维度 | 五帧切片，右端不包含 | 缩放 |
|---|---:|---|---:|
| 身体角速度 | 3 | `[0:15]` | 1 |
| 投影重力 | 3 | `[15:30]` | 1 |
| 速度命令 vx/vy/wz | 3 | `[30:45]` | 1 |
| 电机角度减默认角度 | 12 | `[45:105]` | 1 |
| 电机角速度减默认角速度 | 12 | `[105:165]` | 0.05 |
| 上一策略原始动作 | 12 | `[165:225]` | 1 |
| 步态参数 | 7 | `[225:260]` | 1 |
| 四足相位时钟 | 4 | `[260:280]` | 1 |

所有算法坐标为右手系：X 向前、Y 向左、Z 向上。角度 rad，角速度 rad/s，速度 m/s。平放重力 `[0,0,-1]`，不乘 9.81。

## 电机顺序

`fl_hip_joint, fl_upper_joint, fl_input_joint, fr_hip_joint, fr_upper_joint, fr_input_joint, rl_hip_joint, rl_upper_joint, rl_input_joint, rr_hip_joint, rr_upper_joint, rr_input_joint`。

顺序是左前、右前、左后、右后，每腿为髋侧摆、上部驱动、连杆输入驱动。STM32 工程中的 `Knee` 命名并不意味着它等于模型被动膝关节。必须根据连杆驱动轴验证对应关系和方向。

当前默认电机位置/速度都是零。目标公式：

```text
q_desired = q_default + 0.25 * action
raw = raw_zero + direction * (q_desired - zero_rad) * units_per_rad
```

`raw_zero` 是电机在指定 `zero_rad` 时的实测原始计数；`zero_rad=0` 应对应训练模型机械零位。不能把 `raw_zero=500` 当作所有关节的事实。

LX824 的标称比例 `1000 / radians(240) ≈ 238.7324` 仅是编码换算，不替代装配标定。软件先检查全体目标与机械限位，再做变化率限制；运行时上一动作观测仍保留 actor 原始输出，以符合训练的 `last_action`。限速会改变实际执行器响应，应在后续训练中匹配。

## 条件步态

步态向量 `[frequency_hz, stance_fraction, clearance_m, FL_offset, FR_offset, RL_offset, RR_offset]`。

保存的训练包范围：步频 1–2 Hz，支撑比例 0.55–0.60，抬脚净空 0.015–0.025 m，相位偏移 `[0,0.5,0.5,0]`。配置默认取区间中值。

```text
moving = vx != 0 or vy != 0 or abs(wz) > 0.02
phase_next = (phase + 0.02 * frequency_hz * moving) mod 1
raw_phase = (phase + foot_offset) mod 1
foot_phase = 0.5 * raw_phase / duty                       (raw_phase < duty)
             0.5 + 0.5 * (raw_phase-duty)/(1-duty)        (其余)
clock = sin(2*pi*foot_phase) * moving
```

停车时相位冻结，时钟输出为零；重新运动不强制重置相位。训练复位随机相位，部署初次设为零，这是训练相位区间内的一个起点。

## 需要实测的差异

训练中采用 `MotorPDActuator`：刚度 20、阻尼 0.1，并用近似力矩/速度包络施加仿真力矩。实物接口给 LX824 位置和时间，由舵机内部位置控制器闭环，不能直接写入同样的 Kp/Kd 或任意力矩。必须实测位置阶跃/斜坡跟踪、死区、延迟、反向间隙和带载速度，调整训练执行器模型或域随机化。

从位置回读差分得到角速度，初版低通时间常数 20 ms；LX824 不提供这里所需的直接电机速度观测。回读频率、滤波和噪声都会影响历史输入，需通过日志与训练分布对照。

目前完成了静态契约核对、独立历史单元测试与 ONNX 包装前后输出一致性验证；尚未在用户运行中的 Isaac Lab 环境采集同一轨迹，逐帧比对完整观测。后续上机前宜导出一段无噪声观测/动作作为 golden trace。

参考：[Isaac Lab ObservationManager](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/managers/observation_manager.py)、[CircularBuffer](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/utils/buffers/circular_buffer.py)。其实现定义逐项历史拼接、最旧到最新排列、首次采样填满历史。

