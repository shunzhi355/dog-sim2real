# 比赛任务 FSM（robot-deploy / 香橙派 5 Max）

`src/dog3588/mission.py` 实现一轮配送的确定性、非阻塞状态机。USB 相机和
ToF 都由香橙派采集；FSM 只接收快照，不操作设备、不读取时钟、不 sleep，
不发送串口或 UDP、不执行策略推理。相同配置、mission_id 和输入序列产生相同输出。
每次 `step` 最多切换一个阶段，阶段切换当拍输出零速度。

依据项目 AI 说明：快递为黄/棕色；蓝色是高台预告，不能证明通过；红绿色带
用于分岔提示，规则没有提供通用左右映射；越障后必须稳定至少 3 秒；快递须
完全越过住户门线；返程需实际接触集散中心边界。实现和测试不代表实机验收。

## Runtime 接口

```python
from dog3588.mission import MissionConfig, MissionFSM, MissionInput

mission = MissionFSM(MissionConfig.from_mapping(cfg.get("mission", {})),
                     mission_id="round-unique-id")
# 也可 MissionFSM(cfg["mission"], mission_id=...)。
out = mission.step(MissionInput(
    now=now, start=start_requested,
    vision_stamp=frame_acquired_monotonic,
    tof_stamp=tof_acquired_monotonic, tof_valid=tof_valid,
    line=line_result,                         # LineDetector.detect() 原结果
    colours=tuple(item["colour"] for item in colour_detections),
    parcel_colour=recognised_parcel_colour,    # 来自快递 ROI，不是道路色带
    events=perception_events,
    stable=geometrically_stable,
))
velocity = out.velocity
gait = out.gait if out.gait is not None else configured_gait
```

`MissionInput` 为 frozen dataclass；`now` 必填，其余默认值如下。JSONL 数组可
直接用于 `events`、`colours`，即 `MissionInput(**record)`。调用方须将帧内容与
采集时间戳一起交付，不能通过更新旧帧时间戳让数据重新有效。

| 字段 | 类型 / 默认 | 语义 |
|---|---|---|
| now | float，必填 | 主机单调时钟秒，有限且非负 |
| start | bool / False | 请求启动一轮；一次 true 后已锁存 |
| vision_stamp | float 或 None / None | 当前视觉/融合事件快照的采样时间 |
| tof_stamp | float 或 None / None | 上位机 ToF 有效采样时间 |
| tof_valid | bool / False | 标定、质量筛选后的 ToF 健康状态 |
| line | Mapping / 空字典 | found、error、heading、confidence，与 vision.py 一致 |
| colours | 名称序列 / 空 | blue、red、green、yellow、brown；仅道路语义颜色 |
| parcel_colour | str 或 None / None | yellow 或 brown；独立的快递颜色识别结果 |
| events | 名称集合/序列 / 空 | 下表中的显式感知事件 |
| stable | bool / False | 本帧融合确认稳定、未摔落且满足足端/赛道约束 |
| delivery_ack_id | str 或 None / None | 下位机响应对应的 request_id |
| delivery_ack_status | str 或 None / None | accepted、completed、failed、unsupported |
| delivery_ack_stamp | float 或 None / None | ACK 在主机接收时的单调时钟时间 |
| fault | str 或 None / None | 安全保护/摔倒/掉件/通信等外部故障；立即锁存 |

`MissionOutput` 包含：

| 字段 | 含义 |
|---|---|
| velocity | `(vx, vy, wz)`，米/秒、米/秒、弧度/秒，X 前/Y 左/正 yaw 左转 |
| gait | 可选 7 维 tuple；None 表示保留 runtime 的已验证步态 |
| state | `MissionState`（str Enum），使用 `.value` 序列化 |
| mode | 正常为 state.value；临时感知停车时为 `stop`，state 保留原阶段 |
| delivery_request | `DeliveryRequest(request_id, colour)` 或 None，仅输出一次 |
| reason | 故障或临时停车原因字符串，正常为 None |

`done`、`fault` 都是锁存终态，始终输出零速度，不自动开始下一轮。没有隐式
reset；只有创建新 FSM 并显式启动才开始新一轮。`obstacles_completed` 仅在
几何确认且稳定完成后增加；`delivery_completed` 仅在 ACK 和越门线确认均满足
后变为 True。这些是 FSM 接收证据后的记录，不是自行计算的官方比赛评分。

### 感知事件契约

全部 events 与 `vision_stamp` 绑定。这里的视觉快照可以包含相机、ToF、里程计、
IMU、接触信号融合后的判定，必须保证其依赖数据有效。FSM 不从色块面积猜测
这些几何事件，也不将 ToF 原始距离当成高度特征。

| 事件 | 接收阶段 | 必须由上游确认的事实 |
|---|---|---|
| obstacle_near | approach | 到达已识别障碍的越障入口/控制切换位置 |
| obstacle_crossed | cross | 整机完成障碍通过，并处于有效落地区域；仅蓝色消失不算 |
| turn_complete | turn | 已进入显式 route map 所选出口且方向正确 |
| docked | dock | 目标住户匹配、停靠位置/朝向满足要求，机器人已停止 |
| delivery_confirmed | delivery | 快递已完全越过该住户门线，而非舵机动作结束 |
| return_aligned | return_turn | 投递后掉头完成、朝向返程赛道 |
| home_visible | return | 识别到集散中心，允许进入接近阶段 |
| home_confirmed | home | 投递后的机器人至少一足实际接触/踏入集散中心边界 |

默认需要连续 2 个**不同的新帧**确认事件/颜色，连续 2 帧消失才重新武装。
同一视觉时间戳的重复 step 不累计次数。已持续置位的事件不会在下一阶段或
下一个障碍被重复使用；上游应在对应阶段生成事件，完成后清除。进入阶段前
的缓存事件不能作为阶段完成证据。`home_visible` 和 `home_confirmed` 应按阶段
分别提供，不能在出发时持续携带一个 `home_confirmed` 到返程。

红/绿同时出现是一个 `red+green` 路口签名，不按列表顺序选色。签名本身也需
去抖。红、绿互相闪烁或持续出现时属于同一个路口，必须确认整体离开路口后
才能重触发。蓝色与其他道路标记同时出现、黄棕住户标记同时出现会安全故障，
视觉上游应使用合适的道路 ROI，排除背筐内快递的颜色。

### 投递 ACK 与 bridge

停靠确认后生成 `mission_id + ":delivery:1"`，当拍唯一一次输出请求。
runtime 负责将其可靠排队给 bridge；需要重传时使用**同一个 ID**，不能再创建
新物理投递动作。mission_id 由调用方保证跨实际轮次唯一。不能在仍可能存在
未完成请求时用新 mission_id 重建 FSM。内存中的 FSM 不提供掉电事务恢复。

只有请求 ID 匹配、ACK 时间戳晚于请求发出时刻、未过期且严格晚于此前有效 ACK
的响应才生效。其他请求、旧 ACK、重复 ACK 均忽略。`accepted` 只表示接受；
`completed` 表示机构完成，但还必须收到 `delivery_confirmed` 才能进入返程。
两项证据可先后任意到达。感知临时停车期间仍接收 ACK，恢复前不允许移动。

相邻 STM32 当前没有 delivery 消息。默认 `bridge.delivery` 抛出
`UnsupportedFeature`，runtime 将其转换为安全故障并停止；独立 FSM 的输入接口也接受
匹配 ID 的 `delivery_ack_status="unsupported"`，锁存 `delivery_unsupported`。`failed` 同样故障。
如果接口静默无应答，FSM 最终 `delivery_timeout`。**不能用等待几秒、接口调用
返回、仿真完成、发送成功或 accepted 代替真实 completed/越门线证据。**

### 命令与策略连接

`vision.py` 的 `line_velocity()` 被复用于巡线，图像右偏对应负 yaw。
当前 `commands.py` 的旧 UDP 消息只包含速度/急停，source 只接受 manual/vision；
它不承载 FSM 感知事件、步态或投递请求。runtime 应直接消费本接口或另行增加
结构化通道；不能直接发送 `source="mission"` 并假定旧接收器支持。

策略推理之前，runtime 仍须验证输出速度/7 维 gait 是否处于 manifest 训练范围，
执行现有 SafetyGuard、关节反馈/IMU/控制循环超时检查，以及真机 arm 控制。
FSM 的零速度表示站立/停止行走命令，不替代底层执行器安全保护。`mode="fault"`
应触发现有安全停机流程。相机不能同时绕过 FSM 持续发送另一套运动命令。

## 状态转换与超时

```text
start --显式启动+配置/去抖快递颜色--> follow
follow/return --蓝标--> approach --入口几何--> cross
cross --通过几何--> stabilize --连续稳定 >=3s--> 原 follow/return
follow/return --红绿签名+显式路由--> turn
turn --出口方向确认+新帧重捕获--> 原 follow/return
follow --匹配住户颜色--> dock --停靠确认--> delivery
delivery --匹配 completed ACK + 越门线确认--> return_turn
return_turn --返程朝向确认+新帧重捕获--> return
return --home_visible--> home --home_confirmed--> done
任意活动阶段 --外部故障/超时/无路由--> fault
```

普通巡线/接近/越障/停靠/返程/接近家时，丢线或低置信度立即停车，必须在
配置超时内收到指定数量新鲜有效帧才继续。在 turn/return_turn 中允许暂时看不到
线并执行有时限的原地转向；收到几何转向确认后立即停止转动，再累计新鲜巡线帧
完成视觉重捕获。直行路由丢线时同样不会前进，转向超时仍有效。

所有已启动阶段均检查视觉和 ToF 有效性、采样年龄、未来/乱序时间戳；缺失
立即停车，持续不恢复则故障。恢复窗口不因故障类型交替而重置。阶段和路径
超时按墙钟经过的单调时间计，停车期间不暂停，不会无限等待。

稳定计时只统计新帧中的连续稳定证据。任一不稳定帧、传感器停车、相邻稳定
采样间隔超过 max_vision_age_s 都重置计时；重复缓存帧或跳跃时钟不能凑满 3 秒。
运行方必须持续调用 step，即使传感器断开也不能停调 FSM。now 允许重复，不允许
倒退；CLI 回放可进一步要求 now 严格递增。超时边界 `elapsed >= timeout` 优先
判失败，不接受在期限边界才到达的成功事件。

## 配置

未知配置键拒绝加载。没有隐含左右映射；遇到未配置的实际颜色签名，返回
`unmapped_route:<signature>` 故障。返程路由独立配置，不推测反向行驶的左右。
这个版本按“快递颜色 + 路口颜色签名”匹配；同一轮多个相同签名但要求不同转向
的路口需要上游地图定位/扩展路口 ID，不能从当前标记自动猜测。

```yaml
mission:
  parcel_colour: null             # yellow / brown，null 时等待快递 ROI 识别
  route_map: {}                   # 出程，实测地图确认后填写
  return_route_map: {}            # 返程，实测地图确认后填写
  forward_m_s: 0.12
  approach_m_s: 0.06
  cross_m_s: 0.06
  dock_m_s: 0.03
  turn_rad_s: 0.3
  return_turn_direction: left     # 投递后的掉头方向，不代表红绿路由
  lateral_gain: 0.6
  heading_gain: 0.8
  max_yaw_rad_s: 0.4
  min_line_confidence: 0.2
  max_vision_age_s: 0.25
  max_tof_age_s: 0.25
  max_ack_age_s: 1.0
  sensor_fault_timeout_s: 2.0
  lost_line_timeout_s: 2.0
  startup_timeout_s: 15.0
  path_timeout_s: 120.0           # 出程、返程各一个总时限；delivery 独立限时
  turn_timeout_s: 12.0
  approach_timeout_s: 10.0
  cross_timeout_s: 20.0
  stabilize_timeout_s: 15.0
  stable_duration_s: 3.0          # 不能低于比赛要求的 3 秒
  dock_timeout_s: 15.0
  delivery_timeout_s: 20.0
  home_timeout_s: 15.0
  debounce_frames: 2
  release_frames: 2
  reacquire_frames: 3
  gait: null                     # 可选已验证的 7 维 gait
  obstacle_gait: null             # approach/cross/stabilize 使用；空则沿用 gait
```

路由配置示例仅用于合成回放，**不是已确认比赛地图**：

```python
{"yellow": {"red": "left", "green": "right", "red+green": "left"},
 "brown": {"red": "right", "green": "left", "red+green": "right"}}
```

## 完整离线回放

以下脚本生成含显式假设感知事件的 JSONL，并同时检查完整 FSM。它不操作物理
投递机构，也不证明真实摄像头已经能产生这些事件。时间戳每帧推进 0.1 秒；没有
sleep。CLI 已支持 mission-replay，可将标准输出保存为 JSONL 输入；策略推理
是 runtime/CLI 的可选下一层，不属于 FSM 本身。

```python
import json
from dog3588.mission import MissionFSM, MissionInput

fsm = MissionFSM({"parcel_colour": "yellow",
                  "route_map": {"yellow": {"red+green": "left"}}},
                 mission_id="demo")
now = 0.0

def emit(count=1, **extra):
    global now
    for _ in range(count):
        now = round(now + 0.1, 6)
        row = dict(now=now, vision_stamp=now, tof_stamp=now, tof_valid=True,
                   line=dict(found=True, error=0.0, heading=0.0, confidence=1.0))
        row.update(extra)
        print(json.dumps(row))
        out = fsm.step(MissionInput(**row))
        assert out.state.value != "fault", out.reason
    return out

emit(start=True)
emit(2, colours=["blue"])
emit(2, events=["obstacle_near"])
emit(2, events=["obstacle_crossed"])
emit(31, stable=True)
emit(2, colours=["red", "green"])
emit(2, events=["turn_complete"])
emit(2)
emit(2, colours=["yellow"])
out = emit(2, events=["docked"])
request_id = out.delivery_request.request_id
emit(delivery_ack_id=request_id, delivery_ack_status="completed",
     delivery_ack_stamp=round(now + 0.1, 6))
emit(2, events=["delivery_confirmed"])
emit(2, events=["return_aligned"])
emit(2)
emit(2, events=["home_visible"])
out = emit(2, events=["home_confirmed"])
assert out.state.value == "done"
assert fsm.delivery_completed and fsm.obstacles_completed == 1
```

验证命令：`python -m pytest tests/test_mission.py -q`。测试覆盖黄/棕两轮完整往返、
出返程障碍、路由签名去抖、几何确认、稳定计时、一次性投递、错误/过期 ACK、
unsupported、传感器缺失/乱序/过期、停车重捕获及失败锁存。硬件端尚无 delivery
协议和事件生产器的部分，必须保持 unsupported/等待失败，不能包装成实机完成。
