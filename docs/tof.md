# 香橙派 VL53L5CX：64 区采样与 8 点地形高度

## 数据归属与接线迁移

ToF 直接连接 Orange Pi 5 Max 的 I²C；采集、时间戳、滤波、坐标变换和高度采样
均在上位机完成，不通过 STM32 转发。USB 相机也直接连接上位机。
STM32 负责关节执行、关节反馈和 IMU 反馈；上位机用收到的 IMU 姿态补偿 ToF。
当前 RobotLink 主机接口不依赖单片机 ToF 数据。

```text
VL53L5CX ── I²C2 ──> 香橙派：采集 → 有效性检查 → 姿态补偿 → 地形数据
USB 相机 ── USB ───> 香橙派：视觉处理
STM32 ── RobotLink ─> 香橙派：关节状态、IMU
STM32 <─ RobotLink ── 香橙派：关节目标
```

断电后将模块的 SDA/SCL 从 STM32 移到香橙派 **物理 3 脚 SDA、物理 5 脚 SCL**，
共地；模块电源和 LPn 按所用转接板要求连接，不能仅凭芯片名称确定模块供电。
不要让 STM32 和香橙派同时驱动同一组 I²C 线。

2026-10-18 已在 `192.168.0.102` 验证驱动可导入，并配置下次启动加载
`i2c2-m0`，将 `orangepi` 加入 `i2c` 组。重启后应出现 `/dev/i2c-2`。
接线完成后，桌面“Robot12 策略启动”选择 **7**，或在项目根目录运行：

```bash
.venv/bin/python -m dog3588 --config config/local-model4497.yaml probe-tof --seconds 10
```

该诊断直接打开香橙派 I²C，不需要 STM32，也不需要几何标定；尚未实测传感器。
`model4497` 为 280 维模型，不能直接加入高度数据。其 `tof.enabled: false` 只表示
手动运动主循环不启动地形处理，不影响上述独立诊断；完成几何标定及相应策略接入后
再启用主循环地形处理。将 ToF 接到上位机不要求先替换这个模型。

本模块实现项目 AI 说明 375–400 行的距离有效性检查、安装变换、IMU 重力补偿和八点训练观测采样。所有接口定义在 `dog3588.tof`，复用 `dog3588.types.ImuState`。硬件采集在独立线程中运行；投影不访问硬件。

## 集成接口

```python
from dog3588.tof import ThreadedToF, TerrainProjector, ToFFrame, TerrainState

settings = cfg["tof"]
tof = ThreadedToF(settings)
projector = TerrainProjector(settings)

# armed 前必须调用；传实际训练 manifest，不能传自己生成的匹配副本。
projector.require_calibrated(manifest["height_scan_contract"])
tof.start()  # 异步，包括固件上传
try:
    frame = tof.read()
    # imu 应来自与 frame.stamp 最接近的本机单调时间样本。
    terrain = projector.project(frame, imu, now)
    if not terrain.valid.all():
        raise RuntimeError(f"terrain unavailable: {terrain.reason}")
    # 仅将验证通过的 terrain.heights 交给观测 term；无效时上层执行 stop。
finally:
    tof.close()
```

这是接口示意；主循环应在首帧和其他传感器就绪之前保持未 armed，并在采集异常、过期或任意采样点缺失时停止。模块不会直接控制电机，也不会修改主程序的 arming 逻辑。`require_calibrated()` 只检查配置与 contract，不能代替每帧的 `valid.all()`。

| 类型/字段 | 形状与语义 |
| --- | --- |
| `ToFFrame.distance_m` | `(64,)` float，**光轴深度**，米，原始 zone 顺序；不是径向距离 |
| `target_status` | `(64,)`，ST 的目标状态码 |
| `nb_target_detected` | `(64,)`，每区目标数 |
| `stamp` | 本机 `time.monotonic()` 秒，data-ready 检查开始时刻减配置的测量延迟 |
| `TerrainState.heights` | `(8,)` float，leveled body 的原始 Z，米；无效为 NaN |
| `valid` | `(8,)` NumPy bool；必须逐点消费，不允许只检查总覆盖数 |
| `stamp` | 原始 ToF 时间戳，不会随着重复读取变新 |
| `coverage` | `(8,)` int，每个训练采样点 XY 容差内的有效测量点数；不是覆盖面积比例 |
| `xy_error_m` | `(8,)` float，最近有效测量与训练采样点的 XY 距离；没有任何有效测量时为 inf |
| `calibrated` | 配置、传输及标定确认是否齐全；不表示当前帧有效 |
| `reason` | 正常为 `""`，其余如 `uncalibrated/stale/imu_skew/missing_coverage/excessive_rotation` |

`ThreadedToF.read()` 不做 I2C，也不等待新数据；首帧前返回 stamp=0 的无效帧。每次读取拷贝数组，调用方修改不会污染工作线程。线程故障通过 `RuntimeError` 报告；启动/停帧/关闭超时通过 `TimeoutError` 或包装为 worker `RuntimeError` 报告。`start()` 和 `close()` 可重复调用，但关闭后必须新建实例，不能重新 start。`TerrainProjector` 保存时间戳水位，应由一个主循环使用。

## 主配置初稿

以下各字段直接处于 `cfg["tof"]` 同层。`null` 是刻意保留的待标定项，不会自动生成假射线或假采样位置。

```yaml
tof:
  driver: vl53l5cx_ctypes
  i2c_bus: 2
  i2c_address: 0x29
  frequency_hz: 15
  verified: false
  rays_verified: false
  mount_verified: false
  sampling_verified: false
  reference_verified: false
  ray: null                  # 64×3 单位射线，sensor +Z 是光轴
  sensor_to_body: null        # 3×3，p_body = R @ p_sensor + origin
  origin_body_m: null         # 3 元向量，传感器原点在机身中的位置
  sample_xy_m: null           # 8×2，必须来自训练定义并保持顺序
  max_xy_error_m: null        # XY 最近邻容差，实测决定，必须 >0
  height_reference: body_origin
  min_distance_m: 0.02        # 光轴深度的有效范围，非径向范围
  max_distance_m: 4.0
  min_height_m: -2.0          # 原始 leveled Z 合理范围，越界剔除，非 clip
  max_height_m: 1.0
  valid_statuses: [5]
  frame_timeout_s: 0.1
  imu_timeout_s: 0.1
  max_skew_s: 0.03
  frame_duration_s: 0.0666666667  # 未配置时为 1/frequency_hz
  max_rotation_rad: 0.1
  max_tilt_rad: 1.0
  measurement_latency_s: 0.0
  startup_timeout_s: 30.0
  close_timeout_s: 2.0
  poll_interval_s: 0.002
```

`verified` 包括 I2C 读回、帧率和采样时间/延迟验证；其余四个标志分别确认逐区射线、安装外参、训练采样位置及容差、高度参考与正负号。五个标志必须都为布尔 `true`，几何必须完整，才允许 `require_calibrated()` 成功。仅设 `verified: true` 不够。未提供的标志默认为 false；几何缺失可供离线配置，但投影全部无效；提供了错误形状、非有限值、非单位射线、非正旋转矩阵时立即拒绝构造。

`height_scan_contract` 固定为下列约定；`sample_xy_m` 必须填训练得到的八个二维点，下面省略号只是结构示意：

```yaml
height_scan_contract:
  frame: leveled_body
  units: m
  reference: body_origin
  sample_xy_m: [[x0, y0], [x1, y1], ..., [x7, y7]]
  invalid_policy: stop
```

`require_calibrated(contract)` 校验这五个字段，包括采样位置与顺序，绝对容差 `1e-9 m`。集成端应要求 manifest 存在该字段；不传 contract 只检查标定。投影器不减 nominal height，不估计绝对地面高度，不自动归零。`reference_height_m`、`nominal_height_m`、`height_clip_m` 在本配置中会报错，防止旧参数静默改变观测语义。项目说明建议的 `[-0.05, 0.05]` 不能直接裁剪 body-origin Z；如训练有归一化和裁剪，应在匹配的观测 term 中实施，保持 manifest 一致。

## 开源驱动与香橙派安装

选用 [Pimoroni vl53l5cx-python 主仓库](https://github.com/pimoroni/vl53l5cx-python)，发行包 `vl53l5cx-ctypes==0.0.3`。2026-10-18 核对的 `dev` 提交为 `b1f093a8fa538988315e606d9e9ea6d04392cee3`，查看 [Python 实现](https://github.com/pimoroni/vl53l5cx-python/blob/b1f093a8fa538988315e606d9e9ea6d04392cee3/library/vl53l5cx_ctypes/__init__.py)及 [单目标编译设置](https://github.com/pimoroni/vl53l5cx-python/blob/b1f093a8fa538988315e606d9e9ea6d04392cee3/library/setup.py)。实际 API：

```python
from smbus2 import SMBus
from vl53l5cx_ctypes import VL53L5CX

bus = SMBus(2)
sensor = VL53L5CX(i2c_addr=0x29, i2c_dev=bus)  # 构造时检查芯片并上传固件
sensor.set_resolution(64)
sensor.set_ranging_frequency_hz(15)
sensor.set_ranging_mode(1)   # RANGING_MODE_CONTINUOUS
sensor.set_target_order(1)  # TARGET_ORDER_CLOSEST
sensor.start_ranging()
# sensor.data_ready() -> sensor.get_data()
# 结束 sensor.stop_ranging(); bus.close()
```

上游 setter/start/stop 返回 `None`，不能按真假判断成功；`get_data()` 会在 ULD 错误时抛异常。单目标构建中 distance/status ctypes 数组形状是 `(1,64)`，count 是 `(64,)`；本模块复制并展平为 `(64,)`，将毫米转米。多目标构建、错误长度直接拒绝，不推测布局。驱动通过 `smbus2` 使用 Linux I2C，不依赖树莓派 GPIO 引脚编号；本模块不采用上游示例默认的 bus 1。

在香橙派 Linux 的项目虚拟环境安装：

```sh
sudo apt-get install build-essential python3-dev i2c-tools
python -m pip install 'smbus2>=0.4,<1' 'vl53l5cx-ctypes==0.0.3'
```

若需要从核对过的提交编译，可在工作区外独立克隆，避免覆盖项目文件：

```sh
git clone https://github.com/pimoroni/vl53l5cx-python.git
cd vl53l5cx-python
git checkout b1f093a8fa538988315e606d9e9ea6d04392cee3
git submodule update --init --recursive
python -m pip install ./library
```

已在目标香橙派 ARM64 环境构建并安装驱动，导入检查通过；传感器读数仍待接线实测。项目已将固定版本加入 `tof` 可选依赖，可用 `pip install -e '.[tof]'` 安装。

根据项目集成方已核对的《OrangePi_5_Max_RK3588_用户手册_v1.4》219 页：I2C2_M0 的 SDA 为**物理 3 脚**，SCL 为**物理 5 脚**，设备为 `/dev/i2c-2`；默认关闭，通过 `orangepi-config → System → Hardware → i2c2-m0` 启用并按系统要求重启。以目标系统设备节点核实映射，`verified` 保持 false 直至完成读回验证。使用模块适配板规定的电源和逻辑电平，公共 GND；模块供电/LPn 必须满足适配板要求。本实现不操纵 LPn，也不配置 device tree。Linux 7-bit 地址用 `0x29`，不能把手册中的 8-bit 地址 `0x52` 原样传入。

## 距离、状态与几何

**距离不是径向长度。** ST 工程师说明 VL53L5CX 已在芯片内执行 radial-to-perpendicular 变换：正对平墙时，各区返回近似相同的光轴深度。见 [ST 关于距离语义的说明](https://community.st.com/imaging-sensors-49/vl53l5cx-range-does-not-necessarily-mean-what-you-would-expect-9287)和 [ST 坐标转换讨论](https://community.st.com/imaging-sensors-49/calculate-x-y-z-coordinates-of-each-points-of-the-vl53l5cx-132460)。这里显式固定该语义，不提供会被误配的自动猜测模式。

设标定的单位射线为 `r_i`，光轴为 sensor +Z，原始深度为 `d_i`：

```text
radial_range_i = d_i / r_i.z
p_sensor_i = radial_range_i * r_i
p_body_i = sensor_to_body @ p_sensor_i + origin_body_m
up = -imu.gravity / norm(imu.gravity)
forward = normalize([1,0,0] - up.x * up)
left = cross(up, forward)
p_level_i = [dot(forward,p_body_i), dot(left,p_body_i), dot(up,p_body_i)]
height_i = p_level_i.z
```

机身坐标 +X 向前、+Y 向左、+Z 向上。水平化矩阵等价于 `R_y(pitch) @ R_x(roll)`，与参考 `RobotSensors_Update` 的无 yaw 旋转一致；仅使用 `ImuState.gravity` 即可得到该补偿。安装平移也参与水平化，不能只旋转光线而遗漏安装偏置。重力必须是近似单位向下向量，不是 `m/s²` 加速度；反向、倾倒或不合理范数均无效。

参考 `../dog/Robot/Inc/robot_sensors.h` 的 `RobotTofSample.distance_m` 定义为**径向距离**，与本模块的 `ToFFrame.distance_m` 不同。如接入该 C 接口，先执行 `radial_range = depth / ray[:,2]`，不要把原始深度直接传入，也不要在两端重复转换。

射线须逐 zone 标定，不能凭标称视场生成后标记已验证。ST 的 [UM2884，第 2.2、4.4、5.5 节](https://www.st.com/resource/en/user_manual/um2884-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-ranging-sensor-with-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf)说明镜头会翻转场景、8×8 最大 15 Hz，并给出状态码：5 是有效距离；6 未做 wrap-around，通常是首帧；9 是大脉冲，可能合并了多个目标。这里只默认接受 5，可经验证显式允许 `[5,9]`；6、0、255 等均拒绝。还必须有至少一个目标、有限正深度且在允许距离范围内。显示图像时的 flip/transpose 不应偷偷改变标定射线顺序。

每个训练点 `sample_xy_m[k]` 独立寻找容差内的最近有效测量点，选择其原始 Z；同距时选择驱动顺序靠前的点。允许一个真实测量同时覆盖邻近训练位置，但覆盖数不是独立样本数保证。若训练位置间距小于标定分辨率，应收紧容差或重新设计感知布局，不能扩大 bin 来伪造八点覆盖。高度异常值在最近邻之前剔除，不取 bin 平均，不跨空洞插值，不沿用前一帧高度，不用零填充冒充有效观测。

## 时间、运动与失效

- ToF、IMU 和 `now` 必须使用同一个主机单调时钟。零、未来、非有限和乱序时间戳拒绝；相同帧可在有效期内重复消费，不刷新 stamp。ToF 重启后若时钟基准改变，应同时重建 projector。
- Python 驱动没有硬件测量时间戳。本模块在调用 `data_ready()` **之前**记录主机时间，再减 `measurement_latency_s`；不以读完/消费者读取时刻伪造新鲜度。总线传输/测量延迟超过 `frame_timeout_s` 不发布。`measurement_latency_s=0` 只是未校准默认值，armed 前须用实测时延校准并确认 `verified`。
- 分别验证帧龄、IMU 龄和二者时差。集成端应保留 IMU 历史并选最近样本；不应仅为通过检查放宽 `max_skew_s`。模块不凭陀螺积分伪造过去姿态。
- 用 `norm(imu.omega) * (abs(ToF_stamp - IMU_stamp) + frame_duration_s)` 检查帧内及错时旋转量。8×8 是多区扫描，单一 IMU 姿态无法补偿快速运动；超阈值全帧无效。此检查基于当前角速度，不能保证未观测的瞬态运动，因此仍需静态/动态标定和合适的 IMU 历史。
- 默认 `frame_duration_s=1/frequency_hz`；降低帧率时需同步评估运动阈值、帧超时和策略时效。初始化无数据可等待 `startup_timeout_s`，开始测距后 data-ready 超时为终止故障。
- 停帧、传输异常不自动回退 MockToF。调用方必须处理异常并 stop；缺失覆盖或投影错误返回明确无效状态。`close()` 有界等待，底层 I2C 若堵塞会报错，不能由 Python 线程强制中断；线程独占 bus，主线程不会并发关闭它。
- 上游 ctypes 回调和 setter 的错误传播能力有限。本模块依靠芯片初始化、数据就绪、读取错误、逐区有效性和超时门槛检查运行状态，不能将软件通过测试等同于硬件验收。后续硬件验收须验证断线、I2C 错误和 CPU 负载下的超时行为。

## 标定与软件验证

1. 在未 armed 状态确认 I2C2、地址、zone 顺序、实际刷新间隔和时延，保存驱动/配置版本。
2. 用已知位置和姿态的目标标定 64 条射线、光轴深度语义、安装旋转与平移；旋转必须满足 `R Rᵀ=I`、`det(R)=+1`，射线单位长度且 sensor Z 为正。
3. 从实际训练导出八个采样 XY 及顺序，在相同参考系下用平面和已知高度台阶验证 XY 容差和高度范围。确认平地的原始 Z 约为负机身高度，而非默认零。
4. 对齐 manifest contract，确认观测 term 的 scale/clip 与历史配置。仅在上述验证完成后逐项开启校准标志；运行时仍要求八项 `valid` 全真。

无硬件单元测试：

```sh
python -B -m pytest tests/test_tof.py -q -p no:cacheprovider
```

测试使用独立解析几何和模拟 ctypes/I2C 驱动，覆盖光轴深度恢复、非零安装平移、roll/pitch、台阶正负号与顺序、精确最近邻、无效状态/目标数/异常值、缺失覆盖、时间/旋转失效、manifest 错配、线程生命周期与错误释放。它们不测量模块精度、香橙派硬件时延或真实地面反射率，也不需要安装硬件 Python 驱动。
