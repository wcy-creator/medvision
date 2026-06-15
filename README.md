# 🔬 MedVision — 手术器械视觉追踪与角度测量系统

<p align="center">
  <b>基于 Orbbec Astra 3D 深度摄像头的手术器械姿态测量与实时追踪系统</b>
</p>

---

## 📖 项目简介

MedVision 是一套面向手术机器人的**视觉感知系统**，集成了：

- 🎯 **实时目标追踪** — 颜色检测 + Kalman 预测 + 速度跟随
- 📐 **开合角度测量** — PCA + EMA 平滑，±5° 精度
- 🏗️ **3D 点云扫描** — AI 定位 + 深度测量 + PLY 导出
- 🎮 **云台控制** — CLB-S25 UART 舵机，PID 双轴跟随
- 📹 **实时视频流** — 低延迟 HTTP 服务器
- 📊 **数据记录** — CSV 日志 + 会话记录

### 应用场景

| 场景 | 说明 |
|------|------|
| 手术器械姿态监测 | 实时追踪器械开合角度 |
| 机器人视觉引导 | 3D 定位+云台跟随 |
| 器械操作训练 | 角度数据记录与分析 |
| 远程监控 | Web 实时视频+状态 |

---

## 📦 安装说明

### 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.8+ | 主运行时 |
| OpenCV | 4.8+ | 图像处理 |
| Flask | 3.0+ | Web 服务器 |
| OpenNI2 | - | Astra 深度驱动 |
| 系统 | Linux (Debian/Ubuntu) | 推荐树莓派5 |

### 方法一：pip 安装（推荐）

```bash
# 克隆项目
git clone <repo-url> medvision
cd medvision

# 安装 Python 依赖
pip install -r requirements.txt

# 验证安装
python3 quickstart.py
```

### 方法二：手动安装

```bash
# Python 依赖
pip install opencv-python numpy flask requests openni

# 系统依赖（树莓派）
sudo apt-get install -y cmake libopencv-dev

# Orbbec udev 规则
sudo cp config/56-orbbec-usb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

### 方法三：Docker（实验性）

```bash
docker build -t medvision .
docker run --device=/dev/video0 --device=/dev/ttyUSB0 medvision
```

---

## 🚀 使用方法

### 快速开始（5分钟）

```python
from harness import GimbalTool, CameraTool, DetectTool, AngleTool

# 加载配置
import json
with open("config/default.json") as f:
    cfg = json.load(f)

# 初始化工具
gimbal = GimbalTool(cfg["gimbal"])
camera = CameraTool(cfg["camera"])
detect = DetectTool(cfg["detection"])
angle = AngleTool(cfg["angle"])

# 拍照 + 检测 + 测角
bgr = camera.capture()
result = detect.find(bgr)       # (cx, cy, area) 或 None
a = angle.measure(bgr)          # 角度值
print(f"Target: {result}, Angle: {a:.1f}°")

# 云台跟随
if result:
    cx, cy, _ = result
    err_x = cx - 320
    if abs(err_x) > 25:
        gimbal.nudge(err_x * 0.03, 0)

# 清理
gimbal.center()
gimbal.close()
camera.close()
```

### 命令行运行

```bash
# 启动追踪演示
python3 quickstart.py

# 启动流服务器（浏览器查看）
python3 astra_stream_v5.py
# 浏览器访问 http://localhost:8080

# 启动完整追踪器
python3 track_final_v2.py
```

### 硬件接线

```
Orbbec Astra Pro USB ──→ 树莓派 USB
UC01 USB转串口 ──→ 树莓派 USB
  ├── TX → 云台水平舵机
  └── RX → 云台俯仰舵机
外部电源 5V/2A ──→ 舵机供电
```

### 参数调整

编辑 `config/default.json`：

```json
{
    "tracking": {
        "kp": 0.05,        // 增益（增大=快，减小=稳）
        "dead_zone": 20,    // 死区（像素）
        "max_step": 2.5     // 最大步长（度）
    }
}
```

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────┐
│              MedVision Harness            │
├────────────┬─────────────┬───────────────┤
│  LLM API   │  Tools      │  Hardware     │
│  MiMo/GPT  │  detect     │  V4L2 Camera  │
│  视觉分析   │  angle      │  OpenNI2 Depth│
│            │  gimbal     │  UART Servo   │
│            │  scan3d     │  CLB-S25      │
└────────────┴─────────────┴───────────────┘
```

---

## 📁 项目结构

```
medvision/
├── README.md              # 本文档
├── LICENSE                # MIT 协议
├── requirements.txt       # Python 依赖
├── quickstart.py          # 5分钟快速上手
├── .gitignore             # Git 忽略规则
│
├── harness/               # 核心模块
│   ├── harness_agent.py   # Agent 主循环
│   ├── harness_gimbal.py  # 云台控制
│   ├── harness_camera.py  # 摄像头采集
│   ├── harness_detect.py  # 目标检测
│   └── harness_angle.py   # 角度测量
│
├── config/
│   └── default.json       # 配置参数
│
├── astra_stream_v5.py     # 视频流服务器
├── track_final_v2.py      # Kalman 追踪器
├── track_v7.py            # 多特征追踪器
├── detector.py            # 检测器
├── gimbal_uart.py         # 舵机控制
├── uservo.py              # 舵机 SDK
├── medvision.cpp          # C++ 原型
└── archive/               # 旧版本归档
```

---

## 🔧 API 接口

### HTTP API（流服务器端口 8080）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 实时视频页面 |
| `/snapshot/color` | GET | 彩色快照 JPEG |
| `/snapshot/depth` | GET | 深度快照 JPEG |
| `/mjpeg/color` | GET | 彩色 MJPEG 流 |
| `/mjpeg/depth` | GET | 深度 MJPEG 流 |
| `/status` | GET | JSON 状态信息 |

### Python API

```python
# 工具类 API
gimbal.move_to(pan=0, tilt=25)    # 移动云台
gimbal.nudge(dpan=1, dtilt=0)     # 微调
gimbal.center()                    # 归中
gimbal.query()                     # (pan, tilt)

bgr = camera.capture()             # 640x480 BGR
result = detect.find(bgr)          # (cx, cy, area) or None
angle = angle.measure(bgr)         # 浮点角度值
```

---

## 🧪 测试

```bash
# 快速验证
python3 quickstart.py

# 完整追踪测试
python3 track_final_v2.py

# 流服务器测试
python3 astra_stream_v5.py
# 浏览器 http://localhost:8080
```

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 视频流 FPS | 15-29 |
| 追踪 FPS | 6-15 |
| 角度精度 | ±5° (EMA 平滑后) |
| 检测成功率 | 100% |
| 云台响应 | ~300ms |
| 有效检测距离 | 0.7m - 1.1m |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- [Orbbec](https://www.orbbec.com/) — 3D 深度摄像头
- [OpenCV](https://opencv.org/) — 计算机视觉库
- [FashionStar](https://fashionstar.com/) — CLB-S25 舵机
- [MiMo](https://xiaomi.com/) — AI 视觉推理
