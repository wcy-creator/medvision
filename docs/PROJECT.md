# 🔬 MedVision — 手术器械视觉追踪与角度测量系统

## 完整项目文档 | Introduction, Setup & Usage Guide

---

## 一、项目概述

### 1.1 是什么

MedVision 是一套基于 **Orbbec Astra 3D 深度摄像头**的手术器械视觉感知系统。它能：

- 🎯 **实时追踪**手术器械（颜色检测 + Kalman 预测）
- 📐 **测量开合角度**（PCA + EMA 平滑，±5° 精度）
- 🏗️ **3D 点云扫描**（AI 定位 + 深度测量 + PLY 导出）
- 🎮 **云台自动跟随**（CLB-S25 舵机，PID 双轴控制）
- 📹 **实时视频流**（低延迟 HTTP，浏览器查看）
- 🤖 **LLM 集成**（MiMo/OpenAI 视觉分析）

### 1.2 为谁而做

| 用户 | 场景 |
|------|------|
| 手术医生 | 实时监测器械姿态和开合角度 |
| 手术机器人 | 视觉引导 + 角度反馈 |
| 医学培训 | 角度数据记录与分析 |
| 研究人员 | 3D 扫描 + 点云数据采集 |

### 1.3 技术架构

```
┌─────────────────────────────────────────────┐
│              MedVision Harness              │
├──────────┬──────────┬──────────┬────────────┤
│ LLM API  │ Detection│ Gimbal   │ 3D Scan    │
│ MiMo/GPT │ Color+   │ PID +    │ OpenNI2 +  │
│ Vision   │ Kalman   │ UART     │ Depth      │
├──────────┴──────────┴──────────┴────────────┤
│            Hardware Layer                    │
│  Astra Pro (USB) + CLB-S25 (UART) + Pi5     │
└─────────────────────────────────────────────┘
```

---

## 二、硬件需求

### 2.1 核心硬件

| 设备 | 型号 | 用途 | 连接方式 |
|------|------|------|---------|
| 主控 | 树莓派5 16GB | 运行系统 | - |
| 深度摄像头 | Orbbec Astra Pro | RGB-D 采集 | USB 2.0 |
| 水平舵机 | CLB-S25 | 云台左右 | UART→USB |
| 俯仰舵机 | CLB-S25 | 云台上下 | UART→USB（串联）|
| USB转串口 | UC01 (CH340) | 舵机通信 | USB |

### 2.2 接线图

```
树莓派 USB口1 ──→ Astra Pro 深度摄像头
树莓派 USB口2 ──→ UC01 (CH340)
                    ├── TX → 水平舵机 (ID=0)
                    └── RX → 俯仰舵机 (ID=1, 串联)
外部电源 5V/2A ──→ 两个舵机供电（共地接树莓派GND）
```

### 2.3 软件环境

| 软件 | 版本 | 用途 |
|------|------|------|
| OS | Debian 13 (trixie) | 树莓派5 官方系统 |
| Python | 3.13 | 主运行时 |
| OpenCV | 4.13 | 图像处理 |
| OpenNI2 | 2.2 | Astra 深度驱动 |
| Flask | 3.1 | Web 服务器 |
| onnxruntime | 1.21 | YOLO 推理 |

---

## 三、安装指南

### 3.1 一键安装

```bash
# 方法1：U盘部署（推荐）
cd /path/to/medvision
bash start.sh

# 方法2：pip 安装
git clone <repo-url> medvision
cd medvision
pip install -r requirements.txt
```

### 3.2 手动安装

```bash
# Python 依赖
pip install opencv-python numpy flask requests openni

# 系统依赖（树莓派）
sudo apt-get install -y cmake libopencv-dev

# Orbbec udev 规则（让非 root 用户访问摄像头）
sudo cp config/56-orbbec-usb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

### 3.3 Docker 部署

```bash
docker build -t medvision .
docker run --device=/dev/video0 --device=/dev/ttyUSB0 -p 8080:8080 medvision
```

### 3.4 验证安装

```bash
python3 quickstart.py
# 应该看到：
# [1] Camera test... Captured: (480, 640, 3)
# [2] Detection test... Target: (x, y) area=xxx
# [3] Angle measurement... Angle: xx.x degrees
# [4] Gimbal test... Centered: pan=0.0 tilt=0.0
```

---

## 四、使用流程

### 4.1 快速开始（5分钟）

```python
from harness import GimbalTool, CameraTool, DetectTool, AngleTool
import json

# 加载配置
with open("config/default.json") as f:
    cfg = json.load(f)

# 初始化
gimbal = GimbalTool(cfg["gimbal"])
camera = CameraTool(cfg["camera"])
detect = DetectTool(cfg["detection"])
angle = AngleTool(cfg["angle"])

# 拍照 + 检测 + 测角
bgr = camera.capture()
result = detect.find(bgr)     # (cx, cy, area) 或 None
a = angle.measure(bgr)        # 角度值

if result:
    cx, cy, area = result
    print(f"检测到目标: 位置({cx},{cy}) 面积={area}")
    print(f"开合角度: {a:.1f}°")

# 清理
gimbal.center()
gimbal.close()
camera.close()
```

### 4.2 完整追踪流程

```
步骤1: 启动流服务器（浏览器看画面）
    python3 astra_stream_v5.py
    → 浏览器打开 http://localhost:8080

步骤2: 启动追踪器（新终端）
    python3 track_final_v2.py
    → 自动找到目标 → 开始追踪 → 实时输出角度

步骤3: 观察
    浏览器: 实时画面
    终端: 角度数据流

步骤4: 停止
    Ctrl+C → 云台自动归中 → 输出统计
```

### 4.3 参数调整

编辑 `config/default.json`：

```json
{
    "tracking": {
        "kp": 0.05,        // 增益: 增大=响应快, 减小=更稳
        "dead_zone": 20,    // 死区(像素): 增大=减少抖动
        "max_step": 2.5     // 最大步长(度): 减小=更平滑
    },
    "detection": {
        "min_contour_area": 100  // 最小检测面积
    }
}
```

### 4.4 LLM 视觉分析

```python
from harness_harness_llm import LLMRouter

llm = LLMRouter()

# 文本对话
reply = llm.chat([{"role": "user", "content": "你好"}])

# 图像分析
bgr = camera.capture()
desc = llm.analyze_image(bgr, "描述画面中的物体")
print(desc)

# AI 目标检测
objects = llm.detect_objects(bgr)
print(objects)
```

---

## 五、API 参考

### 5.1 Python API

| 类 | 方法 | 返回值 | 说明 |
|----|------|--------|------|
| `GimbalTool` | `move_to(pan, tilt)` | (pan, tilt) | 移动到指定角度 |
| | `nudge(dpan, dtilt)` | (pan, tilt) | 微调 |
| | `center()` | - | 归中 |
| | `query()` | (pan, tilt) | 查询位置 |
| `CameraTool` | `capture()` | np.array (480,640,3) | 拍照 |
| `DetectTool` | `find(bgr)` | (cx, cy, area) 或 None | 检测红色目标 |
| `AngleTool` | `measure(bgr)` | float | 测量角度 |
| `LLMRouter` | `chat(messages)` | str | 文本对话 |
| | `analyze_image(bgr, prompt)` | str | 图像分析 |

### 5.2 HTTP API（端口 8080）

| 端点 | 方法 | 返回 |
|------|------|------|
| `/` | GET | 实时视频页面 |
| `/snapshot/color` | GET | 彩色 JPEG 快照 |
| `/snapshot/depth` | GET | 深度 JPEG 快照 |
| `/mjpeg/color` | GET | 彩色 MJPEG 流 |
| `/mjpeg/depth` | GET | 深度 MJPEG 流 |
| `/status` | GET | JSON 状态 |

### 5.3 配置文件

| 文件 | 说明 |
|------|------|
| `config/default.json` | 系统参数（追踪/检测/云台） |
| `config/api.json` | LLM API 配置（模型/密钥） |
| `harness/config/default.json` | Harness 工具配置 |

---

## 六、性能指标

| 指标 | 数值 |
|------|------|
| 视频流 FPS | 15-29 |
| 追踪 FPS | 6-15（直接V4L2）|
| 角度精度 | ±5°（EMA平滑后）|
| 检测成功率 | 100% |
| 云台响应 | ~300ms |
| 有效检测距离 | 0.7m - 1.1m |
| 深度有效范围 | 442-3300mm |
| LLM 响应时间 | 5-15秒 |

---

## 七、常见问题

### Q: 摄像头打不开？
```bash
# 检查 USB 设备
lsusb | grep orbbec
# 检查权限
ls -la /dev/video*
# 重新加载 udev
sudo udevadm control --reload-rules
```

### Q: 舵机不动？
```bash
# 检查串口
ls -la /dev/ttyUSB*
# 测试通信
python3 -c "from gimbal_uart import GimbalUART; g=GimbalUART(); g.center(); g.close()"
```

### Q: 追踪不稳定？
- 增大 `dead_zone`（减少抖动）
- 减小 `kp`（更温和的跟随）
- 确保光照均匀（颜色检测受光照影响）

### Q: 角度不准？
- 固定云台在 tilt=30° 测量最准
- 夹子不要反光（哑光表面更好）
- 保持检测距离在 0.7-1.1m

---

## 八、项目结构

```
medvision/
├── README.md                    # 项目介绍
├── LICENSE                      # MIT 协议
├── CONTRIBUTING.md              # 贡献指南
├── requirements.txt             # Python 依赖
├── Dockerfile                   # Docker 部署
├── .gitignore                   # Git 规则
├── .github/workflows/ci.yml    # CI/CD
├── quickstart.py                # 快速上手
├── start.sh                     # 通用启动脚本
│
├── harness/                     # 核心 Harness
│   ├── harness_agent.py         # Agent 主循环
│   ├── harness_gimbal.py        # 云台控制
│   ├── harness_camera.py        # 摄像头采集
│   ├── harness_detect.py        # 目标检测
│   ├── harness_angle.py         # 角度测量
│   └── harness_llm.py           # LLM API 接入
│
├── config/                      # 配置
│   └── default.json
│
├── tests/                       # 测试
│   ├── test_gimbal.py
│   ├── test_camera.py
│   └── test_angle.py
│
├── astra_stream_v5.py           # 视频流服务器
├── track_final_v2.py            # Kalman 追踪器
├── detector.py                  # 多特征检测器
├── gimbal_uart.py               # 舵机控制
├── uservo.py                    # 舵机 SDK
├── medvision.cpp                # C++ 原型
│
└── archive/                     # 旧版本归档
```

---

## 九、许可证

MIT License. 详见 [LICENSE](LICENSE) 文件。

---

## 十、致谢

| 组织 | 贡献 |
|------|------|
| [Orbbec](https://www.orbbec.com/) | Astra 3D 深度摄像头 |
| [OpenCV](https://opencv.org/) | 计算机视觉库 |
| [FashionStar](https://fashionstar.com/) | CLB-S25 UART 舵机 |
| [MiMo (Xiaomi)](https://xiaomi.com/) | AI 视觉推理 API |
| [HyperTrack](https://github.com/coralr-1/hypertrack) | 追踪架构参考 |
| [PyImageSearch](https://pyimagesearch.com/) | PID 追踪教程 |

---

<p align="center">
  <b>MedVision v1.0 | MIT License | 2026</b>
</p>
