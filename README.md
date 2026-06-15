# MedVision - 手术器械视觉追踪系统

> 基于 Orbbec Astra 3D 深度摄像头的手术器械姿态测量与追踪系统

## 🎯 功能特性

- **实时目标追踪**：颜色检测 + Kalman预测 + 速度追踪
- **开合角度测量**：PCA + EMA平滑，±5°精度
- **3D点云扫描**：AI定位 + 深度测量 + PLY导出
- **云台控制**：CLB-S25 UART总线舵机，PID跟随
- **实时视频流**：低延迟 HTTP 服务器，浏览器查看
- **数据记录**：CSV日志 + 会话记录

## 📦 安装

### 依赖要求
- Python 3.8+
- OpenCV 4.x
- OpenNI2 (Astra深度驱动)
- Flask (Web服务器)

### 快速安装
```bash
# 1. 克隆仓库
git clone <repo-url> medvision
cd medvision

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行
python3 quickstart.py
```

### 硬件要求
| 设备 | 用途 |
|------|------|
| Orbbec Astra Pro | 3D深度+彩色摄像头 |
| CLB-S25 舵机 x2 | 云台控制 |
| UC01 USB转串口 | 舵机通信 |
| 树莓派5 (推荐) | 主控平台 |

## 🚀 快速开始

```python
from harness import GimbalTool, CameraTool, DetectTool, AngleTool

# 初始化
gimbal = GimbalTool(config)
camera = CameraTool(config)
detect = DetectTool(config)
angle = AngleTool(config)

# 扫描目标
gimbal.move_to(pan=0, tilt=25)
bgr = camera.capture()
result = detect.find(bgr)

# 测量角度
a = angle.measure(bgr)
print(f"Angle: {a:.1f}°")
```

## 📁 项目结构

```
harness/
├── core/           # Agent主循环
├── tools/          # 功能工具
│   ├── gimbal.py   # 云台控制
│   ├── camera.py   # 摄像头采集
│   ├── detect.py   # 目标检测
│   ├── angle.py    # 角度测量
│   └── llm_api.py  # LLM API接入
├── hardware/       # 硬件驱动
└── web/            # Web UI
```

## 📝 License

MIT License - 详见 LICENSE 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！详见 CONTRIBUTING.md
