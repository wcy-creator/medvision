<div align="center">

# MedVision

**Surgical Instrument Visual Tracking & Angle Measurement System**

Real-time tracking, 3D depth sensing, YOLOv5 detection, and cloud AI — all on Raspberry Pi.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-00ff88?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi5-A22866?style=flat-square&logo=raspberrypi&logoColor=white)](https://raspberrypi.org)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-ONNX-FF6B6B?style=flat-square)](https://github.com/ultralytics/yolov5)

</div>

---

## What is MedVision?

MedVision is a **complete robotic vision system** for tracking and measuring surgical instruments using a 3D depth camera. It combines edge AI (YOLOv5), cloud reasoning (LLM Vision), and precision servo control into one integrated platform.

```
Camera → Detection → Tracking → Measurement → Control
  📷        🎯          🎮          📐           🤖
```

## Features

| Feature | Description |
|---------|-------------|
| **YOLOv5 Detection** | Local ONNX inference at 6.9 FPS (yolov5n) |
| **Kalman Tracking** | Predicts target position when temporarily lost |
| **PID Gimbal Control** | Smooth servo tracking with anti-windup |
| **3D Point Cloud** | Depth sensing + AI-guided 3D measurement |
| **Angle Measurement** | PCA-based ±5° precision for instrument opening |
| **Cloud AI** | MiMo Vision API for intelligent analysis |
| **Web Dashboard** | MJPEG live stream + AI analysis endpoint |
| **Memory System** | Episodic + semantic memory with SQLite |
| **Voice Control** | Text/voice interaction with TTS response |
| **Cross-Platform** | Linux, Windows, macOS (with emulator) |

## Quick Start

```python
from harness import Agent

agent = Agent(use_yolo=True)
agent.cmd_track(on=True)
agent.run()  # Auto-tracks red surgical instruments
```

### Web Dashboard
```bash
python3 astra_stream.py
# Open http://<pi-ip>:8080
```

## Hardware

| Component | Model | Status |
|-----------|-------|--------|
| Computer | Raspberry Pi 5 (16GB) | ✅ Tested |
| Camera | Orbbec Astra Pro (RGB-D) | ✅ Tested |
| Servo | CLB-S25 UART (±0.3°) | ✅ Tested |
| Gimbal | 2-axis (Pan + Tilt) | ✅ Tested |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  MedVision System                    │
├───────────┬───────────┬───────────┬─────────────────┤
│  Vision   │  Track    │  Measure  │   Control       │
│ YOLOv5    │ Kalman    │ PCA +     │ PID +           │
│ Color     │ Filter    │ EMA       │ UART Servo      │
├───────────┴───────────┴───────────┴─────────────────┤
│              Harness Framework                       │
│  Agent │ Camera │ Detect │ LLM │ Memory │ Voice     │
└─────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/wcy-creator/medvision.git
cd medvision
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 harness/harness_agent_v2.py --yolo
```

## Performance

| Metric | Value |
|--------|-------|
| YOLOv5n FPS | 6.9 (CPU) |
| YOLOv5s FPS | 3.6 (CPU) |
| Color Detection FPS | 30+ |
| Angle Precision | ±5° |
| Servo Precision | ±0.3° |

## Related Projects

- **[PiGimbal](https://github.com/wcy-creator/pigimbal)** — Smart gimbal control library
- **[VisionAgent](https://github.com/wcy-creator/visionagent)** — LLM-powered visual agent

## License

MIT License

## Author

**W-cy** — Robotics engineer building intelligent robotic systems.

- GitHub: [@wcy-creator](https://github.com/wcy-creator)
- Projects: [PiGimbal](https://github.com/wcy-creator/pigimbal) | [VisionAgent](https://github.com/wcy-creator/visionagent)

---

If you find MedVision useful, please give it a ⭐!
