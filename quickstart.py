#!/usr/bin/env python3
"""
MedVision Quickstart - 5 minutes to get running.
Demo: Detect red clip, measure angle, control gimbal.
"""
import sys, time
sys.path.insert(0, "/opt/medvision/harness")

from gimbal import GimbalTool
from camera import CameraTool
from detect import DetectTool
from angle import AngleTool

# Load default config
import json
config_path = os.path.join(os.path.dirname(__file__), "config", "default.json")
if not os.path.exists(config_path):
    config_path = "/opt/medvision/harness/config/default.json"
with open(config_path) as f:
    cfg = json.load(f)

print("=" * 50)
print("  MedVision Quickstart Demo")
print("=" * 50)

# Init tools
gimbal = GimbalTool(cfg["gimbal"])
camera = CameraTool(cfg["camera"])
detect = DetectTool(cfg["detection"])
angle = AngleTool(cfg["angle"])

print("\n[1] Camera test...")
bgr = camera.capture()
if bgr is not None:
    print("    Captured: %s" % str(bgr.shape))
else:
    print("    ERROR: Camera not available")
    camera.close()
    sys.exit(1)

print("[2] Detection test...")
result = detect.find(bgr)
if result:
    print("    Target: (%d, %d) area=%d" % result)
else:
    print("    No target found (place red clip in view)")

print("[3] Angle measurement test...")
a = angle.measure(bgr)
print("    Angle: %.1f degrees" % a if a else "    No angle (no target)")

print("[4] Gimbal test...")
gimbal.center()
time.sleep(0.5)
p, t = gimbal.query()
print("    Centered: pan=%.1f tilt=%.1f" % (p, t))

print("[5] Tracking demo (5 seconds)...")
print("    Moving gimbal to follow target...")
for i in range(50):
    bgr = camera.capture()
    if bgr is None:
        continue
    result = detect.find(bgr)
    if result:
        cx, cy, area = result
        err_x = cx - 320
        if abs(err_x) > 25:
            dpan = max(-1, min(1, 0.03 * err_x))
            gimbal.nudge(dpan, 0)
        a = angle.measure(bgr)
        if a and i % 10 == 0:
            print("    Frame %d: pos=(%d,%d) angle=%.1f" % (i, cx, cy, a))
    time.sleep(0.05)

# Cleanup
gimbal.center()
time.sleep(0.3)
gimbal.close()
camera.close()

print("\n" + "=" * 50)
print("  Quickstart Complete!")
print("  Edit config/default.json to customize")
print("=" * 50)
