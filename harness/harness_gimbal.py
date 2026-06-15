"""Gimbal Control Tool - Wrapper around CLB-S25 UART servo."""
import sys, time
sys.path.insert(0, "/opt/medvision")
from gimbal_uart import GimbalUART


class GimbalTool:
    def __init__(self, config):
        self.g = GimbalUART()
        time.sleep(0.5)
        self.g.TILT_MIN = config.get("tilt_min", -60)
        self.g.TILT_MAX = config.get("tilt_max", 60)
        self.g.PAN_MIN = config.get("pan_min", -60)
        self.g.PAN_MAX = config.get("pan_max", 60)

    def move_to(self, pan=None, tilt=None, speed=200):
        self.g.move_to(pan=pan, tilt=tilt, velocity=speed)
        return self.query()

    def nudge(self, dpan, dtilt, speed=200):
        self.g.nudge(dpan, dtilt, velocity=speed)
        return self.query()

    def query(self):
        return self.g.query()

    def center(self):
        self.g.center()

    def close(self):
        self.g.close()
