"""
Hardware Abstraction Layer (inspired by robo-infra + Cyberwave).
Unified interface for camera, gimbal, and sensors.
Supports simulation mode for testing without hardware.
"""
import time, numpy as np


class HAL:
    """Hardware Abstraction Layer - unified interface for all hardware."""

    def __init__(self, config, simulate=False):
        self.simulate = simulate
        self.config = config

        if not simulate:
            from harness_gimbal import GimbalTool
            from harness_camera import CameraTool
            self.gimbal = GimbalTool(config["gimbal"])
            self.camera = CameraTool(config["camera"])
            print("[HAL] Real hardware connected")
        else:
            self.gimbal = SimGimbal()
            self.camera = SimCamera()
            print("[HAL] Simulation mode")

    def capture(self):
        return self.camera.capture()

    def move_gimbal(self, pan=None, tilt=None):
        if pan is not None or tilt is not None:
            self.gimbal.move_to(pan=pan, tilt=tilt)
        return self.gimbal.query()

    def nudge(self, dpan, dtilt):
        return self.gimbal.nudge(dpan, dtilt)

    def center(self):
        self.gimbal.center()

    def close(self):
        self.gimbal.close()
        if hasattr(self.camera, 'close'):
            self.camera.close()


class SimGimbal:
    """Simulated gimbal for testing."""
    def __init__(self):
        self.pan, self.tilt = 0.0, 0.0
    def move_to(self, pan=None, tilt=None, **kw):
        if pan is not None: self.pan = pan
        if tilt is not None: self.tilt = tilt
    def nudge(self, dp, dt, **kw):
        self.pan += dp; self.tilt += dt
    def query(self): return self.pan, self.tilt
    def center(self): self.pan, self.tilt = 0.0, 0.0
    def close(self): pass


class SimCamera:
    """Simulated camera for testing."""
    def capture(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    def close(self): pass
