"""MedVision - Gimbal Controller (v2 - Non-blocking)
PWM always on. Set target angle, PWM holds position.
No sleep, no blocking. Smooth and responsive.
"""
import RPi.GPIO as GPIO
import json

class Gimbal:
    def __init__(self, config_path="/opt/medvision/config.json"):
        with open(config_path) as f:
            cfg = json.load(f)
        g = cfg["gimbal"]
        self.pan_pin = g["servo1_pin"]
        self.tilt_pin = g["servo2_pin"]
        self.pan = float(g.get("center", 90))
        self.tilt = float(g.get("center", 90))
        self.pan_min = g.get("pan_min", 0)
        self.pan_max = g.get("pan_max", 180)
        self.tilt_min = g.get("tilt_min", 0)
        self.tilt_max = g.get("tilt_max", 180)
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pan_pin, GPIO.OUT)
        GPIO.setup(self.tilt_pin, GPIO.OUT)
        self.pwm_pan = GPIO.PWM(self.pan_pin, 50)
        self.pwm_tilt = GPIO.PWM(self.tilt_pin, 50)
        self.pwm_pan.start(self._duty(self.pan))
        self.pwm_tilt.start(self._duty(self.tilt))

    def _duty(self, angle):
        """Convert angle (0-180) to duty cycle (2.5%-12.5%)."""
        return 2.5 + (angle / 180.0) * 10.0

    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def set_pan(self, angle):
        """Set pan angle immediately (non-blocking). PWM stays on."""
        angle = float(self._clamp(angle, self.pan_min, self.pan_max))
        self.pan = angle
        self.pwm_pan.ChangeDutyCycle(self._duty(angle))
        return angle

    def set_tilt(self, angle):
        """Set tilt angle immediately (non-blocking). PWM stays on."""
        angle = float(self._clamp(angle, self.tilt_min, self.tilt_max))
        self.tilt = angle
        self.pwm_tilt.ChangeDutyCycle(self._duty(angle))
        return angle

    def move_to(self, pan=None, tilt=None):
        """Set target angles (non-blocking). PWM always on."""
        if pan is not None:
            self.set_pan(pan)
        if tilt is not None:
            self.set_tilt(tilt)
        return self.pan, self.tilt

    def center(self):
        return self.move_to(90, 90)

    def nudge(self, dpan=0, dtilt=0):
        """Relative move in degrees."""
        return self.move_to(self.pan + dpan, self.tilt + dtilt)

    def stop_pwm(self):
        """Stop PWM (servo will go limp)."""
        self.pwm_pan.ChangeDutyCycle(0)
        self.pwm_tilt.ChangeDutyCycle(0)

    def close(self):
        self.pwm_pan.stop()
        self.pwm_tilt.stop()
        GPIO.cleanup()

    def status(self):
        return {"pan": self.pan, "tilt": self.tilt,
                "pan_range": [self.pan_min, self.pan_max],
                "tilt_range": [self.tilt_min, self.tilt_max]}


if __name__ == "__main__":
    import time
    g = Gimbal()
    print("Gimbal v2 (Non-blocking)")
    print("Pan: %d-%d, Tilt: %d-%d" % (g.pan_min, g.pan_max, g.tilt_min, g.tilt_max))

    print("\n[CENTER]")
    g.center(); time.sleep(2)

    print("[LEFT] Pan -> 45")
    g.move_to(pan=45); time.sleep(2)

    print("[RIGHT] Pan -> 135")
    g.move_to(pan=135); time.sleep(2)

    print("[CENTER]")
    g.center(); time.sleep(2)

    g.close()
    print("\nDone!")
