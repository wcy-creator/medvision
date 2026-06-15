"""MedVision - UART Gimbal (CLB-S25 v2)
Direct angle query, reliable offset handling.
Logical center (0,0) = physical level (pan=+20, tilt=-75).
"""
import serial, struct, time, json, sys
sys.path.insert(0, "/opt/medvision")
from uservo import UartServoManager

class GimbalUART:
    def __init__(self, port="/dev/ttyUSB0"):
        with open("/opt/medvision/gimbal_offset.json") as f:
            off = json.load(f)
        self.pan_offset = off["pan_home"]   # 20
        self.tilt_offset = off["tilt_home"]  # -75
        # Mechanical limits (prevent gimbal jamming)
        self.TILT_MIN = 0    # dont tilt below this (gets stuck)
        self.TILT_MAX = 40    # dont tilt above this
        self.PAN_MIN = -35    # pan left limit
        self.PAN_MAX = 35     # pan right limit
        self.pan_id = 0
        self.tilt_id = 1
        self.uart = serial.Serial(port=port, baudrate=115200,
                                  parity=serial.PARITY_NONE, stopbits=1,
                                  bytesize=8, timeout=0)
        self.uservo = UartServoManager(self.uart, is_scan_servo=True,
                                        srv_num=10, is_debug=False)
        self.pan, self.tilt = self.query()
        print("Ready: pan=%.1f tilt=%.1f" % (self.pan, self.tilt))

    def _raw(self, sid):
        self.uservo.query_servo_angle(sid)
        time.sleep(0.15)
        v = self.uservo.servos[sid].cur_angle
        return float(v) if v is not None else 0.0

    def query(self):
        self.pan = self._raw(self.pan_id) - self.pan_offset
        self.tilt = self._raw(self.tilt_id) - self.tilt_offset
        return self.pan, self.tilt

    def move_to(self, pan=None, tilt=None, velocity=100):
        if pan is not None:
            pan = max(self.PAN_MIN, min(self.PAN_MAX, pan))
            actual = max(-135, min(135, pan + self.pan_offset))
            self.uservo.set_servo_angle(self.pan_id, actual, velocity=velocity)
            self.pan = pan
        if tilt is not None:
            tilt = max(self.TILT_MIN, min(self.TILT_MAX, tilt))
            actual = max(-135, min(135, tilt + self.tilt_offset))
            self.uservo.set_servo_angle(self.tilt_id, actual, velocity=velocity)
            self.tilt = tilt
        time.sleep(0.05)
        return self.pan, self.tilt

    def center(self, velocity=100):
        return self.move_to(0, 0, velocity=velocity)

    def nudge(self, dpan=0, dtilt=0, velocity=100):
        new_pan = max(self.PAN_MIN, min(self.PAN_MAX, self.pan + dpan))
        new_tilt = max(self.TILT_MIN, min(self.TILT_MAX, self.tilt + dtilt))
        return self.move_to(new_pan, new_tilt, velocity)

    def close(self): self.uart.close()

if __name__ == "__main__":
    g = GimbalUART()
    print("\n[Center]"); g.center(); time.sleep(2)
    p,t=g.query(); print("  %.1f, %.1f" % (p,t))
    print("\n[Pan+15]"); g.move_to(pan=15); time.sleep(2)
    p,t=g.query(); print("  %.1f, %.1f" % (p,t))
    print("\n[Pan-15]"); g.move_to(pan=-15); time.sleep(2)
    p,t=g.query(); print("  %.1f, %.1f" % (p,t))
    print("\n[Pan 0]"); g.move_to(pan=0); time.sleep(2)
    print("\n[Tilt+10]"); g.move_to(tilt=10); time.sleep(2)
    p,t=g.query(); print("  %.1f, %.1f" % (p,t))
    print("\n[Tilt-10]"); g.move_to(tilt=-10); time.sleep(2)
    p,t=g.query(); print("  %.1f, %.1f" % (p,t))
    print("\n[Center]"); g.center(); time.sleep(2)
    p,t=g.query(); print("  %.1f, %.1f" % (p,t))
    g.close(); print("\nDone!")
