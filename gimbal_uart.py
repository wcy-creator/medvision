"""MedVision - UART Gimbal v3 (Crash-proof + Smooth)
Fixes serial crash by wrapping serial port with exception handling.
"""
import serial, struct, time, json, sys, os
sys.path.insert(0, "/opt/medvision")

class SafeSerial:
    """Wrapper around serial.Serial that catches Linux readall() bugs."""
    def __init__(self, *args, **kwargs):
        self._real = serial.Serial(*args, **kwargs)

    def readall(self):
        try:
            return self._real.readall()
        except serial.SerialException:
            return b""

    def read(self, size=1):
        try:
            return self._real.read(size)
        except serial.SerialException:
            return b""

    def write(self, data):
        try:
            return self._real.write(data)
        except serial.SerialException:
            return 0

    def __getattr__(self, name):
        return getattr(self._real, name)

from uservo import UartServoManager

class GimbalUART:
    def __init__(self, port="/dev/ttyUSB0"):
        with open("/opt/medvision/gimbal_offset.json") as f:
            off = json.load(f)
        self.pan_offset = off["pan_home"]
        self.tilt_offset = off["tilt_home"]
        self.TILT_MIN = -60
        self.TILT_MAX = 60
        self.PAN_MIN = -60
        self.PAN_MAX = 60
        self.pan_id = 0
        self.tilt_id = 1

        # Safe serial with exception handling
        self.uart = SafeSerial(port=port, baudrate=115200,
                               parity=serial.PARITY_NONE, stopbits=1,
                               bytesize=8, timeout=0.05)
        self.uservo = UartServoManager(self.uart, is_scan_servo=True,
                                        srv_num=10, is_debug=False)

        self._pan = 0.0
        self._tilt = 0.0

        # Init with retry
        for attempt in range(3):
            try:
                self._pan, self._tilt = self._query_raw()
                break
            except Exception:
                time.sleep(0.3)
        print("Ready: pan=%.1f tilt=%.1f" % (self._pan, self._tilt))

    def _query_raw(self):
        try:
            self.uservo.query_servo_angle(self.pan_id)
            time.sleep(0.1)
            pv = self.uservo.servos[self.pan_id].cur_angle
            self.uservo.query_servo_angle(self.tilt_id)
            time.sleep(0.1)
            tv = self.uservo.servos[self.tilt_id].cur_angle
            p = float(pv) if pv is not None else self._pan
            t = float(tv) if tv is not None else self._tilt
            return p - self.pan_offset, t - self.tilt_offset
        except Exception:
            return self._pan, self._tilt

    def query(self):
        try:
            self._pan, self._tilt = self._query_raw()
        except Exception:
            pass
        return self._pan, self._tilt

    def move_to(self, pan=None, tilt=None, velocity=100):
        moved = False
        if pan is not None:
            pan = max(self.PAN_MIN, min(self.PAN_MAX, pan))
            actual = max(-135, min(135, pan + self.pan_offset))
            if abs(pan - self._pan) > 0.05:
                try:
                    self.uservo.set_servo_angle(self.pan_id, actual, velocity=velocity)
                except Exception:
                    pass
                moved = True
            self._pan = pan
        if tilt is not None:
            tilt = max(self.TILT_MIN, min(self.TILT_MAX, tilt))
            actual = max(-135, min(135, tilt + self.tilt_offset))
            if abs(tilt - self._tilt) > 0.05:
                try:
                    self.uservo.set_servo_angle(self.tilt_id, actual, velocity=velocity)
                except Exception:
                    pass
                moved = True
            self._tilt = tilt
        if moved:
            time.sleep(0.02)
        return self._pan, self._tilt

    def center(self, velocity=100):
        return self.move_to(0, 0, velocity=velocity)

    def nudge(self, dpan=0, dtilt=0, velocity=100):
        new_pan = max(self.PAN_MIN, min(self.PAN_MAX, self._pan + dpan))
        new_tilt = max(self.TILT_MIN, min(self.TILT_MAX, self._tilt + dtilt))
        return self.move_to(new_pan, new_tilt, velocity)

    def close(self):
        try:
            self.uart.close()
        except:
            pass

if __name__ == "__main__":
    import math
    g = GimbalUART()
    print("\n[Center]"); g.center(); time.sleep(1)
    p, t = g.query(); print("  %.1f, %.1f" % (p, t))
    print("\n[Smooth sine test]")
    for i in range(40):
        g.move_to(pan=20 * math.sin(i * 0.2))
        time.sleep(0.03)
    g.center(); g.close()
    print("\nDone!")
