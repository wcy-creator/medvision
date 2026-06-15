"""Gimbal Tool - Auto-detect serial port (Linux/Windows)."""
import os, sys, time, glob
sys.path.insert(0, "/opt/medvision")


def find_serial_port():
    """Auto-detect CH340 serial port."""
    # Linux
    for pattern in ["/dev/ttyUSB*", "/dev/ttyACM*"]:
        ports = glob.glob(pattern)
        if ports:
            return ports[0]
    # Windows
    for pattern in ["COM3", "COM4", "COM5", "COM6", "COM7", "COM8"]:
        if os.path.exists("\\\\.\\%s" % pattern):
            return pattern
    # macOS
    for pattern in ["/dev/cu.usbserial*", "/dev/cu.wch*"]:
        ports = glob.glob(pattern)
        if ports:
            return ports[0]
    return None


class GimbalTool:
    def __init__(self, config):
        port = config.get("port") or find_serial_port()
        if not port:
            raise RuntimeError("No serial port found! Connect CH340 USB adapter.")
        print("[Gimbal] Port: %s" % port)

        from gimbal_uart import GimbalUART
        self.g = GimbalUART(port=port)
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
