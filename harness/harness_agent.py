"""
MedVision Harness - Main Agent Loop
Orchestrates tools, manages state, provides CLI interface.
~150 lines core.
"""
import os, sys, time, json
sys.path.insert(0, "/opt/medvision")
from harness_gimbal import GimbalTool
from harness_camera import CameraTool
from harness_detect import DetectTool
from harness_angle import AngleTool


class Agent:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "config", "default.json")
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.gimbal = GimbalTool(self.cfg["gimbal"])
        self.camera = CameraTool(self.cfg["camera"])
        self.detect = DetectTool(self.cfg["detection"])
        self.angle = AngleTool(self.cfg["angle"])
        self.state = "IDLE"
        self.tracking = False
        self.angle_baseline = None
        self.log_dir = self.cfg.get("log_dir", "/opt/medvision/harness/logs")
        os.makedirs(self.log_dir, exist_ok=True)

        print("[Agent] Initialized")
        print("[Agent] Gimbal: pan=%s tilt=%s" % self.gimbal.query())
        print("[Agent] Camera: %s" % ("OK" if self.camera.is_open() else "FAIL"))

    def cmd_move(self, pan=None, tilt=None):
        """Move gimbal to position."""
        if pan is not None or tilt is not None:
            self.gimbal.move_to(pan=pan, tilt=tilt)
            p, t = self.gimbal.query()
            print("[Gimbal] Moved to pan=%.1f tilt=%.1f" % (p, t))
        else:
            self.gimbal.center()
            print("[Gimbal] Centered")

    def cmd_scan(self):
        """Scan for target."""
        print("[Scan] Scanning...")
        best = (0, 0, 0)
        for p in range(-30, 31, 10):
            for t in range(15, 41, 5):
                self.gimbal.move_to(pan=p, tilt=t)
                time.sleep(0.3)
                bgr = self.camera.capture()
                if bgr is None:
                    continue
                r = self.detect.find(bgr)
                if r and r[2] > best[2]:
                    best = (p, t, r[2])
        if best[2] > 0:
            self.gimbal.move_to(pan=best[0], tilt=best[1])
            time.sleep(0.5)
            print("[Scan] Found at pan=%d tilt=%d" % (best[0], best[1]))
        else:
            print("[Scan] Not found")
        return best

    def cmd_calibrate(self):
        """Calibrate angle baseline."""
        print("[Calib] Calibrating baseline (hold clip closed)...")
        angles = []
        for i in range(self.cfg["angle"]["calib_frames"]):
            bgr = self.camera.capture()
            if bgr is not None:
                a = self.angle.measure(bgr)
                if a is not None:
                    angles.append(a)
            time.sleep(0.05)
        if angles:
            self.angle_baseline = float(np.median(angles))
            print("[Calib] Baseline: %.1f deg (%d samples)" % (self.angle_baseline, len(angles)))
        else:
            print("[Calib] FAILED - no angle detected")

    def cmd_track(self, on=True):
        """Start/stop tracking."""
        self.tracking = on
        print("[Track] %s" % ("STARTED" if on else "STOPPED"))

    def cmd_status(self):
        """Print current status."""
        p, t = self.gimbal.query()
        bgr = self.camera.capture()
        detected = False
        pos = None
        if bgr is not None:
            r = self.detect.find(bgr)
            if r:
                detected = True
                pos = r
        print("=== Status ===")
        print("State: %s | Tracking: %s" % (self.state, self.tracking))
        print("Gimbal: pan=%.1f tilt=%.1f" % (p, t))
        print("Target: %s" % ("(%d,%d) area=%d" % (pos[0], pos[1], pos[2]) if pos else "NONE"))
        if self.angle_baseline is not None:
            print("Angle baseline: %.1f deg" % self.angle_baseline)

    def run(self):
        """Main tracking loop."""
        print("\n[Agent] Tracking started. Ctrl+C to stop.\n")
        fc = 0
        t0 = time.time()

        try:
            while True:
                bgr = self.camera.capture()
                if bgr is None:
                    time.sleep(0.005)
                    continue

                fc += 1
                fps = fc / (time.time() - t0) if (time.time() - t0) > 0 else 0

                result = self.detect.find(bgr)

                if result:
                    cx, cy, area = result
                    err_x = cx - 320
                    err_y = cy - 240

                    if self.tracking and (abs(err_x) > 20 or abs(err_y) > 20):
                        dpan = max(-2, min(2, 0.05 * err_x))
                        dtilt = max(-2, min(2, 0.05 * err_y))
                        self.gimbal.nudge(dpan, dtilt)

                    relative_angle = self.angle.measure(bgr, prev=None)

                    if fc % 10 == 0:
                        gp, gt = self.gimbal.query()
                        ang = "%.1f" % relative_angle if relative_angle else "N/A"
                        print("F%04d pos=(%d,%d) angle=%s gimbal=(%+.1f,%+.1f) FPS:%.0f" %
                              (fc, cx, cy, ang, gp, gt, fps))
                else:
                    if fc % 30 == 0:
                        print("F%04d NO TARGET" % fc)

        except KeyboardInterrupt:
            pass
        finally:
            self.gimbal.center()
            self.gimbal.close()
            self.camera.close()
            print("\n[Agent] Done. %d frames" % fc)


def main():
    agent = Agent()
    agent.run()


if __name__ == "__main__":
    main()
