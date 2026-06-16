"""
MedVision Harness - Agent v3
Full integration: YOLO + Kalman + PID + 3D Scan + 7DoF Pose + Multi-view Angle.
"""
import os, sys, time, json, math, argparse
import numpy as np
sys.path.insert(0, "/opt/medvision")
sys.path.insert(0, "/opt/medvision/harness")
from harness_gimbal import GimbalTool
from harness_camera import CameraTool
from harness_detect import DetectTool
from harness_angle import AngleTool

# Optional modules
try:
    from harness_detect_yolo import YOLODetector
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    from harness_angle_v2 import AngleToolV2
    HAS_ANGLE_V2 = True
except ImportError:
    HAS_ANGLE_V2 = False

try:
    from harness_scan3d import Scanner3D
    HAS_SCAN3D = True
except ImportError:
    HAS_SCAN3D = False

try:
    from harness_pose7dof import PoseEstimator7DoF
    HAS_POSE7DOF = True
except ImportError:
    HAS_POSE7DOF = False


# ── Kalman Filter ──
class KalmanTracker:
    def __init__(self, dt=0.033):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 500
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 15.0
        self.initialized = False
        self.last_update = 0

    def update(self, z):
        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            self.last_update = time.time()
            return self.x[:2].copy()
        dt = time.time() - self.last_update
        if dt > 0:
            self.F[0,2] = dt; self.F[1,3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = np.array(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.last_update = time.time()
        return self.x[:2].copy()

    def predict(self, t=0.1):
        if not self.initialized:
            return None
        F = self.F.copy()
        F[0,2] = t; F[1,3] = t
        return (F @ self.x)[:2].copy()

    def get_velocity(self):
        return (float(self.x[2]), float(self.x[3]))

    def reset(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 500
        self.initialized = False


# ── PID Controller ──
class PIDController:
    def __init__(self, kp=0.06, ki=0.002, kd=0.01, imax=50, out_max=3.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.imax, self.out_max = imax, out_max
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0

    def compute(self, error):
        now = time.time()
        dt = max(now - self.prev_time, 0.0001) if self.prev_time > 0 else 0.001
        p = self.kp * error
        self.integral = max(-self.imax, min(self.imax, self.integral + error * dt))
        i = self.ki * self.integral
        d = self.kd * (error - self.prev_error) / dt if self.prev_time > 0 else 0
        self.prev_error = error
        self.prev_time = now
        return max(-self.out_max, min(self.out_max, p + i + d))

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0


# ── Main Agent ──
class Agent:
    STATE_IDLE = "IDLE"
    STATE_SEARCHING = "SEARCHING"
    STATE_TRACKING = "TRACKING"
    STATE_PREDICTING = "PREDICTING"
    STATE_SCANNING = "SCANNING"

    def __init__(self, use_yolo=False):
        config_path = os.path.join(os.path.dirname(__file__), "config", "default.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.cfg = json.load(f)
        else:
            self.cfg = {}

        self.gimbal = GimbalTool(self.cfg.get("gimbal", {}))
        self.camera = CameraTool(self.cfg.get("camera", {}))
        self.detect = DetectTool(self.cfg.get("detection", {}))
        self.angle = AngleTool(self.cfg.get("angle", {}))

        # Optional modules
        self.yolo = None
        self.angle_v2 = AngleToolV2() if HAS_ANGLE_V2 else None
        self.scanner = Scanner3D() if HAS_SCAN3D else None
        self.pose = PoseEstimator7DoF() if HAS_POSE7DOF else None

        if use_yolo and HAS_YOLO:
            model_path = os.path.join(os.path.dirname(__file__), "..", "yolov5n.onnx")
            self.yolo = YOLODetector(model_path, conf_thresh=0.4)

        # Tracking state
        self.kalman = KalmanTracker()
        self.pid_pan = PIDController(kp=0.06, ki=0.002, kd=0.01, out_max=3.0)
        self.pid_tilt = PIDController(kp=0.05, ki=0.001, kd=0.008, out_max=2.5)

        self.state = self.STATE_IDLE
        self.tracking = False
        self.angle_baseline = None
        self.frame_count = 0
        self.fps = 0
        self.fps_counter = 0
        self.fps_time = time.time()
        self.lost_count = 0
        self.max_lost = 15
        self.search_angle = 0
        self.search_dir = 1

        print("[Agent v3] Modules loaded:")
        print("  YOLO: %s" % ("ON" if self.yolo else "OFF"))
        print("  AngleV2: %s" % ("ON" if self.angle_v2 else "OFF"))
        print("  Scan3D: %s" % ("ON" if self.scanner else "OFF"))
        print("  Pose7DoF: %s" % ("ON" if self.pose else "OFF"))
        print("  Gimbal: pan=%s tilt=%s" % self.gimbal.query())

    def _update_fps(self):
        self.fps_counter += 1
        now = time.time()
        if now - self.fps_time >= 1.0:
            self.fps = self.fps_counter / (now - self.fps_time)
            self.fps_counter = 0
            self.fps_time = now

    def _search_sweep(self):
        self.search_angle += 5 * self.search_dir
        if abs(self.search_angle) >= 30:
            self.search_dir *= -1
        pan, tilt = self.gimbal.query()
        self.gimbal.move_to(pan=pan + 5 * self.search_dir, tilt=tilt)
        time.sleep(0.15)

    # ── Core Commands ──

    def cmd_track(self, on=True):
        self.tracking = on
        if on:
            self.state = self.STATE_TRACKING
            self.kalman.reset()
            self.pid_pan.reset()
            self.pid_tilt.reset()
            self.lost_count = 0
            if self.angle_v2:
                self.angle_v2.begin_session()
        else:
            self.state = self.STATE_IDLE
        print("[Track] %s" % ("STARTED" if on else "STOPPED"))

    def cmd_scan(self):
        """Multi-angle scan for 3D reconstruction."""
        self.state = self.STATE_SCANNING
        print("[Scan] Starting multi-angle scan...")
        angles = [(-20, 20), (-10, 25), (0, 20), (10, 25), (20, 20)]
        best = None
        for pan, tilt in angles:
            self.gimbal.move_to(pan=pan, tilt=tilt)
            time.sleep(0.5)
            bgr = self.camera.capture()
            if bgr is None:
                continue
            # Multi-view angle measurement
            if self.angle_v2:
                angle = self.angle_v2.feed(bgr, "p%d_t%d" % (pan, tilt))
                if angle is not None:
                    print("  View (pan=%d, tilt=%d): angle=%.1f" % (pan, tilt, angle))
            # 3D scan
            if self.scanner and best is None:
                result = self.scanner.scan(bgr, np.zeros((480, 640), dtype=np.float32))
                if result and "center_3d" in result:
                    best = result
                    print("  3D: center=(%.0f,%.0f,%.0f)mm" % tuple(result["center_3d"]))

        if self.angle_v2:
            fused = self.angle_v2.fused_result()
            print("\n[Scan] Fused angle: %.1f degrees" % (fused or 0))
        self.state = self.STATE_IDLE

    def cmd_measure(self):
        """Single measurement with all methods."""
        print("[Measure] Taking measurements...")
        bgr = self.camera.capture()
        if bgr is None:
            print("[Measure] No frame")
            return

        # 2D angle (original)
        a1 = self.angle.measure(bgr)
        print("  [PCA]     Angle: %.1f" % (a1 or 0))

        # 2D angle v2 (convex hull)
        if self.angle_v2:
            a2 = self.angle_v2.measure(bgr, method="convex")
            print("  [Convex]  Angle: %.1f" % (a2 or 0))

        # 7DoF pose
        if self.pose:
            result = self.pose.estimate(bgr)
            if result:
                print("  [7DoF]    Angle: %.1f | Confidence: %.0f%%" %
                      (result.get("angle", 0) or 0, result.get("confidence", 0) * 100))
                print("  [7DoF]    Position: (%.0f, %.0f, %.0f) mm" % tuple(result["position"]))
            else:
                print("  [7DoF]    No instrument detected")

        # YOLO detection
        if self.yolo:
            results = self.yolo.detect(bgr)
            print("  [YOLO]    Detections: %d" % len(results))

    def cmd_status(self):
        p, t = self.gimbal.query()
        print("=== MedVision Agent v3 ===")
        print("State: %s | Tracking: %s" % (self.state, self.tracking))
        print("Gimbal: pan=%.1f tilt=%.1f" % (p, t))
        print("FPS: %.1f | Frames: %d" % (self.fps, self.frame_count))
        print("Modules: YOLO=%s AngleV2=%s Scan3D=%s Pose7DoF=%s" % (
            "ON" if self.yolo else "OFF",
            "ON" if self.angle_v2 else "OFF",
            "ON" if self.scanner else "OFF",
            "ON" if self.pose else "OFF",
        ))

    # ── Main Loop ──

    def run(self):
        print("\n[Agent v3] Tracking started. Ctrl+C to stop.\n")
        fc = 0
        t0 = time.time()

        try:
            while True:
                bgr = self.camera.capture()
                if bgr is None:
                    time.sleep(0.005)
                    continue

                fc += 1
                self._update_fps()

                # Detect
                result = self.detect.find(bgr)
                if not result and self.yolo:
                    yr = self.yolo.detect_largest(bgr)
                    if yr:
                        result = yr[:3]

                if result:
                    cx, cy, area = result
                    self.lost_count = 0

                    # Kalman
                    self.kalman.update([cx, cy])

                    # PID
                    err_x = cx - 320
                    err_y = cy - 240
                    dpan = self.pid_pan.compute(err_x)
                    dtilt = self.pid_tilt.compute(err_y)

                    if self.tracking:
                        self.gimbal.nudge(dpan, dtilt)

                    # Angle
                    if self.angle_v2:
                        self.angle_v2.feed(bgr)
                    else:
                        self.angle.measure(bgr)

                    self.state = self.STATE_TRACKING

                    if fc % 10 == 0:
                        gp, gt = self.gimbal.query()
                        vx, vy = self.kalman.get_velocity()
                        angle = self.angle_v2.fused_result() if self.angle_v2 else None
                        ang = "%.1f" % angle if angle else "N/A"
                        print("F%04d pos=(%d,%d) vel=(%.1f,%.1f) angle=%s gimbal=(%+.1f,%+.1f) FPS:%.0f" %
                              (fc, cx, cy, vx, vy, ang, gp, gt, self.fps))
                else:
                    self.lost_count += 1

                    if self.lost_count < self.max_lost and self.kalman.initialized:
                        self.state = self.STATE_PREDICTING
                        pred = self.kalman.predict(0.2)
                        if pred is not None and self.tracking:
                            dpan = self.pid_pan.compute(pred[0] - 320) * 0.5
                            dtilt = self.pid_tilt.compute(pred[1] - 240) * 0.5
                            self.gimbal.nudge(dpan, dtilt)
                    else:
                        self.state = self.STATE_SEARCHING
                        if self.tracking:
                            self._search_sweep()

                    if fc % 30 == 0:
                        print("F%04d state=%s lost=%d" % (fc, self.state, self.lost_count))

        except KeyboardInterrupt:
            pass
        finally:
            self.gimbal.center()
            self.gimbal.close()
            self.camera.close()
            if self.yolo:
                self.yolo.close()
            print("\n[Agent v3] Done. %d frames, %.1f avg FPS" % (fc, self.fps))


def main():
    parser = argparse.ArgumentParser(description="MedVision Agent v3")
    parser.add_argument("--yolo", action="store_true", help="Enable YOLO")
    parser.add_argument("--scan", action="store_true", help="Multi-angle 3D scan")
    parser.add_argument("--measure", action="store_true", help="Single measurement")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args()

    agent = Agent(use_yolo=args.yolo)

    if args.status:
        agent.cmd_status()
    elif args.scan:
        agent.cmd_scan()
    elif args.measure:
        agent.cmd_measure()
    else:
        agent.cmd_track(on=True)
        agent.run()


if __name__ == "__main__":
    main()
