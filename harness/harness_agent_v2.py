"""
MedVision Harness - Main Agent Loop v2
Kalman filter + PID control + smart search + YOLO integration.
~280 lines.
"""
import os, sys, time, json, math
import numpy as np
sys.path.insert(0, "/opt/medvision")
from harness_gimbal import GimbalTool
from harness_camera import CameraTool
from harness_detect import DetectTool
from harness_angle import AngleTool

# Optional: YOLO detector
try:
    from harness_detect_yolo import YOLODetector
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


# ── Kalman Filter for target tracking ──
class KalmanTracker:
    """Simple Kalman filter for 2D position + velocity estimation."""

    def __init__(self, dt=0.033):
        self.dt = dt
        # State: [x, y, vx, vy]
        self.x = np.zeros(4)
        self.P = np.eye(4) * 500  # High initial uncertainty
        # Transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        # Measurement matrix (observe x, y only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.Q = np.eye(4) * 0.1   # Process noise
        self.R = np.eye(2) * 15.0  # Measurement noise
        self.initialized = False
        self.last_update = 0

    def update(self, z):
        """Innovation step with measurement [x, y]."""
        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            self.last_update = time.time()
            return self.x[:2]

        # Predict
        dt = time.time() - self.last_update
        if dt > 0:
            self.F[0, 2] = dt
            self.F[1, 3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        z_arr = np.array(z, dtype=float)
        y = z_arr - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        self.last_update = time.time()
        return self.x[:2]

    def predict(self, seconds_ahead=0.0):
        """Predict future position."""
        if not self.initialized:
            return None
        dt = seconds_ahead
        F = self.F.copy()
        F[0, 2] = dt
        F[1, 3] = dt
        x_pred = F @ self.x
        return x_pred[:2]

    def get_velocity(self):
        return (self.x[2], self.x[3])

    def reset(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 500
        self.initialized = False


# ── PID Controller ──
class PIDController:
    """PID controller with anti-windup and output clamping."""

    def __init__(self, kp=0.06, ki=0.002, kd=0.01, imax=50, out_max=3.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.imax = imax
        self.out_max = out_max
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = now - self.prev_time
        if dt <= 0:
            dt = 0.001

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup
        self.integral += error * dt
        self.integral = max(-self.imax, min(self.imax, self.integral))
        i_term = self.ki * self.integral

        # Derivative
        if self.prev_time > 0:
            d_term = self.kd * (error - self.prev_error) / dt
        else:
            d_term = 0

        self.prev_error = error
        self.prev_time = now

        output = p_term + i_term + d_term
        return max(-self.out_max, min(self.out_max, output))

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()


# ── Main Agent ──
class Agent:
    STATE_IDLE = "IDLE"
    STATE_SEARCHING = "SEARCHING"
    STATE_TRACKING = "TRACKING"
    STATE_PREDICTING = "PREDICTING"

    def __init__(self, use_yolo=False):
        config_path = os.path.join(os.path.dirname(__file__), "config", "default.json")
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.gimbal = GimbalTool(self.cfg["gimbal"])
        self.camera = CameraTool(self.cfg["camera"])
        self.detect = DetectTool(self.cfg["detection"])
        self.angle = AngleTool(self.cfg["angle"])

        # Optional YOLO
        self.yolo = None
        if use_yolo and HAS_YOLO:
            model_path = os.path.join(os.path.dirname(__file__), "..", "yolov5s.onnx")
            self.yolo = YOLODetector(model_path, conf_thresh=0.4)

        # Kalman + PID
        self.kalman = KalmanTracker()
        self.pid_pan = PIDController(kp=0.06, ki=0.002, kd=0.01, out_max=3.0)
        self.pid_tilt = PIDController(kp=0.05, ki=0.001, kd=0.008, out_max=2.5)

        # State
        self.state = self.STATE_IDLE
        self.tracking = False
        self.angle_baseline = None
        self.lost_count = 0
        self.max_lost = 15  # frames before search
        self.frame_count = 0
        self.search_angle = 0  # current search sweep angle
        self.search_dir = 1   # sweep direction

        # Stats
        self.fps = 0
        self.fps_counter = 0
        self.fps_time = time.time()

        # Log
        self.log_dir = self.cfg.get("log_dir", "/opt/medvision/harness/logs")
        os.makedirs(self.log_dir, exist_ok=True)

        print("[Agent v2] Initialized")
        print("[Agent v2] Gimbal: pan=%s tilt=%s" % (self.gimbal.query()))
        print("[Agent v2] Camera: %s" % ("OK" if self.camera.is_open() else "FAIL"))
        print("[Agent v2] YOLO: %s" % ("ON" if self.yolo else "OFF"))

    def _update_fps(self):
        self.fps_counter += 1
        now = time.time()
        if now - self.fps_time >= 1.0:
            self.fps = self.fps_counter / (now - self.fps_time)
            self.fps_counter = 0
            self.fps_time = now

    def _search_sweep(self):
        """Autonomous search: gentle sweep pattern."""
        self.search_angle += 5 * self.search_dir
        if abs(self.search_angle) >= 30:
            self.search_dir *= -1
        pan, tilt = self.gimbal.query()
        new_pan = pan + 5 * self.search_dir
        new_pan = max(-60, min(60, new_pan))
        self.gimbal.move_to(pan=new_pan, tilt=tilt)
        time.sleep(0.15)

    def cmd_track(self, on=True):
        self.tracking = on
        if on:
            self.state = self.STATE_TRACKING
            self.kalman.reset()
            self.pid_pan.reset()
            self.pid_tilt.reset()
            self.lost_count = 0
        else:
            self.state = self.STATE_IDLE
        print("[Track] %s" % ("STARTED" if on else "STOPPED"))

    def cmd_scan(self):
        """Smart scan: rotate to find target, return best position."""
        print("[Scan] Scanning...")
        best = (0, 0, 0)
        for p in range(-30, 31, 10):
            for t in range(10, 41, 5):
                self.gimbal.move_to(pan=p, tilt=t)
                time.sleep(0.25)
                bgr = self.camera.capture()
                if bgr is None:
                    continue
                r = self.detect.find(bgr)
                if r and r[2] > best[2]:
                    best = (p, t, r[2])
        if best[2] > 0:
            self.gimbal.move_to(pan=best[0], tilt=best[1])
            print("[Scan] Found at pan=%d tilt=%d" % (best[0], best[1]))
        else:
            print("[Scan] Not found")
        return best

    def cmd_calibrate(self):
        print("[Calib] Calibrating baseline...")
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
            print("[Calib] FAILED")

    def cmd_status(self):
        p, t = self.gimbal.query()
        print("=== Status v2 ===")
        print("State: %s | Tracking: %s" % (self.state, self.tracking))
        print("Gimbal: pan=%.1f tilt=%.1f" % (p, t))
        print("FPS: %.1f | Frames: %d" % (self.fps, self.frame_count))
        print("Lost: %d/%d" % (self.lost_count, self.max_lost))
        vx, vy = self.kalman.get_velocity()
        print("Kalman vel: vx=%.1f vy=%.1f" % (vx, vy))
        if self.angle_baseline is not None:
            print("Angle baseline: %.1f deg" % self.angle_baseline)

    def run(self):
        """Main tracking loop with Kalman + PID + smart search."""
        print("\n[Agent v2] Tracking started. Ctrl+C to stop.\n")

        try:
            while True:
                bgr = self.camera.capture()
                if bgr is None:
                    time.sleep(0.005)
                    continue

                self.frame_count += 1
                self._update_fps()

                # Detect: try color first, fallback to YOLO
                result = self.detect.find(bgr)
                yolo_result = None
                if self.yolo and not result:
                    yolo_result = self.yolo.detect_largest(bgr)
                    if yolo_result:
                        result = yolo_result[:3]  # (cx, cy, area)

                if result:
                    cx, cy, area = result
                    self.lost_count = 0

                    # Kalman update
                    est = self.kalman.update([cx, cy])

                    # PID compute
                    err_x = cx - 320
                    err_y = cy - 240
                    dpan = self.pid_pan.compute(err_x)
                    dtilt = self.pid_tilt.compute(err_y)

                    if self.tracking:
                        self.gimbal.nudge(dpan, dtilt)

                    # Angle measurement
                    relative_angle = self.angle.measure(bgr)

                    # State
                    self.state = self.STATE_TRACKING

                    if self.frame_count % 10 == 0:
                        gp, gt = self.gimbal.query()
                        vx, vy = self.kalman.get_velocity()
                        ang = "%.1f" % relative_angle if relative_angle else "N/A"
                        cls = yolo_result[4] if yolo_result else "color"
                        print("F%04d pos=(%d,%d) vel=(%.1f,%.1f) angle=%s "
                              "gimbal=(%+.1f,%+.1f) cls=%s FPS:%.0f" %
                              (self.frame_count, cx, cy, vx, vy, ang,
                               gp, gt, cls, self.fps))
                else:
                    self.lost_count += 1

                    # Predict with Kalman
                    if self.lost_count < self.max_lost and self.kalman.initialized:
                        self.state = self.STATE_PREDICTING
                        pred = self.kalman.predict(seconds_ahead=0.2)
                        if pred is not None and self.tracking:
                            err_x = pred[0] - 320
                            err_y = pred[1] - 240
                            dpan = self.pid_pan.compute(err_x)
                            dtilt = self.pid_tilt.compute(err_y)
                            self.gimbal.nudge(dpan * 0.5, dtilt * 0.5)
                    else:
                        # Search mode
                        self.state = self.STATE_SEARCHING
                        if self.tracking:
                            self._search_sweep()

                    if self.frame_count % 30 == 0:
                        print("F%04d state=%s lost=%d" %
                              (self.frame_count, self.state, self.lost_count))

        except KeyboardInterrupt:
            pass
        finally:
            self.gimbal.center()
            self.gimbal.close()
            self.camera.close()
            if self.yolo:
                self.yolo.close()
            print("\n[Agent v2] Done. %d frames, %.1f avg FPS" %
                  (self.frame_count, self.fps))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MedVision Agent v2")
    parser.add_argument("--yolo", action="store_true", help="Enable YOLO detection")
    parser.add_argument("--scan", action="store_true", help="Scan then exit")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args()

    agent = Agent(use_yolo=args.yolo)

    if args.status:
        agent.cmd_status()
    elif args.scan:
        agent.cmd_scan()
    else:
        agent.cmd_track(on=True)
        agent.run()


if __name__ == "__main__":
    main()
