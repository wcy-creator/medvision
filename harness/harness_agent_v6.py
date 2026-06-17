"""
MedVision Agent v6 - Full Integration
EKF + Adaptive PID + ByteTrack + NCNN + Environment Perception
"""
import os, sys, time
import numpy as np
sys.path.insert(0, "/opt/medvision")
sys.path.insert(0, "/opt/medvision/harness")

from harness_agent_v5 import CameraThread
from harness_gimbal import GimbalTool
from harness_detect import DetectTool
from harness_tracker_v2 import ExtendedKalmanFilter, AdaptivePID
from harness_env import EnvDetector, SceneAnalyzer

# Optional modules
try:
    from harness_detect_ncnn import NcnNDetector
    HAS_NCNN = True
except ImportError:
    HAS_NCNN = False

try:
    from harness_byte_track import ByteTracker
    HAS_BYTETRACK = True
except ImportError:
    HAS_BYTETRACK = False

try:
    from harness_angle_v2 import AngleToolV2
    HAS_ANGLE_V2 = True
except ImportError:
    HAS_ANGLE_V2 = False

try:
    from harness_pose7dof import PoseEstimator7DoF
    HAS_POSE = True
except ImportError:
    HAS_POSE = False


class AgentV6:
    """Full-integration agent with all Phase 1+2 modules."""

    def __init__(self):
        # Core
        self.camera = CameraThread()
        self.gimbal = GimbalTool({})
        self.detect_color = DetectTool({})

        # Phase 1: NCNN detection
        self.detector = NcnNDetector() if HAS_NCNN else None

        # Phase 2: EKF + Adaptive PID
        self.ekf = ExtendedKalmanFilter()
        self.pid_pan = AdaptivePID(kp=0.08, ki=0.003, kd=0.015, mx=3.5)
        self.pid_tilt = AdaptivePID(kp=0.06, ki=0.002, kd=0.012, mx=3.0)

        # Phase 2: ByteTrack
        self.tracker = ByteTracker(track_thresh=0.3) if HAS_BYTETRACK else None

        # Optional modules
        self.angle = AngleToolV2() if HAS_ANGLE_V2 else None
        self.pose = PoseEstimator7DoF() if HAS_POSE else None
        self.env = EnvDetector()
        self.scene = SceneAnalyzer()

        # Config
        self.gimbal.g.TILT_MIN = -30
        self.gimbal.g.TILT_MAX = 60
        self.gimbal.g.PAN_MIN = -45
        self.gimbal.g.PAN_MAX = 45

        # State
        self.lost_count = 0
        self.max_lost = 20
        self.frame_count = 0

        print("[Agent v6] Full integration ready")
        print("  NCNN: %s | ByteTrack: %s | AngleV2: %s | Pose: %s" % (
            "ON" if self.detector else "OFF",
            "ON" if self.tracker else "OFF",
            "ON" if self.angle else "OFF",
            "ON" if self.pose else "OFF"))

    def scan_environment(self):
        """Scan environment from multiple angles."""
        print("[Scan] Starting multi-angle scan...")
        self.gimbal.move_to(tilt=-30, speed=200)
        time.sleep(1.5)

        bgr = self.camera.read()
        if bgr is None:
            return

        # Detect people
        if self.detector:
            person = self.detector.detect_person(bgr)
            if person:
                print("[Scan] Person at (%d,%d) conf=%.2f" % (person[0], person[1], person[3]))

        # Environment analysis
        objects = self.env.detect_objects(bgr)
        result = self.scene.analyze(bgr, objects=objects)
        print("[Scan] Lighting: %s | Scene: %s | Objects: %d" % (
            result.get("lighting"), result.get("scene_type"), len(objects)))

    def run_tracking(self, duration=15):
        """Run tracking with EKF + ByteTrack + Adaptive PID."""
        print("[Track] Starting EKF tracking (%ds)..." % duration)

        self.gimbal.move_to(tilt=-30, speed=200)
        time.sleep(1.5)

        fc = 0; t0 = time.time(); lost = 0
        try:
            while time.time() - t0 < duration:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.001); continue

                fc += 1
                fps = fc / max(time.time() - t0, 0.001)

                # Detect (every 3rd frame)
                if fc % 3 == 0 and self.detector:
                    dets = self.detector.detect(frame)
                    # Filter for person class
                    person_dets = [{'bbox': d['bbox'], 'score': d['confidence'], 'class_id': d['class_id']}
                                   for d in dets if d['class_id'] == 0]

                    if self.tracker and person_dets:
                        tracks = self.tracker.update(person_dets)
                        active = self.tracker.get_tracks(class_filter={0})
                        if active:
                            t = active[0]
                            cx, cy = t.get_center()
                            self.ekf.update([cx, cy])
                            lost = 0

                            err_x = cx - 320
                            err_y = cy - 240
                            if abs(err_x) > 15 or abs(err_y) > 15:
                                self.gimbal.nudge_fast(
                                    self.pid_pan(err_x),
                                    self.pid_tilt(err_y)
                                )
                        else:
                            lost += 1
                    elif person_dets:
                        # No tracker, use raw detection
                        d = person_dets[0]
                        cx = (d['bbox'][0] + d['bbox'][2]) / 2
                        cy = (d['bbox'][1] + d['bbox'][3]) / 2
                        self.ekf.update([cx, cy])
                        lost = 0
                        err_x = cx - 320; err_y = cy - 240
                        if abs(err_x) > 15 or abs(err_y) > 15:
                            self.gimbal.nudge_fast(self.pid_pan(err_x), self.pid_tilt(err_y))
                    else:
                        lost += 1
                else:
                    # Use Kalman prediction
                    if lost < self.max_lost and self.ekf.ok:
                        pred = self.ekf.predict(0.2)
                        if pred is not None:
                            self.gimbal.nudge_fast(
                                self.pid_pan(pred[0]-320)*0.5,
                                self.pid_tilt(pred[1]-240)*0.5
                            )
                    else:
                        lost += 1

                if fc % 20 == 0:
                    vel = self.ekf.vel()
                    gp, gt = self.gimbal.query()
                    print("F%04d pan=%+.1f v=(%.0f,%.0f) lost=%d %.1fFPS" %
                          (fc, gp, vel[0], vel[1], lost, fps))

        except KeyboardInterrupt:
            pass

        self.gimbal.center()
        e = time.time() - t0
        print("\n[Track] %d frames / %.1fs = %.1f FPS" % (fc, e, fc/max(e, .001)))

    def measure_angle(self):
        """Take angle measurement with all methods."""
        if not self.angle:
            print("[Angle] AngleV2 not available")
            return

        bgr = self.camera.read()
        if bgr is None:
            return

        # Multi-view measurement
        self.angle.begin_session()
        positions = [(-20, -25), (0, -30), (20, -25)]

        for pan, tilt in positions:
            self.gimbal.move_to(pan=pan, tilt=tilt, speed=200)
            time.sleep(0.5)
            bgr = self.camera.read()
            if bgr is not None:
                a = self.angle.feed(bgr, "pan%d" % pan)
                if a:
                    print("  pan=%+d: angle=%.1f" % (pan, a))

        fused = self.angle.fused_result()
        print("\n[Angle] Fused result: %.1f degrees" % (fused or 0))
        print("  (Multi-view fusion eliminates projection error)")

    def status(self):
        """Print system status."""
        p, t = self.gimbal.query()
        print("=" * 50)
        print("  MedVision Agent v6 Status")
        print("=" * 50)
        print("Gimbal: pan=%.1f tilt=%.1f" % (p, t))
        print("Modules:")
        print("  NCNN:     %s" % ("ON" if self.detector else "OFF"))
        print("  ByteTrack:%s" % ("ON" if self.tracker else "OFF"))
        print("  EKF:      ON")
        print("  AdaptivePID: ON")
        print("  AngleV2:  %s" % ("ON" if self.angle else "OFF"))
        print("  Pose7DoF: %s" % ("ON" if self.pose else "OFF"))
        print("  EnvScan:  ON")
        print("=" * 50)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MedVision Agent v6")
    parser.add_argument("--scan", action="store_true", help="Scan environment")
    parser.add_argument("--track", type=int, default=0, help="Track for N seconds")
    parser.add_argument("--angle", action="store_true", help="Measure angle")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args()

    agent = AgentV6()

    if args.status:
        agent.status()
    elif args.scan:
        agent.scan_environment()
    elif args.track > 0:
        agent.run_tracking(args.track)
    elif args.angle:
        agent.measure_angle()
    else:
        agent.status()


if __name__ == "__main__":
    main()
