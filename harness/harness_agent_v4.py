"""
MedVision Agent v4 - OPTIMIZED for maximum tracking FPS.
Key optimizations:
1. Downscale to 320x240 for detection (4x fewer pixels)
2. ROI-based search (only search near last position)
3. Skip frames with Kalman prediction
4. Simplified morphology (fewer iterations)
"""
import os, sys, time
import numpy as np
import cv2
sys.path.insert(0, "/opt/medvision")
sys.path.insert(0, "/opt/medvision/harness")
from harness_gimbal import GimbalTool
from harness_detect import DetectTool


# ── Camera (picamera2 video mode) ──
class Camera:
    def __init__(self, w=640, h=480):
        self.ok = False
        self.cam = None
        try:
            from picamera2 import Picamera2
            self.cam = Picamera2()
            cfg = self.cam.create_video_configuration(main={"size": (w, h), "format": "RGB888"})
            self.cam.configure(cfg)
            self.cam.start()
            time.sleep(0.5)  # Let camera stabilize
            # Warmup: capture a few frames
            for _ in range(3):
                self.cam.capture_array()
            self.ok = True
            print("[Camera] picamera2 %dx%d" % (w, h))
        except Exception as e:
            print("[Camera] FAIL: %s" % e)

    def read(self):
        if not self.ok: return None
        try:
            arr = self.cam.capture_array()
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except: return None

    def close(self):
        if self.cam: self.cam.stop()


# ── Fast Color Detector (ROI + downscaled) ──
class FastDetector:
    """Optimized color detection: 320x240 + ROI search."""
    def __init__(self, low=(0,100,100), high=(10,255,255), min_area=100):
        self.low = np.array(low, dtype=np.uint8)
        self.high = np.array(high, dtype=np.uint8)
        self.min_area = min_area
        self.last_pos = None  # (cx, cy) in full-res coordinates
        self.roi_margin = 160  # Search radius around last position

    def find(self, bgr):
        h, w = bgr.shape[:2]
        small_h, small_w = h // 2, w // 2

        # Downscale
        small = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # ROI: if we have a last position, only search nearby
        if self.last_pos is not None:
            lx = int(self.last_pos[0] / 2)
            ly = int(self.last_pos[1] / 2)
            margin = self.roi_margin // 2
            x1 = max(0, lx - margin)
            y1 = max(0, ly - margin)
            x2 = min(small_w, lx + margin)
            y2 = min(small_h, ly + margin)

            # Create ROI mask
            roi_mask = np.zeros((small_h, small_w), dtype=np.uint8)
            roi_mask[y1:y2, x1:x2] = 255

            # Detect in ROI
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.low, self.high)
            mask = cv2.bitwise_and(mask, roi_mask)

            # Quick morphology (1 iteration only)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=1)
        else:
            # Full frame detection
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.low, self.high)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=1)

        # Find contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        # Find largest
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            return None

        M = cv2.moments(c)
        if M["m00"] == 0:
            return None

        # Convert back to full-res coordinates
        cx = int(M["m10"] / M["m00"] * 2)
        cy = int(M["m01"] / M["m00"] * 2)
        area_full = int(area * 4)

        self.last_pos = (cx, cy)
        return (cx, cy, area_full)


# ── Kalman ──
class Kalman:
    def __init__(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 500
        self.F = np.array([[1,0,.033,0],[0,1,0,.033],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 15.0
        self.ok = False; self.t0 = 0

    def update(self, z):
        if not self.ok:
            self.x[:2] = z; self.ok = True; self.t0 = time.time()
            return self.x[:2].copy()
        dt = max(time.time() - self.t0, 0.001)
        self.F[0,2] = dt; self.F[1,3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = np.array(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.t0 = time.time()
        return self.x[:2].copy()

    def predict(self, t=0.2):
        if not self.ok: return None
        F = self.F.copy(); F[0,2]=t; F[1,3]=t
        return (F @ self.x)[:2].copy()

    def vel(self):
        return (float(self.x[2]), float(self.x[3]))

    def reset(self):
        self.x = np.zeros(4); self.P = np.eye(4)*500; self.ok = False


# ── PID ──
class PID:
    def __init__(self, kp=0.08, ki=0.003, kd=0.015, mx=3.5):
        self.kp=kp; self.ki=ki; self.kd=kd; self.mx=mx
        self.I=0; self.pe=0; self.pt=0
    def __call__(self, e):
        now = time.time()
        dt = max(now-self.pt, .0001) if self.pt>0 else .001
        p = self.kp * e
        self.I = max(-50, min(50, self.I + e*dt))
        i = self.ki * self.I
        d = self.kd * (e-self.pe)/dt if self.pt>0 else 0
        self.pe=e; self.pt=now
        return max(-self.mx, min(self.mx, p+i+d))
    def reset(self):
        self.I=0; self.pe=0; self.pt=0


# ── Main ──
def main():
    print("="*50)
    print("  MedVision Agent v4 - OPTIMIZED")
    print("="*50)

    cam = Camera()
    gimbal = GimbalTool({})
    det = FastDetector(low=(0,100,100), high=(10,255,255), min_area=80)
    K = Kalman()
    pid_p = PID(kp=0.08, ki=0.003, kd=0.015, mx=3.5)
    pid_t = PID(kp=0.06, ki=0.002, kd=0.012, mx=3.0)

    gimbal.center()
    time.sleep(0.5)

    fc = 0; t0 = time.time()
    lost = 0; tracking = True
    detect_interval = 3  # Detect every N frames, predict otherwise

    print("%-6s %-12s %-12s %-8s %-6s" % ("Frame", "Position", "Velocity", "FPS", "State"))
    print("-" * 55)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.002); continue

            fc += 1
            fps = fc / max(time.time()-t0, 0.001)
            state = "TRACK"

            # Skip detection on some frames (use Kalman prediction)
            do_detect = (fc % detect_interval == 0) or (not K.ok)

            if do_detect:
                r = det.find(frame)
            else:
                r = None

            if r:
                cx, cy, area = r
                lost = 0
                K.update([cx, cy])

                # Only send gimbal command if error is significant (avoid UART bottleneck)
                err_x = cx - 320
                err_y = cy - 240
                if tracking and (abs(err_x)>15 or abs(err_y)>15):
                    dpan = pid_p(err_x)
                    dtilt = pid_t(err_y)
                    gimbal.nudge_fast(dpan, dtilt)

            else:
                lost += 1
                state = "PREDICT" if lost < 20 else "SEARCH"

                if lost < 20 and K.ok:
                    pred = K.predict(0.2)
                    if pred is not None and tracking:
                        gimbal.nudge_fast(pid_p(pred[0]-320)*.5, pid_t(pred[1]-240)*.5)
                elif lost >= 20 and tracking:
                    # Reset ROI to search full frame
                    det.last_pos = None
                    p, t = gimbal.query()
                    gimbal.g.move_to_fast(pan=p+5, tilt=t)
                    time.sleep(0.08)

            if fc % 10 == 0:
                gp, gt = gimbal.query()
                vx, vy = K.vel()
                pos_str = "(%d,%d)" % (cx, cy) if r else "NONE"
                print("F%04d %-12s (%.1f,%.1f)  %.1f  %s" %
                      (fc, pos_str, vx, vy, fps, state))

    except KeyboardInterrupt:
        pass
    finally:
        gimbal.center(); gimbal.close(); cam.close()
        e = time.time()-t0
        print("\n%d frames / %.1fs = %.1f FPS" % (fc, e, fc/max(e,.001)))


if __name__ == "__main__":
    main()
