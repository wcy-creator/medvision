"""
MedVision Agent v5 - ULTRA FAST
INT8 YOLO + Multi-threaded pipeline + picamera2 optimized
"""
import os, sys, time, queue, threading
import numpy as np
import cv2
sys.path.insert(0, "/opt/medvision")
sys.path.insert(0, "/opt/medvision/harness")
from harness_gimbal import GimbalTool
from harness_detect import DetectTool


# ── Camera (picamera2, dedicated thread) ──
class CameraThread:
    """Camera capture in separate thread for max FPS."""
    def __init__(self, w=640, h=480):
        self.frame_queue = queue.Queue(maxsize=2)
        self.ok = False
        self._stop = False
        self._thread = None
        try:
            from picamera2 import Picamera2
            self.cam = Picamera2()
            cfg = self.cam.create_video_configuration(
                main={"size": (w, h), "format": "RGB888"}
            )
            self.cam.configure(cfg)
            self.cam.start()
            self.ok = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            print("[Camera] picamera2 thread started")
        except Exception as e:
            print("[Camera] FAIL: %s" % e)

    def _capture_loop(self):
        while not self._stop:
            try:
                arr = self.cam.capture_array()
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                if self.frame_queue.full():
                    self.frame_queue.get()  # Drop oldest
                self.frame_queue.put(frame)
            except Exception:
                pass

    def read(self):
        try:
            return self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def close(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)
        if hasattr(self, 'cam'):
            self.cam.stop()


# ── Fast Color Detector ──
class FastDetector:
    def __init__(self, low=(0,100,100), high=(10,255,255), min_area=50):
        self.low = np.array(low, dtype=np.uint8)
        self.high = np.array(high, dtype=np.uint8)
        self.min_area = min_area
        self.last_pos = None
        self.roi_margin = 160

    def find(self, bgr):
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
        sh, sw = small.shape[:2]

        if self.last_pos is not None:
            lx, ly = int(self.last_pos[0]/2), int(self.last_pos[1]/2)
            m = self.roi_margin // 2
            roi_mask = np.zeros((sh, sw), dtype=np.uint8)
            roi_mask[max(0,ly-m):min(sh,ly+m), max(0,lx-m):min(sw,lx+m)] = 255
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask = cv2.bitwise_and(cv2.inRange(hsv, self.low, self.high), roi_mask)
        else:
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.low, self.high)

        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_area:
            return None

        M = cv2.moments(c)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"]/M["m00"]*2)
        cy = int(M["m01"]/M["m00"]*2)
        self.last_pos = (cx, cy)
        return (cx, cy, int(cv2.contourArea(c)*4))


# ── Kalman + PID (same as v4) ──
class Kalman:
    def __init__(self):
        self.x = np.zeros(4); self.P = np.eye(4)*500
        self.F = np.array([[1,0,.033,0],[0,1,0,.033],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.eye(4)*0.1; self.R = np.eye(2)*15.0
        self.ok = False; self.t0 = 0
    def update(self, z):
        if not self.ok:
            self.x[:2]=z; self.ok=True; self.t0=time.time(); return self.x[:2].copy()
        dt=max(time.time()-self.t0,.001)
        self.F[0,2]=dt; self.F[1,3]=dt
        self.x=self.F@self.x; self.P=self.F@self.P@self.F.T+self.Q
        y=np.array(z,dtype=float)-self.H@self.x
        S=self.H@self.P@self.H.T+self.R; K=self.P@self.H.T@np.linalg.inv(S)
        self.x+=K@y; self.P=(np.eye(4)-K@self.H)@self.P
        self.t0=time.time(); return self.x[:2].copy()
    def predict(self, t=0.2):
        if not self.ok: return None
        F=self.F.copy(); F[0,2]=t; F[1,3]=t; return (F@self.x)[:2].copy()
    def vel(self): return (float(self.x[2]),float(self.x[3]))
    def reset(self): self.x=np.zeros(4); self.P=np.eye(4)*500; self.ok=False

class PID:
    def __init__(self, kp=0.08, ki=0.003, kd=0.015, mx=3.5):
        self.kp=kp; self.ki=ki; self.kd=kd; self.mx=mx
        self.I=0; self.pe=0; self.pt=0
    def __call__(self, e):
        now=time.time(); dt=max(now-self.pt,.0001) if self.pt>0 else .001
        p=self.kp*e; self.I=max(-50,min(50,self.I+e*dt)); i=self.ki*self.I
        d=self.kd*(e-self.pe)/dt if self.pt>0 else 0
        self.pe=e; self.pt=now; return max(-self.mx,min(self.mx,p+i+d))
    def reset(self): self.I=0; self.pe=0; self.pt=0


def main():
    print("="*50)
    print("  MedVision Agent v5 - ULTRA FAST")
    print("  INT8 YOLO + Multi-thread + picamera2")
    print("="*50)

    cam = CameraThread()
    gimbal = GimbalTool({})
    det = FastDetector(low=(0,100,100), high=(10,255,255), min_area=50)
    K = Kalman()
    pid_p = PID(kp=0.08, ki=0.003, kd=0.015, mx=3.5)
    pid_t = PID(kp=0.06, ki=0.002, kd=0.012, mx=3.0)

    gimbal.move_to(pan=0, tilt=35, speed=400)
    time.sleep(1)

    print("%-6s %-8s %-10s %-6s" % ("Frame", "Target", "Gimbal", "FPS"))
    print("-" * 35)

    fc=0; t0=time.time(); lost=0
    try:
        while time.time()-t0 < 30:
            frame = cam.read()
            if frame is None:
                time.sleep(0.001); continue

            fc += 1
            fps = fc / max(time.time()-t0, 0.001)

            # Detect every 3rd frame
            r = det.find(frame) if fc % 3 == 0 else None

            if r:
                cx, cy, area = r; lost = 0
                K.update([cx, cy])
                err_x = cx - 320; err_y = cy - 240
                if abs(err_x)>15 or abs(err_y)>15:
                    gimbal.nudge_fast(pid_p(err_x), pid_t(err_y))
            else:
                lost += 1
                if lost < 20 and K.ok:
                    pred = K.predict(0.2)
                    if pred is not None:
                        gimbal.nudge_fast(pid_p(pred[0]-320)*.5, pid_t(pred[1]-240)*.5)
                elif lost >= 20:
                    det.last_pos = None

            if fc % 30 == 0:
                gp, gt = gimbal.query()
                pos = "(%d,%d)" % (cx, cy) if r else "NONE"
                print("F%04d %-8s pan=%+.1f  %.1f" % (fc, pos, gp, fps))

    except KeyboardInterrupt:
        pass
    finally:
        gimbal.center(); gimbal.close(); cam.close()
        e = time.time()-t0
        print("\n%d frames / %.1fs = %.1f FPS" % (fc, e, fc/max(e,.001)))


if __name__ == "__main__":
    main()
