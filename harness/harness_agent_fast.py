"""MedVision Agent FAST - Maximum FPS, color detection only."""
import os, sys, time
import numpy as np
import cv2
sys.path.insert(0, "/opt/medvision")
sys.path.insert(0, "/opt/medvision/harness")
from harness_gimbal import GimbalTool
from harness_detect import DetectTool


class PicameraCamera:
    """Use picamera2 in VIDEO mode for max FPS on Astra Pro."""
    def __init__(self, width=640, height=480):
        self._ok = False
        self.picam = None
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            config = self.picam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.picam.configure(config)
            self.picam.start()
            self._ok = True
            print("[Camera] picamera2 video mode (Astra Pro)")
        except Exception as e:
            print("[Camera] picamera2 failed: %s" % e)

    def capture(self):
        if not self._ok:
            return None
        try:
            arr = self.picam.capture_array()
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def is_open(self):
        return self._ok

    def close(self):
        if self.picam:
            self.picam.stop()


class Kalman:
    def __init__(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 500
        self.F = np.array([[1,0,.033,0],[0,1,0,.033],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 15.0
        self.ok = False
        self.t0 = 0

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

    def predict(self, t=0.15):
        if not self.ok: return None
        F = self.F.copy(); F[0,2]=t; F[1,3]=t
        return (F @ self.x)[:2].copy()

    def vel(self):
        return (float(self.x[2]), float(self.x[3]))

    def reset(self):
        self.x = np.zeros(4); self.P = np.eye(4)*500; self.ok = False


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


def main():
    print("[FAST] Starting...")
    gimbal = GimbalTool({})
    camera = PicameraCamera()
    detect = DetectTool({})
    K = Kalman()
    pid_p = PID(kp=0.08, ki=0.003, kd=0.015, mx=3.5)
    pid_t = PID(kp=0.06, ki=0.002, kd=0.012, mx=3.0)

    gimbal.center(); time.sleep(0.5)
    fc=0; t0=time.time(); lost=0

    print("%-6s %-14s %-14s %-6s" % ("Frame", "Position", "Velocity", "FPS"))
    print("-" * 45)

    try:
        while True:
            bgr = camera.capture()
            if bgr is None:
                time.sleep(0.002); continue
            fc += 1
            fps = fc / max(time.time()-t0, 0.001)
            r = detect.find(bgr)
            if r:
                cx, cy, _ = r; lost = 0
                K.update([cx, cy])
                if abs(cx-320)>15 or abs(cy-240)>15:
                    gimbal.nudge(pid_p(cx-320), pid_t(cy-240))
                if fc%5==0:
                    gp,gt = gimbal.query()
                    vx,vy = K.vel()
                    print("F%04d (%d,%d)     (%.1f,%.1f)   %.1f" % (fc,cx,cy,vx,vy,fps))
            else:
                lost += 1
                if lost<20 and K.ok:
                    pred = K.predict(0.2)
                    if pred is not None:
                        gimbal.nudge(pid_p(pred[0]-320)*.5, pid_t(pred[1]-240)*.5)
                elif lost>=20:
                    p,t = gimbal.query()
                    gimbal.move_to(pan=p+5, tilt=t); time.sleep(0.08)
    except KeyboardInterrupt:
        pass
    finally:
        gimbal.center(); gimbal.close(); camera.close()
        e = time.time()-t0
        print("\n[FAST] %d frames / %.1fs = %.1f FPS" % (fc, e, fc/max(e,.001)))

if __name__=="__main__":
    main()
