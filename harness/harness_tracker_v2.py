"""
Enhanced Tracker v2 - EKF + Multi-view Fusion + Adaptive PID
Based on latest research (2025-2026 papers on surgical instrument tracking)
"""
import time
import math
import numpy as np


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for non-linear tracking.
    Better than standard KF for accelerating/decelerating targets.
    Based on: "Adaptive Trajectory Control using EKF-Based Self-Tuning PID" (JRC 2025)
    """

    def __init__(self, dt=0.033):
        self.dt = dt
        # State: [x, y, vx, vy, ax, ay] (position + velocity + acceleration)
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1000

        # State transition (constant acceleration model)
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1,  0, dt, 0],
            [0, 0, 0,  1, 0, dt],
            [0, 0, 0,  0, 1, 0],
            [0, 0, 0,  0, 0, 1],
        ], dtype=float)

        # Measurement matrix (observe position only)
        self.H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]], dtype=float)

        # Process noise
        q = 0.5
        self.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0, q*dt**2/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2, 0, q*dt**2/2],
            [q*dt**3/2, 0, q*dt**2, 0, q*dt, 0],
            [0, q*dt**3/2, 0, q*dt**2, 0, q*dt],
            [q*dt**2/2, 0, q*dt, 0, q, 0],
            [0, q*dt**2/2, 0, q*dt, 0, q],
        ], dtype=float)

        self.R = np.eye(2) * 12.0
        self.ok = False
        self.t0 = 0
        self._residuals = []

    def update(self, z):
        if not self.ok:
            self.x[:2] = z
            self.ok = True
            self.t0 = time.time()
            return self.x[:2].copy()

        dt = max(time.time() - self.t0, 0.001)
        self.F[0, 2] = dt; self.F[0, 4] = 0.5*dt*dt
        self.F[1, 3] = dt; self.F[1, 5] = 0.5*dt*dt
        self.F[2, 4] = dt; self.F[3, 5] = dt

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        z_arr = np.array(z, dtype=float)
        y = z_arr - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        # Adaptive noise (based on residual)
        residual_norm = np.linalg.norm(y)
        self._residuals.append(residual_norm)
        if len(self._residuals) > 20:
            self._residuals.pop(0)
            avg_res = np.mean(self._residuals)
            if avg_res > 20:
                self.R *= 1.1  # Increase measurement noise trust
            elif avg_res < 5:
                self.R *= 0.95  # Decrease measurement noise trust
            self.R = np.clip(self.R, np.eye(2)*5, np.eye(2)*50)

        self.t0 = time.time()
        return self.x[:2].copy()

    def predict(self, t=0.2):
        if not self.ok:
            return None
        F = np.eye(6)
        F[0, 2] = t; F[0, 4] = 0.5*t*t
        F[1, 3] = t; F[1, 5] = 0.5*t*t
        F[2, 4] = t; F[3, 5] = t
        x_pred = F @ self.x
        return x_pred[:2].copy()

    def vel(self):
        return (float(self.x[2]), float(self.x[3]))

    def accel(self):
        return (float(self.x[4]), float(self.x[5]))

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1000
        self.ok = False
        self._residuals = []


class AdaptivePID:
    """
    PID with automatic parameter tuning based on error dynamics.
    Based on: "HBO-PID Dynamic Tuning" (IROS 2025)
    """

    def __init__(self, kp=0.08, ki=0.003, kd=0.015, mx=3.5):
        self.kp_base = kp
        self.ki_base = ki
        self.kd_base = kd
        self.mx = mx
        self.I = 0; self.pe = 0; self.pt = 0
        self.error_history = []

    def __call__(self, e):
        now = time.time()
        dt = max(now - self.pt, 0.0001) if self.pt > 0 else 0.001

        # Track error for adaptive tuning
        self.error_history.append(abs(e))
        if len(self.error_history) > 30:
            self.error_history.pop(0)

        # Adaptive gains based on error magnitude
        avg_err = np.mean(self.error_history) if self.error_history else 0
        if avg_err > 100:
            # Large error: increase P, decrease D
            kp = self.kp_base * 1.3
            ki = self.ki_base * 0.5
            kd = self.kd_base * 0.7
        elif avg_err < 20:
            # Small error: decrease P, increase I
            kp = self.kp_base * 0.7
            ki = self.ki_base * 1.5
            kd = self.kd_base * 1.2
        else:
            kp = self.kp_base
            ki = self.ki_base
            kd = self.kd_base

        p = kp * e
        self.I = max(-50, min(50, self.I + e * dt))
        i = ki * self.I
        d = kd * (e - self.pe) / dt if self.pt > 0 else 0
        self.pe = e; self.pt = now
        return max(-self.mx, min(self.mx, p + i + d))

    def reset(self):
        self.I = 0; self.pe = 0; self.pt = 0; self.error_history = []


class MultiViewTracker:
    """
    Multi-view tracking with measurement fusion.
    Fuses detections from multiple camera angles for more robust tracking.
    """

    def __init__(self, n_views=3):
        self.n_views = n_views
        self.view_results = {}  # view_name -> (cx, cy, confidence)
        self.fused_position = None
        self.fused_confidence = 0

    def add_measurement(self, view_name, cx, cy, confidence):
        """Add a measurement from a specific camera view."""
        self.view_results[view_name] = (cx, cy, confidence)

    def fuse(self):
        """Fuse measurements from all views using weighted average."""
        if not self.view_results:
            return self.fused_position

        positions = []
        weights = []
        for name, (cx, cy, conf) in self.view_results.items():
            positions.append([cx, cy])
            weights.append(conf)

        positions = np.array(positions)
        weights = np.array(weights)
        weights = weights / weights.sum()

        fused = np.average(positions, axis=0, weights=weights)
        self.fused_position = fused.tolist()
        self.fused_confidence = float(np.mean(weights))
        self.view_results = {}  # Clear for next frame

        return self.fused_position

    def get_position(self):
        return self.fused_position

    def get_confidence(self):
        return self.fused_confidence

    def reset(self):
        self.view_results = {}
        self.fused_position = None
        self.fused_confidence = 0
