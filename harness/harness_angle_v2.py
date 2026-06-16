"""
Angle Measurement v2 - Multi-method + Multi-view fusion.
Methods: PCA, Convex Hull, 3D Point Cloud
Improvement: Multi-view averaging eliminates projection error.
"""
import math
import numpy as np
import cv2


class AngleToolV2:
    """
    Enhanced angle measurement with multi-view fusion.

    Usage:
        tool = AngleToolV2()
        # Single view
        angle = tool.measure(bgr)
        # Multi-view (call from different gimbal positions)
        tool.begin_session()
        tool.feed(bgr, view_name="center")
        tool.feed(bgr, view_name="left")
        tool.feed(bgr, view_name="right")
        angle = tool.fused_result()  # Best estimate
    """

    def __init__(self, config=None):
        cfg = config or {}
        self.ema_alpha = cfg.get("ema_alpha", 0.4)
        self.smoothed = None
        self.min_area = cfg.get("min_area", 100)
        self.views = {}  # view_name -> angle
        self.all_angles = []

    def measure(self, bgr, method="auto"):
        """Single measurement. method: auto/pca/convex/3d"""
        if method == "auto":
            # Try convex hull first (more robust), fallback to PCA
            a = self._convex_hull(bgr)
            if a is None:
                a = self._pca(bgr)
            elif a == 0:
                return 0.0  # Closed clip
        elif method == "pca":
            a = self._pca(bgr)
        elif method == "convex":
            a = self._convex_hull(bgr)
        else:
            a = self._pca(bgr)

        if a is None:
            return self.smoothed

        # EMA smoothing
        if self.smoothed is None:
            self.smoothed = a
        else:
            self.smoothed = self.ema_alpha * a + (1 - self.ema_alpha) * self.smoothed

        return self.smoothed

    def measure_3d(self, bgr, depth, camera_params=None):
        """3D point cloud angle measurement.不受投影影响!"""
        if camera_params is None:
            camera_params = {"fx": 580, "fy": 580, "cx": 320, "cy": 240}

        # Segment red object
        mask = self._segment(bgr)
        valid = (depth > 100) & (depth < 4000)
        combined = (mask > 0) & valid

        if np.sum(combined) < 20:
            return None

        # Get 3D points
        h, w = depth.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        fx, fy = camera_params["fx"], camera_params["fy"]
        cx, cy = camera_params["cx"], camera_params["cy"]

        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        pts = np.column_stack([
            X[combined].flatten(),
            Y[combined].flatten(),
            Z[combined].flatten()
        ])

        if len(pts) < 10:
            return None

        # Remove outliers (statistical)
        pts = self._remove_outliers(pts, std_ratio=2.0)

        if len(pts) < 10:
            return None

        # PCA on 3D points
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (largest = primary axis)
        idx = np.argsort(eigenvalues)[::-1]
        eig1 = eigenvectors[:, idx[0]]  # Primary axis
        eig2 = eigenvectors[:, idx[1]]  # Secondary axis

        # Project points onto primary axis
        proj = centered @ eig1

        # Split into two arms
        half = np.median(proj)
        arm1 = centered[proj < half]
        arm2 = centered[proj >= half]

        if len(arm1) < 5 or len(arm2) < 5:
            return None

        # Fit line to each arm using PCA
        _, ev1 = np.linalg.eigh(np.cov(arm1.T))
        _, ev2 = np.linalg.eigh(np.cov(arm2.T))
        dir1 = ev1[:, -1]
        dir2 = ev2[:, -1]

        # Angle between two arms
        cos_angle = abs(np.dot(dir1, dir2))
        angle = math.degrees(math.acos(min(cos_angle, 1.0)))

        return angle

    def _remove_outliers(self, pts, std_ratio=2.0):
        """Remove statistical outliers."""
        mean = np.mean(pts, axis=0)
        std = np.std(pts, axis=0)
        mask = np.all(np.abs(pts - mean) < std_ratio * std, axis=1)
        return pts[mask]

    # ── Multi-view fusion ──

    def begin_session(self):
        """Start a new multi-view measurement session."""
        self.views = {}
        self.all_angles = []

    def feed(self, bgr, view_name=None):
        """Add a measurement from a specific view."""
        angle = self.measure(bgr)
        if angle is not None:
            if view_name:
                self.views[view_name] = angle
            self.all_angles.append(angle)
        return angle

    def fused_result(self):
        """Get the best estimate from all views."""
        if not self.all_angles:
            return self.smoothed

        # Remove outliers from all views
        arr = np.array(self.all_angles)
        if len(arr) >= 3:
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            # Keep only close-to-median values
            good = arr[np.abs(arr - median) < 2 * max(mad, 1.0)]
            if len(good) > 0:
                return float(np.mean(good))

        return float(np.mean(arr))

    # ── Internal methods ──

    def _segment(self, bgr):
        """Red color segmentation."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([155, 80, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def _pca(self, bgr):
        """PCA-based angle measurement."""
        mask = self._segment(bgr)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        body = None
        for c in cnts:
            if cv2.contourArea(c) > self.min_area:
                body = c
                break
        if body is None or len(body) < 5:
            return None

        pts = body.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            return None

        mean, eig = cv2.PCACompute(pts, None)
        proj = (pts - mean[0]) @ eig[0]
        s1, s2 = pts[proj < 0], pts[proj >= 0]

        if len(s1) < 3 or len(s2) < 3:
            return None

        [vx1, vy1, _, _] = cv2.fitLine(s1, cv2.DIST_L2, 0, 0.01, 0.01)
        [vx2, vy2, _, _] = cv2.fitLine(s2, cv2.DIST_L2, 0, 0.01, 0.01)
        a1 = math.degrees(math.atan2(float(vy1), float(vx1)))
        a2 = math.degrees(math.atan2(float(vy2), float(vx2)))
        ca = abs(a1 - a2)
        if ca > 180:
            ca = 360 - ca
        if ca > 90:
            ca = 180 - ca
        return ca

    def _convex_hull(self, bgr):
        """Convex hull defect method. Returns 0 for closed clips."""
        mask = self._segment(bgr)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        body = None
        for c in cnts:
            if cv2.contourArea(c) > self.min_area:
                body = c
                break
        if body is None or len(body) < 5:
            return None

        hull = cv2.convexHull(body, returnPoints=False)
        if len(hull) < 4:
            return None

        defects = cv2.convexityDefects(body, hull)
        if defects is None or len(defects) == 0:
            return 0.0  # No defect = closed

        max_depth = 0
        deepest_idx = 0
        for i in range(defects.shape[0]):
            _, _, _, d = defects[i, 0]
            if d > max_depth:
                max_depth = d
                deepest_idx = i

        if max_depth < 200:
            return 0.0  # Too shallow = nearly closed

        s, e, f, _ = defects[deepest_idx, 0]
        p_start = tuple(body[s][0])
        p_end = tuple(body[e][0])
        p_far = tuple(body[f][0])

        v1 = np.array(p_start, dtype=float) - np.array(p_far, dtype=float)
        v2 = np.array(p_end, dtype=float) - np.array(p_far, dtype=float)
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = math.degrees(math.acos(np.clip(cos_a, -1, 1)))
        return angle
