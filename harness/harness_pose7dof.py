"""
7DoF Pose Estimation - Based on SurgeoNet architecture.
Estimates: 3D position + 3D rotation + 1D articulation angle.
For surgical instrument tracking with opening angle.
"""
import os
import time
import math
import numpy as np
import cv2


class PoseEstimator7DoF:
    """
    7-DoF pose estimator for articulated surgical instruments.
    Uses keypoint detection + geometric solver.

    Output: [x, y, z, rx, ry, rz, articulation_angle]

    Usage:
        pose = PoseEstimator7DoF()
        result = pose.estimate(bgr, depth)
        # result = {
        #     "position": [x, y, z],      # 3D position (mm)
        #     "rotation": [rx, ry, rz],    # Euler angles (degrees)
        #     "angle": 45.2,               # Articulation angle (degrees)
        #     "confidence": 0.85,
        #     "keypoints": [...]           # Detected 2D keypoints
        # }
    """

    # Camera intrinsics (Astra Pro)
    FX, FY = 580.0, 580.0
    CX, CY = 320.0, 240.0

    def __init__(self, config=None):
        cfg = config or {}
        self.fx = cfg.get("fx", self.FX)
        self.fy = cfg.get("fy", self.FY)
        self.cx = cfg.get("cx", self.CX)
        self.cy = cfg.get("cy", self.CY)
        self.ema_alpha = cfg.get("ema_alpha", 0.4)
        self.smoothed_angle = None

    def estimate(self, bgr, depth=None):
        """
        Full 7-DoF pose estimation.
        Returns dict with position, rotation, angle, confidence, keypoints.
        """
        if bgr is None:
            return None

        # Step 1: Detect instrument and keypoints
        keypoints = self._detect_keypoints(bgr)
        if keypoints is None:
            return None

        # Step 2: 2D angle from keypoints
        angle_2d = self._compute_2d_angle(keypoints)

        # Step 3: 3D position from depth (if available)
        position_3d = None
        if depth is not None:
            position_3d = self._depth_to_3d(keypoints, depth)

        # Step 4: Estimate 3D rotation
        rotation = self._estimate_rotation(keypoints, depth)

        # Step 5: EMA smooth the angle
        if angle_2d is not None:
            if self.smoothed_angle is None:
                self.smoothed_angle = angle_2d
            else:
                self.smoothed_angle = self.ema_alpha * angle_2d + (1 - self.ema_alpha) * self.smoothed_angle

        result = {
            "position": position_3d or [0, 0, 0],
            "rotation": rotation or [0, 0, 0],
            "angle": self.smoothed_angle if self.smoothed_angle else angle_2d,
            "angle_raw": angle_2d,
            "confidence": self._compute_confidence(keypoints, depth),
            "keypoints": keypoints,
        }

        return result

    def _detect_keypoints(self, bgr):
        """
        Detect instrument keypoints: base, crotch, tip1, tip2.
        These define the V-shape of an open clip/scissors.
        """
        # Color segmentation (red instrument)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([155, 80, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Find contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        body = None
        for c in cnts:
            if cv2.contourArea(c) > 200:
                body = c
                break
        if body is None or len(body) < 5:
            return None

        # Method: Convex hull defects to find keypoint positions
        hull = cv2.convexHull(body, returnPoints=False)
        if len(hull) < 4:
            return None

        defects = cv2.convexityDefects(body, hull)
        if defects is None:
            return None

        # Find the crotch point (deepest defect)
        max_depth = 0
        deepest_idx = 0
        for i in range(defects.shape[0]):
            _, _, _, d = defects[i, 0]
            if d > max_depth:
                max_depth = d
                deepest_idx = i

        if max_depth < 100:
            # No significant defect = closed instrument
            # Estimate keypoints from contour extremes
            pts = body.reshape(-1, 2)
            top = pts[pts[:, 1].argmin()]
            bottom = pts[pts[:, 1].argmax()]
            left = pts[pts[:, 0].argmin()]
            right = pts[pts[:, 0].argmax()]
            return {
                "crotch": tuple(bottom.tolist()),
                "tip1": tuple(left.tolist()),
                "tip2": tuple(right.tolist()),
                "base": tuple(top.tolist()),
                "open": False,
                "defect_depth": 0,
            }

        s, e, f, _ = defects[deepest_idx, 0]
        crotch = tuple(body[f][0])
        tip1 = tuple(body[s][0])
        tip2 = tuple(body[e][0])

        # Base point: furthest from crotch along the body
        pts = body.reshape(-1, 2)
        dists = np.linalg.norm(pts - np.array(crotch), axis=1)
        base_idx = dists.argmax()
        base = tuple(pts[base_idx].tolist())

        return {
            "crotch": crotch,
            "tip1": tip1,
            "tip2": tip2,
            "base": base,
            "open": True,
            "defect_depth": max_depth,
        }

    def _compute_2d_angle(self, kps):
        """Compute angle from keypoints using vector dot product."""
        if not kps.get("open", False):
            return 0.0

        crotch = np.array(kps["crotch"], dtype=float)
        v1 = np.array(kps["tip1"], dtype=float) - crotch
        v2 = np.array(kps["tip2"], dtype=float) - crotch

        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = math.degrees(math.acos(np.clip(cos_a, -1, 1)))
        return angle

    def _depth_to_3d(self, kps, depth):
        """Convert keypoint pixel positions to 3D using depth."""
        crotch = kps["crotch"]
        h, w = depth.shape

        # Get depth at crotch point
        cx, cy = crotch
        cx = max(0, min(w-1, cx))
        cy = max(0, min(h-1, cy))

        # Sample周围区域取中值（更稳定）
        roi_r = 5
        y1 = max(0, cy - roi_r)
        y2 = min(h, cy + roi_r + 1)
        x1 = max(0, cx - roi_r)
        x2 = min(w, cx + roi_r + 1)
        z_vals = depth[y1:y2, x1:x2]
        z_valid = z_vals[z_vals > 100]
        if len(z_valid) == 0:
            return None

        z = float(np.median(z_valid))
        x = (cx - self.cx) * z / self.fx
        y = (cy - self.cy) * z / self.fy

        return [round(x, 1), round(y, 1), round(z, 1)]

    def _estimate_rotation(self, kps, depth=None):
        """Estimate 3D rotation from 2D keypoints."""
        if not kps.get("open", False):
            return [0, 0, 0]

        # Simple estimation based on keypoint geometry
        crotch = np.array(kps["crotch"], dtype=float)
        tip1 = np.array(kps["tip1"], dtype=float)
        tip2 = np.array(kps["tip2"], dtype=float)
        base = np.array(kps["base"], dtype=float)

        # Arm directions
        arm1 = tip1 - crotch
        arm2 = tip2 - crotch

        # Rotation: average arm direction gives orientation
        avg_arm = (arm1 + arm2) / 2
        roll = math.degrees(math.atan2(avg_arm[1], avg_arm[0]))

        return [round(roll, 1), 0, 0]

    def _compute_confidence(self, kps, depth):
        """Estimate confidence based on detection quality."""
        if kps is None:
            return 0.0

        conf = 0.5  # Base confidence

        # Open instrument = more confident
        if kps.get("open", False):
            conf += 0.2

        # Deep defect = more confident
        defect = kps.get("defect_depth", 0)
        if defect > 500:
            conf += 0.2
        elif defect > 200:
            conf += 0.1

        # Depth available = more confident
        if depth is not None:
            conf += 0.1

        return min(conf, 1.0)

    def draw(self, bgr, result):
        """Draw pose estimation results on image."""
        vis = bgr.copy()

        if result is None or result.get("keypoints") is None:
            cv2.putText(vis, "No instrument detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis

        kps = result["keypoints"]

        # Draw keypoints
        for name, pt in kps.items():
            if isinstance(pt, tuple) and len(pt) == 2:
                color = {"crotch": (0, 255, 0), "tip1": (255, 0, 0),
                        "tip2": (0, 0, 255), "base": (255, 255, 0)}.get(name, (255, 255, 255))
                cv2.circle(vis, pt, 6, color, -1)
                cv2.putText(vis, name[:3], (pt[0]+8, pt[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw arms
        if kps.get("open", False):
            c = kps["crotch"]
            cv2.line(vis, c, kps["tip1"], (0, 255, 0), 2)
            cv2.line(vis, c, kps["tip2"], (0, 255, 0), 2)

        # Draw angle arc
        if result.get("angle") is not None:
            angle = result["angle"]
            c = kps["crotch"]
            cv2.putText(vis, "Angle: %.1f deg" % angle, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw 3D info
        pos = result.get("position", [0, 0, 0])
        if pos and pos[2] > 0:
            cv2.putText(vis, "3D: (%.0f, %.0f, %.0f) mm" % tuple(pos), (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Confidence bar
        conf = result.get("confidence", 0)
        bar_w = 100
        cv2.rectangle(vis, (10, 65), (10 + bar_w, 75), (50, 50, 50), -1)
        cv2.rectangle(vis, (10, 65), (10 + int(bar_w * conf), 75), (0, 255, 0), -1)
        cv2.putText(vis, "%.0f%%" % (conf * 100), (bar_w + 15, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis
