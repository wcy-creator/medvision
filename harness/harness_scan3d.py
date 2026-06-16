"""
3D Point Cloud Scanner v2 - AI-guided + Multi-angle scanning.
Improved depth handling and 3D pose estimation.
"""
import os
import time
import math
import numpy as np
import cv2


class Scanner3D:
    """
    3D point cloud scanner with AI-guided object detection.

    Usage:
        scanner = Scanner3D()
        # Single scan
        result = scanner.scan(bgr, depth)
        # Multi-angle scan
        result = scanner.scan_multi(gimbal, camera, angles=[(-20,20), (0,20), (20,20)])
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
        self.min_depth = cfg.get("min_depth", 100)
        self.max_depth = cfg.get("max_depth", 4000)
        self.outlier_std = cfg.get("outlier_std", 2.0)

    def scan(self, bgr, depth, roi=None):
        """
        Single frame 3D scan.
        Args:
            bgr: BGR image (640x480)
            depth: Depth map (640x480, mm)
            roi: Optional (x1,y1,x2,y2) region of interest
        Returns:
            dict with points, center, dimensions, color_info
        """
        if bgr is None or depth is None:
            return None

        # Segment object (red)
        mask = self._segment(bgr)

        # Apply ROI if provided
        if roi is not None:
            roi_mask = np.zeros_like(mask)
            x1, y1, x2, y2 = roi
            roi_mask[y1:y2, x1:x2] = 255
            mask = cv2.bitwise_and(mask, roi_mask)

        # Depth validity
        valid = (depth > self.min_depth) & (depth < self.max_depth)
        combined = (mask > 0) & valid

        n_points = np.sum(combined)
        if n_points < 20:
            return {"error": "insufficient_points", "n_points": int(n_points)}

        # Get 3D points
        h, w = depth.shape
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        X = (x_grid - self.cx) * depth / self.fx
        Y = (y_grid - self.cy) * depth / self.fy
        Z = depth

        pts = np.column_stack([
            X[combined].flatten(),
            Y[combined].flatten(),
            Z[combined].flatten()
        ])

        # Color info
        colors = bgr[combined]

        # Remove outliers
        pts_clean = self._remove_outliers(pts)
        if len(pts_clean) < 10:
            return {"error": "too_few_after_clean", "n_points": int(n_points)}

        # Compute metrics
        center_3d = np.mean(pts_clean, axis=0)
        dims = np.max(pts_clean, axis=0) - np.min(pts_clean, axis=0)
        distance = np.linalg.norm(center_3d)

        # Direction angles
        azimuth = math.degrees(math.atan2(center_3d[0], center_3d[2]))
        elevation = math.degrees(math.atan2(center_3d[1], center_3d[2]))

        # Color analysis
        mean_color = np.mean(colors, axis=0)
        dominant_hsv = self._dominant_color(colors)

        # PCA orientation
        orientation = self._compute_orientation(pts_clean)

        result = {
            "points": pts_clean,
            "n_points": len(pts_clean),
            "center_3d": center_3d.tolist(),
            "dimensions_mm": dims.tolist(),
            "distance_mm": float(distance),
            "azimuth_deg": float(azimuth),
            "elevation_deg": float(elevation),
            "mean_bgr": mean_color.tolist(),
            "dominant_hsv": dominant_hsv,
            "orientation": orientation,
            "quality": self._assess_quality(n_points, len(pts_clean), distance),
        }

        return result

    def scan_multi(self, gimbal, camera, angles=None, depth_func=None):
        """
        Multi-angle scan for better 3D reconstruction.
        Args:
            gimbal: GimbalController with move_to(pan, tilt)
            camera: CameraTool with capture()
            angles: List of (pan, tilt) positions
            depth_func: Function to get depth map
        Returns:
            Merged 3D scan result
        """
        if angles is None:
            angles = [(-20, 20), (0, 20), (20, 20), (0, 30), (0, 15)]

        all_points = []
        all_colors = []
        results = []

        for pan, tilt in angles:
            gimbal.move_to(pan=pan, tilt=tilt)
            time.sleep(0.5)

            bgr = camera.capture()
            if bgr is None:
                continue

            depth = depth_func() if depth_func else None
            if depth is None:
                continue

            result = self.scan(bgr, depth)
            if result and "points" in result:
                # Transform points to global frame
                pts = result["points"]
                # Apply rotation based on gimbal angle
                pts_global = self._rotate_points(pts, pan, tilt)
                all_points.append(pts_global)
                results.append(result)

        if not all_points:
            return {"error": "no_valid_scans"}

        merged = np.vstack(all_points)

        # Final metrics
        center = np.mean(merged, axis=0)
        dims = np.max(merged, axis=0) - np.min(merged, axis=0)

        return {
            "merged_points": len(merged),
            "center_3d": center.tolist(),
            "dimensions_mm": dims.tolist(),
            "n_views": len(results),
            "individual_results": results,
        }

    def _rotate_points(self, pts, pan_deg, tilt_deg):
        """Rotate 3D points based on gimbal angles."""
        pan = math.radians(pan_deg)
        tilt = math.radians(tilt_deg)

        # Rotation matrix: pan around Y, tilt around X
        R_pan = np.array([
            [math.cos(pan), 0, math.sin(pan)],
            [0, 1, 0],
            [-math.sin(pan), 0, math.cos(pan)]
        ])
        R_tilt = np.array([
            [1, 0, 0],
            [0, math.cos(tilt), -math.sin(tilt)],
            [0, math.sin(tilt), math.cos(tilt)]
        ])

        return (R_tilt @ R_pan @ pts.T).T

    def _segment(self, bgr):
        """Red color segmentation."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([155, 80, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def _remove_outliers(self, pts):
        """Statistical outlier removal."""
        mean = np.mean(pts, axis=0)
        std = np.std(pts, axis=0)
        mask = np.all(np.abs(pts - mean) < self.outlier_std * std, axis=1)
        return pts[mask]

    def _dominant_color(self, bgr_pixels):
        """Find dominant HSV color."""
        hsv = cv2.cvtColor(bgr_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)
        h = hsv[0, :, 0]
        return [int(np.median(h)), int(np.mean(hsv[0, :, 1])), int(np.mean(hsv[0, :, 2]))]

    def _compute_orientation(self, pts):
        """PCA orientation of point cloud."""
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        return {
            "primary_axis": eigenvectors[:, idx[0]].tolist(),
            "eigenvalues": eigenvalues[idx].tolist(),
        }

    def _assess_quality(self, raw_count, clean_count, distance):
        """Assess scan quality."""
        ratio = clean_count / max(raw_count, 1)
        if distance < 300:
            return "TOO_CLOSE"
        elif distance > 1500:
            return "TOO_FAR"
        elif ratio < 0.3:
            return "NOISY"
        elif clean_count > 500:
            return "EXCELLENT"
        elif clean_count > 100:
            return "GOOD"
        else:
            return "FAIR"

    def save_ply(self, path, points, colors=None):
        """Save point cloud as PLY file."""
        n = len(points)
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write("element vertex %d\n" % n)
            f.write("property float x\nproperty float y\nproperty float z\n")
            if colors is not None:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(n):
                line = "%.2f %.2f %.2f" % (points[i, 0], points[i, 1], points[i, 2])
                if colors is not None:
                    line += " %d %d %d" % (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
                f.write(line + "\n")
        return n

    def visualize(self, bgr, result, depth=None):
        """Draw 3D scan results on image."""
        vis = bgr.copy()

        if result is None or "error" in result:
            cv2.putText(vis, "No data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis

        # Draw center point (project back to 2D)
        cx, cy, cz = result["center_3d"]
        px = int(cx * self.fx / max(cz, 1) + self.cx)
        py = int(cy * self.fy / max(cz, 1) + self.cy)
        cv2.circle(vis, (px, py), 8, (0, 255, 0), -1)

        # Info text
        info = [
            "3D Position: (%.0f, %.0f, %.0f) mm" % (cx, cy, cz),
            "Distance: %.0f mm" % result["distance_mm"],
            "Azimuth: %.1f deg" % result["azimuth_deg"],
            "Elevation: %.1f deg" % result["elevation_deg"],
            "Dimensions: %.0fx%.0fx%.0f mm" % tuple(result["dimensions_mm"]),
            "Quality: %s (%d pts)" % (result["quality"], result["n_points"]),
        ]

        for i, text in enumerate(info):
            cv2.putText(vis, text, (10, 25 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return vis
