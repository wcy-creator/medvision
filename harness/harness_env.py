"""
MedVision - Environmental Perception System
3D Space Scanning + Object Detection + Scene Understanding
"""
import os, sys, time, math, json
import numpy as np
import cv2
sys.path.insert(0, "/opt/medvision")
sys.path.insert(0, "/opt/medvision/harness")


# ── 3D Space Scanner ──
class SpaceScanner:
    """Scan the environment using depth camera + gimbal rotation."""

    FX, FY = 580.0, 580.0
    CX, CY = 320.0, 240.0

    def __init__(self, gimbal=None, camera_func=None):
        self.gimbal = gimbal
        self.camera_func = camera_func  # Function to get depth frame

    def scan_single(self, bgr, depth):
        """Single frame 3D scan."""
        mask = self._segment(bgr)
        valid = (depth > 100) & (depth < 4000)
        combined = (mask > 0) & valid

        n = np.sum(combined)
        if n < 20:
            return None

        h, w = depth.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        X = (x - self.CX) * depth / self.FX
        Y = (y - self.CY) * depth / self.FY
        Z = depth

        pts = np.column_stack([X[combined], Y[combined], Z[combined]])
        pts = self._remove_outliers(pts)

        if len(pts) < 10:
            return None

        center = np.mean(pts, axis=0)
        dims = np.max(pts, axis=0) - np.min(pts, axis=0)
        distance = np.linalg.norm(center)
        azimuth = math.degrees(math.atan2(center[0], center[2]))
        elevation = math.degrees(math.atan2(center[1], center[2]))

        return {
            "center_3d": center.tolist(),
            "dimensions": dims.tolist(),
            "distance": float(distance),
            "azimuth": float(azimuth),
            "elevation": float(elevation),
            "n_points": len(pts),
        }

    def scan_panorama(self, gimbal, camera_func, depth_func, angles=None):
        """Full panorama scan."""
        if angles is None:
            angles = [(-30, 20), (-15, 25), (0, 20), (15, 25), (30, 20),
                      (-30, 35), (-15, 40), (0, 35), (15, 40), (30, 35)]

        all_points = []
        object_map = []

        for pan, tilt in angles:
            gimbal.move_to(pan=pan, tilt=tilt, speed=400)
            time.sleep(0.5)

            bgr = camera_func()
            depth = depth_func() if depth_func else None

            if bgr is None:
                continue

            result = self.scan_single(bgr, depth)
            if result:
                # Transform to global frame
                pts_global = self._rotate_points(
                    np.array(result["center_3d"]).reshape(1, 3), pan, tilt
                )
                all_points.append(pts_global[0])
                object_map.append({
                    "position": result["center_3d"],
                    "distance": result["distance"],
                    "pan": pan, "tilt": tilt,
                })

        if not all_points:
            return {"objects": [], "bounds": None}

        all_pts = np.array(all_points)
        return {
            "objects": object_map,
            "bounds": {
                "min": all_pts.min(axis=0).tolist(),
                "max": all_pts.max(axis=0).tolist(),
            },
            "n_views": len(object_map),
        }

    def _segment(self, bgr):
        """Segment foreground from background."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def _remove_outliers(self, pts, std_ratio=2.0):
        mean = np.mean(pts, axis=0)
        std = np.std(pts, axis=0)
        mask = np.all(np.abs(pts - mean) < std_ratio * std, axis=1)
        return pts[mask]

    def _rotate_points(self, pts, pan_deg, tilt_deg):
        pan = math.radians(pan_deg)
        tilt = math.radians(tilt_deg)
        R_pan = np.array([[math.cos(pan), 0, math.sin(pan)],
                          [0, 1, 0],
                          [-math.sin(pan), 0, math.cos(pan)]])
        R_tilt = np.array([[1, 0, 0],
                           [0, math.cos(tilt), -math.sin(tilt)],
                           [0, math.sin(tilt), math.cos(tilt)]])
        return (R_tilt @ R_pan @ pts.T).T


# ── Object Detector ──
class EnvDetector:
    """Detect and classify objects in the scene."""

    # Common object categories
    CATEGORIES = {
        "person": "human",
        "chair": "furniture",
        "desk": "furniture",
        "laptop": "electronics",
        "cell phone": "electronics",
        "keyboard": "electronics",
        "mouse": "electronics",
        "bottle": "container",
        "cup": "container",
        "book": "object",
        "backpack": "object",
        "potted plant": "decoration",
        "tv": "electronics",
        "monitor": "electronics",
    }

    def __init__(self, yolo_model=None):
        self.yolo = yolo_model
        self.detected_objects = []
        self.last_scan = None

    def detect_objects(self, bgr, depth=None):
        """Detect all objects in frame."""
        results = []

        # YOLO detection
        if self.yolo is not None:
            try:
                detections = self.yolo.detect(bgr)
                for det in detections:
                    cx, cy = det["center"]
                    cat = self.CATEGORIES.get(det["class_name"], "unknown")

                    # Estimate distance from depth if available
                    dist = None
                    if depth is not None:
                        h, w = depth.shape
                        px = min(max(cx, 0), w - 1)
                        py = min(max(cy, 0), h - 1)
                        roi = depth[max(0, py-5):min(h, py+5),
                                    max(0, px-5):min(w, px+5)]
                        valid = roi[roi > 100]
                        if len(valid) > 0:
                            dist = float(np.median(valid))

                    results.append({
                        "name": det["class_name"],
                        "category": cat,
                        "position_2d": (cx, cy),
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                        "distance_mm": dist,
                        "area": det["size"][0] * det["size"][1],
                    })
            except Exception as e:
                print("[Detector] YOLO error: %s" % e)

        # Color-based detection (always available)
        red_result = self._detect_color(bgr, (0, 100, 100), (10, 255, 255), "red_tool")
        if red_result:
            results.append(red_result)

        self.detected_objects = results
        return results

    def _detect_color(self, bgr, low, high, name):
        """Detect object by color."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 100:
            return None

        M = cv2.moments(c)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return {
            "name": name,
            "category": "tool",
            "position_2d": (cx, cy),
            "confidence": min(area / 5000, 1.0),
            "area": int(area),
            "distance_mm": None,
        }

    def get_scene_summary(self):
        """Get summary of detected objects."""
        if not self.detected_objects:
            return "No objects detected"

        summary = {}
        for obj in self.detected_objects:
            cat = obj["category"]
            if cat not in summary:
                summary[cat] = []
            summary[cat].append(obj["name"])

        parts = []
        for cat, names in summary.items():
            unique = list(set(names))
            parts.append("%s: %s" % (cat, ", ".join(unique)))

        return "Detected: " + "; ".join(parts)


# ── Scene Analyzer ──
class SceneAnalyzer:
    """Analyze the scene using AI and geometric data."""

    def __init__(self, llm_func=None):
        self.llm_func = llm_func  # Function to call LLM

    def analyze(self, bgr, depth=None, objects=None, scan_result=None):
        """Full scene analysis."""
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "objects": objects or [],
            "n_objects": len(objects) if objects else 0,
        }

        # Brightness analysis
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        result["brightness"] = float(brightness)
        result["lighting"] = "dark" if brightness < 80 else "normal" if brightness < 180 else "bright"

        # Edge density (scene complexity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        result["complexity"] = float(edge_density)
        result["scene_type"] = "complex" if edge_density > 0.1 else "simple"

        # Color distribution
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(hsv[:, :, 0])
        result["dominant_hue"] = float(mean_h)

        # 3D info
        if depth is not None:
            valid = depth[(depth > 100) & (depth < 4000)]
            if len(valid) > 0:
                result["depth_min"] = float(np.percentile(valid, 5))
                result["depth_max"] = float(np.percentile(valid, 95))
                result["depth_mean"] = float(np.mean(valid))
                result["depth_range"] = float(result["depth_max"] - result["depth_min"])

        # Object summary
        if objects:
            result["object_summary"] = self._summarize_objects(objects)

        # AI analysis (if available)
        if self.llm_func:
            try:
                prompt = self._build_analysis_prompt(result)
                ai_result = self.llm_func(bgr, prompt)
                result["ai_analysis"] = ai_result
            except Exception as e:
                result["ai_error"] = str(e)

        return result

    def _summarize_objects(self, objects):
        """Summarize detected objects."""
        categories = {}
        for obj in objects:
            cat = obj.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(obj.get("name", "object"))

        summary = []
        for cat, names in categories.items():
            unique = list(set(names))
            summary.append("%s (%d): %s" % (cat, len(unique), ", ".join(unique)))

        return "; ".join(summary) if summary else "No objects"

    def _build_analysis_prompt(self, result):
        """Build prompt for AI analysis."""
        parts = ["Analyze this scene:"]
        parts.append("- Lighting: %s (brightness: %.0f)" %
                     (result.get("lighting", "unknown"), result.get("brightness", 0)))
        parts.append("- Scene complexity: %s" % result.get("scene_type", "unknown"))

        if "depth_mean" in result:
            parts.append("- Depth range: %.0f - %.0f mm" %
                        (result.get("depth_min", 0), result.get("depth_max", 0)))

        if "object_summary" in result:
            parts.append("- Objects: %s" % result["object_summary"])

        parts.append("Describe what you see and any potential hazards.")
        return "\n".join(parts)

    def generate_report(self, result):
        """Generate human-readable report."""
        lines = ["=" * 50]
        lines.append("  ENVIRONMENT PERCEPTION REPORT")
        lines.append("  Time: %s" % result.get("timestamp", "unknown"))
        lines.append("=" * 50)
        lines.append("")

        # Lighting
        lines.append("[Lighting] %s (brightness: %.0f)" %
                     (result.get("lighting", "?").upper(),
                      result.get("brightness", 0)))

        # Scene
        lines.append("[Scene] %s" % result.get("scene_type", "unknown"))
        lines.append("[Complexity] %.1f%%" % (result.get("complexity", 0) * 100))

        # Objects
        n = result.get("n_objects", 0)
        lines.append("[Objects] %d detected" % n)
        if "object_summary" in result:
            lines.append("  %s" % result["object_summary"])

        # Depth
        if "depth_mean" in result:
            lines.append("[Depth] %.0f - %.0f mm (mean: %.0f mm)" %
                        (result.get("depth_min", 0), result.get("depth_max", 0),
                         result.get("depth_mean", 0)))

        # AI analysis
        if "ai_analysis" in result:
            lines.append("[AI Analysis]")
            lines.append("  %s" % result["ai_analysis"])

        lines.append("=" * 50)
        return "\n".join(lines)


# ── Main Demo ──
def main():
    from harness_agent_v5 import CameraThread, FastDetector

    print("=" * 50)
    print("  Environmental Perception System")
    print("=" * 50)

    cam = CameraThread()
    det_env = EnvDetector()
    analyzer = SceneAnalyzer()

    print("\n[Init] Camera started")
    time.sleep(2)

    # Capture and analyze
    print("[Scan] Capturing frame...")
    bgr = cam.read()
    if bgr is None:
        print("[ERROR] No frame captured")
        cam.close()
        return

    # Detect objects
    objects = det_env.detect_objects(bgr)
    print("[Detect] Found %d objects" % len(objects))
    for obj in objects:
        print("  - %s (%s) conf=%.2f" % (obj["name"], obj["category"], obj["confidence"]))

    # Scene analysis
    result = analyzer.analyze(bgr, objects=objects)
    report = analyzer.generate_report(result)
    print("\n" + report)

    # Save report
    report_path = "/opt/medvision/snapshots/env_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print("[Report] Saved to %s" % report_path)

    cam.close()
    print("[Done]")


if __name__ == "__main__":
    main()
