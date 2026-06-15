"""
Multi-Feature Detector for surgical clip tracking.
Combines: Color + Shape + Size + Template + CSRT tracker.
Much more robust than color-only detection.
"""
import os, sys, time, math, numpy as np, cv2

CSRT_AVAILABLE = False
try:
    cv2.TrackerCSRT_create()
    CSRT_AVAILABLE = True
except:
    try:
        cv2.legacy.TrackerCSRT_create()
        CSRT_AVAILABLE = True
    except:
        pass

class MultiFeatureDetector:
    """Robust detector using multiple visual features."""

    def __init__(self):
        # Template from first detection
        self.template = None
        self.template_contour = None
        self.template_hist = None
        self.csrt = None
        self.last_bbox = None
        self.lost_count = 0
        self.init_done = False

    def init(self, bgr, cx, cy, contour=None):
        """Initialize with first detection result."""
        h, w = bgr.shape[:2]
        pad = 50
        x1 = max(0, cx - pad)
        y1 = max(0, cy - pad)
        x2 = min(w, cx + pad)
        y2 = min(h, cy + pad)
        self.template = bgr[y1:y2, x1:x2].copy()

        if contour is not None:
            self.template_contour = contour.copy()

        hsv_roi = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
        self.template_hist = cv2.calcHist([hsv_roi], [0, 1], None, [32, 32], [0, 180, 0, 255])
        cv2.normalize(self.template_hist, self.template_hist, 0, 255)

        if CSRT_AVAILABLE:
            try:
                self.csrt = cv2.TrackerCSRT_create()
                self.csrt.init(bgr, (x1, y1, x2 - x1, y2 - y1))
            except:
                self.csrt = None

        self.last_bbox = (x1, y1, x2, y2)
        self.init_done = True
        print("[Detector] Template initialized (%dx%d)" % (x2-x1, y2-y1))

    def detect(self, bgr):
        """Multi-feature detection. Returns (cx, cy, score) or None."""
        if bgr is None:
            return None, 0

        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # === Feature 1: Color (HSV red mask) ===
        m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([155, 80, 80]), np.array([180, 255, 255]))
        color_mask = cv2.bitwise_or(m1, m2)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # === Find contours with multi-feature scoring ===
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = 0
        best_result = None

        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                continue

            score = 0

            # --- Feature 1: Size (clip should be 500-20000 px) ---
            if 500 < area < 20000:
                score += 30
            elif area > 200:
                score += 10

            # --- Feature 2: Shape (aspect ratio, rectangularity) ---
            rect = cv2.minAreaRect(c)
            rx, ry = rect[1]
            if rx > 0 and ry > 0:
                aspect = max(rx, ry) / min(rx, ry)
                fill = area / (rx * ry)
                if 0.3 < aspect < 4.0 and fill > 0.2:
                    score += 20
                if 0.5 < aspect < 2.5:
                    score += 10

            # --- Feature 3: V-shape (convexity defect) ---
            if len(c) >= 5:
                hull = cv2.convexHull(c, returnPoints=False)
                if len(hull) > 3:
                    defects = cv2.convexityDefects(c, hull)
                    if defects is not None:
                        max_defect = max(defects[:, 0, 3])
                        if max_defect > 500:  # significant V-shape
                            score += 20

            # --- Feature 4: Template match (color histogram correlation) ---
            if self.template_hist is not None:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    tcx = int(M["m10"] / M["m00"])
                    tcy = int(M["m01"] / M["m00"])
                    # Check color similarity in ROI
                    pad = 20
                    tx1 = max(0, tcx - pad)
                    ty1 = max(0, tcy - pad)
                    tx2 = min(w, tcx + pad)
                    ty2 = min(h, tcy + pad)
                    if tx2 > tx1 and ty2 > ty1:
                        roi_hsv = hsv[ty1:ty2, tx1:tx2]
                        roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [32, 32], [0, 180, 0, 255])
                        cv2.normalize(roi_hist, roi_hist, 0, 255)
                        corr = cv2.compareHist(self.template_hist, roi_hist, cv2.HISTCMP_CORREL)
                        if corr > 0.5:
                            score += 25
                        elif corr > 0.3:
                            score += 10

            # --- Feature 5: Distance from last known position (if tracking) ---
            if self.last_bbox is not None:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    tcx = int(M["m10"] / M["m00"])
                    tcy = int(M["m01"] / M["m00"])
                    last_cx = (self.last_bbox[0] + self.last_bbox[2]) / 2
                    last_cy = (self.last_bbox[1] + self.last_bbox[3]) / 2
                    dist = math.sqrt((tcx - last_cx)**2 + (tcy - last_cy)**2)
                    if dist < 100:  # close to last position
                        score += 15
                    elif dist < 200:
                        score += 5

            if score > best_score:
                best_score = score
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, bw, bh = cv2.boundingRect(c)
                    best_result = (cx, cy, x, y, bw, bh, area, score)

        # === CSRT fallback ===
        if best_result is None and self.csrt and self.init_done:
            ok, box = self.csrt.update(bgr)
            if ok:
                x, y, bw, bh = [int(v) for v in box]
                cx = x + bw // 2
                cy = y + bh // 2
                best_result = (cx, cy, x, y, bw, bh, bw*bh, 15)
                # Re-init CSRT at new position
                x1 = max(0, cx - 50)
                y1 = max(0, cy - 50)
                x2 = min(w, cx + 50)
                y2 = min(h, cy + 50)
                try:
                    self.csrt = cv2.TrackerCSRT_create()
                    self.csrt.init(bgr, (x1, y1, x2-x1, y2-y1))
                except:
                    pass

        if best_result:
            cx, cy, x, y, bw, bh, area, score = best_result
            self.last_bbox = (x, y, x + bw, y + bh)
            self.lost_count = 0
            return (cx, cy), score
        else:
            self.lost_count += 1
            return None, 0
