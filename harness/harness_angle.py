"""Angle Measurement Tool - PCA + EMA smoothed."""
import math, numpy as np, cv2


class AngleTool:
    def __init__(self, config):
        self.alpha = config.get("ema_alpha", 0.4)
        self.smoothed = None

    def measure(self, bgr, prev=None, alpha=None):
        """Measure clip opening angle using PCA. Returns smoothed angle."""
        if alpha is None:
            alpha = self.alpha
        raw = self._pca(bgr)
        if raw is None:
            return prev

        if prev is None:
            prev = raw
        smoothed = alpha * raw + (1 - alpha) * prev
        return smoothed

    def _pca(self, bgr):
        if bgr is None:
            return None
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([155, 80, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        body = None
        for c in cnts:
            if cv2.contourArea(c) > 100:
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
        a1 = math.degrees(math.atan2(vy1.item(), vx1.item()))
        a2 = math.degrees(math.atan2(vy2.item(), vx2.item()))
        ca = abs(a1 - a2)
        if ca > 180: ca = 360 - ca
        if ca > 90: ca = 180 - ca
        return ca
