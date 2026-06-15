"""Detection Tool - Color + shape + size multi-feature detection."""
import numpy as np, cv2


class DetectTool:
    def __init__(self, config):
        self.min_area = config.get("min_contour_area", 100)
        self.hsv_lower1 = np.array(config.get("hsv_red_lower1", [0, 80, 80]))
        self.hsv_upper1 = np.array(config.get("hsv_red_upper1", [15, 255, 255]))
        self.hsv_lower2 = np.array(config.get("hsv_red_lower2", [155, 80, 80]))
        self.hsv_upper2 = np.array(config.get("hsv_red_upper2", [180, 255, 255]))

    def find(self, bgr):
        """Find largest red object. Returns (cx, cy, area) or None."""
        if bgr is None:
            return None
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        m2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(c) > self.min_area:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), int(cv2.contourArea(c))
        return None
