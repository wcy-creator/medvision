"""Tests for DetectTool."""
import sys, pytest, numpy as np, cv2
sys.path.insert(0, "/opt/medvision/harness")
from harness_detect import DetectTool

CONFIG = {"min_contour_area": 100, "hsv_red_lower1": [0,80,80], "hsv_red_upper1": [15,255,255],
          "hsv_red_lower2": [155,80,80], "hsv_red_upper2": [180,255,255]}


def test_find_red():
    """Test detection of red object in synthetic image."""
    det = DetectTool(CONFIG)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 50, (0, 0, 255), -1)
    result = det.find(img)
    assert result is not None, "Failed to detect red circle"
    cx, cy, area = result
    assert abs(cx - 320) < 20, "X wrong: %d" % cx
    assert abs(cy - 240) < 20, "Y wrong: %d" % cy
    print("  find_red: pos=(%d,%d) area=%d OK" % (cx, cy, area))


def test_no_red():
    """Test no detection in all-blue image."""
    det = DetectTool(CONFIG)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    result = det.find(img)
    assert result is None, "False positive: %s" % result
    print("  no_red: None OK")


def test_multiple_red():
    """Test detection picks largest red object."""
    det = DetectTool(CONFIG)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 15, (0, 0, 255), -1)  # Small
    cv2.circle(img, (400, 300), 50, (0, 0, 255), -1)  # Large
    result = det.find(img)
    assert result is not None, "No detection"
    cx, cy, area = result
    assert 350 < cx < 450, "Should pick large: cx=%d" % cx
    print("  multi_red: picks largest at (%d,%d) OK" % (cx, cy))


def test_red_rectangle():
    """Test detection of rectangular red object."""
    det = DetectTool(CONFIG)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (250, 150), (390, 330), (0, 0, 255), -1)
    result = det.find(img)
    assert result is not None, "Rectangle not detected"
    cx, cy, area = result
    assert 300 < cx < 340, "X wrong: %d" % cx
    print("  red_rect: pos=(%d,%d) area=%d OK" % (cx, cy, area))


def test_ema_stability():
    """Test EMA smoothing reduces jitter across frames."""
    det = DetectTool(CONFIG)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 40, (0, 0, 255), -1)
    det.find(img)  # init
    results = []
    for _ in range(5):
        r = det.find(img)
        if r:
            results.append(r[:2])
    if len(results) > 1:
        xs = [r[0] for r in results]
        std = np.std(xs)
        assert std < 5, "EMA too noisy: std=%.1f" % std
        print("  ema_stable: std=%.2f OK" % std)


if __name__ == "__main__":
    test_find_red()
    test_no_red()
    test_multiple_red()
    test_red_rectangle()
    test_ema_stability()
    print("\nAll detect tests passed!")
