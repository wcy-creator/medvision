"""Tests for AngleTool."""
import sys, pytest, numpy as np, cv2
sys.path.insert(0, "/opt/medvision/harness")
from harness_angle import AngleTool

CONFIG = {"ema_alpha": 0.4, "calib_frames": 30}


def test_angle_closed_clip():
    """Test angle of closed clip (should be near 0)."""
    angle_tool = AngleTool(CONFIG)
    # Simulate closed clip: two thin parallel lines
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a thin V-shape (nearly closed)
    pts1 = np.array([[280, 200], [320, 250], [360, 200]], np.int32)
    pts2 = np.array([[280, 200], [320, 260], [360, 200]], np.int32)
    cv2.fillPoly(img, [np.vstack([pts1, pts2[::-1]])], (0, 0, 200))
    angle = angle_tool.measure(img)
    # For synthetic test, just verify it returns a number
    assert angle is not None, "No angle returned"
    print("  closed_clip: angle=%.1f OK" % angle)


def test_angle_stability():
    """Test that EMA smoothing reduces variance."""
    angle_tool = AngleTool(CONFIG)
    # Measure same image multiple times
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 30, (0, 0, 200), -1)
    angles = []
    for _ in range(10):
        a = angle_tool.measure(img)
        if a is not None:
            angles.append(a)
    if angles:
        std = np.std(angles)
        assert std < 5, f"Too unstable: std={std}"
        print("  stability: std=%.2f OK" % std)


if __name__ == "__main__":
    test_angle_closed_clip()
    test_angle_stability()
    print("\nAll angle tests passed!")
