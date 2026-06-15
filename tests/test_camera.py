"""Tests for CameraTool."""
import sys, pytest, time
sys.path.insert(0, "/opt/medvision/harness")
from harness_camera import CameraTool

CONFIG = {"device_id": 0, "width": 640, "height": 480, "buffer_size": 1}


def test_capture():
    """Test camera capture."""
    cam = CameraTool(CONFIG)
    assert cam.is_open(), "Camera not opened"
    bgr = cam.capture()
    assert bgr is not None, "Capture returned None"
    assert bgr.shape == (480, 640, 3), f"Wrong shape: {bgr.shape}"
    print("  capture: %s OK" % str(bgr.shape))
    cam.close()


def test_capture_speed():
    """Test capture frame rate."""
    cam = CameraTool(CONFIG)
    t0 = time.time()
    for _ in range(10):
        cam.capture()
    fps = 10 / (time.time() - t0)
    print("  speed: %.1f FPS" % fps)
    assert fps > 5, f"Too slow: {fps} FPS"
    cam.close()


if __name__ == "__main__":
    test_capture()
    test_capture_speed()
    print("\nAll camera tests passed!")
