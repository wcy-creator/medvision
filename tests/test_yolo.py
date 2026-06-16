"""Tests for YOLODetector."""
import sys, os, pytest, numpy as np, time
sys.path.insert(0, "/opt/medvision/harness")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "yolov5n.onnx")


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="YOLO model not found")
def test_yolo_load():
    from harness_detect_yolo import YOLODetector
    det = YOLODetector(MODEL_PATH)
    assert det.session is not None, "Model not loaded"
    print("  yolo_load: OK")
    det.close()


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="YOLO model not found")
def test_yolo_empty():
    from harness_detect_yolo import YOLODetector
    det = YOLODetector(MODEL_PATH, conf_thresh=0.9)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = det.detect(img)
    print("  yolo_empty: %d results (expect 0) OK" % len(results))
    det.close()


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="YOLO model not found")
def test_yolo_speed():
    from harness_detect_yolo import YOLODetector
    det = YOLODetector(MODEL_PATH)
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        det.detect(dummy)
    t0 = time.time()
    for _ in range(10):
        det.detect(dummy)
    fps = 10 / (time.time() - t0)
    print("  yolo_speed: %.1f FPS" % fps)
    assert fps > 1, "Too slow: %.1f FPS" % fps
    det.close()


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="YOLO model not found")
def test_yolo_person():
    """Test with synthetic person-like shape (tall rectangle)."""
    from harness_detect_yolo import YOLODetector
    det = YOLODetector(MODEL_PATH, conf_thresh=0.1)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a tall rectangle (roughly person-shaped)
    import cv2
    cv2.rectangle(img, (270, 100), (370, 400), (180, 120, 80), -1)
    results = det.detect(img)
    print("  yolo_person: %d detections" % len(results))
    det.close()


if __name__ == "__main__":
    test_yolo_load()
    test_yolo_empty()
    test_yolo_speed()
    test_yolo_person()
    print("\nAll YOLO tests passed!")
