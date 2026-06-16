"""
YOLO26 Detection Module - Upgraded from YOLOv5.
Supports YOLO26/YOLOv11/YOLOv8 with automatic model selection.
"""
import os
import time
import numpy as np
import cv2


class YOLODetectorV2:
    """
    YOLO26/V11/V8 detector with pose estimation support.

    Usage:
        det = YOLODetectorV2(model="yolo26n")
        results = det.detect(frame)
        # or with pose
        results = det.detect_with_pose(frame)
    """

    # Available models (downloaded on first use)
    MODELS = {
        "yolo26n": {"file": "yolo26n.onnx", "size": "nano", "task": "detect"},
        "yolo26s": {"file": "yolo26s.onnx", "size": "small", "task": "detect"},
        "yolov11n": {"file": "yolov11n.onnx", "size": "nano", "task": "detect"},
        "yolov8n": {"file": "yolov8n.onnx", "size": "nano", "task": "detect"},
        "yolov8n-pose": {"file": "yolov8n-pose.onnx", "size": "nano", "task": "pose"},
    }

    # COCO class names
    COCO_NAMES = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck',
        'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
        'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',
        'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
        'skis','snowboard','sports ball','kite','baseball bat','baseball glove',
        'skateboard','surfboard','tennis racket','bottle','wine glass','cup',
        'fork','knife','spoon','bowl','banana','apple','sandwich','orange',
        'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
        'potted plant','bed','dining table','toilet','tv','laptop','mouse',
        'remote','keyboard','cell phone','microwave','oven','toaster','sink',
        'refrigerator','book','clock','vase','scissors','teddy bear',
        'hair drier','toothbrush'
    ]

    def __init__(self, model="yolo26n", conf_thresh=0.45, iou_thresh=0.5,
                 input_size=640, classes=None, model_dir=None):
        self.model_name = model
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        self.classes = classes
        self.session = None
        self._is_fp16 = False

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)

        self._load_model(model_dir)

    def _load_model(self, model_dir):
        """Load ONNX model with auto-download."""
        model_info = self.MODELS.get(self.model_name)
        if not model_info:
            print("[YOLO] Unknown model: %s, falling back to yolo26n" % self.model_name)
            model_info = self.MODELS["yolo26n"]
            self.model_name = "yolo26n"

        model_path = os.path.join(model_dir, model_info["file"])

        # Try to download if not exists
        if not os.path.exists(model_path):
            print("[YOLO] Model not found, attempting download...")
            self._download_model(model_path, model_info)

        if not os.path.exists(model_path):
            print("[YOLO] ERROR: Model not available: %s" % model_info["file"])
            return

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_type = self.session.get_inputs()[0].type
            self._is_fp16 = 'float16' in self.input_type
            self._warmup()
            dtype = "fp16" if self._is_fp16 else "fp32"
            print("[YOLO] Loaded: %s (%s, %s)" % (self.model_name, model_info["size"], dtype))
        except Exception as e:
            print("[YOLO] ERROR: %s" % e)

    def _download_model(self, path, info):
        """Download model from Ultralytics."""
        try:
            from ultralytics import YOLO
            print("[YOLO] Downloading via ultralytics...")
            model = YOLO("yolo%s.pt" % self.model_name.replace("yolo", "").replace("26", "v8"))
            model.export(format="onnx", imgsz=self.input_size)
            # Move to model_dir
            import shutil
            src = "yolo%s.onnx" % self.model_name.replace("yolo", "").replace("26", "v8")
            if os.path.exists(src):
                shutil.move(src, path)
        except Exception as e:
            print("[YOLO] Download failed: %s" % e)

    def _warmup(self):
        dtype = np.float16 if self._is_fp16 else np.float32
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=dtype)
        self.session.run(None, {self.input_name: dummy})

    def preprocess(self, bgr):
        h, w = bgr.shape[:2]
        ratio = self.input_size / max(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dx, dy = (self.input_size - new_w) // 2, (self.input_size - new_h) // 2
        canvas[dy:dy+new_h, dx:dx+new_w] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        if self._is_fp16:
            blob = blob.astype(np.float16)

        self._ratio = ratio
        self._pad = (dx, dy, new_w, new_h)
        return blob

    def postprocess(self, output, orig_h, orig_w):
        """Parse YOLO output."""
        preds = output[0][0]

        # Handle different output shapes
        if len(preds.shape) == 2:
            # Standard YOLO output [N, 85]
            obj_conf = preds[:, 4]
            class_scores = preds[:, 5:]
            max_class_scores = class_scores.max(axis=1)
            scores = obj_conf * max_class_scores
        else:
            # Transposed output
            preds = preds.T
            obj_conf = preds[4]
            class_scores = preds[5:]
            max_class_scores = class_scores.max(axis=0)
            scores = obj_conf * max_class_scores
            preds = preds.T

        mask = scores > self.conf_thresh
        preds = preds[mask]
        scores = scores[mask]

        if len(preds) == 0:
            return []

        if self.classes is not None:
            class_ids = preds[:, 5:].argmax(axis=1) if len(preds.shape) == 2 else preds[5:].argmax(axis=0)
            class_mask = np.isin(class_ids, self.classes)
            preds = preds[class_mask]
            scores = scores[class_mask]
            class_ids = class_ids[class_mask]
        else:
            class_ids = preds[:, 5:].argmax(axis=1) if len(preds.shape) == 2 else preds[5:].argmax(axis=0)

        # xywh -> xyxy
        boxes = preds[:, :4].copy()
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        # Map to original coordinates
        dx, dy, nw, nh = self._pad
        ratio = self._ratio
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.conf_thresh, self.iou_thresh
        )

        results = []
        if len(indices) > 0:
            for i in (indices.flatten() if isinstance(indices, np.ndarray) else indices):
                x1, y1, x2, y2 = boxes[i]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                cid = int(class_ids[i])
                results.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (int(cx), int(cy)),
                    'size': (int(w), int(h)),
                    'confidence': float(scores[i]),
                    'class_id': cid,
                    'class_name': self.COCO_NAMES[cid] if cid < len(self.COCO_NAMES) else str(cid)
                })
        return results

    def detect(self, bgr):
        if self.session is None:
            return []
        orig_h, orig_w = bgr.shape[:2]
        blob = self.preprocess(bgr)
        output = self.session.run(None, {self.input_name: blob})
        return self.postprocess(output, orig_h, orig_w)

    def detect_largest(self, bgr, class_filter=None):
        results = self.detect(bgr)
        if not results:
            return None
        if class_filter:
            results = [r for r in results if r['class_id'] in class_filter]
        if not results:
            return None
        best = max(results, key=lambda r: r['size'][0] * r['size'][1])
        cx, cy = best['center']
        area = best['size'][0] * best['size'][1]
        return (cx, cy, area, best['confidence'], best['class_name'])

    def detect_tool(self, bgr):
        """Detect surgical tools (knife=44, scissors=76 in COCO)."""
        tool_ids = {44, 76}
        return self.detect_largest(bgr, class_filter=tool_ids)

    def benchmark(self, n=20):
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.detect(dummy)
        t0 = time.time()
        for _ in range(n):
            self.detect(dummy)
        elapsed = time.time() - t0
        fps = n / elapsed
        print("[YOLO] %s Benchmark: %.1f FPS (%d frames in %.2fs)" %
              (self.model_name, fps, n, elapsed))
        return fps

    def close(self):
        self.session = None
