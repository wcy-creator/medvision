"""Camera Tool - Cross-platform (Linux V4L2 + Windows DirectShow)."""
import os, sys, time, numpy as np, cv2

class CameraTool:
    def __init__(self, config):
        device_id = config.get("device_id", 0)
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("width", 640))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("height", 480))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.get("buffer_size", 1))
        time.sleep(0.3)
        if not self.cap.isOpened():
            # Try common Windows indices
            for idx in [0, 1, 2]:
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    print("[Camera] Found on device index %d" % idx)
                    break

    def is_open(self):
        return self.cap.isOpened()

    def capture(self):
        ret, bgr = self.cap.read()
        return bgr if ret else None

    def close(self):
        self.cap.release()
