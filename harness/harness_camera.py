"""Camera Tool - V4L2 capture + optional OpenNI2 depth."""
import time, numpy as np, cv2


class CameraTool:
    def __init__(self, config):
        self.cap = cv2.VideoCapture(config.get("device_id", 0))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("width", 640))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("height", 480))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.get("buffer_size", 1))
        time.sleep(0.3)

    def is_open(self):
        return self.cap.isOpened()

    def capture(self):
        ret, bgr = self.cap.read()
        return bgr if ret else None

    def get_depth(self):
        """Capture depth via OpenNI2 (separate call, slow)."""
        from openni import openni2
        openni2.initialize()
        dev = openni2.Device.open_any()
        ds = dev.create_depth_stream()
        ds.start()
        time.sleep(0.5)
        df = ds.read_frame()
        dd = np.array(df.get_buffer_as_triplet()).reshape([480, 640, 2])
        depth = np.asarray(dd[:, :, 0], "float32") + np.asarray(dd[:, :, 1], "float32") * 255
        ds.stop()
        dev.close()
        openni2.unload()
        return depth

    def close(self):
        self.cap.release()
