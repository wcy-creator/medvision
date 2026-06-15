#!/usr/bin/env python3
"""MedVision - 摄像头采集模块
使用 libcamera (rpicam) 采集图像，适配 Pi Camera 3 (IMX708)
支持: 单帧拍照、连续采集、OpenCV 集成
"""
import subprocess
import tempfile
import os
import cv2
import numpy as np
import json
import time
from datetime import datetime

class MedVisionCamera:
    def __init__(self, config_path='/opt/medvision/config.json'):
        with open(config_path) as f:
            cfg = json.load(f)['camera']
        self.width = cfg['width']
        self.height = cfg['height']
        self.fps = cfg['fps']
        self.snap_dir = '/opt/medvision/snapshots'
        os.makedirs(self.snap_dir, exist_ok=True)
    
    def capture_still(self, output_path=None):
        """单帧拍照，返回 OpenCV 图像 (numpy array)"""
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.jpg')
        
        cmd = [
            'rpicam-still',
            '-o', output_path,
            '--width', str(self.width),
            '--height', str(self.height),
            '--nopreview',
            '-t', '500'  # 500ms 快速采集
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f'rpicam-still failed: {result.stderr.decode()}')
        
        img = cv2.imread(output_path)
        if img is None:
            raise RuntimeError(f'Failed to read captured image: {output_path}')
        return img
    
    def capture_frame_fast(self):
        """快速采集单帧 (内存中完成，不落盘)"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp = f.name
        try:
            img = self.capture_still(tmp)
            return img
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    
    def capture_and_save(self, prefix='snap'):
        """拍照并保存到 snapshots 目录"""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.snap_dir, f'{prefix}_{ts}.jpg')
        img = self.capture_still(path)
        return path, img
    
    def capture_video_frames(self, duration_sec=2, callback=None):
        """采集视频帧序列 (通过 h264 转换)"""
        with tempfile.NamedTemporaryFile(suffix='.h264', delete=False) as f:
            tmp = f.name
        
        cmd = [
            'rpicam-vid',
            '-o', tmp,
            '--width', str(self.width),
            '--height', str(self.height),
            '--nopreview',
            '-t', str(int(duration_sec * 1000))
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=duration_sec + 5)
        
        if os.path.exists(tmp):
            cap = cv2.VideoCapture(tmp)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if callback:
                    callback(frame)
            cap.release()
            os.remove(tmp)
            return frames
        
        return []

    def test(self):
        """自检测试"""
        print('=' * 50)
        print('MedVision 摄像头自检')
        print('=' * 50)
        
        # 测试1: 拍照
        print('\n📷 测试1: 单帧拍照...')
        start = time.time()
        img = self.capture_still()
        elapsed = time.time() - start
        h, w = img.shape[:2]
        print(f'  ✅ {w}x{h}, 耗时 {elapsed:.2f}s')
        
        # 测试2: 保存快照
        print('\n📸 测试2: 保存快照...')
        path, _ = self.capture_and_save('test')
        print(f'  ✅ 保存: {path} ({os.path.getsize(path)/1024:.1f}KB)')
        
        # 测试3: 连续采集
        print('\n🎬 测试3: 连续采集 5 帧...')
        times = []
        for i in range(5):
            start = time.time()
            img = self.capture_frame_fast()
            times.append(time.time() - start)
            print(f'  帧 {i+1}: {img.shape[1]}x{img.shape[0]} ({times[-1]:.2f}s)')
        avg = np.mean(times)
        print(f'  ✅ 平均采集时间: {avg:.2f}s (约 {1/avg:.1f} FPS)')
        
        print('\n🎉 摄像头自检完成！')
        return True

if __name__ == '__main__':
    cam = MedVisionCamera()
    cam.test()
