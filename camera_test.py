#!/usr/bin/env python3
"""MedVision - 摄像头测试脚本
测试 Pi Camera 3 (IMX708) 的图像采集功能
使用 rpicam-still 采集（适配树莓派5 libcamera 架构）
"""
import cv2
import time
import sys
import os
import json
import subprocess
import tempfile
from datetime import datetime

def load_config():
    with open('/opt/medvision/config.json') as f:
        return json.load(f)

def test_camera_rpicam(config):
    """主方案: 使用 rpicam-still 采集（Pi Camera 3 推荐方式）"""
    cam_cfg = config['camera']
    w, h = cam_cfg['width'], cam_cfg['height']
    
    print(f'\n📷 分辨率: {w}x{h}')
    print(f'  采集方式: rpicam-still (libcamera)')
    
    # 测试1: 单帧拍照
    print('\n⏳ 测试1: 单帧拍照...')
    snap_path = f'/opt/medvision/snapshots/test_{datetime.now():%Y%m%d_%H%M%S}.jpg'
    cmd = ['rpicam-still', '-o', snap_path, '--width', str(w), '--height', str(h), '--nopreview', '-t', '500']
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, timeout=15)
    elapsed = time.time() - start
    
    if result.returncode == 0 and os.path.exists(snap_path):
        size = os.path.getsize(snap_path)
        img = cv2.imread(snap_path)
        if img is not None:
            ih, iw = img.shape[:2]
            print(f'  ✅ {iw}x{ih}, {size/1024:.1f}KB, 耗时 {elapsed:.2f}s')
            print(f'  📸 快照已保存: {snap_path}')
        else:
            print(f'  ❌ 图片读取失败')
    else:
        err = result.stderr.decode() if result.stderr else 'unknown'
        print(f'  ❌ 拍照失败: {err}')
        return False
    
    # 测试2: 连续采集 5 帧
    print('\n🎬 测试2: 连续采集 5 帧...')
    times = []
    for i in range(5):
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp = f.name
        try:
            start = time.time()
            result = subprocess.run(
                ['rpicam-still', '-o', tmp, '--width', str(w), '--height', str(h), '--nopreview', '-t', '500'],
                capture_output=True, timeout=15
            )
            elapsed = time.time() - start
            times.append(elapsed)
            
            if result.returncode == 0 and os.path.exists(tmp):
                img = cv2.imread(tmp)
                if img is not None:
                    print(f'  帧 {i+1}: {img.shape[1]}x{img.shape[0]} ✅ ({elapsed:.2f}s)')
                else:
                    print(f'  帧 {i+1}: 读取失败 ❌')
            else:
                print(f'  帧 {i+1}: 采集失败 ❌')
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    
    if times:
        import numpy as np
        avg = np.mean(times)
        print(f'  ✅ 平均采集时间: {avg:.2f}s (约 {1/avg:.1f} FPS)')
    
    print('\n✅ 摄像头测试完成')
    return True

def test_camera_v4l2_fallback(config):
    """备用方案: 尝试 V4L2（某些配置可能可用）"""
    cam_cfg = config['camera']
    print('\n⚠️  尝试 V4L2 备用方案...')
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print('  ❌ V4L2 无法打开')
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg['height'])
    cap.set(cv2.CAP_PROP_FPS, cam_cfg['fps'])
    time.sleep(1)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        print(f'  ✅ V4L2 可用: {frame.shape[1]}x{frame.shape[0]}')
        return True
    else:
        print('  ❌ V4L2 read 失败（Pi Camera 3 需要 libcamera）')
        return False

def main():
    config = load_config()
    
    print('='*50)
    print('MedVision 摄像头测试')
    print('='*50)
    
    # 优先使用 rpicam (libcamera) 方式
    success = test_camera_rpicam(config)
    
    if not success:
        # 回退到 V4L2
        test_camera_v4l2_fallback(config)

if __name__ == '__main__':
    main()
