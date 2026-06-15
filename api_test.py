#!/usr/bin/env python3
"""MedVision - 云端API测试脚本
测试 MiMo API 图像识别功能
"""
import json
import time
import base64
import sys
import os
import requests
from datetime import datetime

def load_config():
    with open('/opt/medvision/config.json') as f:
        return json.load(f)

def test_api(image_path=None):
    config = load_config()
    api = config['api']
    
    print('='*50)
    print('MedVision 云端API测试')
    print('='*50)
    print(f'\n🌐 URL: {api["base_url"]}')
    print(f'🤖 模型: {api["model"]}')
    
    # 如果没有图片，先拍一张
    if not image_path:
        image_path = '/opt/medvision/snapshots/api_test.jpg'
        print('\n📷 拍摄测试图片...')
        ret = os.system(f'rpicam-still -o {image_path} --width 640 --height 480 --nopreview -t 1000')
        if ret != 0:
            print('  ❌ 拍摄失败，使用测试模式')
            # 创建一个简单的测试图片
            try:
                import cv2
                import numpy as np
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                img[200:280, 280:360] = [0, 255, 0]  # 绿色方块
                cv2.imwrite(image_path, img)
                print('  ⚠️ 已生成测试图片')
            except Exception as e:
                print(f'  ❌ 无法创建测试图片: {e}')
                return False
    
    if not os.path.exists(image_path):
        print(f'  ❌ 图片不存在: {image_path}')
        return False
    
    # 编码图片
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    print(f'  图片大小: {os.path.getsize(image_path)/1024:.1f}KB')
    print(f'  Base64 长度: {len(img_b64)}')
    
    # 发送API请求
    headers = {
        'Authorization': f'Bearer {api["api_key"]}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': api['model'],
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': '请识别图片中的手术器械，返回JSON格式: {"instruments": [{"name": "器械名称", "confidence": 置信度, "bbox": [x,y,w,h]}]}'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{img_b64}'
                        }
                    }
                ]
            }
        ],
        'max_tokens': 512,
        'temperature': 0.1
    }
    
    print('\n📡 发送API请求...')
    start = time.time()
    
    try:
        response = requests.post(
            f'{api["base_url"]}/chat/completions',
            headers=headers,
            json=payload,
            timeout=api['timeout']
        )
        elapsed = time.time() - start
        
        print(f'  ⏱️ 响应时间: {elapsed:.2f}s')
        print(f'  📊 状态码: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f'  ✅ API响应成功!')
            print(f'  📝 响应内容:')
            print(f'     {content[:500]}')
            return True
        else:
            print(f'  ❌ API错误: {response.text[:200]}')
            return False
            
    except requests.exceptions.Timeout:
        print(f'  ❌ 请求超时 ({api["timeout"]}s)')
        return False
    except Exception as e:
        print(f'  ❌ 请求失败: {e}')
        return False

if __name__ == '__main__':
    image = sys.argv[1] if len(sys.argv) > 1 else None
    test_api(image)
