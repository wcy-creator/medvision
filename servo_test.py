#!/usr/bin/env python3
"""MedVision - 舵机测试脚本
测试 SG90 舵机控制 (杜邦线到货后使用)
接线: GPIO18 -> SG90 信号线, GND -> GND, 5V -> VCC
"""
import json
import time
import sys

def load_config():
    with open('/opt/medvision/config.json') as f:
        return json.load(f)

def test_servo_gpiozero():
    """使用 gpiozero 测试舵机 (推荐，Pi5兼容)"""
    from gpiozero import Servo, Device
    from gpiozero.pins.lgpio import LGPIOFactory
    
    config = load_config()
    gimbal = config['gimbal']
    
    print('='*50)
    print('MedVision 舵机测试 (gpiozero)')
    print('='*50)
    print(f'\n🔧 舵机1 引脚: GPIO{gimbal["servo1_pin"]}')
    print(f'🔧 舵机2 引脚: GPIO{gimbal["servo2_pin"]}')
    
    # 使用 lgpio 后端 (Pi 5 原生支持)
    factory = LGPIOFactory()
    
    # 限制 PWM 脉宽范围 (SG90: 500-2500us)
    servo1 = Servo(
        gimbal['servo1_pin'],
        min_pulse_width=gimbal['min_pulse']/1_000_000,
        max_pulse_width=gimbal['max_pulse']/1_000_000,
        pin_factory=factory
    )
    servo2 = Servo(
        gimbal['servo2_pin'],
        min_pulse_width=gimbal['min_pulse']/1_000_000,
        max_pulse_width=gimbal['max_pulse']/1_000_000,
        pin_factory=factory
    )
    
    try:
        # 居中测试
        print('\n📍 测试1: 居中 (90°)')
        servo1.mid()
        servo2.mid()
        time.sleep(1)
        print('  ✅ 居中完成')
        
        # 左转测试
        print('\n📍 测试2: 左转 (0°)')
        servo1.min()
        servo2.min()
        time.sleep(1)
        print('  ✅ 左转完成')
        
        # 右转测试
        print('\n📍 测试3: 右转 (180°)')
        servo1.max()
        servo2.max()
        time.sleep(1)
        print('  ✅ 右转完成')
        
        # 平滑移动测试
        print('\n📍 测试4: 平滑移动')
        steps = 20
        for i in range(steps + 1):
            val = -1.0 + (2.0 * i / steps)  # -1.0 to 1.0
            servo1.value = val
            servo2.value = val
            angle = 90 * (val + 1)  # 0 to 180
            print(f'  角度: {angle:.0f}°', end='\r')
            time.sleep(0.05)
        print(f'  角度: 180° ✅')
        
        # 回中
        print('\n📍 回到中心...')
        servo1.mid()
        servo2.mid()
        time.sleep(0.5)
        
        print('\n🎉 舵机测试全部通过！')
        return True
        
    except Exception as e:
        print(f'\n❌ 舵机测试失败: {e}')
        return False
    finally:
        servo1.close()
        servo2.close()

def test_servo_manual():
    """不接线的模拟测试 (验证代码逻辑)"""
    import RPi.GPIO as GPIO
    
    config = load_config()
    gimbal = config['gimbal']
    
    print('='*50)
    print('MedVision 舵机测试 (手动PWM - RPi.GPIO)')
    print('='*50)
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(gimbal['servo1_pin'], GPIO.OUT)
    GPIO.setup(gimbal['servo2_pin'], GPIO.OUT)
    
    pwm1 = GPIO.PWM(gimbal['servo1_pin'], 50)
    pwm2 = GPIO.PWM(gimbal['servo2_pin'], 50)
    pwm1.start(0)
    pwm2.start(0)
    
    def set_angle(pwm, angle):
        duty = 2.5 + (angle / 180.0) * 10.0
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.3)
        pwm.ChangeDutyCycle(0)
    
    try:
        angles = [0, 45, 90, 135, 180, 90]
        for angle in angles:
            print(f'  设置角度: {angle}°')
            set_angle(pwm1, angle)
            set_angle(pwm2, angle)
            time.sleep(0.5)
        
        print('\n🎉 舵机测试完成')
    finally:
        pwm1.stop()
        pwm2.stop()
        GPIO.cleanup()

if __name__ == '__main__':
    if '--manual' in sys.argv:
        test_servo_manual()
    else:
        test_servo_gpiozero()
