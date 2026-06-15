@echo off
echo ============================================
echo   MedVision Harness - Windows Start
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.8+
    pause
    exit /b 1
)
echo [OK] Python found

REM Install dependencies
echo [Setup] Installing dependencies...
python -m pip install -q opencv-python numpy flask requests pyserial 2>nul
echo [OK] Dependencies installed

REM Check camera
python -c "import cv2; c=cv2.VideoCapture(0); print('[OK] Camera:', 'Found' if c.isOpened() else 'Not found'); c.release()" 2>nul

REM Check serial
python -c "import serial.tools.list_ports; ports = list(serial.tools.list_ports.comports()); print('[OK] COM ports:', [p.device for p in ports])" 2>nul

REM Start agent
echo.
echo Starting MedVision Agent...
python -c "import sys; sys.path.insert(0, 'harness'); from harness_agent import Agent; a = Agent(); a.tracking = True; a.run()"

pause
