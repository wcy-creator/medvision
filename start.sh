#!/bin/bash
# ============================================
# MedVision Harness - Universal Start Script
# Works on: Linux, macOS, WSL
# Insert USB → Run this script → Ready to go
# ============================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════╗"
echo "║  MedVision Harness - Starting...     ║"
echo "╚══════════════════════════════════════╝"

# --- Step 1: Detect Python ---
PYTHON=""
for p in python3 python; do
    if command -v $p &>/dev/null; then
        PYTHON=$p
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python not found. Please install Python 3.8+"
    exit 1
fi
echo "[OK] Python: $($PYTHON --version)"

# --- Step 2: Install dependencies ---
echo "[Setup] Installing dependencies..."
$PYTHON -m pip install -q -r requirements.txt 2>/dev/null || \
$PYTHON -m pip install -q opencv-python numpy flask requests openni 2>/dev/null
echo "[OK] Dependencies installed"

# --- Step 3: Check hardware ---
echo "[Check] Detecting hardware..."
if [ -e /dev/ttyUSB0 ]; then
    echo "[OK] Serial port: /dev/ttyUSB0"
elif [ -e /dev/ttyACM0 ]; then
    echo "[OK] Serial port: /dev/ttyACM0"
else
    echo "[WARN] No serial port found (gimbal may not be connected)"
fi

if [ -e /dev/video0 ]; then
    echo "[OK] Camera: /dev/video0"
else
    echo "[WARN] No camera found"
fi

# --- Step 4: Load API config ---
if [ -f config/api.json ]; then
    API_PROVIDER=$(cat config/api.json | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('provider','none'))")
    echo "[OK] LLM Provider: $API_PROVIDER"
else
    echo "[WARN] No API config found. Copy config/api.json.example to config/api.json and fill in your API key."
fi

# --- Step 5: Start ---
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  Starting MedVision Harness...       ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Start web server in background
if [ -f harness/web/server.js ] && command -v node &>/dev/null; then
    cd harness/web && node server.js &
    WEB_PID=$!
    echo "[OK] Web UI: http://localhost:3000 (PID=$WEB_PID)"
fi

# Start agent
$PYTHON -c "
import sys; sys.path.insert(0, 'harness')
from harness_agent import Agent
agent = Agent()
agent.tracking = True
agent.run()
"
