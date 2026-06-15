"""MedVision Unified Server v1.0
Astra S 3D Stream + AI Analysis + PID Tracking + Gimbal Control
One server, one port (8080), browser-based control.

Access: http://10.93.111.126:8080/
"""
import os, sys, time, json, base64, threading, tempfile, numpy as np, cv2
from datetime import datetime
from flask import Flask, Response, jsonify, request
sys.path.insert(0, "/opt/medvision")

# ─── Config ───
SNAP_DIR = "/opt/medvision/snapshots"
LOG_DIR = "/opt/medvision/logs"
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def load_config():
    with open("/opt/medvision/config.json") as f:
        return json.load(f)

# ─── Globals ───
app = Flask(__name__)
_color_jpeg = None
_depth_jpeg = None
_frame_lock = threading.Lock()
_running = True
_fc = 0
_t0 = time.time()
_depth_info = {"min": 0, "max": 0, "valid": 0}

# Tracking state
tracking = False
method = "motion"
dead_zone = 30
target_pos = None
depth_mm = 0
_last_analysis = None
_ai_busy = False

# Gimbal & PID
_gimbal = None
_pid_pan = None
_pid_tilt = None

# ─── PID Controller ───
class PIDController:
    def __init__(self, kp=0.06, ki=0.002, kd=0.025, limit=4.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.integral = 0.0
        self.prev_err = 0.0

    def update(self, err):
        self.integral = max(-self.limit, min(self.limit, self.integral + err))
        derivative = err - self.prev_err
        self.prev_err = err
        out = self.kp * err + self.ki * self.integral + self.kd * derivative
        return max(-self.limit, min(self.limit, out))

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0

# ─── Camera + Depth Thread ───
def camera_loop():
    global _color_jpeg, _depth_jpeg, _depth_info, _fc, _running

    from openni import openni2

    # Init depth (OpenNI2)
    openni2.initialize()
    dev = openni2.Device.open_any()
    print("[Astra] Device: %s" % dev.get_device_info().name)
    ds = dev.create_depth_stream()
    ds.start()
    time.sleep(1)

    # Init color (V4L2)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(0.5)
    print("[Astra] V4L2: %s" % ("OK" if cap.isOpened() else "FAIL"))

    enc = [cv2.IMWRITE_JPEG_QUALITY, 60]

    while _running:
        try:
            # Depth
            df = ds.read_frame()
            dd = np.array(df.get_buffer_as_triplet()).reshape([480, 640, 2])
            dpt = np.asarray(dd[:, :, 0], dtype="float32") + \
                  np.asarray(dd[:, :, 1], dtype="float32") * 255
            valid = dpt[dpt > 0]
            if len(valid) > 0:
                dmin, dmax = float(np.min(valid)), float(np.max(valid))
                dpt_vis = np.clip(dpt / max(dmax, 1) * 255, 0, 255).astype(np.uint8)
            else:
                dmin, dmax, dpt_vis = 0, 0, np.zeros((480, 640), dtype=np.uint8)
            dpt_color = cv2.applyColorMap(dpt_vis, cv2.COLORMAP_JET)
            _, db = cv2.imencode(".jpg", dpt_color, enc)

            # Color
            ret, bgr = cap.read()
            c_bytes = None
            if ret:
                _, cb = cv2.imencode(".jpg", bgr, enc)
                c_bytes = cb.tobytes()

            with _frame_lock:
                _depth_jpeg = db.tobytes()
                if c_bytes:
                    _color_jpeg = c_bytes
                _depth_info = {"min": dmin, "max": dmax, "valid": len(valid)}
                _fc += 1
        except Exception as e:
            print("[Astra] err: %s" % e)
            time.sleep(0.5)

# ─── AI Analysis ───
def ai_analyze(frame_bytes, prompt=None):
    """Analyze a frame using MiMo Vision API. Returns text result."""
    import requests as hr
    cfg = load_config()["api"]
    if not cfg.get("api_key") or len(cfg["api_key"]) < 10:
        return {"success": False, "error": "no api key"}

    img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    sc = 320 / max(h, w)
    if sc < 1:
        img = cv2.resize(img, (int(w * sc), int(h * sc)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    b64 = base64.b64encode(buf.tobytes()).decode()

    if not prompt:
        prompt = "Analyze this image. Identify objects, especially surgical instruments. Describe type, position, angle, and open/closed state."

    headers = {"Authorization": "Bearer " + cfg["api_key"], "Content-Type": "application/json"}
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}
        ]}],
        "max_tokens": 500,
        "temperature": 0.1
    }

    t1 = time.time()
    try:
        r = hr.post(cfg["base_url"] + "/chat/completions", headers=headers, json=payload, timeout=cfg["timeout"])
        if r.status_code == 200:
            return {"success": True,
                    "analysis": r.json().get("choices", [{}])[0].get("message", {}).get("content", ""),
                    "time": round(time.time() - t1, 1),
                    "image_bytes": len(b64)}
        return {"success": False, "error": str(r.status_code), "time": round(time.time() - t1, 1)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def local_analyze(frame_bytes):
    """Fast local OpenCV analysis (no API needed)."""
    img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    brightness = float(np.mean(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_pct = np.count_nonzero(edges) / (h * w) * 100
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_contours = [c for c in contours if cv2.contourArea(c) > 500]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    metal_mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    metal_pct = np.count_nonzero(metal_mask) / (h * w) * 100
    return {
        "brightness": round(brightness, 1),
        "edge_density": round(edge_pct, 2),
        "contour_count": len(big_contours),
        "metal_pct": round(metal_pct, 1),
        "summary": "%dx%d %s" % (w, h, "OK" if 40 < brightness < 200 else "BAD")
    }

# ─── Tracking Thread ───
def tracking_loop():
    global tracking, method, dead_zone, target_pos, depth_mm
    global _gimbal, _pid_pan, _pid_tilt, _running

    from gimbal_uart import GimbalUART
    _gimbal = GimbalUART()
    _gimbal.center()
    _pid_pan = PIDController()
    _pid_tilt = PIDController()
    time.sleep(1)
    print("[Gimbal] Ready")

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)

    while _running:
        time.sleep(0.03)  # ~30Hz loop

        if not tracking:
            target_pos = None
            depth_mm = 0
            continue

        # Get current color frame
        with _frame_lock:
            cj = _color_jpeg
        if cj is None:
            continue

        bgr = cv2.imdecode(np.frombuffer(cj, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        # Detection
        result = None
        if method == "motion":
            mask = bg_sub.apply(bgr)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 500:
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        x, y, w, h = cv2.boundingRect(largest)
                        result = {"cx": cx, "cy": cy, "area": cv2.contourArea(largest), "bbox": (x, y, w, h)}

        elif method == "bright":
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 300:
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        x, y, w, h = cv2.boundingRect(largest)
                        result = {"cx": cx, "cy": cy, "area": cv2.contourArea(largest), "bbox": (x, y, w, h)}

        if result is None:
            target_pos = None
            depth_mm = 0
            continue

        cx, cy = result["cx"], result["cy"]
        target_pos = (cx, cy)
        err_x = cx - 320
        err_y = cy - 240

        if abs(err_x) > dead_zone or abs(err_y) > dead_zone:
            dpan = -_pid_pan.update(err_x / 320.0)
            dtilt = _pid_tilt.update(err_y / 240.0)
            _gimbal.nudge(dpan, dtilt)

# ─── MJPEG Generator ───
def mjpeg_gen(getter, name):
    boundary = b"--frame"
    while _running:
        with _frame_lock:
            d = getter()
        if d:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + \
                  str(len(d)).encode() + b"\r\n\r\n" + d + b"\r\n"
        else:
            time.sleep(0.02)

# ─── HTML Page ───
HTML = """<!DOCTYPE html><html><head><meta charset=utf-8><title>MedVision Unified</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f1a;color:#e0e0e0;font-family:'Segoe UI',Arial,sans-serif;min-height:100vh}
.hdr{background:linear-gradient(90deg,#1a1a3e,#0d1b2a);padding:10px 20px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #333}
.hdr h1{color:#00d4ff;font-size:18px}
.hdr .st{padding:4px 12px;border-radius:12px;font-size:12px;font-weight:bold}
.main{display:flex;flex-wrap:wrap;gap:8px;padding:8px;justify-content:center}
.streams{display:flex;gap:6px;flex-wrap:wrap;justify-content:center}
.sb{position:relative;border:2px solid #2a2a4a;border-radius:8px;overflow:hidden;background:#111}
.sb img{display:block;max-width:100%}
.lbl{position:absolute;top:6px;left:8px;background:#000000aa;color:#00ff88;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold}
.panel{width:100%;max-width:900px}
.p{background:#16213e;border-radius:8px;padding:10px;margin-bottom:6px}
.p h3{margin:0 0 8px;color:#00d4ff;font-size:14px}
.ctrl{display:flex;justify-content:center;gap:6px;padding:6px;flex-wrap:wrap}
.btn{background:#1a1a3e;color:#00d4ff;border:1px solid #00d4ff44;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:13px;transition:all .2s}
.btn:hover{background:#00d4ff22}
.btn.on{background:#00ff8833;color:#00ff88;border-color:#00ff88}
.btn.ai{background:#ff6b35;color:#fff;border-color:#ff6b35}
.btn.ai:hover{background:#ff6b3588}
.bar{background:#16213e;padding:8px;text-align:center;color:#888;font-size:12px;display:flex;justify-content:center;gap:20px;flex-wrap:wrap}
.bar span{color:#00d4ff;font-weight:bold}
.r{display:flex;justify-content:space-between;padding:3px 0;font-size:13px}
.lb{color:#888}.vl{color:#eee;font-weight:bold}
.ai-box{background:#0d1b2a;border:1px solid #333;border-radius:6px;padding:10px;min-height:50px;font-size:13px;line-height:1.5;margin-top:6px;max-height:200px;overflow-y:auto}
.spin{display:inline-block;width:12px;height:12px;border:2px solid #555;border-top-color:#00d4ff;border-radius:50%;animation:s .6s linear infinite;margin-right:4px;vertical-align:middle}
@keyframes s{to{transform:rotate(360deg)}}
</style></head><body>
<div class="hdr">
<h1>MedVision Unified Server</h1>
<span class="st" id="st" style="color:#ff6b35">INIT</span>
</div>

<div class="main">
<div class="streams">
<div class="sb"><span class="lbl">COLOR</span><img id="ci" width="640" height="480"></div>
<div class="sb"><span class="lbl">DEPTH</span><img id="di" width="640" height="480"></div>
</div>

<div class="panel">
<div class="p">
<h3>Tracking</h3>
<div class="ctrl">
<button class="btn" id="bt" onclick="cmd('track')">Tracking OFF</button>
<button class="btn" id="bm1" onclick="cmd('m1')">1:Motion</button>
<button class="btn" id="bm2" onclick="cmd('m2')">2:Bright</button>
<button class="btn" onclick="cmd('reset')">Reset</button>
<button class="btn" onclick="cmd('snap')">Snapshot</button>
</div>
</div>

<div class="p">
<h3>AI Vision</h3>
<button class="btn ai" id="ab" onclick="aiAnalyze()">AI Analyze</button>
<div class="ai-box" id="ar">Press AI Analyze to analyze current frame</div>
</div>

<div class="p">
<h3>Info</h3>
<div class="r"><span class="lb">FPS</span><span class="vl" id="fps">--</span></div>
<div class="r"><span class="lb">Frames</span><span class="vl" id="fc">--</span></div>
<div class="r"><span class="lb">Depth Range</span><span class="vl" id="dr">--</span></div>
<div class="r"><span class="lb">Gimbal</span><span class="vl" id="gm">--</span></div>
<div class="r"><span class="lb">Target</span><span class="vl" id="tg">None</span></div>
<div class="r"><span class="lb">Method</span><span class="vl" id="md">motion</span></div>
<div class="r"><span class="lb">Uptime</span><span class="vl" id="up">--</span></div>
</div>
</div>
</div>

<div class="bar">
<span>Port: 8080</span>
<span>Astra S 3D</span>
<span>CLB-S25 Gimbal</span>
<span>MiMo Vision API</span>
</div>

<script>
// Frame polling
setInterval(function(){
var t=Date.now();
document.getElementById("ci").src="/snapshot/color?"+t;
document.getElementById("di").src="/snapshot/depth?"+t;
},100);

// Status polling
setInterval(function(){
fetch("/api/status").then(function(r){return r.json()}).then(function(d){
document.getElementById("fps").textContent=d.fps;
document.getElementById("fc").textContent=d.frames;
document.getElementById("dr").textContent=d.depth_range;
document.getElementById("up").textContent=d.uptime;
document.getElementById("gm").textContent=d.gimbal;
document.getElementById("tg").textContent=d.target;
document.getElementById("md").textContent=d.method;
var s=document.getElementById("st");
if(d.tracking){s.textContent="TRACKING";s.style.color="#00ff88";}
else{s.textContent="IDLE";s.style.color="#00d4ff";}
var bt=document.getElementById("bt");
bt.className=d.tracking?"btn on":"btn";
bt.textContent=d.tracking?"Tracking ON":"Tracking OFF";
}).catch(function(){});
},1500);

function cmd(c){fetch("/cmd/"+c).then(function(r){return r.json()}).then(function(d){
if(d.snap)console.log("Snap:",d.snap);
});}

function aiAnalyze(){
var b=document.getElementById("ab"),ar=document.getElementById("ar");
b.disabled=1;b.innerHTML='<span class="spin"></span>Analyzing...';
ar.innerHTML='<span class="spin"></span> Calling MiMo Vision API...';
fetch("/api/analyze").then(function(r){return r.json()}).then(function(d){
b.disabled=0;b.innerHTML="AI Analyze";
if(d.local){
ar.innerHTML="<b>Local:</b> Bright="+d.local.brightness+" Edge="+d.local.edge_density+"% Metal="+d.local.metal_pct+"%<br>";
}
if(d.cloud&&d.cloud.success){
ar.innerHTML+="<b>AI:</b><br>"+d.cloud.analysis.replace(/\\n/g,"<br>");
}else if(d.cloud){
ar.innerHTML+='<span style="color:red">'+d.cloud.error+'</span>';
}
}).catch(function(e){
b.disabled=0;b.innerHTML="AI Analyze";
ar.innerHTML='<span style="color:red">Error: '+e+'</span>';
});
}
</script></body></html>"""

# ─── Flask Routes ───
@app.route("/")
def index():
    return HTML

@app.route("/snapshot/color")
def snap_color():
    with _frame_lock:
        d = _color_jpeg
    if d:
        return Response(d, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-cache, no-store"})
    return "", 204

@app.route("/snapshot/depth")
def snap_depth():
    with _frame_lock:
        d = _depth_jpeg
    if d:
        return Response(d, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-cache, no-store"})
    return "", 204

@app.route("/mjpeg/color")
def mjpeg_color():
    return Response(mjpeg_gen(lambda: _color_jpeg, "color"),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/mjpeg/depth")
def mjpeg_depth():
    return Response(mjpeg_gen(lambda: _depth_jpeg, "depth"),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/status")
def api_status():
    global tracking, method, target_pos, depth_mm
    el = time.time() - _t0
    m, s = divmod(int(el), 60)
    h, m = divmod(m, 60)
    with _frame_lock:
        di = _depth_info.copy()
    gp = "N/A"
    if _gimbal:
        gp = "P:%.1f T:%.1f" % (_gimbal.pan, _gimbal.tilt)
    tgt = "None"
    if target_pos:
        tgt = "(%d,%d)" % target_pos
    return jsonify({
        "status": "running", "frames": _fc,
        "fps": round(_fc / el, 1) if el > 0 else 0,
        "depth_range": "%.0f-%.0f mm" % (di["min"], di["max"]),
        "uptime": "%dh%02dm%02ds" % (h, m, s),
        "tracking": tracking, "method": method,
        "target": tgt, "gimbal": gp,
        "depth_mm": round(depth_mm, 0)
    })

@app.route("/api/analyze")
def api_analyze():
    global _last_analysis, _ai_busy
    if _ai_busy:
        return jsonify({"error": "busy"})
    _ai_busy = True
    try:
        with _frame_lock:
            cj = _color_jpeg
        if not cj:
            return jsonify({"error": "no frame"})

        # Save snapshot
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = os.path.join(SNAP_DIR, "u_%s.jpg" % ts)
        with open(snap_path, "wb") as f:
            f.write(cj)

        # Local + Cloud analysis
        lo = local_analyze(cj)
        prompt = request.args.get("prompt", None)
        cl = ai_analyze(cj, prompt)
        result = {"timestamp": ts, "snapshot": snap_path, "local": lo, "cloud": cl}
        _last_analysis = result
        return jsonify(result)
    finally:
        _ai_busy = False

@app.route("/api/last_analysis")
def api_last():
    if _last_analysis:
        return jsonify(_last_analysis)
    return jsonify({"error": "no analysis yet"})

# ─── Commands ───
@app.route("/cmd/track")
def cmd_track():
    global tracking
    tracking = not tracking
    if tracking:
        _pid_pan.reset()
        _pid_tilt.reset()
    print("[Track] %s" % ("ON" if tracking else "OFF"))
    return jsonify({"tracking": tracking})

@app.route("/cmd/m1")
def cmd_m1():
    global method
    method = "motion"
    print("[Method] motion")
    return jsonify({"method": "motion"})

@app.route("/cmd/m2")
def cmd_m2():
    global method
    method = "bright"
    print("[Method] bright")
    return jsonify({"method": "bright"})

@app.route("/cmd/reset")
def cmd_reset():
    global tracking
    tracking = False
    _pid_pan.reset()
    _pid_tilt.reset()
    _gimbal.center()
    print("[Reset] done")
    return jsonify({"reset": True})

@app.route("/cmd/snap")
def cmd_snap():
    with _frame_lock:
        cj = _color_jpeg
        dj = _depth_jpeg
    ts = time.strftime("%Y%m%d_%H%M%S")
    if cj:
        with open(os.path.join(SNAP_DIR, "snap_color_%s.jpg" % ts), "wb") as f:
            f.write(cj)
    if dj:
        with open(os.path.join(SNAP_DIR, "snap_depth_%s.jpg" % ts), "wb") as f:
            f.write(dj)
    print("[Snap] %s" % ts)
    return jsonify({"snap": ts})

# ─── Main ───
if __name__ == "__main__":
    print("=" * 60)
    print("  MedVision Unified Server v1.0")
    print("  Astra S 3D + AI Analysis + PID Tracking")
    print("=" * 60)

    # Start camera thread
    threading.Thread(target=camera_loop, daemon=True).start()
    time.sleep(2)

    # Start tracking thread
    threading.Thread(target=tracking_loop, daemon=True).start()
    time.sleep(1)

    print("[Server] http://0.0.0.0:8080/")
    print("[Server] Endpoints: /snapshot/color /snapshot/depth /api/status /api/analyze /cmd/track")
    app.run(host="0.0.0.0", port=8080, threaded=True)
