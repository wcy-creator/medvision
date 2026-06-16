#!/usr/bin/env python3
"""MedVision - Vision Analyzer Module"""
import subprocess, base64, json, os, time, cv2, numpy as np, tempfile, requests
from datetime import datetime

class VisionAnalyzer:
    def __init__(self, config_path="/opt/medvision/config.json"):
        with open(config_path) as f:
            cfg = json.load(f)
        self.cam_cfg = cfg["camera"]
        self.api_cfg = cfg["api"]
        self.det_cfg = cfg["detection"]
        self.snap_dir = "/opt/medvision/snapshots"
        os.makedirs(self.snap_dir, exist_ok=True)

    def capture(self, width=None, height=None):
        w = width or self.cam_cfg["width"]
        h = height or self.cam_cfg["height"]
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = f.name
        try:
            cmd = ["rpicam-still", "-o", tmp, "--width", str(w), "--height", str(h), "--nopreview", "-t", "500", "--rotation", "180"]
            result = subprocess.run(cmd, capture_output=True, timeout=15)
            if result.returncode != 0:
                raise RuntimeError(f"rpicam-still failed: {result.stderr.decode()[:200]}")
            img = cv2.imread(tmp)
            if img is None:
                raise RuntimeError("Failed to read captured image")
            return img
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def compress_for_api(self, img, max_side=320, quality=60):
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def analyze_local(self, img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))
        std_b = float(np.std(gray))
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = int(np.count_nonzero(edges)) / (h * w)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sig = [c for c in contours if cv2.contourArea(c) > 500]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        metal = float(np.count_nonzero(cv2.inRange(hsv, (0,0,180), (180,50,255)))) / (h*w)
        tissue = float(np.count_nonzero(cv2.inRange(hsv, (0,80,80), (10,255,255)))) / (h*w)
        result = {"timestamp": datetime.now().isoformat(), "resolution": f"{w}x{h}",
                  "brightness": round(mean_b,1), "brightness_std": round(std_b,1),
                  "edge_density": round(edge_ratio*100,2), "contour_count": len(sig),
                  "metal_area_pct": round(metal*100,2), "tissue_area_pct": round(tissue*100,2)}
        parts = [f"Image {w}x{h}"]
        if mean_b < 50: parts.append("TOO DARK")
        elif mean_b > 200: parts.append("TOO BRIGHT")
        else: parts.append("brightness OK")
        if metal > 0.05: parts.append(f"metal {metal*100:.1f}%")
        if tissue > 0.02: parts.append(f"tissue {tissue*100:.1f}%")
        if len(sig) > 0: parts.append(f"{len(sig)} contours")
        result["summary"] = ", ".join(parts)
        return result

    def analyze_cloud(self, img, prompt=None):
        img_bytes = self.compress_for_api(img)
        img_b64 = base64.b64encode(img_bytes).decode()
        if prompt is None:
            prompt = "Analyze this surgical image. Identify instruments, estimate position/angle, describe open/close state. Return JSON."
        headers = {"Authorization": f"Bearer {self.api_cfg['api_key']}", "Content-Type": "application/json"}
        payload = {"model": self.api_cfg["model"], "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}], "max_tokens": 512, "temperature": 0.1}
        start = time.time()
        try:
            resp = requests.post(f"{self.api_cfg['base_url']}/chat/completions", headers=headers, json=payload, timeout=self.api_cfg["timeout"])
            elapsed = time.time() - start
            if resp.status_code == 200:
                content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"success": True, "analysis": content, "response_time": round(elapsed,2), "image_size_bytes": len(img_bytes)}
            return {"success": False, "error": f"API {resp.status_code}: {resp.text[:200]}", "response_time": round(elapsed,2)}
        except requests.exceptions.Timeout:
            return {"success": False, "error": f"API timeout ({self.api_cfg['timeout']}s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def analyze(self, prompt=None, save_snapshot=False):
        print("Capturing...")
        img = self.capture()
        h, w = img.shape[:2]
        print(f"  Got {w}x{h}")
        if save_snapshot:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = os.path.join(self.snap_dir, f"analyze_{ts}.jpg")
            cv2.imwrite(p, img)
            print(f"  Snapshot: {p}")
        print("Local analysis...")
        lr = self.analyze_local(img)
        print(f"  {lr['summary']}")
        cr = None
        if self.api_cfg["api_key"] and len(self.api_cfg["api_key"]) > 10:
            print("Cloud API...")
            cr = self.analyze_cloud(img, prompt)
            if cr["success"]: print(f"  API OK ({cr['response_time']}s)")
            else: print(f"  API fail: {cr['error']}")
        else:
            print("  API Key not set, skip cloud")
        return {"timestamp": lr["timestamp"], "resolution": lr["resolution"],
                "local_analysis": lr, "cloud_analysis": cr, "status": "ok"}

    def monitor(self, interval=5, max_frames=10, prompt=None):
        results = []
        print(f"Monitor (interval {interval}s, max {max_frames})")
        prev = None
        for i in range(max_frames):
            print(f"\n--- Frame {i+1}/{max_frames} ---")
            img = self.capture()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            motion = float(np.mean(cv2.absdiff(prev, gray))) if prev is not None else 0
            prev = gray
            lr = self.analyze_local(img)
            lr["motion_score"] = round(motion,2)
            fr = {"frame": i+1, "local": lr, "motion": round(motion,2)}
            if motion > self.det_cfg["motion_threshold"] and self.api_cfg["api_key"] and len(self.api_cfg["api_key"]) > 10:
                print("  Motion! calling cloud...")
                fr["cloud"] = self.analyze_cloud(img, prompt)
            results.append(fr)
            print(f"  {lr['summary']} | motion: {motion:.1f}")
            if i < max_frames - 1: time.sleep(interval)
        print(f"\nDone, {len(results)} frames")
        return results

def main():
    import sys
    a = VisionAnalyzer()
    prompt = sys.argv[1] if len(sys.argv) > 1 else None
    r = a.analyze(prompt=prompt, save_snapshot=True)
    print("\n" + "="*50)
    print("Result (TEXT)")
    print("="*50)
    print(json.dumps(r, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
