import os, json, base64, requests, cv2


class LLMRouter:
    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                self.cfg = json.load(f)
        else:
            self.cfg = {
                "provider": "mimo",
                "base_url": "https://api.xiaomimimo.com/v1",
                "api_key": "sk-cuep8vloqzmcnvkd4sgpk7wb6lo8pog30sgeb9p9vmsu54ps",
                "model": "mimo-v2-omni",
                "max_tokens": 800,
                "timeout": 30
            }

    def chat(self, messages, **kwargs):
        model = kwargs.get("model", self.cfg.get("model", "mimo-v2-omni"))
        max_tokens = kwargs.get("max_tokens", self.cfg.get("max_tokens", 800))
        headers = {"Authorization": "Bearer " + self.cfg["api_key"], "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": kwargs.get("temperature", 0.1)}
        try:
            r = requests.post(self.cfg["base_url"] + "/chat/completions", headers=headers, json=payload, timeout=self.cfg.get("timeout", 30))
            data = r.json()
            return data["choices"][0]["message"].get("content", "") or data["choices"][0]["message"].get("reasoning_content", "")
        except Exception as e:
            return "API Error: " + str(e)

    def analyze_image(self, bgr, prompt="Describe what you see"):
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 55])
        b64 = base64.b64encode(buf.tobytes()).decode()
        url = "data:image/jpeg;base64," + b64
        return self.chat([{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url}}
        ]}])

    def detect_objects(self, bgr):
        return self.analyze_image(bgr, "List all objects visible. For each: label, position (left/center/right, top/middle/bottom).")

    def describe_scene(self, bgr):
        return self.analyze_image(bgr, "Describe this scene concisely.")
