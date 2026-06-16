"""
Voice Interaction Module - Text + TTS via MiMo API.
Supports: text input -> LLM processing -> voice output.
"""
import os, sys, time, json, base64, requests, subprocess
sys.path.insert(0, "/opt/medvision/harness")

class VoiceInterface:
    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                self.api_cfg = json.load(f)
        else:
            self.api_cfg = {
                "base_url": "https://api.xiaomimimo.com/v1",
                "api_key": "sk-cuep8vloqzmcnvkd4sgpk7wb6lo8pog30sgeb9p9vmsu54ps",
                "model": "mimo-v2.5"
            }
        self.tts_url = self.api_cfg["base_url"] + "/audio/speech"
        self.tts_api_key = self.api_cfg["api_key"]
        self.tts_voice = "mimo-v2-tts"

    def speak(self, text):
        """Text-to-speech using MiMo TTS API."""
        try:
            r = requests.post(
                self.tts_url,
                headers={"Authorization": "Bearer " + self.tts_api_key},
                json={"model": self.tts_voice, "input": text, "voice": "alloy"},
                timeout=15)
            if r.status_code == 200:
                # Save audio and play
                audio_path = "/tmp/tts_output.mp3"
                with open(audio_path, "wb") as f:
                    f.write(r.content)
                # Play audio (try different methods)
                for player in ["mpv", "aplay", "ffplay", "paplay"]:
                    try:
                        subprocess.run([player, audio_path], timeout=10,
                                      capture_output=True)
                        break
                    except:
                        continue
                print("[TTS] %s" % text)
            else:
                print("[TTS] API error: %d" % r.status_code)
        except Exception as e:
            print("[TTS] Error: %s (fallback: text only)" % str(e))
            print("[SPEAK] %s" % text)

    def listen(self):
        """Text input (simple terminal interface)."""
        try:
            return input("[You] > ").strip()
        except EOFError:
            return None

    def chat(self, llm, message, speak=True):
        """Process user message through LLM and respond."""
        # Add system context
        system_prompt = """你是 MedVision 手术器械追踪系统的 AI 助手。
你可以控制云台、查看摄像头、测量角度、执行3D扫描。
请用简洁的中文回答。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        reply = llm.chat(messages)

        if speak:
            self.speak(reply)
        else:
            print("[AI] %s" % reply)

        return reply

    def interactive_loop(self, llm, hal):
        """Interactive voice/text loop."""
        print("=" * 50)
        print("  MedVision Voice Interface")
        print("  输入文字指令，系统语音回复")
        print("  输入 'quit' 退出")
        print("=" * 50)

        while True:
            user_input = self.listen()
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            if "追踪" in user_input or "track" in user_input.lower():
                hal.center()
                bgr = hal.capture()
                r = hal.detect.find(bgr)
                if r:
                    cx, cy, area = r
                    err_x = cx - 320
                    if abs(err_x) > 25:
                        hal.nudge(err_x * 0.03, 0)
                    reply = "检测到目标在位置(%d,%d)，已对准" % (cx, cy)
                else:
                    reply = "未检测到目标"
            elif "角度" in user_input or "angle" in user_input.lower():
                bgr = hal.capture()
                a = hal.angle.measure(bgr)
                reply = "当前角度: %.1f度" % a if a else "无法测量角度"
            elif "扫描" in user_input or "scan" in user_input.lower():
                reply = "开始扫描桌面..."
            elif "归中" in user_input or "center" in user_input.lower():
                hal.center()
                reply = "云台已归中"
            else:
                reply = self.chat(llm, user_input, speak=False)

            print("[AI] %s" % reply)
            self.speak(reply)

def main():
    from harness_llm import LLMRouter
    from harness_hal import HAL

    config_path = "/opt/medvision/harness/config/default.json"
    api_path = "/opt/medvision/config/api.json"

    llm = LLMRouter(api_path)
    hal = HAL(json.load(open(config_path)), simulate=True)  # simulate for demo

    voice = VoiceInterface(api_path)
    voice.interactive_loop(llm, hal)

if __name__ == "__main__":
    main()
