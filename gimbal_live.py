"""MedVision - Live Gimbal Control with Camera Preview
Real-time camera view + keyboard control + position recording.
W/S = Tilt, A/D = Pan, SPACE = Record, Q = Quit
"""
import os
os.environ["SDL_VIDEODRIVER"] = "x11"

import sys, cv2, numpy as np, subprocess, tempfile, time, json, io
import pygame

sys.path.insert(0, "/opt/medvision")
from gimbal import Gimbal

pygame.init()
W, H = 800, 600
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("MedVision Gimbal Control")
pygame.key.set_repeat(200, 50)  # 200ms delay, 50ms repeat
font_s = pygame.font.SysFont("monospace", 16)
font_m = pygame.font.SysFont("monospace", 20, bold=True)
font_l = pygame.font.SysFont("monospace", 24, bold=True)
clock = pygame.time.Clock()

# Colors
BG = (20, 20, 35)
GRN = (0, 255, 100)
RED = (255, 60, 60)
YEL = (255, 255, 0)
BLU = (80, 160, 255)
WHT = (255, 255, 255)
PNL = (30, 30, 50)

# Init camera (picamera2 continuous)
use_p2 = False
p2 = None
try:
    from picamera2 import Picamera2
    p2 = Picamera2()
    p2.configure(p2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    p2.start()
    time.sleep(0.5)
    use_p2 = True
    print("Camera: picamera2")
except Exception as e:
    print("Camera fail:", e)

# Init gimbal
gimbal = Gimbal()
pan = gimbal.pan
tilt = gimbal.tilt
step = 5
records = []
last_frame = None
running = True

def get_frame():
    global last_frame
    if use_p2 and p2:
        try:
            arr = p2.capture_array()
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
            last_frame = buf.tobytes()
            return arr
        except:
            pass
    # Fallback: rpicam-still
    tmp = tempfile.mktemp(suffix=".jpg")
    try:
        subprocess.run(["rpicam-still", "-o", tmp, "--width", "640", "--height", "480",
                         "--nopreview", "-t", "200", "--immediate", "--rotation", "180"],
                        capture_output=True, timeout=5)
        with open(tmp, "rb") as f:
            last_frame = f.read()
        return cv2.imdecode(np.frombuffer(last_frame, np.uint8), cv2.IMREAD_COLOR)
    except:
        return None
    finally:
        if os.path.exists(tmp): os.remove(tmp)

def draw_hud(scr, frame_rgb):
    # Camera feed
    if frame_rgb is not None:
        fh, fw = frame_rgb.shape[:2]
        sc = min((W - 220) / fw, (H - 80) / fh)
        nw, nh = int(fw * sc), int(fh * sc)
        ox = 210 + (W - 220 - nw) // 2
        oy = 40 + (H - 80 - nh) // 2
        # Crosshair on frame
        frame_draw = frame_rgb.copy()
        cx, cy = fw // 2, fh // 2
        cv2.drawMarker(frame_draw, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        surf = pygame.surfarray.make_surface(np.rot90(frame_draw))
        surf = pygame.transform.scale(surf, (nw, nh))
        scr.blit(surf, (ox, oy))
        pygame.draw.rect(scr, GRN, (ox-2, oy-2, nw+4, nh+4), 2)

    # Top bar
    pygame.draw.rect(scr, PNL, (0, 0, W, 36))
    scr.blit(font_l.render("Gimbal Control", True, GRN), (10, 6))
    scr.blit(font_m.render("WASD=Move SPACE=Record Q=Quit", True, (120,120,120)), (250, 8))

    # Left panel - position
    pygame.draw.rect(scr, PNL, (5, 42, 200, 280))
    pygame.draw.rect(scr, GRN, (5, 42, 200, 280), 1)
    y = 50
    items = [
        ("Pan", "%.0f deg" % pan, BLU),
        ("Tilt", "%.0f deg" % tilt, BLU),
        ("Step", "%d deg" % step, WHT),
        ("Records", "%d" % len(records), YEL),
    ]
    for label, val, col in items:
        scr.blit(font_s.render(label, True, (150,150,150)), (12, y))
        scr.blit(font_m.render(val, True, col), (12, y+20))
        y += 45

    # Controls hint
    y = 240
    hints = ["W=Up", "S=Down", "A=Left", "D=Right", "+/-=Speed"]
    for h in hints:
        scr.blit(font_s.render(h, True, (100,100,100)), (12, y))
        y += 18

    # Records list
    if records:
        y = 40
        pygame.draw.rect(scr, PNL, (W-180, 42, 175, min(len(records)*22+30, 200)))
        pygame.draw.rect(scr, YEL, (W-180, 42, 175, min(len(records)*22+30, 200)), 1)
        scr.blit(font_s.render("Recorded:", True, YEL), (W-175, y))
        y += 22
        for i, r in enumerate(records[-7:]):
            scr.blit(font_s.render("#%d P=%.0f T=%.0f" % (i+1, r["pan"], r["tilt"]), True, WHT), (W-175, y))
            y += 20

    # Bottom bar
    pygame.draw.rect(scr, PNL, (0, H-32, W, 32))
    msg = "Pan=%.0f Tilt=%.0f  Step=%d" % (pan, tilt, step)
    scr.blit(font_s.render(msg, True, WHT), (10, H-26))

print("=" * 50)
print("MedVision Gimbal + Camera Live")
print("WASD=Move SPACE=Record Q=Quit")
print("=" * 50)

while running:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_q or ev.key == pygame.K_ESCAPE:
                running = False

            elif ev.key == pygame.K_s:
                tilt = min(180, tilt + step)
                gimbal.move_to(tilt=tilt)

            elif ev.key == pygame.K_w:
                tilt = max(0, tilt - step)
                gimbal.move_to(tilt=tilt)

            elif ev.key == pygame.K_d:
                pan = max(0, pan - step)
                gimbal.move_to(pan=pan)

            elif ev.key == pygame.K_a:
                pan = min(180, pan + step)
                gimbal.move_to(pan=pan)

            elif ev.key == pygame.K_EQUALS or ev.key == pygame.K_PLUS:
                step = min(20, step + 1)

            elif ev.key == pygame.K_MINUS:
                step = max(1, step - 1)

            elif ev.key == pygame.K_SPACE:
                records.append({"pan": round(pan), "tilt": round(tilt)})
                print("RECORDED: Pan=%.0f Tilt=%.0f" % (pan, tilt))

            elif ev.key == pygame.K_c:
                pan = 90; tilt = 90
                gimbal.center()

            elif ev.key == pygame.K_r:
                if records:
                    last = records[-1]
                    pan = last["pan"]; tilt = last["tilt"]
                    gimbal.move_to(pan=pan, tilt=tilt)

    screen.fill(BG)
    frame = get_frame()
    draw_hud(screen, frame)
    pygame.display.flip()
    clock.tick(15)

# Cleanup
pygame.quit()
if p2:
    p2.stop(); p2.close()
gimbal.close()

# Save records
if records:
    with open("/opt/medvision/screen_positions.json", "w") as f:
        json.dump(records, f, indent=2)
    print("\nSaved %d positions to screen_positions.json" % len(records))
    for i, r in enumerate(records):
        print("  #%d: Pan=%d Tilt=%d" % (i+1, r["pan"], r["tilt"]))
print("Done!")
