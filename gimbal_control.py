"""MedVision - Interactive Gimbal Control
Keyboard control with position recording.
W/S = Tilt Up/Down, A/D = Pan Left/Right
Q = Quit, SPACE = Record current position
"""
import sys, time, json
sys.path.insert(0, "/opt/medvision")
from gimbal import Gimbal

g = Gimbal()
pan = g.pan
tilt = g.tilt
step = 5  # degrees per keypress
records = []

print("=" * 50)
print("MedVision Gimbal Controller")
print("=" * 50)
print("Controls:")
print("  W = Tilt UP    S = Tilt DOWN")
print("  A = Pan LEFT   D = Pan RIGHT")
print("  + = Faster     - = Slower")
print("  SPACE = Record position")
print("  R = Return to recorded position")
print("  C = Center (90,90)")
print("  Q = Quit")
print("=" * 50)
print("Current: Pan=%.0f Tilt=%.0f  Step=%d deg" % (pan, tilt, step))
print("")

import tty, termios

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

try:
    while True:
        ch = getch()

        if ch == "q" or ch == "Q":
            print("\nQuit!")
            break

        elif ch == "w" or ch == "W":
            tilt = max(30, tilt - step)
            g.move_to(tilt=tilt)
            print("Tilt UP  -> Pan=%.0f Tilt=%.0f" % (pan, tilt))

        elif ch == "s" or ch == "S":
            tilt = min(150, tilt + step)
            g.move_to(tilt=tilt)
            print("Tilt DOWN -> Pan=%.0f Tilt=%.0f" % (pan, tilt))

        elif ch == "a" or ch == "A":
            pan = max(30, pan - step)
            g.move_to(pan=pan)
            print("Pan LEFT -> Pan=%.0f Tilt=%.0f" % (pan, tilt))

        elif ch == "d" or ch == "D":
            pan = min(150, pan + step)
            g.move_to(pan=pan)
            print("Pan RIGHT -> Pan=%.0f Tilt=%.0f" % (pan, tilt))

        elif ch == "+":
            step = min(20, step + 1)
            print("Step: %d deg" % step)

        elif ch == "-":
            step = max(1, step - 1)
            print("Step: %d deg" % step)

        elif ch == " ":
            records.append({"pan": pan, "tilt": tilt})
            print("*** RECORDED: Pan=%.0f Tilt=%.0f (total %d) ***" % (pan, tilt, len(records)))

        elif ch == "r" or ch == "R":
            if records:
                last = records[-1]
                pan = last["pan"]
                tilt = last["tilt"]
                g.move_to(pan=pan, tilt=tilt)
                print("Returned to: Pan=%.0f Tilt=%.0f" % (pan, tilt))
            else:
                print("No position recorded yet!")

        elif ch == "c" or ch == "C":
            pan = 90; tilt = 90
            g.center()
            print("Centered: Pan=90 Tilt=90")

        else:
            print("Key: %s (WASD to move, SPACE to record, Q to quit)" % repr(ch))

except KeyboardInterrupt:
    print("\nInterrupted!")

finally:
    g.close()

# Save records
if records:
    print("\n=== Recorded Positions ===")
    for i, r in enumerate(records):
        print("  %d: Pan=%.0f Tilt=%.0f" % (i+1, r["pan"], r["tilt"]))
    # Save to file
    with open("/opt/medvision/screen_positions.json", "w") as f:
        json.dump(records, f, indent=2)
    print("Saved to /opt/medvision/screen_positions.json")
else:
    print("\nNo positions recorded.")
print("Done!")
