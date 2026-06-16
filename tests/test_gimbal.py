"""Tests for GimbalTool."""
import sys, time, pytest
sys.path.insert(0, "/opt/medvision/harness")
from harness_gimbal import GimbalTool

CONFIG = {"tilt_min": -60, "tilt_max": 60, "pan_min": -60, "pan_max": 60}

@pytest.fixture
def gimbal():
    return GimbalTool(CONFIG)

def test_center(gimbal):
    """Test gimbal centers correctly."""
    gimbal.center()
    time.sleep(1)
    p, t = gimbal.query()
    assert abs(p) < 5, f"Pan not centered: {p}"
    assert abs(t) < 5, f"Tilt not centered: {t}"
    print("  center: pan=%.1f tilt=%.1f OK" % (p, t))

def test_limits(gimbal):
    """Test gimbal respects limits."""
    gimbal.move_to(pan=50, tilt=50)  # Should be clamped
    time.sleep(0.5)
    p, t = gimbal.query()
    assert -60 <= p <= 60, f"Pan out of range: {p}"
    assert -60 <= t <= 60, f"Tilt out of range: {t}"
    print("  limits: pan=%.1f tilt=%.1f OK" % (p, t))
    gimbal.center()
    time.sleep(0.5)

def test_move_to(gimbal):
    """Test move to specific position."""
    gimbal.move_to(pan=10, tilt=15)
    time.sleep(0.5)
    p, t = gimbal.query()
    assert abs(p - 10) < 5, f"Pan wrong: {p}"
    assert abs(t - 15) < 5, f"Tilt wrong: {t}"
    print("  move_to: pan=%.1f tilt=%.1f OK" % (p, t))
    gimbal.center()
    time.sleep(0.5)

if __name__ == "__main__":
    g = GimbalTool(CONFIG)
    test_center(g)
    test_limits(g)
    test_move_to(g)
    g.close()
    print("\nAll gimbal tests passed!")
