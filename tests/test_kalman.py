"""Tests for KalmanTracker and PIDController."""
import sys, time, math, numpy as np, pytest
sys.path.insert(0, "/opt/medvision/harness")
from harness_agent_v2 import KalmanTracker, PIDController


def test_kalman_init():
    """Test Kalman initializes on first measurement."""
    kf = KalmanTracker()
    assert not kf.initialized
    pos = kf.update([100, 200])
    assert kf.initialized
    assert abs(pos[0] - 100) < 1
    assert abs(pos[1] - 200) < 1
    print("  kalman_init: (%.1f, %.1f) OK" % (pos[0], pos[1]))


def test_kalman_tracking():
    """Test Kalman follows steady movement."""
    kf = KalmanTracker()
    positions = [(100, 200), (110, 210), (120, 220), (130, 230)]
    for x, y in positions:
        time.sleep(0.01)
        kf.update([x, y])
    pos = kf.update([140, 240])
    # Should roughly follow
    assert abs(pos[0] - 140) < 30, "Kalman X drifted: %.1f" % pos[0]
    print("  kalman_track: (%.1f, %.1f) OK" % (pos[0], pos[1]))


def test_kalman_predict():
    """Test Kalman predicts future position."""
    kf = KalmanTracker()
    for x, y in [(100, 200), (110, 210), (120, 220)]:
        time.sleep(0.01)
        kf.update([x, y])
    pred = kf.predict(seconds_ahead=0.1)
    assert pred is not None
    # Should predict ~130, ~230
    print("  kalman_predict: (%.1f, %.1f) OK" % (pred[0], pred[1]))


def test_kalman_velocity():
    """Test velocity estimation."""
    kf = KalmanTracker()
    for x, y in [(100, 100), (120, 100), (140, 100)]:
        time.sleep(0.02)
        kf.update([x, y])
    vx, vy = kf.get_velocity()
    assert vx > 0, "Should have positive X velocity"
    print("  kalman_vel: vx=%.1f vy=%.1f OK" % (vx, vy))


def test_kalman_reset():
    """Test reset clears state."""
    kf = KalmanTracker()
    kf.update([100, 200])
    kf.reset()
    assert not kf.initialized
    print("  kalman_reset: OK")


def test_pid_proportional():
    """Test PID proportional response."""
    pid = PIDController(kp=0.1, ki=0, kd=0, out_max=20.0)
    out = pid.compute(100)
    assert out == pytest.approx(10.0, abs=0.1)
    print("  pid_p: err=100 out=%.2f OK" % out)


def test_pid_clamping():
    """Test PID output clamping."""
    pid = PIDController(kp=0.1, ki=0, kd=0, out_max=3.0)
    out = pid.compute(1000)
    assert out == 3.0, "Should clamp to 3.0, got %.1f" % out
    print("  pid_clamp: out=%.2f OK" % out)


def test_pid_anti_windup():
    """Test integral anti-windup."""
    pid = PIDController(kp=0, ki=0.1, kd=0, imax=10, out_max=100)
    for _ in range(1000):
        pid.compute(100)
    assert abs(pid.integral) <= 10, "Integral exceeded limit: %.1f" % pid.integral
    print("  pid_windup: integral=%.1f OK" % pid.integral)


def test_pid_damping():
    """Test PID responds to changing error (gradual decrease, no derivative spike)."""
    pid = PIDController(kp=0.05, ki=0.001, kd=0, out_max=5.0)
    # Large error -> large output
    out1 = pid.compute(100)
    time.sleep(0.01)
    # Gradual decrease
    out2 = pid.compute(60)
    time.sleep(0.01)
    out3 = pid.compute(20)
    # Output should decrease as error decreases
    assert out1 > out2 > out3, "Should decrease: %.2f -> %.2f -> %.2f" % (out1, out2, out3)
    print("  pid_damp: %.2f -> %.2f -> %.2f OK" % (out1, out2, out3))


def test_pid_reset():
    """Test PID reset."""
    pid = PIDController(kp=0.1, ki=0.1, kd=0.01)
    pid.compute(100)
    pid.reset()
    assert pid.integral == 0
    assert pid.prev_error == 0
    print("  pid_reset: OK")


if __name__ == "__main__":
    test_kalman_init()
    test_kalman_tracking()
    test_kalman_predict()
    test_kalman_velocity()
    test_kalman_reset()
    test_pid_proportional()
    test_pid_clamping()
    test_pid_anti_windup()
    test_pid_damping()
    test_pid_reset()
    print("\nAll Kalman+PID tests passed!")
