"""Unit tests for PID controller."""
import numpy as np
from core.controllers import PIDController
from config.base_config import PIDConfig


def test_pid_zero_error_returns_zero():
    controller = PIDController(PIDConfig())
    torque = controller.compute_control(target=0.0, current=0.0, dt=0.01)
    assert torque == 0.0


def test_pid_integral_windup_clamped():
    controller = PIDController(PIDConfig(integral_limit=0.5))
    for _ in range(100):
        controller.compute_control(target=1.0, current=0.0, dt=0.1)
    assert abs(controller.integral) <= 0.5


def test_pid_derivative_state_updates():
    controller = PIDController(PIDConfig())
    controller.compute_control(target=1.0, current=0.0, dt=0.1)
    state = controller.get_state()
    assert state['prev_derivative'] != 0.0
