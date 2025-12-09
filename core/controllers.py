"""Controllers for the pendulum system."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from config.base_config import PIDConfig


class ControllerBase(ABC):
    """Abstract base class for controllers."""

    @abstractmethod
    def compute_control(self, target: float, current: float, dt: float) -> float:
        """Compute control effort."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset internal controller state."""
        raise NotImplementedError


class PIDController(ControllerBase):
    """PID controller with anti-windup and optional derivative filtering."""

    def __init__(self, config: PIDConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

    def compute_control(self, target: float, current: float, dt: float) -> float:
        error = target - current
        p_term = self.config.kp * error

        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -self.config.integral_limit, self.config.integral_limit))
        i_term = self.config.ki * self.integral

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        if self.config.derivative_filter_tau > 0:
            alpha = dt / (self.config.derivative_filter_tau + dt)
            derivative = alpha * derivative + (1.0 - alpha) * self.prev_derivative
        d_term = self.config.kd * derivative

        self.prev_derivative = derivative
        self.prev_error = error
        return float(p_term + i_term + d_term)

    def get_state(self) -> Dict[str, float]:
        return {
            'integral': float(self.integral),
            'prev_error': float(self.prev_error),
            'prev_derivative': float(self.prev_derivative),
        }

    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        self.config.kp = kp
        self.config.ki = ki
        self.config.kd = kd


class AdaptivePIDController(PIDController):
    """Placeholder for RL-driven PID gain adaptation."""

    def __init__(self, config: PIDConfig):
        super().__init__(config)
        self.gain_history = []

    def adapt_gains(self, observation, rl_agent) -> None:
        # TODO: Integrate RL policy to update gains
        return None
