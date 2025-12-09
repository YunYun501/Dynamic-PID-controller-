"""Pendulum dynamics and kinematics utilities."""
from typing import Tuple
import numpy as np

from config.base_config import PhysicsConfig


class PendulumDynamics:
    """Compute pendulum dynamics and integrate forward one step."""

    def __init__(self, config: PhysicsConfig):
        self.config = config

    def effective_inertia(self) -> float:
        """Total inertia including optional base mass at the end of the arm."""
        return self.config.moment_of_inertia + self.config.base_mass * (self.config.arm_length ** 2)

    def compute_passive_torques(self, angle: float, velocity: float) -> Tuple[float, float, float]:
        """Return (gravity, damping, friction) torques for the current state."""
        arm = self.config
        length = arm.arm_length
        tau_gravity = -(
            arm.arm_mass * arm.gravity * (length / 2.0)
            + arm.base_mass * arm.gravity * length
        ) * np.sin(angle)
        tau_damping = -arm.damping_coefficient * velocity
        tau_friction = -arm.friction_coefficient * np.sign(velocity)
        return tau_gravity, tau_damping, tau_friction

    def compute_acceleration(self, angle: float, velocity: float, torque: float) -> float:
        """Compute angular acceleration from torque and current state."""
        tau_gravity, tau_damping, tau_friction = self.compute_passive_torques(angle, velocity)
        tau_total = torque + tau_gravity + tau_damping + tau_friction
        inertia = self.effective_inertia()
        return tau_total / inertia

    def integrate_step(self, angle: float, velocity: float, torque: float, dt: float) -> Tuple[float, float, float]:
        """Integrate one timestep using semi-implicit Euler."""
        acceleration = self.compute_acceleration(angle, velocity, torque)
        new_velocity = velocity + acceleration * dt
        new_angle = angle + new_velocity * dt
        new_angle = np.clip(new_angle, -self.config.max_angle, self.config.max_angle)
        new_velocity = np.clip(new_velocity, -self.config.max_velocity, self.config.max_velocity)
        return float(new_angle), float(new_velocity), float(acceleration)


class PendulumKinematics:
    """Forward kinematics for the pendulum."""

    def __init__(self, arm_length: float):
        self.arm_length = arm_length

    def get_end_effector_position(self, angle: float) -> Tuple[float, float]:
        """Return (x, y) position for a given angle (0 = down, positive clockwise)."""
        x = self.arm_length * np.sin(angle)
        y = self.arm_length * np.cos(angle)
        return float(x), float(y)
