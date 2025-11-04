import numpy as np
from typing import Tuple


class Kinematics:
    """Handles kinematic calculations for the soft robotic arm."""
    
    def __init__(self, arm_length: float):
        self.arm_length = arm_length
    
    def calculate_reference_trajectory(self, step: int, time_step: float, sinusoidal_magnitude: float, sinusoidal_frequency: float) -> Tuple[float, float, float]:
        """
        Compute reference targets (theta, x, y) using sinusoidal trajectory.
        
        Args:
            step: Current step number
            time_step: Simulation time step
            sinusoidal_magnitude: Magnitude of sinusoidal trajectory
            sinusoidal_frequency: Frequency of sinusoidal trajectory
            
        Returns:
            Tuple of (theta_target, x_target, y_target) - target angle and Cartesian coordinates
        """
        theta_target = sinusoidal_magnitude * np.sin(2 * np.pi * sinusoidal_frequency * step * time_step)
        x_target = self.arm_length * np.sin(theta_target)
        y_target = self.arm_length * np.cos(theta_target)
        return float(theta_target), float(x_target), float(y_target)

    def calculate_kinematics(self, arm_angle: float) -> Tuple[float, float]:
        """
        Return (x, y) end-effector position from angle.
        
        Args:
            arm_angle: Current arm angle (radians)
            
        Returns:
            Tuple of (x_position, y_position) - Cartesian coordinates of end-effector
        """
        x_position = self.arm_length * np.sin(arm_angle)
        y_position = self.arm_length * np.cos(arm_angle)
        return float(x_position), float(y_position)