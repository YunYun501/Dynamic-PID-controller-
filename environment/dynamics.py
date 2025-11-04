import numpy as np
from typing import Tuple


class Dynamics:
    """Handles dynamic calculations for the soft robotic arm."""
    
    def calculate_dynamics(
        self,
        torque: float,
        arm_angle: float,
        angular_velocity: float,
        time_step: float,
        arm_mass: float,
        gravitational_acceleration: float,
        distance_from_pivot_to_center_of_mass: float,
        damping_coefficient: float,
        stiffness_coefficient: float,
        effective_moment_of_inertia: float,
        theta_limit: float,
        object_mass: float = 0.0,
        arm_length: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Integrate one step of rotational dynamics using semi-implicit Euler.
        
        Args:
            torque: Applied torque (N*m)
            arm_angle: Current arm angle
            angular_velocity: Current angular velocity
            time_step: Simulation time step
            arm_mass: Mass of the arm
            gravitational_acceleration: Gravitational acceleration
            distance_from_pivot_to_center_of_mass: Distance from pivot to center of mass
            damping_coefficient: Damping coefficient
            stiffness_coefficient: Stiffness coefficient
            effective_moment_of_inertia: Effective moment of inertia
            theta_limit: Limit for arm angle
            object_mass: Mass of object being manipulated (default: 0.0)
            arm_length: Length of the arm (default: 1.0)
            
        Returns:
            Tuple of (arm_angle, angular_velocity, angular_acceleration) - new state values
        """
        # Calculate gravitational torque from both arm and object
        arm_gravitational_torque = - arm_mass * gravitational_acceleration * distance_from_pivot_to_center_of_mass * np.sin(arm_angle)
        
        # Object gravitational torque (assumed to be at end of arm)
        object_gravitational_torque = - object_mass * gravitational_acceleration * arm_length * np.sin(arm_angle)
        
        # Total gravitational torque
        gravitational_torque = arm_gravitational_torque + object_gravitational_torque
        
        # Calculate angular acceleration
        angular_acceleration = (torque + gravitational_torque - damping_coefficient * angular_velocity - stiffness_coefficient * arm_angle) / effective_moment_of_inertia
        angular_velocity_new = angular_velocity + time_step * angular_acceleration
        arm_angle_new = arm_angle + time_step * angular_velocity_new

        # Clamp to limits
        arm_angle_new = float(np.clip(arm_angle_new, -theta_limit, theta_limit))
        angular_velocity_new = float(np.clip(angular_velocity_new, -10.0, 10.0))
        angular_acceleration = float(np.clip(angular_acceleration, -100.0, 100.0))
        return arm_angle_new, angular_velocity_new, angular_acceleration