from environment.base import SoftRoboticBase
from environment.kinematics import Kinematics
from environment.dynamics import Dynamics
from environment.observations import Observations
from environment.actions import Actions
from environment.rendering import Rendering
import numpy as np


class SoftRobotic(SoftRoboticBase):
    """Main SoftRobotic environment class that integrates all components."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize components
        self.kinematics = Kinematics(self.arm_length)
        self.dynamics = Dynamics()
        self.observations = Observations()
        self.actions = Actions()
        self.renderer = Rendering()
    
    def _calculate_reference_trajectory(self, step: int):
        """Backward compatibility method."""
        return self.kinematics.calculate_reference_trajectory(
            step, self.time_step, self.sinusoidal_magnitude, self.sinusoidal_frequency)
    
    def _calculate_kinematics(self, arm_angle: float):
        """Backward compatibility method."""
        return self.kinematics.calculate_kinematics(arm_angle)
    
    def _calculate_dynamics(self, torque: float):
        """Backward compatibility method."""
        return self.dynamics.calculate_dynamics(
            torque, self.arm_angle, self.angular_velocity, self.time_step,
            self.arm_mass, self.gravitational_acceleration, 
            self.distance_from_pivot_to_center_of_mass, self.damping_coefficient,
            self.stiffness_coefficient, self.effective_moment_of_inertia,
            self.theta_limit)
    
    def _get_observation(self):
        """Backward compatibility method."""
        return self.observations.get_observation(
            self.kinematics, self.step_count, self.arm_angle, self.angular_velocity,
            self.angular_acceleration, self.torque_applied, self.time_step,
            self.sinusoidal_magnitude, self.sinusoidal_frequency)
    
    def _get_information(self):
        """Backward compatibility method."""
        return self.observations.get_information(
            self.kinematics, self.step_count, self.arm_angle, self.angular_velocity,
            self.angular_acceleration, self.torque_applied, self.time_step,
            self.sinusoidal_magnitude, self.sinusoidal_frequency,
            self.left_actuator_force, self.right_actuator_force,
            self.left_force_x_position, self.right_force_x_position,
            self.effective_moment_of_inertia)
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment to its initial state."""
        if seed is not None:
            self.numpy_random = np.random.default_rng(seed)

        self.step_count = 0
        if self.random_start:
            self.arm_angle = float(self.numpy_random.uniform(-0.3, 0.3))
            self.angular_velocity = float(self.numpy_random.uniform(-0.1, 0.1))
        else:
            self.arm_angle = 0.0
            self.angular_velocity = 0.0
        self.angular_acceleration = 0.0
        self.torque_applied = 0.0

        observation = self._get_observation()
        information = self._get_information()
        return observation, information

    def step(self, action):
        """Execute one time step in the environment."""
        # Process action to determine torque
        torque = self.actions.process_action(
            self.control_mode, action, self.step_count, self.arm_angle, 
            self.angular_velocity, self.time_step, self.arm_mass, 
            self.gravitational_acceleration, self.distance_from_pivot_to_center_of_mass,
            self.effective_moment_of_inertia, self.damping_coefficient, 
            self.stiffness_coefficient, self.maximum_torque, self.kinematics,
            self.sinusoidal_magnitude, self.sinusoidal_frequency)

        # Integrate dynamics
        self.torque_applied = torque
        self.last_torque_applied = torque
        arm_angle, angular_velocity, angular_acceleration = self._calculate_dynamics(torque)
        self.arm_angle, self.angular_velocity, self.angular_acceleration = arm_angle, angular_velocity, angular_acceleration
        self.step_count += 1

        # Calculate reward
        theta_reference, _, _ = self._calculate_reference_trajectory(self.step_count)
        theta_error = float(theta_reference - self.arm_angle)
        reward = -abs(theta_error)
        reward -= 0.01 * (self.torque_applied ** 2)

        # Episode termination
        terminated = False
        truncated = self.step_count >= self.maximum_steps

        observation = self._get_observation()
        information = self._get_information()
        return observation, reward, terminated, truncated, information

    def render(self):
        """Render the current state of the environment."""
        return self.renderer.render(self)

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.is_window_open = False