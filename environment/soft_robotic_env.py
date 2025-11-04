from .base import SoftRoboticBase
from .kinematics import Kinematics
from .dynamics import Dynamics
from .observations import Observations
from .actions import Actions
from .rendering import Rendering
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
        
        # Reset PID controller state
        self.actions.prev_error = 0.0
        self.actions.integral = 0.0
        
        # For object manipulation tasks
        self.object_mass = 0.0
        self.object_attached = False
        self.last_left_force = 0.0
        self.last_right_force = 0.0
        self.last_action = None
    
    def attach_object(self, mass: float):
        """Attach an object of given mass to the end of the arm."""
        self.object_mass = float(mass)
        self.object_attached = True
        
    def detach_object(self):
        """Detach any attached object."""
        self.object_mass = 0.0
        self.object_attached = False
    
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
            self.theta_limit, self.object_mass, self.arm_length)
    
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
        self.left_actuator_force = 0.0
        self.right_actuator_force = 0.0
        self.last_left_force = 0.0
        self.last_right_force = 0.0
        self.last_action = None
        
        # Reset PID controller state
        self.actions.prev_error = 0.0
        self.actions.integral = 0.0

        observation = self._get_observation()
        information = self._get_information()
        return observation, information

    def step(self, action):
        """Execute one time step in the environment."""
        # Store action for reward calculation
        self.last_action = action
        
        # Process action to determine torque and forces
        result = self.actions.process_action(
            self.control_mode, action, self.step_count, self.arm_angle, 
            self.angular_velocity, self.time_step, self.arm_mass, 
            self.gravitational_acceleration, self.distance_from_pivot_to_center_of_mass,
            self.effective_moment_of_inertia, self.damping_coefficient, 
            self.stiffness_coefficient, self.maximum_torque, self.kinematics,
            self.sinusoidal_magnitude, self.sinusoidal_frequency,
            self.left_force_x_position, self.right_force_x_position)
        
        torque, left_force, right_force = result
        
        if self.control_mode == "force":
            self.left_actuator_force = left_force if left_force is not None else 0.0
            self.right_actuator_force = right_force if right_force is not None else 0.0
        else:
            # For non-force control modes, forces are not directly controlled
            # but we can still track them for reward calculation
            self.left_actuator_force = 0.0
            self.right_actuator_force = 0.0

        # Integrate dynamics
        self.torque_applied = torque
        self.last_torque_applied = torque
        arm_angle, angular_velocity, angular_acceleration = self._calculate_dynamics(torque)
        self.arm_angle, self.angular_velocity, self.angular_acceleration = arm_angle, angular_velocity, angular_acceleration
        self.step_count += 1

        # Calculate improved reward
        reward = self._calculate_reward()

        # Episode termination
        terminated = False
        truncated = self.step_count >= self.maximum_steps

        observation = self._get_observation()
        information = self._get_information()
        
        # Store current forces for next step
        self.last_left_force = self.left_actuator_force
        self.last_right_force = self.right_actuator_force
        
        return observation, reward, terminated, truncated, information

    def _calculate_reward(self):
        """Calculate improved reward function suitable for reinforcement learning."""
        # Extract observation data
        observation = self._get_observation()
        theta_error = observation[11]  # etheta
        theta_dot = observation[3]     # theta_dot
        
        # Base tracking reward
        tracking_reward = -abs(theta_error)
        
        # Add velocity penalty for smoother movements
        velocity_penalty = -0.01 * abs(theta_dot)
        
        # Add energy penalty
        energy_penalty = -0.001 * (self.torque_applied ** 2)
        
        # Add smoothness penalty (torque changes)
        torque_change_penalty = 0.0
        if hasattr(self, 'last_torque_applied'):
            torque_change = abs(self.torque_applied - self.last_torque_applied)
            torque_change_penalty = -0.005 * torque_change
            
        # For force control mode, include force-related rewards
        force_reward = 0.0
        if self.control_mode == "force":
            # Object manipulation rewards (if object is attached)
            if self.object_attached and self.object_mass > 0:
                # Reward for maintaining balanced forces (good grip)
                force_balance = abs(self.left_actuator_force - self.right_actuator_force)
                total_force = abs(self.left_actuator_force) + abs(self.right_actuator_force)
                
                if total_force > 0:
                    balance_reward = 1.0 - min(force_balance / total_force, 1.0)
                    force_reward += balance_reward
                
                # Reward for sufficient force to hold object
                min_force = self.object_mass * self.gravitational_acceleration * 0.6  # 60% safety margin
                if total_force >= min_force:
                    force_reward += 1.0
                else:
                    # Penalize insufficient force
                    force_reward -= (min_force - total_force) / min_force

            # Stability reward: Penalize excessive force changes
            force_change_penalty = 0.0
            if self.step_count > 1:  # Only apply after first step
                left_force_change = abs(self.left_actuator_force - self.last_left_force)
                right_force_change = abs(self.right_actuator_force - self.last_right_force)
                force_change_penalty = 0.005 * (left_force_change + right_force_change)
            force_reward -= force_change_penalty

        # Overall reward
        reward = tracking_reward + velocity_penalty + energy_penalty + torque_change_penalty + force_reward
        
        return float(reward)

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