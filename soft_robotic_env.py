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
        """Calculate improved reward function for force control tasks."""
        if self.control_mode == "force":
            # Primary reward: Object manipulation success (maintaining grip)
            grip_reward = 0.0

            # If we're simulating object pickup, reward for maintaining appropriate forces
            # to hold the object without dropping it or applying excessive force
            if self.object_attached and self.object_mass > 0:
                # Reward for maintaining balanced forces (good grip)
                force_balance = abs(self.left_actuator_force - self.right_actuator_force)
                total_force = abs(self.left_actuator_force) + abs(self.right_actuator_force)
                
                if total_force > 0:
                    balance_reward = 1.0 - min(force_balance / total_force, 1.0)
                    grip_reward += balance_reward
                
                # Reward for sufficient force to hold object
                min_force = self.object_mass * self.gravitational_acceleration * 0.6  # 60% safety margin
                if total_force >= min_force:
                    grip_reward += 1.0
                else:
                    # Penalize insufficient force
                    grip_reward -= (min_force - total_force) / min_force

            # Stability reward: Penalize excessive force changes
            force_change_penalty = 0.0
            if self.step_count > 1:  # Only apply after first step
                left_force_change = abs(self.left_actuator_force - self.last_left_force)
                right_force_change = abs(self.right_actuator_force - self.last_right_force)
                force_change_penalty = 0.01 * (left_force_change + right_force_change)

            # Energy efficiency: Penalize excessive force magnitude
            energy_penalty = 0.001 * (self.left_actuator_force**2 + self.right_actuator_force**2)

            # Overall reward
            reward = grip_reward - force_change_penalty - energy_penalty
        else:
            # For other control modes, use the appropriate reward function
            if hasattr(self, 'last_action') and self.last_action is not None:
                # Use the action as the target for reward calculation
                if self.control_mode == "position":
                    theta_reference = float(self.last_action[0])
                elif self.control_mode == "velocity":
                    # For velocity control, we don't have a specific angle target
                    # Use sinusoidal reference for consistency
                    theta_reference, _, _ = self._calculate_reference_trajectory(self.step_count)
                elif self.control_mode == "acceleration":
                    # For acceleration control, we don't have a specific angle target
                    # Use sinusoidal reference for consistency
                    theta_reference, _, _ = self._calculate_reference_trajectory(self.step_count)
                else:
                    theta_reference, _, _ = self._calculate_reference_trajectory(self.step_count)
            else:
                # Fallback to sinusoidal reference
                theta_reference, _, _ = self._calculate_reference_trajectory(self.step_count)
            
            theta_error = float(theta_reference - self.arm_angle)
            reward = -abs(theta_error)
            reward -= 0.01 * (self.torque_applied ** 2)
            
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