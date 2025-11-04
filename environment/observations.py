import numpy as np


class Observations:
    """Handles observation generation for the soft robotic arm."""
    
    def get_observation(self, kinematics, step_count, arm_angle, angular_velocity, angular_acceleration, torque_applied, time_step, sinusoidal_magnitude, sinusoidal_frequency):
        """
        Generate the current observation vector.
        
        Returns:
            np.ndarray: Observation vector containing positions, velocities, and errors
        """
        theta_reference, x_reference, y_reference = kinematics.calculate_reference_trajectory(
            step_count, time_step, sinusoidal_magnitude, sinusoidal_frequency)
        x_position, y_position = kinematics.calculate_kinematics(arm_angle)
        x_error = x_reference - x_position
        y_error = y_reference - y_position
        theta_error = theta_reference - arm_angle
        
        observation = np.array(
            [
                x_position,
                y_position,
                arm_angle,
                angular_velocity,
                angular_acceleration,
                torque_applied,
                x_reference,
                y_reference,
                theta_reference,
                x_error,
                y_error,
                theta_error,
            ],
            dtype=np.float32,
        )
        return observation

    def get_information(self, kinematics, step_count, arm_angle, angular_velocity, angular_acceleration, torque_applied, time_step, sinusoidal_magnitude, sinusoidal_frequency, left_actuator_force, right_actuator_force, left_force_x_position, right_force_x_position, effective_moment_of_inertia):
        """
        Generate additional information about the current state.
        
        Returns:
            Dict[str, float]: Dictionary containing detailed state information
        """
        x_position, y_position = kinematics.calculate_kinematics(arm_angle)
        theta_reference, x_reference, y_reference = kinematics.calculate_reference_trajectory(
            step_count, time_step, sinusoidal_magnitude, sinusoidal_frequency)
        return {
            "step": float(step_count),
            "theta": float(arm_angle),
            "theta_dot": float(angular_velocity),
            "theta_ddot": float(angular_acceleration),
            "tau": float(torque_applied),
            "x": float(x_position),
            "y": float(y_position),
            "theta_target": float(theta_reference),
            "x_target": float(x_reference),
            "y_target": float(y_reference),
            "force_left": float(left_actuator_force),
            "force_right": float(right_actuator_force),
            "x_force_left": float(left_force_x_position),
            "x_force_right": float(right_force_x_position),
            "I_eff": float(effective_moment_of_inertia),
        }