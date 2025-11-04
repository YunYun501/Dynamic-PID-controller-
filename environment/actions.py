import numpy as np


class Actions:
    """Handles action processing for the soft robotic arm."""
    
    def process_action(self, control_mode, action, step_count, arm_angle, angular_velocity, time_step, arm_mass, gravitational_acceleration, distance_from_pivot_to_center_of_mass, effective_moment_of_inertia, damping_coefficient, stiffness_coefficient, maximum_torque, kinematics, sinusoidal_magnitude, sinusoidal_frequency):
        """
        Process the action based on the control mode.
        
        Args:
            control_mode: Control mode ("position", "velocity", or "acceleration")
            action: Control action (1D array with control signal)
            step_count: Current step count
            arm_angle: Current arm angle
            angular_velocity: Current angular velocity
            time_step: Simulation time step
            arm_mass: Mass of the arm
            gravitational_acceleration: Gravitational acceleration
            distance_from_pivot_to_center_of_mass: Distance from pivot to center of mass
            effective_moment_of_inertia: Effective moment of inertia
            damping_coefficient: Damping coefficient
            stiffness_coefficient: Stiffness coefficient
            maximum_torque: Maximum torque that can be applied
            kinematics: Kinematics object
            sinusoidal_magnitude: Magnitude of sinusoidal trajectory
            sinusoidal_frequency: Frequency of sinusoidal trajectory
            
        Returns:
            float: Torque to apply
        """
        control_signal = float(action[0])
        
        # Determine torque based on control mode
        if control_mode == "position":
            # Position control: drive towards target angle using proportional control
            theta_reference, _, _ = kinematics.calculate_reference_trajectory(
                step_count, time_step, sinusoidal_magnitude, sinusoidal_frequency)
            angle_error = theta_reference - arm_angle
            torque = 10.0 * angle_error
            torque = float(np.clip(torque, -maximum_torque, maximum_torque))
        elif control_mode == "velocity":
            # Velocity control: compute required torque to achieve target velocity
            target_velocity = control_signal
            gravitational_torque = - arm_mass * gravitational_acceleration * distance_from_pivot_to_center_of_mass * np.sin(arm_angle)
            torque = effective_moment_of_inertia * (target_velocity - angular_velocity) / time_step + \
                     damping_coefficient * angular_velocity + \
                     stiffness_coefficient * arm_angle - \
                     gravitational_torque
            torque = float(np.clip(torque, -maximum_torque, maximum_torque))
        else:  # acceleration
            # Acceleration control: compute required torque to achieve target acceleration
            target_acceleration = control_signal
            gravitational_torque = - arm_mass * gravitational_acceleration * distance_from_pivot_to_center_of_mass * np.sin(arm_angle)
            torque = effective_moment_of_inertia * target_acceleration + \
                     damping_coefficient * angular_velocity + \
                     stiffness_coefficient * arm_angle - \
                     gravitational_torque
            torque = float(np.clip(torque, -maximum_torque, maximum_torque))
            
        return torque