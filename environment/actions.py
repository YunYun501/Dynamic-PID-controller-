import numpy as np


class Actions:
    """Handles action processing for the soft robotic arm."""
    
    def __init__(self):
        # Initialize PID controller state
        self.prev_error = 0.0
        self.integral = 0.0
    
    def process_action(self, control_mode, action, step_count, arm_angle, angular_velocity, time_step, arm_mass, gravitational_acceleration, distance_from_pivot_to_center_of_mass, effective_moment_of_inertia, damping_coefficient, stiffness_coefficient, maximum_torque, kinematics, sinusoidal_magnitude, sinusoidal_frequency, left_force_x_position, right_force_x_position):
        """
        Process the action based on the control mode.
        
        Args:
            control_mode: Control mode ("position", "velocity", "acceleration", or "force")
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
            left_force_x_position: X position of left actuator
            right_force_x_position: X position of right actuator
            
        Returns:
            tuple: (torque, left_force, right_force) - torque to apply and forces (None for non-force modes)
        """
        # Initialize forces to None for non-force control modes
        left_force = None
        right_force = None
        
# Determine torque based on control mode
        if control_mode == "position":
            # Position control: drive towards target angle using PID control
            control_signal = float(action[0])
            theta_reference = control_signal
            angle_error = theta_reference - arm_angle
            
            # PID controller
            self.integral += angle_error * time_step
            derivative = (angle_error - self.prev_error) / time_step if time_step > 0 else 0.0
            
            # PID gains
            Kp = 100.0  # Proportional gain
            Ki = 10.0   # Integral gain
            Kd = 20.0   # Derivative gain
            
            torque = Kp * angle_error + Ki * self.integral + Kd * derivative
            torque = float(np.clip(torque, -maximum_torque, maximum_torque))
            
            # Update previous error
            self.prev_error = angle_error
        elif control_mode == "velocity":
            # Velocity control: compute required torque to achieve target velocity
            control_signal = float(action[0])
            target_velocity = control_signal
            gravitational_torque = - arm_mass * gravitational_acceleration * distance_from_pivot_to_center_of_mass * np.sin(arm_angle)
            torque = effective_moment_of_inertia * (target_velocity - angular_velocity) / time_step + \
                     damping_coefficient * angular_velocity + \
                     stiffness_coefficient * arm_angle - \
                     gravitational_torque
            torque = float(np.clip(torque, -maximum_torque, maximum_torque))
        elif control_mode == "acceleration":
            # Acceleration control: compute required torque to achieve target acceleration
            control_signal = float(action[0])
            target_acceleration = control_signal
            gravitational_torque = - arm_mass * gravitational_acceleration * distance_from_pivot_to_center_of_mass * np.sin(arm_angle)
            torque = effective_moment_of_inertia * target_acceleration + \
                     damping_coefficient * angular_velocity + \
                     stiffness_coefficient * arm_angle - \
                     gravitational_torque
            torque = float(np.clip(torque, -maximum_torque, maximum_torque))
        else:  # force control
            # Force control: directly set left and right actuator forces
            left_force = float(action[0])
            right_force = float(action[1])
            
            # Calculate torque from forces applied at different x positions
            # Torque = force * lever_arm (perpendicular distance)
            left_torque = left_force * left_force_x_position
            right_torque = right_force * right_force_x_position
            torque = left_torque + right_torque
            
        return float(torque), left_force, right_force