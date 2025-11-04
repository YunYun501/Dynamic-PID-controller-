from __future__ import annotations

from typing import Optional
from gymnasium import spaces
import gymnasium as gym
import numpy as np


class SoftRoboticBase(gym.Env):
    """
    Base class for the SoftRobotic environment with three control modes:
    1. Position control - directly set the target angle
    2. Angular velocity control - set the rate of change of angle
    3. Angular acceleration control - set the rate of change of angular velocity
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        time_step: float = 0.02,
        arm_length: float = 1.0,
        moment_of_inertia: float = 1.0,
        damping_coefficient: float = 0.05,
        stiffness_coefficient: float = 0.0,
        arm_mass: float = 1.0,
        gravitational_acceleration: float = 9.81,
        center_of_mass_ratio: float = 1.0,
        include_rod_inertia: bool = True,
        rod_mass: Optional[float] = None,
        rod_inertia_factor: float = 1.0 / 3.0,
        maximum_torque: float = 2.0,
        theta_limit: float = np.pi / 2,
        maximum_steps: int = 2000,
        control_mode: str = "position",
        sinusoidal_magnitude: float = 0.5,
        sinusoidal_frequency: float = 0.5,
        random_start: bool = False,
        seed: Optional[int] = None,
        screen_dimension: int = 600,
        pivot_y_fraction: float = 0.15,
    ) -> None:
        """
        Initialize the SoftRobotic environment.
        
        Args:
            render_mode: Rendering mode ("human" or "rgb_array")
            time_step: Simulation time step (seconds)
            arm_length: Length of the robotic arm (meters)
            moment_of_inertia: Moment of inertia of the arm (kg*m^2)
            damping_coefficient: Damping coefficient
            stiffness_coefficient: Stiffness coefficient
            arm_mass: Mass of the arm (kg)
            gravitational_acceleration: Gravitational acceleration (m/s^2)
            center_of_mass_ratio: Ratio of center of mass to arm length
            include_rod_inertia: Whether to include rod inertia in calculations
            rod_mass: Mass of the rod (kg)
            rod_inertia_factor: Inertia factor for the rod
            maximum_torque: Maximum torque that can be applied (N*m)
            theta_limit: Limit for arm angle (radians)
            maximum_steps: Maximum steps per episode
            control_mode: Control mode ("position", "velocity" or "acceleration")
            sinusoidal_magnitude: Magnitude of sinusoidal reference trajectory
            sinusoidal_frequency: Frequency of sinusoidal reference trajectory (Hz)
            random_start: Whether to start with random initial conditions
            seed: Random seed
            screen_dimension: Screen dimension for rendering
            pivot_y_fraction: Fraction of screen height for pivot position
        """
        super().__init__()

        assert control_mode in ("position", "velocity", "acceleration", "force"), "control_mode must be 'position', 'velocity' or 'acceleration' or 'force'"

        # Simulation parameters
        self.time_step = float(time_step)
        self.arm_length = float(arm_length)
        self.moment_of_inertia = float(moment_of_inertia)
        self.damping_coefficient = float(damping_coefficient)
        self.stiffness_coefficient = float(stiffness_coefficient)
        self.arm_mass = float(arm_mass)
        self.gravitational_acceleration = float(gravitational_acceleration)
        self.include_rod_inertia = bool(include_rod_inertia)
        self.rod_mass = float(rod_mass) if rod_mass is not None else self.arm_mass
        self.rod_inertia_factor = float(rod_inertia_factor)
        self.maximum_torque = float(maximum_torque)
        self.theta_limit = float(theta_limit)
        self.center_of_mass_ratio = float(center_of_mass_ratio)
        self.distance_from_pivot_to_center_of_mass = self.arm_length * self.center_of_mass_ratio

        # Effective rotational inertia
        self.effective_moment_of_inertia = self.moment_of_inertia + (self.rod_inertia_factor * self.rod_mass * (self.arm_length ** 2) if self.include_rod_inertia else 0.0)

        # Reference trajectory parameters
        self.maximum_steps = int(maximum_steps)

        # Sinusoidal control parameters
        self.sinusoidal_magnitude = float(sinusoidal_magnitude)
        self.sinusoidal_frequency = float(sinusoidal_frequency)

        # Control mode
        self.control_mode = control_mode

        # Random number generator
        self.numpy_random = np.random.default_rng(seed)
        self.random_start = bool(random_start)

        # Disturbance placeholders
        self.left_actuator_force = 0.0
        self.right_actuator_force = 0.0
        self.left_force_x_position = -0.5 * self.arm_length
        self.right_force_x_position = 0.5 * self.arm_length
        
        # Object handling (for force control mode)
        self.object_mass = 0.0  # Mass of object being manipulated
        self.object_attached = False
        self.last_left_force = 0.0
        self.last_right_force = 0.0

        # Render parameters
        self.render_mode = render_mode
        self.screen_dimension = int(screen_dimension)
        self.pivot_y_fraction = float(pivot_y_fraction)
        self.screen = None
        self.clock = None
        self.surface = None
        self.is_window_open = True
        self.last_torque_applied = 0.0

        # Define action space based on control mode
        if self.control_mode == "position":
            self.action_space = spaces.Box(
                low=-self.theta_limit,
                high=self.theta_limit,
                shape=(1,),
                dtype=np.float32,
            )
        elif self.control_mode == "velocity":
            self.action_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(1,),
                dtype=np.float32,
            )
        elif self.control_mode == "acceleration":
            self.action_space = spaces.Box(
                low=-100.0,
                high=100.0,
                shape=(1,),
                dtype=np.float32,
            )
        else:  # force control
            # Action space: [left_force, right_force]
            self.action_space = spaces.Box(
                low=-100.0,
                high=100.0,
                shape=(2,),
                dtype=np.float32,
            )

        # Observation space bounds
        arm_length = self.arm_length
        theta_limit_value = self.theta_limit
        theta_dot_limit = 10.0
        theta_ddot_limit = 100.0
        observation_low = np.array(
            [
                -arm_length,
                0.0,
                -theta_limit_value,
                -theta_dot_limit,
                -theta_ddot_limit,
                -self.maximum_torque,
                -arm_length,
                0.0,
                -theta_limit_value,
                -2 * arm_length,
                -arm_length,
                -2 * theta_limit_value,
            ],
            dtype=np.float32,
        )
        observation_high = np.array(
            [
                arm_length,
                arm_length,
                theta_limit_value,
                theta_dot_limit,
                theta_ddot_limit,
                self.maximum_torque,
                arm_length,
                arm_length,
                theta_limit_value,
                2 * arm_length,
                arm_length,
                2 * theta_limit_value,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)

        # Internal state
        self.arm_angle = 0.0
        self.angular_velocity = 0.0
        self.angular_acceleration = 0.0
        self.torque_applied = 0.0
        self.step_count = 0