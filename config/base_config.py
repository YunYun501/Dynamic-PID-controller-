"""Base configuration dataclasses for the pendulum control system."""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import numpy as np


@dataclass
class PhysicsConfig:
    """Physical parameters of the pendulum."""
    arm_length: float = 1.0
    arm_mass: float = 1.0
    base_mass: float = 0.0  # point mass attached at the distal end
    moment_of_inertia: float = 1.0
    gravity: float = 9.81
    damping_coefficient: float = 0.1
    friction_coefficient: float = 0.05
    max_torque: float = 10.0
    max_angle: float = np.pi
    max_velocity: float = 10.0


@dataclass
class PIDConfig:
    """PID controller parameters."""
    kp: float = 100.0
    ki: float = 10.0
    kd: float = 20.0
    integral_limit: float = 1.0
    derivative_filter_tau: float = 0.0


@dataclass
class SimulationConfig:
    """Simulation parameters."""
    dt: float = 0.01
    max_steps: int = 1000
    control_mode: str = 'position'
    include_pid_gains: bool = False
    initial_angle: float = 0.0
    initial_velocity: float = 0.0
    random_initial_state: bool = False


@dataclass
class RenderConfig:
    """Rendering parameters."""
    screen_size: int = 600
    render_mode: str = 'human'
    fps: int = 50
    pivot_y_fraction: float = 0.15


@dataclass
class LoggingConfig:
    """Data logging parameters."""
    output_dir: str = 'outputs'
    log_frequency: int = 1
    save_video: bool = False
    auto_plot: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration container."""
    name: str = 'pendulum_pid'
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        assert self.physics.arm_length > 0, 'arm_length must be positive'
        assert self.physics.moment_of_inertia > 0, 'moment_of_inertia must be positive'
        assert self.physics.base_mass >= 0, 'base_mass cannot be negative'
        assert self.simulation.dt > 0, 'dt must be positive'
        assert self.simulation.max_steps > 0, 'max_steps must be positive'
        assert self.simulation.control_mode in {'position', 'velocity', 'acceleration'}, 'control_mode invalid'
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from a nested dictionary."""
        return cls(
            name=data.get('name', 'pendulum_pid'),
            physics=PhysicsConfig(**data.get('physics', {})),
            pid=PIDConfig(**data.get('pid', {})),
            simulation=SimulationConfig(**data.get('simulation', {})),
            render=RenderConfig(**data.get('render', {})),
            logging=LoggingConfig(**data.get('logging', {})),
        )
