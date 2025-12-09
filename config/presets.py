"""Pre-defined configuration presets from the design plan."""
from .base_config import ExperimentConfig


def get_config(preset_name: str) -> ExperimentConfig:
    """Return a preset ExperimentConfig by name."""
    preset = preset_name.lower()

    if preset == 'default':
        return ExperimentConfig()
    if preset == 'high_frequency':
        config = ExperimentConfig()
        config.simulation.dt = 0.001
        config.simulation.max_steps = 10000
        return config
    if preset == 'aggressive_pid':
        config = ExperimentConfig()
        config.pid.kp = 200.0
        config.pid.ki = 50.0
        config.pid.kd = 40.0
        return config

    # Add new presets here as experiments evolve
    raise ValueError(f"Unknown preset: {preset_name}")
