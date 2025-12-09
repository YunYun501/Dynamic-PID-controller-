"""Reward helpers for RL experiments."""

def tracking_reward(error: float, torque: float, velocity: float) -> float:
    """Baseline reward: penalize tracking error, effort, and high velocity."""
    return -abs(error) - 0.01 * (torque ** 2) - 0.001 * (velocity ** 2)


def placeholder_reward(*args, **kwargs) -> float:
    """Extend with custom reward shaping for experiments."""
    return tracking_reward(kwargs.get('error', 0.0), kwargs.get('torque', 0.0), kwargs.get('velocity', 0.0))
