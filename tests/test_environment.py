"""Smoke tests for the Gymnasium environment."""
import numpy as np
import pytest

from config.base_config import ExperimentConfig
from environment.pendulum_env import PendulumEnv


def test_env_reset_and_step():
    cfg = ExperimentConfig()
    cfg.render.render_mode = None
    env = PendulumEnv(cfg)
    obs, info = env.reset()
    action = np.array([0.1], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape[0] == 4
    assert isinstance(reward, float)
    assert terminated is False
    assert isinstance(truncated, bool)


@pytest.mark.parametrize('mode', ['position', 'velocity', 'acceleration'])
def test_action_space_modes(mode):
    cfg = ExperimentConfig()
    cfg.simulation.control_mode = mode
    cfg.render.render_mode = None
    env = PendulumEnv(cfg)
    assert env.action_space.shape == (1,)


def test_observation_with_pid_gains_enabled():
    cfg = ExperimentConfig()
    cfg.simulation.include_pid_gains = True
    cfg.render.render_mode = None
    env = PendulumEnv(cfg)
    obs, _ = env.reset()
    assert obs.shape[0] == 7


def test_acceleration_mode_feedforward_torque():
    cfg = ExperimentConfig()
    cfg.simulation.control_mode = 'acceleration'
    cfg.render.render_mode = None
    env = PendulumEnv(cfg)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
