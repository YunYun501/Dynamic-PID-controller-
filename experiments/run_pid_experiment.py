"""PID experiment runner."""
import sys
from pathlib import Path
import numpy as np

# Ensure project root is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.base_config import ExperimentConfig
from environment.pendulum_env import PendulumEnv
from sim_logging.data_logger import DataLogger
from sim_logging.visualizer import Visualizer


def generate_target(step: int, config: ExperimentConfig) -> float:
    #t = step * config.simulation.dt
    #amplitude = 0.5
    #frequency = 0.5
    #return float(amplitude * np.sin(2 * np.pi * frequency * t))
    CONST_TARGET = 30.0 * (np.pi / 180.0)  # 30 degrees in radians
    return float(CONST_TARGET)


def run_pid_experiment(config: ExperimentConfig, num_episodes: int = 1):
    config.validate()
    env = PendulumEnv(config)
    logger = DataLogger(config.logging, config.name)
    dt = config.simulation.dt

    for episode in range(num_episodes):
        _, _ = env.reset()
        logger.reset()
        episode_reward = 0.0

        for step in range(config.simulation.max_steps):
            target = generate_target(step, config)
            action = np.array([target], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            logger.log_step(info, reward, dt)
            episode_reward += reward
            if config.render.render_mode == 'human':
                env.render()
            if terminated or truncated:
                break

        csv_path, json_path = logger.save(episode, config)
        plot_path = None
        if config.logging.auto_plot:
            plot_path = Visualizer.plot_episode(csv_path)
        msg = f"Episode {episode}: reward={episode_reward:.2f} -> {csv_path}"
        if plot_path:
            msg += f" (plot {plot_path})"
        print(msg)

    env.close()


if __name__ == "__main__":
    cfg = ExperimentConfig()
    cfg.render.render_mode = None
    cfg.logging.auto_plot = False
    run_pid_experiment(cfg, num_episodes=1)
