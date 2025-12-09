"""Gymnasium environment skeleton for the pendulum control task."""
from typing import Any, Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config.base_config import ExperimentConfig
from core.physics import PendulumDynamics, PendulumKinematics
from core.controllers import PIDController
from core.rewards import tracking_reward


class PendulumEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.config.validate()
        self.dynamics = PendulumDynamics(config.physics)
        self.kinematics = PendulumKinematics(config.physics.arm_length)
        self.controller = PIDController(config.pid)
        self.renderer = None
        self.include_pid_gains = config.simulation.include_pid_gains
        self.dt = config.simulation.dt

        if config.render.render_mode:
            try:
                from .renderer import PendulumRenderer
                self.renderer = PendulumRenderer(config.render)
            except ImportError as exc:
                raise ImportError(
                    "Rendering requested but pygame is not installed. "
                    "Install pygame or set render_mode=None."
                ) from exc
            except Exception as exc:
                raise RuntimeError("Renderer failed to initialize") from exc

        self._setup_spaces()

        self.angle = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.torque = 0.0
        self.step_count = 0
        self.target = 0.0

    def _setup_spaces(self) -> None:
        physics = self.config.physics
        mode = self.config.simulation.control_mode

        if mode == 'position':
            self.action_space = spaces.Box(low=-physics.max_angle, high=physics.max_angle, shape=(1,), dtype=np.float32)
        elif mode == 'velocity':
            self.action_space = spaces.Box(low=-physics.max_velocity, high=physics.max_velocity, shape=(1,), dtype=np.float32)
        elif mode == 'acceleration':
            max_accel = physics.max_torque / max(self.dynamics.effective_inertia(), 1e-6)
            self.action_space = spaces.Box(low=-max_accel, high=max_accel, shape=(1,), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported control_mode: {mode}")

        obs_low = [
            -physics.max_angle,
            -physics.max_velocity,
            -physics.max_angle,
            -2 * physics.max_angle,
        ]
        obs_high = [
            physics.max_angle,
            physics.max_velocity,
            physics.max_angle,
            2 * physics.max_angle,
        ]

        if self.include_pid_gains:
            pid_range = 1e6
            obs_low.extend([-pid_range, -pid_range, -pid_range])
            obs_high.extend([pid_range, pid_range, pid_range])

        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        physics = self.config.physics
        sim = self.config.simulation

        if sim.random_initial_state:
            self.angle = float(self.np_random.uniform(-physics.max_angle, physics.max_angle))
            self.velocity = float(self.np_random.uniform(-0.5, 0.5))
        else:
            self.angle = sim.initial_angle
            self.velocity = sim.initial_velocity

        self.acceleration = 0.0
        self.torque = 0.0
        self.step_count = 0
        self.target = 0.0
        self.controller.reset()

        return self._get_observation(), self._get_info()

    def step(self, action):
        sim = self.config.simulation
        physics = self.config.physics

        clipped_action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        self.target = float(clipped_action[0])

        if sim.control_mode == 'position':
            self.torque = self.controller.compute_control(self.target, self.angle, sim.dt)
        elif sim.control_mode == 'velocity':
            self.torque = self.controller.compute_control(self.target, self.velocity, sim.dt)
        elif sim.control_mode == 'acceleration':
            desired_accel = self.target
            tau_gravity, tau_damping, tau_friction = self.dynamics.compute_passive_torques(self.angle, self.velocity)
            inertia = self.dynamics.effective_inertia()
            feedforward_torque = desired_accel * inertia - (tau_gravity + tau_damping + tau_friction)
            self.torque = feedforward_torque
        else:
            raise ValueError(f"Unsupported control_mode: {sim.control_mode}")

        self.torque = float(np.clip(self.torque, -physics.max_torque, physics.max_torque))

        self.angle, self.velocity, self.acceleration = self.dynamics.integrate_step(
            self.angle, self.velocity, self.torque, sim.dt
        )
        self.step_count += 1

        reward = self._compute_reward()
        terminated = False
        truncated = self.step_count >= sim.max_steps

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        base_obs = [
            self.angle,
            self.velocity,
            self.target,
            self.target - self.angle,
        ]
        if self.include_pid_gains:
            base_obs.extend([self.controller.config.kp, self.controller.config.ki, self.controller.config.kd])
        return np.array(base_obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            'angle': self.angle,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'torque': self.torque,
            'target': self.target,
            'error': self.target - self.angle,
            'step': self.step_count,
            'pid_state': self.controller.get_state(),
        }

    def _compute_reward(self) -> float:
        return float(tracking_reward(self.target - self.angle, self.torque, self.velocity))

    def render(self):
        if self.renderer is not None:
            return self.renderer.render(self)
        return None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
