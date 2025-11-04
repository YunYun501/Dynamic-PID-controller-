import csv
import json
import math
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

from soft_robotic_env import SoftRobotic
from simulation.logging import start_episode_log, append_log, persist_episode
from simulation.visualization import plot_observations


def run(
    control_mode: str = "position",
    steps: int = 10000,
    screen_dim: int = 600,
    random_start: bool = False,
    mass: Optional[float] = None,
    gravity: Optional[float] = None,
    com_ratio: Optional[float] = None,
    out_dir: str = "runs",
    # Sinusoidal control parameters
    sinusoidal_magnitude: float = 0.5,
    sinusoidal_frequency: float = 0.5,
    # Action parameters for each control mode
    position_action: float = 0.0,
    velocity_action: float = 0.0,
    acceleration_action: float = 0.0,
    # Object manipulation parameters
    object_mass: float = 0.0,
):
    kwargs = dict(
        render_mode="human",
        control_mode=control_mode,
        screen_dimension=screen_dim,
        random_start=random_start,
        sinusoidal_magnitude=sinusoidal_magnitude,
        sinusoidal_frequency=sinusoidal_frequency,
    )
    if mass is not None:
        kwargs["arm_mass"] = float(mass)
    if gravity is not None:
        kwargs["gravitational_acceleration"] = float(gravity)
    if com_ratio is not None:
        kwargs["center_of_mass_ratio"] = float(com_ratio)
    env = SoftRobotic(**kwargs)
    
    # Attach object if specified
    if object_mass > 0:
        env.attach_object(object_mass)

    obs, info = env.reset()
    print("SoftRobotic demo running - close the window or Ctrl+C to exit.")

    # Prepare logging (per-episode)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    obs_headers = [
        "x",
        "y",
        "theta",
        "theta_dot",
        "theta_ddot",
        "tau",
        "x_target",
        "y_target",
        "theta_target",
        "ex",
        "ey",
        "etheta",
    ]

    extra_headers = ["reward", "reward_cum", "force_left", "force_right"]

    episode_idx = 1
    ep_log = start_episode_log(obs_headers, extra_headers)
    ep_step = 0
    ret = 0.0
    append_log(ep_log, t_val=ep_step * env.time_step, step_idx=ep_step, obs_vec=obs,
               reward=0.0, reward_cum=ret,
               force_left=getattr(env, 'left_actuator_force', 0.0), force_right=getattr(env, 'right_actuator_force', 0.0))

    # Detailed logging for agent evaluation
    decision_log = []

    def log_decision(step, action, obs, reward, info, action_value=None):
        """Log detailed control decisions for agent evaluation."""
        decision_entry = {
            "step": step,
            "timestamp": step * env.time_step,
            "action": action.tolist() if hasattr(action, 'tolist') else action,
            "action_value": action_value,
            "observation": {
                "theta": float(obs[2]) if len(obs) > 2 else 0.0,
                "theta_dot": float(obs[3]) if len(obs) > 3 else 0.0,
                "theta_target": float(obs[8]) if len(obs) > 8 else 0.0,
                "error": float(obs[11]) if len(obs) > 11 else 0.0,
            },
            "reward": float(reward),
            "info": {
                "torque": info.get("tau", 0.0),
                "force_left": info.get("force_left", 0.0),
                "force_right": info.get("force_right", 0.0),
            }
        }
        decision_log.append(decision_entry)

    try:
        for i in range(steps):
            # For all control modes, we can use a sinusoidal action or a fixed action
            t = i * env.time_step
            action_value = None
            if control_mode == "position":
                # Position control: action is target angle
                action_value = position_action + sinusoidal_magnitude * math.sin(2.0 * math.pi * sinusoidal_frequency * t)
                action = np.array([action_value], dtype=np.float32)
            elif control_mode == "velocity":
                # Velocity control: action is target angular velocity
                action_value = velocity_action + sinusoidal_magnitude * math.sin(2.0 * math.pi * sinusoidal_frequency * t)
                action = np.array([action_value], dtype=np.float32)
            elif control_mode == "acceleration":
                # Acceleration control: action is target angular acceleration
                action_value = acceleration_action + sinusoidal_magnitude * math.sin(2.0 * math.pi * sinusoidal_frequency * t)
                action = np.array([action_value], dtype=np.float32)
            else:  # force control
                # Force control: action is [left_force, right_force]
                # For demo purposes, we'll use sinusoidal forces
                left_force = sinusoidal_magnitude * math.sin(2.0 * math.pi * sinusoidal_frequency * t)
                right_force = sinusoidal_magnitude * math.sin(2.0 * math.pi * sinusoidal_frequency * t + math.pi/4)
                action = np.array([left_force, right_force], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            ret += float(reward)
            env.render()
            
            # Log decision for agent evaluation
            log_decision(i, action, obs, reward, info, action_value)

            if terminated or truncated:
                # End of episode: save and reset
                persist_episode(episode_idx, ep_log, control_mode, sinusoidal_magnitude, 
                              sinusoidal_frequency, position_action, velocity_action, 
                              acceleration_action, decision_log, out_path, env)
                episode_idx += 1
                obs, info = env.reset()
                ep_log = start_episode_log(obs_headers, extra_headers)
                ep_step = 0
                ret = 0.0
                decision_log.clear()  # Reset decision log for new episode
                append_log(ep_log, t_val=ep_step * env.time_step, step_idx=ep_step, obs_vec=obs,
                           reward=0.0, reward_cum=ret,
                           force_left=getattr(env, 'left_actuator_force', 0.0), force_right=getattr(env, 'right_actuator_force', 0.0))
            else:
                ep_step += 1
                append_log(ep_log, t_val=ep_step * env.time_step, step_idx=ep_step, obs_vec=obs,
                           reward=reward, reward_cum=ret,
                           force_left=getattr(env, 'left_actuator_force', 0.0), force_right=getattr(env, 'right_actuator_force', 0.0))

    except KeyboardInterrupt:
        pass
    finally:
        env.close()