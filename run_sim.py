import argparse
import csv
import math
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

from soft_robotic_env import SoftRobotic


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

    def start_episode_log():
        d = {"t": [], "step": []}
        for name in obs_headers + extra_headers:
            d[name] = []
        return d

    def append_log(log, t_val, step_idx, obs_vec, reward=None, reward_cum=None, force_left=None, force_right=None):
        log["t"].append(float(t_val))
        log["step"].append(int(step_idx))
        for i, name in enumerate(obs_headers):
            log[name].append(float(obs_vec[i]))
        if reward is not None:
            log["reward"].append(float(reward))
        else:
            log["reward"].append(0.0 if step_idx == 0 else float(log["reward"][-1]))
        if reward_cum is not None:
            log["reward_cum"].append(float(reward_cum))
        else:
            prev = 0.0 if step_idx == 0 else float(log["reward_cum"][-1])
            log["reward_cum"].append(prev)
        if force_left is not None:
            log["force_left"].append(float(force_left))
        else:
            log["force_left"].append(0.0 if step_idx == 0 else float(log["force_left"][-1]))
        if force_right is not None:
            log["force_right"].append(float(force_right))
        else:
            log["force_right"].append(0.0 if step_idx == 0 else float(log["force_right"][-1]))

    def persist_episode(ep_idx, log):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = out_path / f"episode_{ep_idx:03d}_{ts}"
        csv_file = base.with_suffix(".csv")
        png_file = base.with_suffix(".png")

        # Write CSV with headers
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "step"] + obs_headers + extra_headers)
            for i in range(len(log["t"])):
                row = [log["t"][i], log["step"][i]] + [log[name][i] for name in obs_headers + extra_headers]
                writer.writerow(row)

        # Plot observations over time
        try:
            import matplotlib.pyplot as plt
            cols = 3
            plot_headers = obs_headers + extra_headers
            rows = int(math.ceil(len(plot_headers) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.6), sharex=True)
            # Normalize axes to 2D array
            if rows == 1:
                axes = np.array([axes])
            t_arr = np.array(log["t"], dtype=float)
            for idx, name in enumerate(plot_headers):
                r, c = divmod(idx, cols)
                ax = axes[r, c]
                ax.plot(t_arr, np.array(log[name], dtype=float))
                ax.set_title(name)
                ax.grid(True, alpha=0.3)
            # Hide any unused axes
            total = rows * cols
            for k in range(len(plot_headers), total):
                r, c = divmod(k, cols)
                axes[r, c].axis("off")
            axes[rows - 1, 0].set_xlabel("time (s)")
            fig.suptitle(f"SoftRobotic observations - episode {ep_idx}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(png_file, dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"Plotting skipped (matplotlib missing or error): {e}")

        print(f"Saved CSV: {csv_file}")
        print(f"Saved plot: {png_file}")

    episode_idx = 1
    ep_log = start_episode_log()
    ep_step = 0
    ret = 0.0
    append_log(ep_log, t_val=ep_step * env.time_step, step_idx=ep_step, obs_vec=obs,
               reward=0.0, reward_cum=ret,
               force_left=getattr(env, 'left_actuator_force', 0.0), force_right=getattr(env, 'right_actuator_force', 0.0))

    try:
        for i in range(steps):
            # For all control modes, we can use a sinusoidal action or a fixed action
            t = i * env.time_step
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

            if terminated or truncated:
                # End of episode: save and reset
                persist_episode(episode_idx, ep_log)
                episode_idx += 1
                obs, info = env.reset()
                ep_log = start_episode_log()
                ep_step = 0
                ret = 0.0
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


def parse_args(argv):
    p = argparse.ArgumentParser(description="Run the SoftRobotic environment visual demo.")
    p.add_argument("--mode", choices=["position", "velocity", "acceleration", "force"], default="position", help="Control mode")
    p.add_argument("--steps", type=int, default=2000, help="Number of steps to simulate")
    p.add_argument("--screen-dim", type=int, default=600, help="Window size in pixels")
    p.add_argument("--random-start", action="store_true", help="Start from a random small angle")
    p.add_argument("--out-dir", type=str, default="runs", help="Directory to save CSV and plots")
    # Sinusoidal trajectory parameters
    p.add_argument("--sinusoidal-magnitude", type=float, default=0.5, help="Magnitude of sinusoidal reference trajectory")
    p.add_argument("--sinusoidal-frequency", type=float, default=0.5, help="Frequency of sinusoidal reference trajectory (Hz)")
    # Action parameters for each control mode
    p.add_argument("--position-action", type=float, default=0.0, help="Base position action value")
    p.add_argument("--velocity-action", type=float, default=0.0, help="Base velocity action value")
    p.add_argument("--acceleration-action", type=float, default=0.0, help="Base acceleration action value")
    # Physical params (optional overrides)
    p.add_argument("--mass", type=float, default=None, help="Mass for gravity torque (kg)")
    p.add_argument("--gravity", type=float, default=None, help="Gravity acceleration (m/s^2)")
    p.add_argument("--com-ratio", type=float, default=None, help="Center of mass ratio of L (0..1)")
    # Object manipulation
    p.add_argument("--object-mass", type=float, default=0.0, help="Mass of object to manipulate (kg)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(
        control_mode=args.mode,
        steps=args.steps,
        screen_dim=args.screen_dim,
        random_start=args.random_start,
        mass=args.mass,
        gravity=args.gravity,
        com_ratio=args.com_ratio,
        out_dir=args.out_dir,
        sinusoidal_magnitude=args.sinusoidal_magnitude,
        sinusoidal_frequency=args.sinusoidal_frequency,
        position_action=args.position_action,
        velocity_action=args.velocity_action,
        acceleration_action=args.acceleration_action,
        object_mass=args.object_mass,
    )
