import argparse
import csv
import math
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

from custom_env import SoftRobotic


def run(
    control_mode: str = "pid",
    steps: int = 10000,
    screen_dim: int = 600,
    tau_amp: float = 1.0,
    tau_freq: float = 0.5,
    kp: float = 5.0,
    ki: float = 0.5,
    kd: float = 1.0,
    random_start: bool = False,
    mass: Optional[float] = None,
    gravity: Optional[float] = None,
    com_ratio: Optional[float] = None,
    out_dir: str = "runs",
    force_base: float = 5.0,
    force_amp: float = 5.0,
    force_freq: float = 0.5,
    tune_mode: str = "sine",
    kp_amp: float = 2.0,
    ki_amp: float = 0.5,
    kd_amp: float = 0.5,
    tune_freq: float = 0.2,
    tune_sigma: float = 0.1,
):
    kwargs = dict(
        render_mode="human",
        control_mode=control_mode,
        screen_dim=screen_dim,
        random_start=random_start,
    )
    if mass is not None:
        kwargs["mass"] = float(mass)
    if gravity is not None:
        kwargs["gravity"] = float(gravity)
    if com_ratio is not None:
        kwargs["com_ratio"] = float(com_ratio)
    env = SoftRobotic(**kwargs)

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
        "p_error",
        "i_error",
        "d_error",
    ]

    extra_headers = ["kp", "ki", "kd", "reward", "reward_cum", "force_left", "force_right"]

    def start_episode_log():
        d = {"t": [], "step": []}
        for name in obs_headers + extra_headers:
            d[name] = []
        return d

    def append_log(log, t_val, step_idx, obs_vec, kp=None, ki=None, kd=None, reward=None, reward_cum=None, force_left=None, force_right=None):
        log["t"].append(float(t_val))
        log["step"].append(int(step_idx))
        for i, name in enumerate(obs_headers):
            log[name].append(float(obs_vec[i]))
        if kp is not None:
            log["kp"].append(float(kp))
        else:
            log["kp"].append(float('nan'))
        if ki is not None:
            log["ki"].append(float(ki))
        else:
            log["ki"].append(float('nan'))
        if kd is not None:
            log["kd"].append(float(kd))
        else:
            log["kd"].append(float('nan'))
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
    append_log(ep_log, t_val=ep_step * env.dt, step_idx=ep_step, obs_vec=obs,
               kp=getattr(env, 'Kp', None), ki=getattr(env, 'Ki', None), kd=getattr(env, 'Kd', None),
               reward=0.0, reward_cum=ret,
               force_left=getattr(env, 'force_left', 0.0), force_right=getattr(env, 'force_right', 0.0))

    # Gain tuning helpers
    rng = np.random.default_rng()
    kp_state, ki_state, kd_state = kp, ki, kd

    def dynamic_gains(step_idx: int, t: float):
        nonlocal kp_state, ki_state, kd_state
        if tune_mode == "fixed":
            kp_dyn, ki_dyn, kd_dyn = kp, ki, kd
        elif tune_mode == "sine":
            kp_dyn = kp + kp_amp * math.sin(2.0 * math.pi * tune_freq * t)
            ki_dyn = ki + ki_amp * math.sin(2.0 * math.pi * tune_freq * t + 2.0)
            kd_dyn = kd + kd_amp * math.sin(2.0 * math.pi * tune_freq * t + 4.0)
        elif tune_mode == "random":
            kp_state = kp_state + rng.normal(0.0, tune_sigma)
            ki_state = ki_state + rng.normal(0.0, tune_sigma)
            kd_state = kd_state + rng.normal(0.0, tune_sigma)
            kp_dyn, ki_dyn, kd_dyn = kp_state, ki_state, kd_state
        else:
            kp_dyn, ki_dyn, kd_dyn = kp, ki, kd
        # Clip to env bounds if available
        kp_dyn = float(max(float(env.pid_low[0]), min(float(env.pid_high[0]), kp_dyn)))
        ki_dyn = float(max(float(env.pid_low[1]), min(float(env.pid_high[1]), ki_dyn)))
        kd_dyn = float(max(float(env.pid_low[2]), min(float(env.pid_high[2]), kd_dyn)))
        return kp_dyn, ki_dyn, kd_dyn

    try:
        for i in range(steps):
            if control_mode == "forces":
                # Sinusoidal differential forces around a base level
                t = i * env.dt
                kp_dyn, ki_dyn, kd_dyn = dynamic_gains(i, t)
                Fdiff = force_amp * math.sin(2.0 * math.pi * force_freq * t)
                F_left = max(0.0, min(env.max_force, force_base - Fdiff))
                F_right = max(0.0, min(env.max_force, force_base + Fdiff))
                action = np.array([F_left, F_right, kp_dyn, ki_dyn, kd_dyn], dtype=np.float32)
            elif control_mode == "torque":
                # Simple sinusoidal torque for visible motion
                t = i * env.dt
                kp_dyn, ki_dyn, kd_dyn = dynamic_gains(i, t)
                tau = tau_amp * math.sin(2.0 * math.pi * tau_freq * t)
                action = np.array([tau, kp_dyn, ki_dyn, kd_dyn], dtype=np.float32)
            else:
                # Fixed PID gains each step; env computes torque internally
                t = i * env.dt
                kp_dyn, ki_dyn, kd_dyn = dynamic_gains(i, t)
                action = np.array([kp_dyn, ki_dyn, kd_dyn], dtype=np.float32)

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
                append_log(ep_log, t_val=ep_step * env.dt, step_idx=ep_step, obs_vec=obs,
                           kp=getattr(env, 'Kp', None), ki=getattr(env, 'Ki', None), kd=getattr(env, 'Kd', None),
                           reward=0.0, reward_cum=ret,
                           force_left=getattr(env, 'force_left', 0.0), force_right=getattr(env, 'force_right', 0.0))
            else:
                ep_step += 1
                append_log(ep_log, t_val=ep_step * env.dt, step_idx=ep_step, obs_vec=obs,
                           kp=getattr(env, 'Kp', None), ki=getattr(env, 'Ki', None), kd=getattr(env, 'Kd', None),
                           reward=reward, reward_cum=ret,
                           force_left=getattr(env, 'force_left', 0.0), force_right=getattr(env, 'force_right', 0.0))

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def parse_args(argv):
    p = argparse.ArgumentParser(description="Run the SoftRobotic environment visual demo.")
    p.add_argument("--mode", choices=["forces", "torque", "pid"], default="forces", help="Control mode")
    p.add_argument("--steps", type=int, default=2000, help="Number of steps to simulate")
    p.add_argument("--screen-dim", type=int, default=600, help="Window size in pixels")
    p.add_argument("--random-start", action="store_true", help="Start from a random small angle")
    p.add_argument("--out-dir", type=str, default="runs", help="Directory to save CSV and plots")
    # Forces-mode params
    p.add_argument("--force-base", type=float, default=5.0, help="Base force per side (N) in forces mode")
    p.add_argument("--force-amp", type=float, default=5.0, help="Force sine amplitude (N)")
    p.add_argument("--force-freq", type=float, default=0.5, help="Force sine frequency (Hz)")
    # Torque-mode params
    p.add_argument("--tau-amp", type=float, default=1.0, help="Torque amplitude (Nm)")
    p.add_argument("--tau-freq", type=float, default=0.5, help="Torque frequency (Hz)")
    # PID-mode params
    p.add_argument("--kp", type=float, default=10.0, help="Kp gain (pid mode)")
    p.add_argument("--ki", type=float, default=0.5, help="Ki gain (pid mode)")
    p.add_argument("--kd", type=float, default=1.0, help="Kd gain (pid mode)")
    # PID dynamic tuning profile
    p.add_argument("--tune-mode", choices=["fixed", "sine", "random"], default="sine", help="Dynamic gain profile")
    p.add_argument("--kp-amp", type=float, default=0.0, help="Sine amplitude for Kp")
    p.add_argument("--ki-amp", type=float, default=0.0, help="Sine amplitude for Ki")
    p.add_argument("--kd-amp", type=float, default=0.0, help="Sine amplitude for Kd")
    p.add_argument("--tune-freq", type=float, default=0.2, help="Sine frequency for PID tuning (Hz)")
    p.add_argument("--tune-sigma", type=float, default=0.1, help="Random-walk sigma for PID tuning")
    # Physical params (optional overrides)
    p.add_argument("--mass", type=float, default=None, help="Mass for gravity torque (kg)")
    p.add_argument("--gravity", type=float, default=None, help="Gravity acceleration (m/s^2)")
    p.add_argument("--com-ratio", type=float, default=None, help="Center of mass ratio of L (0..1)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(
        control_mode=args.mode,
        steps=args.steps,
        screen_dim=args.screen_dim,
        tau_amp=args.tau_amp,
        tau_freq=args.tau_freq,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        random_start=args.random_start,
        mass=args.mass,
        gravity=args.gravity,
        com_ratio=args.com_ratio,
        out_dir=args.out_dir,
        force_base=args.force_base,
        force_amp=args.force_amp,
        force_freq=args.force_freq,
        tune_mode=args.tune_mode,
        kp_amp=args.kp_amp,
        ki_amp=args.ki_amp,
        kd_amp=args.kd_amp,
        tune_freq=args.tune_freq,
        tune_sigma=args.tune_sigma,
    )
