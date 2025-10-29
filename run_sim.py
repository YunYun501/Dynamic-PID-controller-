import argparse
import math
import sys

from typing import Optional

import numpy as np

from custom_env import SoftRobotic


def run(
    control_mode: str = "torque",
    steps: int = 1000,
    screen_dim: int = 600,
    tau_amp: float = 1.0,
    tau_freq: float = 0.5,
    kp: float = 10.0,
    ki: float = 0.5,
    kd: float = 1.0,
    random_start: bool = False,
    mass: Optional[float] = None,
    gravity: Optional[float] = None,
    com_ratio: Optional[float] = None,
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
    print("SoftRobotic demo running — close the window or Ctrl+C to exit.")
    try:
        for i in range(steps):
            if control_mode == "torque":
                # Simple sinusoidal torque for visible motion
                t = i * env.dt
                tau = tau_amp * math.sin(2.0 * math.pi * tau_freq * t)
                action = np.array([tau], dtype=np.float32)
            else:
                # Fixed PID gains each step; env computes torque internally
                action = np.array([kp, ki, kd], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                obs, info = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def parse_args(argv):
    p = argparse.ArgumentParser(description="Run the SoftRobotic environment visual demo.")
    p.add_argument("--mode", choices=["torque", "pid"], default="torque", help="Control mode")
    p.add_argument("--steps", type=int, default=2000, help="Number of steps to simulate")
    p.add_argument("--screen-dim", type=int, default=600, help="Window size in pixels")
    p.add_argument("--random-start", action="store_true", help="Start from a random small angle")
    # Torque-mode params
    p.add_argument("--tau-amp", type=float, default=1.0, help="Torque amplitude (Nm)")
    p.add_argument("--tau-freq", type=float, default=0.5, help="Torque frequency (Hz)")
    # PID-mode params
    p.add_argument("--kp", type=float, default=10.0, help="Kp gain (pid mode)")
    p.add_argument("--ki", type=float, default=0.5, help="Ki gain (pid mode)")
    p.add_argument("--kd", type=float, default=1.0, help="Kd gain (pid mode)")
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
    )
