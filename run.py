"""Convenience CLI to run pendulum simulations with common overrides."""
import argparse
import sys
from pathlib import Path

# Make sure local packages are importable when invoked as a script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.base_config import ExperimentConfig
from config.presets import get_config
from experiments.run_pid_experiment import run_pid_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pendulum PID simulation")
    parser.add_argument('--preset', default='default', help='Config preset name (default, high_frequency, aggressive_pid)')
    parser.add_argument('--name', default='pendulum_pid', help='Experiment name')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--steps', type=int, default=None, help='Max steps per episode (override preset)')
    parser.add_argument('--dt', type=float, default=None, help='Timestep dt (override preset)')
    parser.add_argument('--control-mode', choices=['position', 'velocity', 'acceleration'], default=None, help='Control mode')
    parser.add_argument('--render', choices=['human', 'none'], default='human', help='Render mode (human or none)')
    parser.add_argument('--fps', type=int, default=None, help='Render FPS')
    parser.add_argument('--auto-plot', action='store_true', help='Enable auto-plot after each episode')
    parser.add_argument('--no-auto-plot', dest='auto_plot', action='store_false', help='Disable auto-plot after each episode')
    parser.set_defaults(auto_plot=True)
    parser.add_argument('--log-frequency', type=int, default=None, help='Log every N steps')
    parser.add_argument('--output-dir', default=None, help='Output directory for logs/plots')

    # Physics overrides
    parser.add_argument('--arm-length', type=float, default=None, help='Pendulum arm length (m)')
    parser.add_argument('--arm-mass', type=float, default=None, help='Pendulum arm mass (kg)')
    parser.add_argument('--base-mass', type=float, default=None, help='Point mass at arm end (kg)')
    parser.add_argument('--moment-of-inertia', type=float, default=None, help='Base moment of inertia (kg*m^2)')
    parser.add_argument('--max-torque', type=float, default=None, help='Torque limit (N*m)')

    # PID overrides
    parser.add_argument('--kp', type=float, default=10, help='Proportional gain')
    parser.add_argument('--ki', type=float, default=10, help='Integral gain')
    parser.add_argument('--kd', type=float, default=10, help='Derivative gain')
    parser.add_argument('--include-pid-gains', action='store_true', help='Append PID gains to observation space')

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = get_config(args.preset)
    cfg.name = args.name

    if args.steps is not None:
        cfg.simulation.max_steps = args.steps
    if args.dt is not None:
        cfg.simulation.dt = args.dt
    if args.control_mode is not None:
        cfg.simulation.control_mode = args.control_mode
    cfg.simulation.include_pid_gains = bool(args.include_pid_gains)

    if args.render == 'none':
        cfg.render.render_mode = None
    else:
        cfg.render.render_mode = 'human'
    if args.fps is not None:
        cfg.render.fps = args.fps

    if args.log_frequency is not None:
        cfg.logging.log_frequency = args.log_frequency
    if args.output_dir is not None:
        cfg.logging.output_dir = args.output_dir
    cfg.logging.auto_plot = bool(args.auto_plot)

    # Physics overrides
    if args.arm_length is not None:
        cfg.physics.arm_length = args.arm_length
    if args.arm_mass is not None:
        cfg.physics.arm_mass = args.arm_mass
    if args.base_mass is not None:
        cfg.physics.base_mass = args.base_mass
    if args.moment_of_inertia is not None:
        cfg.physics.moment_of_inertia = args.moment_of_inertia
    if args.max_torque is not None:
        cfg.physics.max_torque = args.max_torque

    # PID overrides
    if args.kp is not None:
        cfg.pid.kp = args.kp
    if args.ki is not None:
        cfg.pid.ki = args.ki
    if args.kd is not None:
        cfg.pid.kd = args.kd

    cfg.validate()
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    run_pid_experiment(cfg, num_episodes=args.episodes)


if __name__ == '__main__':
    main()
