import sys

from simulation.runner import run
from simulation.arguments import parse_args


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