import argparse
import sys


def parse_args(argv):
    """Parse command line arguments for the simulation."""
    p = argparse.ArgumentParser(description="Run the SoftRobotic environment visual demo.")
    p.add_argument("--mode", choices=["position", "velocity", "acceleration", "force"], default="position", help="Control mode")
    p.add_argument("--steps", type=int, default=5000, help="Number of steps to simulate")
    p.add_argument("--screen-dim", type=int, default=600, help="Window size in pixels")
    p.add_argument("--random-start", action="store_true", help="Start from a random small angle")
    p.add_argument("--out-dir", type=str, default="runs", help="Directory to save CSV and plots")
    # Sinusoidal trajectory parameters
    p.add_argument("--sinusoidal-magnitude", type=float, default=0.5, help="Magnitude of sinusoidal reference trajectory")
    p.add_argument("--sinusoidal-frequency", type=float, default=0.2, help="Frequency of sinusoidal reference trajectory (Hz)")
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