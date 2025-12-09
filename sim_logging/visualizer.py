"""Plotting utilities for logged pendulum episodes."""
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


class Visualizer:
    """Generate plots from logged episode CSV files."""

    @staticmethod
    def plot_episode(csv_path: Path, output_dir: Optional[Path] = None):
        output_dir = Path(output_dir) if output_dir else csv_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(csv_path)
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))

        axes[0, 0].plot(df['time'], df['angle'], label='actual')
        axes[0, 0].plot(df['time'], df['target'], '--', label='target')
        axes[0, 0].set_ylabel('Angle (rad)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(df['time'], df['error'])
        axes[0, 1].set_ylabel('Error (rad)')
        axes[0, 1].grid(True)

        axes[1, 0].plot(df['time'], df['velocity'])
        axes[1, 0].set_ylabel('Velocity (rad/s)')
        axes[1, 0].grid(True)

        axes[1, 1].plot(df['time'], df['torque'])
        axes[1, 1].set_ylabel('Torque (N*m)')
        axes[1, 1].grid(True)

        axes[2, 0].plot(df['time'], df['reward'])
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Reward')
        axes[2, 0].grid(True)

        axes[2, 1].plot(df['time'], df['pid_integral'], label='integral')
        axes[2, 1].plot(df['time'], df['pid_derivative'], label='derivative')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('PID state')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        plt.tight_layout()
        plot_path = output_dir / f"{csv_path.stem}_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path
