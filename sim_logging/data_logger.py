"""High-frequency data logger scaffold."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any

import numpy as np
import pandas as pd

from config.base_config import LoggingConfig, ExperimentConfig


class DataLogger:
    """Efficient data logging for pendulum experiments."""

    def __init__(self, logging_config: LoggingConfig, experiment_name: str):
        self.config = logging_config
        self.experiment_name = experiment_name
        self.data_dir = Path(logging_config.output_dir) / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data = {
            'time': [],
            'angle': [],
            'velocity': [],
            'acceleration': [],
            'torque': [],
            'target': [],
            'error': [],
            'reward': [],
            'pid_integral': [],
            'pid_derivative': [],
        }

    def log_step(self, info: Dict[str, Any], reward: float, dt: float) -> None:
        """Log a single timestep based on env info."""
        step_idx = info.get('step', 0)
        if step_idx % max(self.config.log_frequency, 1) != 0:
            return

        self.data['time'].append(step_idx * dt)
        self.data['angle'].append(info.get('angle', 0.0))
        self.data['velocity'].append(info.get('velocity', 0.0))
        self.data['acceleration'].append(info.get('acceleration', 0.0))
        self.data['torque'].append(info.get('torque', 0.0))
        self.data['target'].append(info.get('target', 0.0))
        self.data['error'].append(info.get('error', 0.0))
        self.data['reward'].append(reward)
        pid_state = info.get('pid_state', {})
        self.data['pid_integral'].append(pid_state.get('integral', 0.0))
        self.data['pid_derivative'].append(pid_state.get('prev_derivative', 0.0))

    def save(self, episode: int, experiment_config: ExperimentConfig):
        """Persist logged data to CSV and metadata JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{self.experiment_name}_ep{episode:03d}_{timestamp}"

        arrays = {k: np.array(v) for k, v in self.data.items()}
        csv_path = self.data_dir / f"{base_name}.csv"
        pd.DataFrame(arrays).to_csv(csv_path, index=False)

        metadata = {
            'experiment_name': self.experiment_name,
            'episode': episode,
            'timestamp': timestamp,
            'config': experiment_config.to_dict(),
            'num_steps': len(self.data['time']),
            'final_error': float(arrays['error'][-1]) if len(arrays['error']) else 0.0,
            'mean_error': float(np.mean(np.abs(arrays['error']))) if len(arrays['error']) else 0.0,
            'max_torque': float(np.max(np.abs(arrays['torque']))) if len(arrays['torque']) else 0.0,
            'log_frequency': self.config.log_frequency,
        }

        json_path = self.data_dir / f"{base_name}_metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return csv_path, json_path

    def reset(self) -> None:
        for key in self.data:
            self.data[key].clear()
