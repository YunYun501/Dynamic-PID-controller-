"""Future RL training entrypoint placeholder."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_rl_experiment():
    """Hook RL library here when ready."""
    raise NotImplementedError('RL experiment runner not implemented yet.')


if __name__ == "__main__":
    raise SystemExit("run_rl_experiment is not implemented yet. Use run_pid_experiment instead.")
