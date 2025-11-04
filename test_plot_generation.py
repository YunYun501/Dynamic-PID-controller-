import numpy as np
from simulation.visualization import plot_tracking_performance
from pathlib import Path

# Create test data similar to what would be in ep_log
test_log = {
    "t": np.linspace(0, 10, 100).tolist(),
    "theta": (np.sin(np.linspace(0, 10, 100)) * 0.5).tolist(),
    "theta_target": (np.sin(np.linspace(0, 10, 100) + 0.1) * 0.5).tolist()
}

# Test the plot generation
output_dir = Path("plots/test")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "test_tracking"

try:
    plot_tracking_performance(test_log, 1, output_file)
    print("Tracking plot generation test completed successfully!")
    print(f"Check {output_file.with_name(f'{output_file.stem}_tracking{output_file.suffix}')} for the generated plot")
except Exception as e:
    print(f"Error in plot generation: {e}")
    import traceback
    traceback.print_exc()