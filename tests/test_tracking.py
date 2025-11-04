import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from environment.soft_robotic_env import SoftRobotic

def run_tracking_test():
    # Create environment
    env = SoftRobotic(
        render_mode=None,  # No rendering for testing
        control_mode="position",
        sinusoidal_magnitude=0.5,
        sinusoidal_frequency=0.5
    )

    # Run simulation
    obs, info = env.reset()
    theta_values = []
    theta_target_values = []
    time_values = []

    for i in range(200):
        # Get target from environment
        theta_ref, x_ref, y_ref = env.kinematics.calculate_reference_trajectory(
            i, env.time_step, env.sinusoidal_magnitude, env.sinusoidal_frequency)
        
        # Apply position control action
        action = np.array([theta_ref], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store values for plotting
        theta_values.append(obs[2])  # Current theta
        theta_target_values.append(obs[8])  # Target theta
        time_values.append(i * env.time_step)
        
        if terminated or truncated:
            break

    # Calculate tracking error
    tracking_error = np.abs(np.array(theta_values) - np.array(theta_target_values))
    avg_error = np.mean(tracking_error)
    max_error = np.max(tracking_error)
    
    print(f"Average tracking error: {avg_error:.4f}")
    print(f"Maximum tracking error: {max_error:.4f}")

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Tracking performance
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_values, theta_values, label='Actual Theta')
    plt.plot(time_values, theta_target_values, label='Target Theta', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (rad)')
    plt.title('Position Control Tracking Performance')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_values, tracking_error, 'r', label='Tracking Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('Tracking Error Over Time')
    plt.legend()
    plt.grid(True)
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Avg Error: {avg_error:.4f}\nMax Error: {max_error:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    tracking_plot_file = plots_dir / f"tracking_performance_{timestamp}.png"
    plt.savefig(tracking_plot_file, dpi=120)
    plt.show()
    
    print(f"Saved tracking plot: {tracking_plot_file}")
    
    # Plot 2: Detailed view of a segment
    if len(time_values) > 50:
        start_idx = 50
        end_idx = min(150, len(time_values))
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_values[start_idx:end_idx], theta_values[start_idx:end_idx], label='Actual Theta', marker='o', markersize=3)
        plt.plot(time_values[start_idx:end_idx], theta_target_values[start_idx:end_idx], label='Target Theta', linestyle='--', marker='s', markersize=3)
        plt.xlabel('Time (s)')
        plt.ylabel('Theta (rad)')
        plt.title('Position Control Tracking Performance (Detailed View)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        detail_plot_file = plots_dir / f"tracking_detail_{timestamp}.png"
        plt.savefig(detail_plot_file, dpi=120)
        plt.show()
        
        print(f"Saved detail plot: {detail_plot_file}")

if __name__ == "__main__":
    run_tracking_test()