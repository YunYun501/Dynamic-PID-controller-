import math
import numpy as np


def plot_observations(ep_log, env, ep_idx, png_file):
    """Plot observations over time and save to file."""
    try:
        import matplotlib.pyplot as plt
        
        # Select only essential headers for plotting based on control mode
        essential_headers = ["theta", "theta_dot", "tau", "reward"]
        
        # Add force information for force control mode
        if env.control_mode == "force":
            essential_headers.extend(["force_left", "force_right"])
        else:
            # Add target information for other control modes
            essential_headers.append("theta_target")
        
        # Add cumulative reward
        essential_headers.append("reward_cum")
        
        cols = 3
        rows = int(math.ceil(len(essential_headers) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.6), sharex=True)
        # Normalize axes to 2D array
        if rows == 1:
            axes = np.array([axes])
        t_arr = np.array(ep_log["t"], dtype=float)
        for idx, name in enumerate(essential_headers):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            ax.plot(t_arr, np.array(ep_log[name], dtype=float))
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
        # Hide any unused axes
        total = rows * cols
        for k in range(len(essential_headers), total):
            r, c = divmod(k, cols)
            axes[r, c].axis("off")
        axes[rows - 1, 0].set_xlabel("time (s)")
        fig.suptitle(f"SoftRobotic observations - episode {ep_idx}")
        fig.tight_layout()
        fig.savefig(png_file, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"Plotting skipped (matplotlib missing or error): {e}")


def plot_tracking_performance(ep_log, ep_idx, png_file_base):
    """Plot tracking performance comparison."""
    try:
        import matplotlib.pyplot as plt
        
        # Check if we have the required data for tracking plots
        if "theta" not in ep_log or "theta_target" not in ep_log or "t" not in ep_log:
            return
            
        t_arr = np.array(ep_log["t"], dtype=float)
        theta_actual = np.array(ep_log["theta"], dtype=float)
        theta_target = np.array(ep_log["theta_target"], dtype=float)
        
        # Calculate tracking error
        tracking_error = np.abs(theta_actual - theta_target)
        
        # Create tracking performance plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot actual vs target
        ax1.plot(t_arr, theta_actual, label='Actual Theta')
        ax1.plot(t_arr, theta_target, label='Target Theta', linestyle='--')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Theta (rad)')
        ax1.set_title('Tracking Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot tracking error
        ax2.plot(t_arr, tracking_error, 'r', label='Tracking Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (rad)')
        ax2.set_title('Tracking Error Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        avg_error = np.mean(tracking_error)
        max_error = np.max(tracking_error)
        ax2.text(0.02, 0.98, f'Avg Error: {avg_error:.4f}\nMax Error: {max_error:.4f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle(f'Tracking Performance - Episode {ep_idx}')
        fig.tight_layout()
        
        # Save tracking plot with different name
        tracking_plot_file = png_file_base.with_name(f"{png_file_base.stem}_tracking{png_file_base.suffix}")
        fig.savefig(tracking_plot_file, dpi=120)
        plt.close(fig)
        
        print(f"Saved tracking plot: {tracking_plot_file}")
    except Exception as e:
        print(f"Tracking plot skipped (error): {e}")