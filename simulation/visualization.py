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