import csv
import json
from pathlib import Path
from datetime import datetime


def start_episode_log(obs_headers, extra_headers):
    """Initialize a new episode log."""
    d = {"t": [], "step": []}
    for name in obs_headers + extra_headers:
        d[name] = []
    return d


def append_log(log, t_val, step_idx, obs_vec, reward=None, reward_cum=None, force_left=None, force_right=None):
    """Append data to the episode log."""
    log["t"].append(float(t_val))
    log["step"].append(int(step_idx))
    
    # Find observation headers from the log keys (excluding t and step)
    obs_headers = [k for k in log.keys() if k not in ["t", "step", "reward", "reward_cum", "force_left", "force_right"]]
    
    for i, name in enumerate(obs_headers):
        if i < len(obs_vec):
            log[name].append(float(obs_vec[i]))
        else:
            log[name].append(0.0)
            
    if reward is not None:
        log["reward"].append(float(reward))
    else:
        log["reward"].append(0.0 if step_idx == 0 else float(log["reward"][-1]))
        
    if reward_cum is not None:
        log["reward_cum"].append(float(reward_cum))
    else:
        prev = 0.0 if step_idx == 0 else float(log["reward_cum"][-1])
        log["reward_cum"].append(prev)
        
    if force_left is not None:
        log["force_left"].append(float(force_left))
    else:
        log["force_left"].append(0.0 if step_idx == 0 else float(log["force_left"][-1]))
        
    if force_right is not None:
        log["force_right"].append(float(force_right))
    else:
        log["force_right"].append(0.0 if step_idx == 0 else float(log["force_right"][-1]))


def persist_episode(ep_idx, log, control_mode, sinusoidal_magnitude, sinusoidal_frequency,
                   position_action, velocity_action, acceleration_action, decision_log, 
                   out_path, env):
    """Save episode data to CSV, JSON, and plot files."""
    from simulation.visualization import plot_observations
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_path / f"episode_{ep_idx:03d}_{ts}"
    csv_file = base.with_suffix(".csv")
    png_file = base.with_suffix(".png")
    analysis_file = base.with_suffix(".analysis.json")
    decisions_file = base.with_suffix(".decisions.json")

    # Write CSV with headers
    obs_headers = [k for k in log.keys() if k not in ["t", "step", "reward", "reward_cum", "force_left", "force_right"]]
    extra_headers = ["reward", "reward_cum", "force_left", "force_right"]
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "step"] + obs_headers + extra_headers)
        for i in range(len(log["t"])):
            row = [log["t"][i], log["step"][i]] + [log[name][i] for name in obs_headers + extra_headers]
            writer.writerow(row)

    # Write analysis file for agent evaluation
    analysis_data = {
        "episode": ep_idx,
        "control_mode": control_mode,
        "total_steps": len(log["t"]),
        "timestamp": ts,
        "parameters": {
            "sinusoidal_magnitude": sinusoidal_magnitude,
            "sinusoidal_frequency": sinusoidal_frequency,
            "position_action": position_action,
            "velocity_action": velocity_action,
            "acceleration_action": acceleration_action,
        },
        "final_state": {
            "theta": log["theta"][-1] if log["theta"] else 0.0,
            "theta_dot": log["theta_dot"][-1] if log["theta_dot"] else 0.0,
            "theta_target": log["theta_target"][-1] if log["theta_target"] else 0.0,
            "error": log["etheta"][-1] if log["etheta"] else 0.0,
            "cumulative_reward": log["reward_cum"][-1] if log["reward_cum"] else 0.0,
        },
        "performance_metrics": {
            "max_error": max([abs(e) for e in log["etheta"]]) if log["etheta"] else 0.0,
            "avg_error": sum([abs(e) for e in log["etheta"]]) / len(log["etheta"]) if log["etheta"] else 0.0,
            "max_torque": max([abs(t) for t in log["tau"]]) if log["tau"] else 0.0,
            "avg_torque": sum([abs(t) for t in log["tau"]]) / len(log["tau"]) if log["tau"] else 0.0,
        }
    }
    
    # Add force metrics for force control mode
    if control_mode == "force":
        analysis_data["force_metrics"] = {
            "max_force_left": max([abs(f) for f in log["force_left"]]) if log["force_left"] else 0.0,
            "max_force_right": max([abs(f) for f in log["force_right"]]) if log["force_right"] else 0.0,
            "avg_force_left": sum([abs(f) for f in log["force_left"]]) / len(log["force_left"]) if log["force_left"] else 0.0,
            "avg_force_right": sum([abs(f) for f in log["force_right"]]) / len(log["force_right"]) if log["force_right"] else 0.0,
        }

    with open(analysis_file, "w") as f:
        json.dump(analysis_data, f, indent=2)

    # Write detailed decisions log for agent evaluation
    with open(decisions_file, "w") as f:
        json.dump(decision_log, f, indent=2)
        
    # Plot observations
    plot_observations(log, env, ep_idx, png_file)

    print(f"Saved CSV: {csv_file}")
    print(f"Saved analysis: {analysis_file}")
    print(f"Saved decisions log: {decisions_file}")
    print(f"Saved plot: {png_file}")