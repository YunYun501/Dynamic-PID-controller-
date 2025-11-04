#!/usr/bin/env python3
"""
Simulation demo using RL-enhanced environment.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.rl_wrapper import create_rl_environment
import matplotlib.pyplot as plt


def run_rl_simulation():
    """Run a simulation with RL-enhanced environment and visualize results."""
    print("Running RL-Enhanced Simulation Demo")
    print("=" * 40)
    
    # Create RL-enhanced environment
    env = create_rl_environment(
        control_mode="position",
        reward_type="shaped",  # Use shaped rewards for better learning signals
        action_type="delta",   # Use delta actions for better exploration
        sinusoidal_magnitude=0.25,
        sinusoidal_frequency=0.1
    )
    
    print("Environment created with:")
    print("  - Position control mode")
    print("  - Shaped rewards")
    print("  - Delta actions")
    print("  - Sinusoidal trajectory (mag=0.25, freq=0.1 Hz)")
    
    # Initialize tracking arrays
    theta_values = []
    theta_target_values = []
    reward_values = []
    time_values = []
    error_values = []
    
    # Reset environment
    obs, info = env.reset()
    total_reward = 0.0
    
    print(f"\nInitial state:")
    print(f"  Theta: {obs[2]:.3f}")
    print(f"  Target: {obs[8]:.3f}")
    print(f"  Error: {abs(obs[11]):.3f}")
    
    # Run simulation
    steps = 100
    print(f"\nRunning simulation for {steps} steps...")
    
    for i in range(steps):
        # Time-based action (following sinusoidal reference)
        t = i * env.env.time_step
        action_value = 0.03 * np.sin(2.0 * np.pi * 0.1 * t)
        action = np.array([action_value], dtype=np.float32)
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Store data for plotting
        theta_values.append(obs[2])
        theta_target_values.append(obs[8])
        reward_values.append(reward)
        time_values.append(t)
        error_values.append(abs(obs[11]))
        
        # Print progress
        if i % 20 == 0:
            print(f"  Step {i:3d}: Theta={obs[2]:.3f}, Target={obs[8]:.3f}, "
                  f"Error={abs(obs[11]):.3f}, Reward={reward:.4f}")
        
        if terminated or truncated:
            print("Episode terminated early!")
            break
    
    # Final state
    print(f"\nFinal state:")
    print(f"  Theta: {obs[2]:.3f}")
    print(f"  Target: {obs[8]:.3f}")
    print(f"  Error: {abs(obs[11]):.3f}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Average reward: {total_reward/steps:.4f}")
    
    # Close environment
    env.close()
    
    # Create visualization
    print("\nGenerating performance plots...")
    create_performance_plots(time_values, theta_values, theta_target_values, 
                           reward_values, error_values)
    
    return total_reward


def create_performance_plots(time_values, theta_values, theta_target_values, 
                           reward_values, error_values):
    """Create performance visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Tracking performance
    axes[0, 0].plot(time_values, theta_values, label='Actual Theta', linewidth=2)
    axes[0, 0].plot(time_values, theta_target_values, label='Target Theta', 
                    linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Theta (rad)')
    axes[0, 0].set_title('Tracking Performance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Tracking error
    axes[0, 1].plot(time_values, error_values, 'r', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Error (rad)')
    axes[0, 1].set_title('Tracking Error Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Rewards
    axes[1, 0].plot(time_values, reward_values, 'g', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Reward Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error histogram
    axes[1, 1].hist(error_values, bins=20, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Error (rad)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics to the error plot
    avg_error = np.mean(error_values)
    max_error = np.max(error_values)
    axes[0, 1].axhline(y=avg_error, color='b', linestyle=':', 
                       label=f'Avg Error: {avg_error:.4f}')
    axes[0, 1].legend()
    
    plt.tight_layout()
    plt.savefig('rl_simulation_demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Performance metrics:")
    print(f"  Average tracking error: {avg_error:.4f}")
    print(f"  Maximum tracking error: {max_error:.4f}")
    print(f"  Total reward: {np.sum(reward_values):.4f}")
    print(f"  Plot saved as 'rl_simulation_demo_results.png'")


def compare_control_modes():
    """Compare different control modes with RL enhancements."""
    print("\n\nComparing Control Modes with RL Enhancements")
    print("=" * 50)
    
    control_modes = ["position", "velocity", "acceleration"]
    results = {}
    
    for mode in control_modes:
        print(f"\nTesting {mode.upper()} control:")
        print("-" * 25)
        
        # Create environment
        env = create_rl_environment(
            control_mode=mode,
            reward_type="shaped",
            action_type="delta",
            sinusoidal_magnitude=0.2,
            sinusoidal_frequency=0.08
        )
        
        # Run short simulation
        obs, info = env.reset()
        total_reward = 0.0
        final_error = 0.0
        
        for i in range(50):
            # Simple sinusoidal action
            t = i * env.env.time_step
            if mode == "position":
                action_val = 0.02 * np.sin(2.0 * np.pi * 0.08 * t)
                action = np.array([action_val], dtype=np.float32)
            elif mode == "velocity":
                action_val = 0.3 * np.sin(2.0 * np.pi * 0.08 * t)
                action = np.array([action_val], dtype=np.float32)
            else:  # acceleration
                action_val = 2.0 * np.sin(2.0 * np.pi * 0.08 * t)
                action = np.array([action_val], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated or i == 49:
                final_error = abs(obs[11])
                break
        
        results[mode] = {
            'reward': total_reward,
            'error': final_error
        }
        
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Final error: {final_error:.4f}")
        
        env.close()
    
    # Print comparison
    print(f"\nControl Mode Comparison:")
    print("-" * 30)
    for mode, result in results.items():
        print(f"  {mode.capitalize():>12}: Reward={result['reward']:>8.4f}, "
              f"Error={result['error']:>8.4f}")


if __name__ == "__main__":
    print("SoftRobotic RL Simulation Demo")
    print("This demo shows the RL-friendly improvements to the environment.")
    
    try:
        # Run main simulation
        total_reward = run_rl_simulation()
        
        # Compare control modes
        compare_control_modes()
        
        print(f"\n" + "=" * 50)
        print("SIMULATION DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("The RL-enhanced environment provides:")
        print("1. Better reward shaping for learning")
        print("2. Flexible action transformations")
        print("3. Enhanced exploration capabilities")
        print("4. Improved visualization and analysis tools")
        print("\nCheck the generated plot for detailed performance metrics.")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()