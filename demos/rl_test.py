#!/usr/bin/env python3
"""
Test script demonstrating the RL-friendly enhancements to the SoftRobotic environment.
"""

import numpy as np
import time
# Import locally to avoid circular imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.rl_wrapper import create_rl_environment


def test_rl_environments():
    """Test different RL environment configurations."""
    print("Testing RL-friendly SoftRobotic environments...")
    print("=" * 50)
    
    # Test different reward types
    reward_types = ["shaped", "sparse", "multi_objective"]
    control_modes = ["position", "velocity", "acceleration"]
    
    for reward_type in reward_types:
        print(f"\nTesting {reward_type.upper()} reward type:")
        print("-" * 30)
        
        for control_mode in control_modes:
            print(f"  {control_mode.capitalize()} control:")
            
            # Create RL environment
            env = create_rl_environment(
                control_mode=control_mode,
                reward_type=reward_type,
                action_type="delta",  # Use delta actions for better exploration
                sinusoidal_magnitude=0.3,
                sinusoidal_frequency=0.2
            )
            
            # Run a short episode
            obs, info = env.reset()
            total_reward = 0.0
            steps = 50
            
            for i in range(steps):
                # Simple policy: small random actions for exploration
                if control_mode == "position":
                    action = np.array([np.random.uniform(-0.1, 0.1)], dtype=np.float32)
                elif control_mode == "velocity":
                    action = np.array([np.random.uniform(-0.5, 0.5)], dtype=np.float32)
                else:  # acceleration
                    action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            print(f"    Total reward: {total_reward:.4f}")
            print(f"    Final error: {abs(obs[11]):.4f} rad")
            env.close()


def demonstrate_action_types():
    """Demonstrate different action transformation types."""
    print("\n\nDemonstrating action transformation types:")
    print("=" * 50)
    
    action_types = ["absolute", "delta", "normalized"]
    
    for action_type in action_types:
        print(f"\n{action_type.capitalize()} actions:")
        print("-" * 20)
        
        env = create_rl_environment(
            control_mode="position",
            reward_type="shaped",
            action_type=action_type,
            sinusoidal_magnitude=0.2,
            sinusoidal_frequency=0.1
        )
        
        obs, info = env.reset()
        total_reward = 0.0
        steps = 30
        
        print(f"  Initial target: {obs[8]:.4f}")
        
        for i in range(steps):
            # Different action strategies for different action types
            if action_type == "normalized":
                # Actions in [-1, 1] range
                action = np.array([np.sin(i * 0.2)], dtype=np.float32)
            elif action_type == "delta":
                # Small changes to target
                action = np.array([0.02 * np.sin(i * 0.3)], dtype=np.float32)
            else:  # absolute
                # Direct target angles
                action = np.array([0.2 * np.sin(i * 0.1)], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if i % 10 == 0:  # Print every 10 steps
                print(f"    Step {i}: Target={obs[8]:.4f}, Actual={obs[2]:.4f}, Error={abs(obs[11]):.4f}")
            
            if terminated or truncated:
                break
        
        print(f"  Final reward: {total_reward:.4f}")
        print(f"  Final error: {abs(obs[11]):.4f} rad")
        env.close()


def compare_with_baseline():
    """Compare RL-enhanced environment with baseline."""
    print("\n\nComparing RL-enhanced vs. baseline environment:")
    print("=" * 50)
    
    # Import locally to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from environment.soft_robotic_env import SoftRobotic
    
    # Baseline environment (original)
    print("Baseline environment (original reward):")
    baseline_env = SoftRobotic(
        control_mode="position",
        sinusoidal_magnitude=0.25,
        sinusoidal_frequency=0.15
    )
    
    # RL-enhanced environment
    print("RL-enhanced environment (shaped reward):")
    rl_env = create_rl_environment(
        control_mode="position",
        reward_type="shaped",
        action_type="delta",
        sinusoidal_magnitude=0.25,
        sinusoidal_frequency=0.15
    )
    
    environments = [
        ("Baseline", baseline_env),
        ("RL-Enhanced", rl_env)
    ]
    
    for name, env in environments:
        obs, info = env.reset()
        total_reward = 0.0
        steps = 40
        
        for i in range(steps):
            # Simple sinusoidal action
            action_value = 0.1 * np.sin(i * 0.15)
            action = np.array([action_value], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"  {name}: Total reward = {total_reward:.4f}, Final error = {abs(obs[11]):.4f} rad")
        env.close()


if __name__ == "__main__":
    print("SoftRobotic RL Enhancement Demo")
    print("This script demonstrates the RL-friendly improvements to the environment.")
    
    try:
        test_rl_environments()
        demonstrate_action_types()
        compare_with_baseline()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey improvements demonstrated:")
        print("1. Multiple reward types (shaped, sparse, multi-objective)")
        print("2. Action transformations (absolute, delta, normalized)")
        print("3. Exploration bonuses and smoothness rewards")
        print("4. Better reward shaping for learning")
        print("\nThese enhancements make the environment more suitable for")
        print("reinforcement learning while preserving the PID control focus.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()