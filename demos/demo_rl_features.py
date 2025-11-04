#!/usr/bin/env python3
"""
Demonstration of RL-friendly features in the SoftRobotic environment.
"""

import numpy as np
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.rl_wrapper import SoftRoboticRLWrapper
from environment.soft_robotic_env import SoftRobotic


def demonstrate_reward_types():
    """Demonstrate different reward types available in the RL wrapper."""
    print("=== RL Reward Types Demonstration ===\n")
    
    # Create environments with different reward types
    reward_types = ["shaped", "sparse", "multi_objective"]
    
    for reward_type in reward_types:
        print(f"{reward_type.upper()} REWARD TYPE:")
        print("-" * 30)
        
        # Create environment
        base_env = SoftRobotic(control_mode="position", sinusoidal_magnitude=0.2)
        rl_env = SoftRoboticRLWrapper(base_env, reward_type=reward_type)
        
        # Run a short episode
        obs, info = rl_env.reset()
        total_reward = 0.0
        step_rewards = []
        
        print(f"Initial state - Theta: {obs[2]:.3f}, Target: {obs[8]:.3f}, Error: {abs(obs[11]):.3f}")
        
        # Execute a simple sinusoidal policy
        for i in range(20):
            # Action that tries to follow a sinusoidal target
            target_change = 0.05 * np.sin(i * 0.3)
            action = np.array([target_change], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = rl_env.step(action)
            total_reward += reward
            step_rewards.append(reward)
            
            if i in [5, 10, 15, 19]:  # Print at intervals
                print(f"  Step {i:2d} - Action: {action[0]:.3f}, Reward: {reward:.4f}, "
                      f"Error: {abs(obs[11]):.3f}")
            
            if terminated or truncated:
                break
        
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Average reward per step: {np.mean(step_rewards):.4f}")
        print(f"  Final error: {abs(obs[11]):.4f}\n")
        
        rl_env.close()


def demonstrate_action_transformations():
    """Demonstrate different action transformation types."""
    print("=== Action Transformation Types ===\n")
    
    action_types = ["absolute", "delta", "normalized"]
    
    for action_type in action_types:
        print(f"{action_type.upper()} ACTIONS:")
        print("-" * 25)
        
        # Create environment
        base_env = SoftRobotic(control_mode="position", sinusoidal_magnitude=0.15)
        rl_env = SoftRoboticRLWrapper(base_env, action_type=action_type)
        
        obs, info = rl_env.reset()
        print(f"Initial target: {obs[8]:.3f}")
        
        # Execute actions
        for i in range(10):
            # Different action strategies for different types
            if action_type == "normalized":
                # Actions in [-1, 1] range
                action = np.array([np.sin(i * 0.5)], dtype=np.float32)
            elif action_type == "delta":
                # Small changes to target
                action = np.array([0.03 * np.sin(i * 0.4)], dtype=np.float32)
            else:  # absolute
                # Direct target angles
                action = np.array([0.15 * np.sin(i * 0.2)], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = rl_env.step(action)
            
            if i in [3, 6, 9]:  # Print at intervals
                print(f"  Step {i}: Action={action[0]:.3f}, Target={obs[8]:.3f}, "
                      f"Actual={obs[2]:.3f}")
            
            if terminated or truncated:
                break
        
        print(f"  Final target: {obs[8]:.3f}\n")
        rl_env.close()


def compare_rl_vs_traditional():
    """Compare RL-enhanced environment with traditional environment."""
    print("=== RL vs Traditional Environment ===\n")
    
    # Traditional environment
    print("TRADITIONAL ENVIRONMENT:")
    print("-" * 25)
    traditional_env = SoftRobotic(control_mode="position", sinusoidal_magnitude=0.2)
    obs, info = traditional_env.reset()
    traditional_reward = 0.0
    
    print(f"Initial - Theta: {obs[2]:.3f}, Error: {abs(obs[11]):.3f}")
    
    for i in range(15):
        action = np.array([0.1 * np.sin(i * 0.3)], dtype=np.float32)
        obs, reward, terminated, truncated, info = traditional_env.step(action)
        traditional_reward += reward
        
        if i == 14:
            print(f"Final - Theta: {obs[2]:.3f}, Error: {abs(obs[11]):.3f}, "
                  f"Reward: {traditional_reward:.4f}")
    
    traditional_env.close()
    
    # RL-enhanced environment
    print("\nRL-ENHANCED ENVIRONMENT:")
    print("-" * 25)
    base_env = SoftRobotic(control_mode="position", sinusoidal_magnitude=0.2)
    rl_env = SoftRoboticRLWrapper(base_env, reward_type="shaped")
    obs, info = rl_env.reset()
    rl_reward = 0.0
    
    print(f"Initial - Theta: {obs[2]:.3f}, Error: {abs(obs[11]):.3f}")
    
    for i in range(15):
        action = np.array([0.05 * np.sin(i * 0.3)], dtype=np.float32)
        obs, reward, terminated, truncated, info = rl_env.step(action)
        rl_reward += reward
        
        if i == 14:
            print(f"Final - Theta: {obs[2]:.3f}, Error: {abs(obs[11]):.3f}, "
                  f"Reward: {rl_reward:.4f}")
    
    rl_env.close()
    
    print(f"\nComparison:")
    print(f"  Traditional reward: {traditional_reward:.4f}")
    print(f"  RL-enhanced reward: {rl_reward:.4f}")
    print(f"  Difference: {rl_reward - traditional_reward:.4f}")


if __name__ == "__main__":
    print("SoftRobotic RL Features Demonstration")
    print("=" * 40)
    print("This script demonstrates the RL-friendly improvements to the environment.\n")
    
    try:
        demonstrate_reward_types()
        demonstrate_action_transformations()
        compare_rl_vs_traditional()
        
        print("\n" + "=" * 50)
        print("DEMONSTRATION COMPLETE")
        print("=" * 50)
        print("Key RL improvements implemented:")
        print("1. Multiple reward types for different learning scenarios")
        print("2. Action transformations for better exploration")
        print("3. Enhanced reward shaping for faster learning")
        print("4. Exploration bonuses and smoothness rewards")
        print("\nThese features make the environment more suitable for")
        print("reinforcement learning while maintaining PID control focus.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()