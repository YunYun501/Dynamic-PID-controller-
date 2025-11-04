#!/usr/bin/env python3
"""
Simple test to verify RL wrapper functionality.
"""

import numpy as np
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Test basic import
try:
    from utils.rl_wrapper import SoftRoboticRLWrapper
    print("RL wrapper imported successfully!")
except Exception as e:
    print(f"Error importing RL wrapper: {e}")

# Test creating environment
try:
    from environment.soft_robotic_env import SoftRobotic
    env = SoftRobotic(control_mode="position")
    print("Base environment created successfully!")
    
    # Test RL wrapper
    rl_env = SoftRoboticRLWrapper(env)
    print("RL wrapper instantiated successfully!")
    
    # Test reset
    obs, info = rl_env.reset()
    print(f"Environment reset successful! Observation shape: {obs.shape}")
    
    # Test step
    action = np.array([0.1], dtype=np.float32)
    obs, reward, terminated, truncated, info = rl_env.step(action)
    print(f"Step executed successfully! Reward: {reward:.4f}")
    
    rl_env.close()
    print("Environment closed successfully!")
    
except Exception as e:
    print(f"Error testing environments: {e}")
    import traceback
    traceback.print_exc()

print("\nSimple RL test completed!")