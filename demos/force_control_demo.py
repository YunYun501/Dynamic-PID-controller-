#!/usr/bin/env python3
"""
Force Control Demo - Demonstrates significant actuator movement using large forces.
"""

import numpy as np
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from environment.soft_robotic_env import SoftRobotic

def force_control_demo():
    """Demonstrate force control with large forces for significant movement."""
    print("Force Control Demo - Large Forces for Significant Movement")
    print("=" * 55)
    
    # Create force control environment
    env = SoftRobotic(control_mode="force", render_mode=None)
    obs, info = env.reset()
    
    print(f"Initial state - Theta: {obs[2]:.4f} radians ({np.degrees(obs[2]):.2f} degrees)")
    print(f"System parameters:")
    print(f"  Arm length: {env.arm_length}m")
    print(f"  Arm mass: {env.arm_mass}kg")
    print(f"  Gravity: {env.gravitational_acceleration}m/s²")
    print()
    
    # Apply large, unbalanced forces to create significant movement
    print("Applying large unbalanced forces:")
    print("  Left force: 800N (strong positive torque)")
    print("  Right force: 100N (weak negative torque)")
    print()
    
    initial_theta = obs[2]
    max_theta = initial_theta
    min_theta = initial_theta
    
    # Apply consistent unbalanced forces for 200 steps
    for i in range(200):
        # Large unbalanced forces - this should create significant rotation
        left_force = 800.0   # Strong force on left actuator
        right_force = 100.0  # Weak force on right actuator
        action = np.array([left_force, right_force], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track movement range
        max_theta = max(max_theta, obs[2])
        min_theta = min(min_theta, obs[2])
        
        # Print progress every 40 steps
        if i % 40 == 0 or i == 199:
            print(f"Step {i:3d} - Theta: {obs[2]:.4f} rad ({np.degrees(obs[2]):.2f} deg) - "
                  f"Forces: [{left_force:.0f}, {right_force:.0f}]N - "
                  f"Torque: {info.get('tau', 0):.2f} N*m")
    
    final_theta = obs[2]
    total_movement = abs(final_theta - initial_theta)
    range_movement = abs(max_theta - min_theta)
    
    print()
    print("Results:")
    print(f"  Initial theta: {np.degrees(initial_theta):.2f}deg")
    print(f"  Final theta:   {np.degrees(final_theta):.2f}deg")
    print(f"  Total movement: {np.degrees(total_movement):.2f}deg")
    print(f"  Range of movement: {np.degrees(range_movement):.2f}deg")
    
    # Test with opposite forces to show bidirectional control
    print()
    print("Now applying opposite forces:")
    print("  Left force: 100N")
    print("  Right force: 800N")
    
    for i in range(100):
        # Opposite unbalanced forces
        left_force = 100.0   # Weak force on left actuator
        right_force = 800.0  # Strong force on right actuator
        action = np.array([left_force, right_force], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 30 == 0 or i == 99:
            print(f"Step {200+i:3d} - Theta: {obs[2]:.4f} rad ({np.degrees(obs[2]):.2f}deg) - "
                  f"Forces: [{left_force:.0f}, {right_force:.0f}]N")
    
    env.close()
    print()
    print("Force control demo completed successfully!")

def sinusoidal_force_demo():
    """Demonstrate sinusoidal force control with larger magnitudes."""
    print("\n" + "=" * 55)
    print("Sinusoidal Force Control Demo - Large Magnitudes")
    print("=" * 55)
    
    # Create force control environment
    env = SoftRobotic(control_mode="force", render_mode=None)
    obs, info = env.reset()
    
    print(f"Initial theta: {obs[2]:.4f} radians ({np.degrees(obs[2]):.2f} degrees)")
    print("Applying sinusoidal forces with 600N magnitude")
    
    # Apply sinusoidal forces with larger magnitude
    for i in range(200):
        t = i * env.time_step
        # Use larger magnitude (600N instead of 0.5N)
        magnitude = 600.0
        frequency = 0.5  # 0.5 Hz
        
        left_force = magnitude * np.sin(2.0 * np.pi * frequency * t)
        right_force = magnitude * np.sin(2.0 * np.pi * frequency * t + np.pi/3)
        action = np.array([left_force, right_force], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 50 == 0:
            print(f"Step {i:3d} - Theta: {obs[2]:.4f} rad ({np.degrees(obs[2]):.2f}deg) - "
                  f"Forces: [{left_force:6.1f}, {right_force:6.1f}]N")
    
    print(f"Final theta: {obs[2]:.4f} radians ({np.degrees(obs[2]):.2f} degrees)")
    env.close()

if __name__ == "__main__":
    force_control_demo()
    sinusoidal_force_demo()