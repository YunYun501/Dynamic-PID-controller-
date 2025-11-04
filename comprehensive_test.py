import numpy as np
import matplotlib.pyplot as plt
from soft_robotic_env import SoftRobotic

def test_position_control():
    """Test position control tracking performance."""
    print("Testing position control...")
    
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
    errors = []

    for i in range(500):
        # Get target from environment
        theta_ref, x_ref, y_ref = env.kinematics.calculate_reference_trajectory(
            i, env.time_step, env.sinusoidal_magnitude, env.sinusoidal_frequency)
        
        # Apply position control action
        action = np.array([theta_ref], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store values for analysis
        theta_values.append(obs[2])  # Current theta
        theta_target_values.append(obs[8])  # Target theta
        time_values.append(i * env.time_step)
        errors.append(abs(obs[2] - obs[8]))  # Tracking error
        
        if terminated or truncated:
            break

    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"Position Control - Average error: {avg_error:.4f}, Max error: {max_error:.4f}")
    
    # Plot results
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
    plt.plot(time_values, errors, 'r', label='Tracking Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('Position Control Tracking Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('position_control_test.png')
    plt.show()
    
    return avg_error, max_error

def test_velocity_control():
    """Test velocity control tracking performance."""
    print("Testing velocity control...")
    
    # Create environment
    env = SoftRobotic(
        render_mode=None,  # No rendering for testing
        control_mode="velocity",
        sinusoidal_magnitude=0.5,
        sinusoidal_frequency=0.5
    )

    # Run simulation
    obs, info = env.reset()
    theta_values = []
    theta_target_values = []
    theta_dot_values = []
    theta_dot_target_values = []
    time_values = []
    errors = []

    for i in range(500):
        # Get target from environment
        theta_ref, x_ref, y_ref = env.kinematics.calculate_reference_trajectory(
            i, env.time_step, env.sinusoidal_magnitude, env.sinusoidal_frequency)
        
        # Calculate target velocity (numerical differentiation)
        if i > 0:
            prev_theta_ref, _, _ = env.kinematics.calculate_reference_trajectory(
                i-1, env.time_step, env.sinusoidal_magnitude, env.sinusoidal_frequency)
            target_velocity = (theta_ref - prev_theta_ref) / env.time_step
        else:
            target_velocity = 0.0
            
        # Apply velocity control action
        action = np.array([target_velocity], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store values for analysis
        theta_values.append(obs[2])  # Current theta
        theta_target_values.append(obs[8])  # Target theta
        theta_dot_values.append(obs[3])  # Current angular velocity
        theta_dot_target_values.append(target_velocity)  # Target angular velocity
        time_values.append(i * env.time_step)
        errors.append(abs(obs[2] - obs[8]))  # Tracking error
        
        if terminated or truncated:
            break

    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"Velocity Control - Average error: {avg_error:.4f}, Max error: {max_error:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(time_values, theta_values, label='Actual Theta')
    plt.plot(time_values, theta_target_values, label='Target Theta', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (rad)')
    plt.title('Velocity Control - Angle Tracking Performance')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_values, theta_dot_values, label='Actual Theta Dot')
    plt.plot(time_values, theta_dot_target_values, label='Target Theta Dot', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta Dot (rad/s)')
    plt.title('Velocity Control - Angular Velocity Tracking Performance')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_values, errors, 'r', label='Angle Tracking Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('Velocity Control - Angle Tracking Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('velocity_control_test.png')
    plt.show()
    
    return avg_error, max_error

if __name__ == "__main__":
    print("Running comprehensive control tests...")
    
    pos_avg, pos_max = test_position_control()
    print()
    vel_avg, vel_max = test_velocity_control()
    
    print("\nSUMMARY:")
    print(f"Position Control - Avg Error: {pos_avg:.4f}, Max Error: {pos_max:.4f}")
    print(f"Velocity Control - Avg Error: {vel_avg:.4f}, Max Error: {vel_max:.4f}")