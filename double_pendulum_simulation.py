import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint


def double_pendulum_derivatives(state, t, L1, L2, m1, m2, g):
    """
    Compute the derivatives of the double pendulum state.
    
    Parameters:
    state : array
        The current state [theta1, omega1, theta2, omega2]
    t : float
        Time (not used explicitly in this autonomous system)
    L1, L2 : float
        Lengths of the pendulum rods
    m1, m2 : float
        Masses of the pendulum bobs
    g : float
        Acceleration due to gravity
        
    Returns:
    derivatives : array
        Time derivatives of the state [dtheta1/dt, domega1/dt, dtheta2/dt, domega2/dt]
    """
    theta1, omega1, theta2, omega2 = state
    
    # Trigonometric values
    cos_diff = np.cos(theta1 - theta2)
    sin_diff = np.sin(theta1 - theta2)
    sin1 = np.sin(theta1)
    sin2 = np.sin(theta2)
    cos1 = np.cos(theta1)
    
    # Denominator for the angular accelerations
    denom = L1 * (2 * m1 + m2 - m2 * np.cos(2 * (theta1 - theta2)))
    
    # Angular accelerations
    domega1_dt = (-g * (2 * m1 + m2) * sin1 - 
                  m2 * g * sin1 * cos_diff - 
                  m2 * sin_diff * (omega2**2 * L2 + omega1**2 * L1 * cos_diff)) / denom
    
    domega2_dt = (2 * sin_diff * (omega1**2 * L1 * (m1 + m2) + 
                                  g * (m1 + m2) * cos1 + 
                                  omega2**2 * L2 * m2 * cos_diff)) / denom
    
    # Return derivatives
    return [omega1, domega1_dt, omega2, domega2_dt]


def simulate_double_pendulum(initial_state, t, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
    """
    Simulate the double pendulum motion.
    
    Parameters:
    initial_state : array
        Initial state [theta1, omega1, theta2, omega2]
    t : array
        Time points for the simulation
    L1, L2 : float
        Lengths of the pendulum rods
    m1, m2 : float
        Masses of the pendulum bobs
    g : float
        Acceleration due to gravity
        
    Returns:
    solution : array
        Solution array with columns [theta1, omega1, theta2, omega2]
    """
    solution = odeint(double_pendulum_derivatives, initial_state, t, args=(L1, L2, m1, m2, g))
    return solution


def animate_double_pendulum(solution, t, L1=1.0, L2=1.0):
    """
    Animate the double pendulum motion.
    
    Parameters:
    solution : array
        Solution array with columns [theta1, omega1, theta2, omega2]
    t : array
        Time points for the simulation
    L1, L2 : float
        Lengths of the pendulum rods
    """
    # Extract angles
    theta1 = solution[:, 0]
    theta2 = solution[:, 2]
    
    # Convert to Cartesian coordinates
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = L1 * np.sin(theta1) + L2 * np.sin(theta2)
    y2 = -L1 * np.cos(theta1) - L2 * np.cos(theta2)
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-L1 - L2 - 0.5, L1 + L2 + 0.5)
    ax.set_ylim(-L1 - L2 - 0.5, 0.5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Double Pendulum Animation')
    
    # Create lines for the pendulum rods
    line1, = ax.plot([], [], 'o-', lw=2, markersize=8, color='blue')
    line2, = ax.plot([], [], 'o-', lw=2, markersize=8, color='red')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Trail for the second pendulum
    trail, = ax.plot([], [], 'r-', lw=0.5, alpha=0.5)
    trail_x, trail_y = [], []
    
    def init():
        """Initialize the animation."""
        line1.set_data([], [])
        line2.set_data([], [])
        time_text.set_text('')
        trail.set_data([], [])
        return line1, line2, time_text, trail
    
    def animate(i):
        """Update the animation for frame i."""
        # Update pendulum positions
        line1_x = [0, x1[i]]
        line1_y = [0, y1[i]]
        line1.set_data(line1_x, line1_y)
        
        line2_x = [x1[i], x2[i]]
        line2_y = [y1[i], y2[i]]
        line2.set_data(line2_x, line2_y)
        
        # Update trail
        trail_x.append(x2[i])
        trail_y.append(y2[i])
        # Keep only the last 200 points for the trail
        if len(trail_x) > 200:
            trail_x.pop(0)
            trail_y.pop(0)
        trail.set_data(trail_x, trail_y)
        
        # Update time text
        time_text.set_text(f'Time: {t[i]:.2f} s')
        
        return line1, line2, time_text, trail
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, frames=len(t), 
        init_func=init, interval=20, blit=True
    )
    
    plt.tight_layout()
    plt.show()
    
    return ani


def main():
    """
    Main function to run the double pendulum simulation and animation.
    """
    # Parameters
    L1 = 1.0  # Length of first pendulum (m)
    L2 = 1.0  # Length of second pendulum (m)
    m1 = 1.0  # Mass of first bob (kg)
    m2 = 1.0  # Mass of second bob (kg)
    g = 9.81  # Acceleration due to gravity (m/s^2)
    
    # Initial conditions: [theta1, omega1, theta2, omega2]
    # theta is measured from the vertical (0 = hanging down)
    initial_state = [np.pi/2, 0, np.pi/2, 0]  # Both pendulums at 90 degrees with zero initial velocity
    
    # Time points
    t = np.linspace(0, 10, 500)  # 10 seconds, 500 points
    
    # Simulate the double pendulum
    solution = simulate_double_pendulum(initial_state, t, L1, L2, m1, m2, g)
    
    # Animate the double pendulum
    animate_double_pendulum(solution, t, L1, L2)


if __name__ == "__main__":
    main()