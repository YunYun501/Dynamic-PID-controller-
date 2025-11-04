# Dynamic PID Controller for Soft Robotic Arm

This is a code base to optimize real-time PID control for a soft robotic arm simulation environment. The project provides a Gymnasium-based environment for controlling a single-link robotic arm with various control modes.

## Project Structure

```
Dynamic-PID-controller-/
├── environment/              # Core environment modules
│   ├── __init__.py          # Environment package initialization
│   ├── base.py             # Base environment class with physics parameters
│   ├── kinematics.py       # Kinematic calculations (position, trajectory)
│   ├── dynamics.py         # Dynamic calculations (torque, motion integration)
│   ├── observations.py     # Observation space generation
│   ├── actions.py          # Action processing and PID control
│   └── rendering.py        # Visualization using pygame
├── simulation/              # Simulation execution modules
│   ├── __init__.py         # Simulation package initialization
│   ├── runner.py           # Main simulation execution logic
│   ├── arguments.py        # Command-line argument parsing
│   ├── logging.py          # Data logging and persistence
│   └── visualization.py    # Plotting and visualization
├── plots/                   # Generated tracking performance plots
├── runs/                    # Simulation data output (CSV, JSON, PNG)
├── soft_robotic_env.py     # Main environment class integration
├── run_sim.py              # Main entry point for simulations
├── test_tracking.py        # Simple tracking performance test
├── comprehensive_test.py   # Detailed control performance analysis
├── generate_tracking_plots.py # Generate plots from existing CSV data
├── test_plot_generation.py # Test plot generation functionality
├── AGENTS.md               # Development guidelines and commands
├── LICENSE                 # Project license
└── README.md              # This file
```

## Core Environment Modules

### `environment/base.py`
**Main Function**: Defines the base SoftRobotic environment class that extends Gymnasium's Env class.

**Key Features**:
- Physical parameters (arm length, mass, inertia, damping, etc.)
- Control mode configuration (position, velocity, acceleration, force)
- Observation and action space definitions
- Environment state initialization and reset functionality
- Rendering setup and configuration

### `environment/kinematics.py`
**Main Function**: Handles all kinematic calculations for the robotic arm.

**Key Features**:
- `calculate_reference_trajectory()`: Generates sinusoidal target trajectories
- `calculate_kinematics()`: Computes end-effector position from arm angle
- Cartesian coordinate transformations

### `environment/dynamics.py`
**Main Function**: Manages the dynamic behavior and motion integration of the arm.

**Key Features**:
- `calculate_dynamics()`: Computes new state based on applied torques
- Physics integration using numerical methods
- Gravity, damping, and stiffness effects
- Motion constraints and limits

### `environment/observations.py`
**Main Function**: Generates observation vectors and information dictionaries.

**Key Features**:
- `get_observation()`: Creates the main observation vector with positions, velocities, and errors
- `get_information()`: Provides detailed state information for analysis
- Reference trajectory integration

### `environment/actions.py`
**Main Function**: Processes control actions and implements PID control.

**Key Features**:
- Control mode-specific action processing
- PID controller implementation for position control
- Torque calculation for all control modes
- Force-to-torque conversion for force control

### `environment/rendering.py`
**Main Function**: Provides visual representation of the robotic arm using pygame.

**Key Features**:
- Real-time visualization of arm position
- T-shaped joint representation
- Coordinate system and reference indicators
- Cross-platform rendering support

## Simulation Modules

### `simulation/runner.py`
**Main Function**: Executes the main simulation loop and controls the experiment flow.

**Key Features**:
- Simulation orchestration and control
- Episode management and reset handling
- Action generation for different control modes
- Decision logging for analysis
- Integration with environment modules

### `simulation/arguments.py`
**Main Function**: Parses and validates command-line arguments for simulation configuration.

**Key Features**:
- Control mode selection (position, velocity, acceleration, force)
- Simulation parameters (steps, screen dimensions, etc.)
- Physical parameter overrides
- Sinusoidal trajectory configuration
- Object manipulation settings

### `simulation/logging.py`
**Main Function**: Handles data persistence and experiment logging.

**Key Features**:
- Episode data logging (CSV format)
- Performance analysis JSON generation
- Decision log persistence
- Episode management and file naming
- Data structure initialization

### `simulation/visualization.py`
**Main Function**: Generates plots and visualizations for experiment analysis.

**Key Features**:
- `plot_observations()`: Creates comprehensive observation plots
- `plot_tracking_performance()`: Generates tracking comparison plots
- Performance metrics visualization
- Error analysis plots

## Main Entry Points

### `soft_robotic_env.py`
**Main Function**: Integrates all environment components into a single cohesive class.

**Key Features**:
- Component composition (kinematics, dynamics, observations, etc.)
- Backward compatibility methods
- Object manipulation capabilities
- Reward calculation and episode termination

### `run_sim.py`
**Main Function**: Main entry point for running simulations and experiments.

**Key Features**:
- Command-line interface
- Experiment execution
- Data collection and storage
- Real-time visualization

## Utility Scripts

### `test_tracking.py`
**Main Function**: Simple script to test and visualize tracking performance.

**Key Features**:
- Quick performance validation
- Basic tracking visualization
- Error calculation and reporting

### `comprehensive_test.py`
**Main Function**: Detailed analysis of control performance across different modes.

**Key Features**:
- Multi-mode performance comparison
- Detailed error analysis
- Comprehensive visualization
- Statistical reporting

### `generate_tracking_plots.py`
**Main Function**: Generates tracking plots from existing CSV data files.

**Key Features**:
- Post-processing analysis
- Batch plot generation
- Performance metrics extraction
- Data visualization

### `test_plot_generation.py`
**Main Function**: Tests the plotting functionality with sample data.

**Key Features**:
- Visualization system validation
- Plot generation testing
- Error handling verification

### `rl_wrapper.py`
**Main Function**: RL-friendly wrapper for the SoftRobotic environment.

**Key Features**:
- Multiple reward types (shaped, sparse, multi-objective)
- Action transformations (absolute, delta, normalized)
- Exploration bonuses and smoothness rewards
- Better reward shaping for reinforcement learning

### `rl_test.py`
**Main Function**: Demonstrates RL-friendly environment enhancements.

**Key Features**:
- Comparison of different reward types
- Action transformation demonstrations
- RL vs. traditional control comparison
- Performance metrics for learning

## Usage Examples

### Basic Simulation
```bash
python run_sim.py --mode position --steps 1000
```

### Velocity Control with Custom Parameters
```bash
python run_sim.py --mode velocity --velocity-action 0.5 --sinusoidal-magnitude 0.3
```

### Force Control with Object Manipulation
```bash
python run_sim.py --mode force --object-mass 0.5 --steps 2000
```

### RL-Enhanced Environment
```bash
python simple_rl_test.py
python rl_test.py
```

## Reinforcement Learning Features

The environment now includes RL-friendly enhancements:

### Reward Functions
- **Shaped Rewards**: Multi-component rewards for smooth learning
- **Sparse Rewards**: Task-completion based rewards
- **Multi-Objective Rewards**: Weighted combination of objectives

### Action Transformations
- **Absolute Actions**: Direct control signals
- **Delta Actions**: Changes to current targets (better exploration)
- **Normalized Actions**: Scaled actions for consistent learning

### Exploration Features
- Exploration bonuses for visiting new states
- Smoothness rewards to encourage stable policies
- Energy efficiency considerations

## Development Guidelines

See `AGENTS.md` for:
- Build, lint, and test commands
- Code style guidelines
- Environment details
- Special notes for contributors

## Data Output

Simulation results are saved in the `runs/` directory:
- `.csv`: Time-series data for analysis
- `.json`: Performance metrics and experiment parameters
- `.png`: Visualization plots
- `.decisions.json`: Detailed control decisions for agent evaluation

Tracking performance plots are saved in the `plots/` directory for visual analysis of controller performance.