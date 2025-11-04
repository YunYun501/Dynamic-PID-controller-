import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any


class SoftRoboticRLWrapper(gym.Wrapper):
    """
    RL-friendly wrapper for the SoftRobotic environment.
    Provides enhanced reward functions, action transformations, and RL-specific features.
    """
    
    def __init__(self, env: gym.Env, reward_type: str = "shaped", action_type: str = "delta"):
        """
        Initialize the RL wrapper.
        
        Args:
            env: The base SoftRobotic environment
            reward_type: Type of reward function ("shaped", "sparse", "multi_objective")
            action_type: Type of action transformation ("absolute", "delta", "normalized")
        """
        super().__init__(env)
        self.reward_type = reward_type
        self.action_type = action_type
        
        # For delta actions, we need to track the current target
        self.current_target = 0.0
        
        # For exploration tracking
        self.visited_states = set()
        
        # For reward shaping
        self.last_torque = 0.0
        self.last_theta_error = 0.0
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment and initialize tracking variables."""
        observation, info = self.env.reset(seed=seed, options=options)
        
        # Reset tracking variables
        self.current_target = observation[8]  # theta_target from observation
        self.last_torque = 0.0
        self.last_theta_error = observation[11]  # etheta from observation
        self.visited_states.clear()
        
        return observation, info
    
    def step(self, action):
        """Execute one step with RL-friendly action processing and reward calculation."""
        # Transform action based on action_type
        transformed_action = self._transform_action(action)
        
        # Execute the action in the base environment
        observation, _, terminated, truncated, info = self.env.step(transformed_action)
        
        # Calculate RL-friendly reward
        reward = self._calculate_rl_reward(action, observation, info)
        
        # Track visited states for exploration bonus
        state_key = self._get_state_key(observation)
        self.visited_states.add(state_key)
        
        # Update tracking variables
        self.last_torque = info.get("tau", 0.0)
        self.last_theta_error = observation[11]  # etheta
        
        return observation, reward, terminated, truncated, info
    
    def _transform_action(self, action):
        """Transform action based on the selected action type."""
        if self.action_type == "delta":
            # Action represents change in target rather than absolute target
            if self.env.control_mode == "position":
                self.current_target += float(action[0])
                # Keep within limits
                self.current_target = np.clip(self.current_target, 
                                            -self.env.theta_limit, 
                                            self.env.theta_limit)
                return np.array([self.current_target], dtype=np.float32)
            else:
                # For other control modes, use action as-is
                return action
        elif self.action_type == "normalized":
            # Normalize action to environment limits
            if self.env.control_mode == "position":
                # Scale from [-1, 1] to [-theta_limit, theta_limit]
                scaled_action = float(action[0]) * self.env.theta_limit
                return np.array([scaled_action], dtype=np.float32)
            elif self.env.control_mode == "velocity":
                # Scale from [-1, 1] to [-10, 10]
                scaled_action = float(action[0]) * 10.0
                return np.array([scaled_action], dtype=np.float32)
            elif self.env.control_mode == "acceleration":
                # Scale from [-1, 1] to [-100, 100]
                scaled_action = float(action[0]) * 100.0
                return np.array([scaled_action], dtype=np.float32)
            else:  # force control
                # Scale from [-1, 1] to [-100, 100] for both forces
                scaled_left = float(action[0]) * 100.0
                scaled_right = float(action[1]) * 100.0
                return np.array([scaled_left, scaled_right], dtype=np.float32)
        else:
            # "absolute" - use action as-is
            return action
    
    def _calculate_rl_reward(self, action, observation, info):
        """Calculate RL-friendly reward based on reward_type."""
        if self.reward_type == "sparse":
            return self._calculate_sparse_reward(observation)
        elif self.reward_type == "multi_objective":
            return self._calculate_multi_objective_reward(action, observation, info)
        else:  # "shaped" (default)
            return self._calculate_shaped_reward(action, observation, info)
    
    def _calculate_shaped_reward(self, action, observation, info):
        """Calculate shaped reward with multiple components."""
        # Extract relevant values
        theta_error = observation[11]  # etheta
        theta_dot = observation[3]     # theta_dot
        torque = info.get("tau", 0.0)
        
        # Tracking reward (primary objective)
        tracking_reward = -abs(theta_error) - 0.1 * abs(theta_dot)
        
        # Smoothness reward (penalize jerky movements)
        torque_change = abs(torque - self.last_torque)
        smoothness_reward = -0.01 * torque_change
        
        # Energy efficiency reward
        energy_reward = -0.001 * (torque ** 2)
        
        # Exploration bonus (encourage visiting new states)
        state_key = self._get_state_key(observation)
        exploration_bonus = 0.1 if state_key not in self.visited_states else 0.0
        
        # Total reward
        total_reward = tracking_reward + smoothness_reward + energy_reward + exploration_bonus
        
        return float(total_reward)
    
    def _calculate_sparse_reward(self, observation):
        """Calculate sparse reward (only at task completion or milestones)."""
        theta_error = observation[11]  # etheta
        time_penalty = -0.01  # Small penalty for each step
        
        # Success reward when close to target
        success_reward = 0.0
        if abs(theta_error) < 0.05:  # Within 0.05 radians of target
            success_reward = 1.0
            
        return time_penalty + success_reward
    
    def _calculate_multi_objective_reward(self, action, observation, info):
        """Calculate multi-objective reward with weighted components."""
        theta_error = observation[11]  # etheta
        theta_dot = observation[3]     # theta_dot
        torque = info.get("tau", 0.0)
        
        # Tracking objective (weight: 0.5)
        tracking_obj = -abs(theta_error) - 0.1 * abs(theta_dot)
        tracking_reward = 0.5 * tracking_obj
        
        # Energy efficiency objective (weight: 0.3)
        energy_obj = -0.001 * (torque ** 2)
        energy_reward = 0.3 * energy_obj
        
        # Smoothness objective (weight: 0.2)
        torque_change = abs(torque - self.last_torque)
        smoothness_obj = -0.01 * torque_change
        smoothness_reward = 0.2 * smoothness_obj
        
        total_reward = tracking_reward + energy_reward + smoothness_reward
        
        return float(total_reward)
    
    def _get_state_key(self, observation):
        """Create a discrete key for the current state for exploration tracking."""
        # Discretize continuous state variables
        theta_disc = int(observation[2] * 100)  # theta
        theta_dot_disc = int(observation[3] * 10)  # theta_dot
        return (theta_disc, theta_dot_disc)


# Convenience functions for creating different RL variants
def create_rl_environment(control_mode: str = "position", 
                         reward_type: str = "shaped", 
                         action_type: str = "delta",
                         **kwargs) -> SoftRoboticRLWrapper:
    """
    Create an RL-friendly SoftRobotic environment.
    
    Args:
        control_mode: Control mode ("position", "velocity", "acceleration", "force")
        reward_type: Type of reward function ("shaped", "sparse", "multi_objective")
        action_type: Type of action transformation ("absolute", "delta", "normalized")
        **kwargs: Additional arguments passed to SoftRobotic environment
        
    Returns:
        SoftRoboticRLWrapper: RL-friendly environment
    """
    # Import locally to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from environment.soft_robotic_env import SoftRobotic
    
    # Create base environment
    base_env = SoftRobotic(control_mode=control_mode, **kwargs)
    
    # Wrap with RL enhancements
    rl_env = SoftRoboticRLWrapper(base_env, reward_type=reward_type, action_type=action_type)
    
    return rl_env