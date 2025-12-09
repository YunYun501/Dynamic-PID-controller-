"""
PYGAME RENDERING BACKUP - T-SHAPED ROBOTIC ARM JOINT
=====================================================

This file contains the complete pygame rendering code for the T-shaped robotic arm.
Use this as a reference when rebuilding the visualization component from scratch.

Key Features:
- Fixed upper T-shape (base/pivot mount)
- Dynamic lower T-shape (rotating arm)
- Target position marker
- Torque indicator
- Smooth anti-aliased rendering
"""

import pygame
from pygame import gfxdraw
import numpy as np
from os import path


class RobotRenderer:
    """
    Complete rendering system for the T-shaped robotic arm.
    
    Required env attributes:
    - screen_dimension: int (e.g., 600)
    - pivot_y_fraction: float (e.g., 0.15)
    - arm_angle: float (radians)
    - render_mode: str ("human" or "rgb_array")
    - screen, clock, surface: pygame objects (can be None initially)
    - last_torque_applied: float
    - maximum_torque: float
    - metadata: dict with "render_fps"
    
    Optional for target marker:
    - last_action: array
    - control_mode: str
    - time_step: float
    - angular_velocity: float
    - sinusoidal_magnitude, sinusoidal_frequency: float
    - step_count: int
    """
    
    def __init__(self):
        pass
    
    def render(self, env):
        """
        Render the current state of the environment with T-shaped components.
        
        Args:
            env: Environment object with required attributes
            
        Returns:
            None (human mode) or np.ndarray RGB array (rgb_array mode)
        """
        if env.render_mode is None:
            return
        
        # Import pygame (lazy import)
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise ImportError(
                'pygame is not installed, run `pip install pygame`'
            ) from e
        
        # Initialize display on first call
        if env.screen is None:
            pygame.init()
            if env.render_mode == "human":
                pygame.display.init()
                env.screen = pygame.display.set_mode(
                    (env.screen_dimension, env.screen_dimension)
                )
                pygame.display.set_caption("Soft Robotic Arm")
            else:  # rgb_array
                env.screen = pygame.Surface(
                    (env.screen_dimension, env.screen_dimension)
                )
        
        if env.clock is None:
            env.clock = pygame.time.Clock()
        
        # Create drawing surface
        env.surface = pygame.Surface(
            (env.screen_dimension, env.screen_dimension)
        )
        env.surface.fill((255, 255, 255))  # White background
        
        # =====================================================
        # COORDINATE SYSTEM SETUP
        # =====================================================
        boundary = 2.2  # World space extends from -2.2 to +2.2
        scale = env.screen_dimension / (boundary * 2)
        offset_x = env.screen_dimension // 2  # Center horizontally
        offset_y = int(env.screen_dimension * env.pivot_y_fraction)
        
        # =====================================================
        # T-SHAPE DIMENSIONS
        # =====================================================
        t_width = 0.3 * scale      # Half-width of horizontal bar
        t_height = 0.4 * scale     # Height of vertical bar
        t_thickness = 0.1 * scale  # Thickness of bars
        
        # =====================================================
        # DRAW FIXED UPPER T-SHAPE (BASE)
        # =====================================================
        # Color: Gray (100, 100, 100)
        base_color = (100, 100, 100)
        
        # Horizontal bar of upper T
        h_bar_upper = [
            (int(offset_x - t_width), int(offset_y - t_thickness)),
            (int(offset_x - t_width), int(offset_y + t_thickness)),
            (int(offset_x + t_width), int(offset_y + t_thickness)),
            (int(offset_x + t_width), int(offset_y - t_thickness))
        ]
        gfxdraw.aapolygon(env.surface, h_bar_upper, base_color)
        gfxdraw.filled_polygon(env.surface, h_bar_upper, base_color)
        
        # Vertical bar of upper T
        v_bar_upper = [
            (int(offset_x - t_thickness), int(offset_y)),
            (int(offset_x - t_thickness), int(offset_y + t_height)),
            (int(offset_x + t_thickness), int(offset_y + t_height)),
            (int(offset_x + t_thickness), int(offset_y))
        ]
        gfxdraw.aapolygon(env.surface, v_bar_upper, base_color)
        gfxdraw.filled_polygon(env.surface, v_bar_upper, base_color)
        
        # =====================================================
        # DRAW DYNAMIC LOWER T-SHAPE (ROTATING ARM)
        # =====================================================
        # Color: Reddish (204, 77, 77)
        arm_color = (204, 77, 77)
        
        # Connection point (bottom of upper T's vertical bar)
        connection_y = offset_y + t_height
        
        # Define T-shape in local coordinates (before rotation)
        # The lower T is "flipped" - horizontal bar at bottom, vertical bar extends up
        
        # Horizontal bar of lower T
        horizontal_coords = [
            pygame.math.Vector2(-t_width, t_height),
            pygame.math.Vector2(-t_width, t_height + t_thickness),
            pygame.math.Vector2(t_width, t_height + t_thickness),
            pygame.math.Vector2(t_width, t_height)
        ]
        
        # Vertical bar of lower T (points upward to connection)
        vertical_coords = [
            pygame.math.Vector2(-t_thickness, t_height),
            pygame.math.Vector2(-t_thickness, 0),
            pygame.math.Vector2(t_thickness, 0),
            pygame.math.Vector2(t_thickness, t_height)
        ]
        
        # Rotate and translate coordinates based on arm_angle
        rotated_horizontal = []
        rotated_vertical = []
        
        for coord in horizontal_coords:
            rotated_coord = coord.rotate_rad(env.arm_angle)
            rotated_coord.x += offset_x
            rotated_coord.y += connection_y
            rotated_horizontal.append(
                (int(rotated_coord.x), int(rotated_coord.y))
            )
        
        for coord in vertical_coords:
            rotated_coord = coord.rotate_rad(env.arm_angle)
            rotated_coord.x += offset_x
            rotated_coord.y += connection_y
            rotated_vertical.append(
                (int(rotated_coord.x), int(rotated_coord.y))
            )
        
        # Draw the rotated lower T
        gfxdraw.aapolygon(env.surface, rotated_horizontal, arm_color)
        gfxdraw.filled_polygon(env.surface, rotated_horizontal, arm_color)
        gfxdraw.aapolygon(env.surface, rotated_vertical, arm_color)
        gfxdraw.filled_polygon(env.surface, rotated_vertical, arm_color)
        
        # =====================================================
        # DRAW TARGET POSITION MARKER
        # =====================================================
        # Color: Green (0, 180, 0)
        target_color = (0, 180, 0)
        
        # Calculate target angle (theta_reference)
        theta_reference = self._get_target_angle(env)
        
        # Calculate target position (end of lower T)
        target_end_x = offset_x + t_height * np.sin(theta_reference)
        target_end_y = connection_y + t_height * np.cos(theta_reference)
        target_radius = max(2, int(0.07 * scale))
        
        gfxdraw.aacircle(
            env.surface,
            int(target_end_x),
            int(target_end_y),
            target_radius,
            target_color
        )
        gfxdraw.filled_circle(
            env.surface,
            int(target_end_x),
            int(target_end_y),
            target_radius,
            target_color
        )
        
        # =====================================================
        # DRAW TORQUE INDICATOR
        # =====================================================
        if hasattr(env, 'last_torque_applied') and env.last_torque_applied is not None:
            # Try to load clockwise icon first
            torque_drawn = self._draw_torque_icon(
                env, scale, offset_x, offset_y
            )
            
            # Fallback to colored circle if icon not available
            if not torque_drawn:
                self._draw_torque_circle(
                    env, scale, offset_x, connection_y
                )
        
        # =====================================================
        # DRAW CONNECTION POINT (PIVOT)
        # =====================================================
        # Color: Black (0, 0, 0)
        pivot_color = (0, 0, 0)
        pivot_radius = int(0.05 * scale)
        
        gfxdraw.aacircle(
            env.surface,
            int(offset_x),
            int(connection_y),
            pivot_radius,
            pivot_color
        )
        gfxdraw.filled_circle(
            env.surface,
            int(offset_x),
            int(connection_y),
            pivot_radius,
            pivot_color
        )
        
        # =====================================================
        # FINALIZE RENDERING
        # =====================================================
        env.screen.blit(env.surface, (0, 0))
        
        if env.render_mode == "human":
            pygame.event.pump()
            env.clock.tick(env.metadata.get("render_fps", 50))
            pygame.display.flip()
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(env.screen)),
                axes=(1, 0, 2)
            )
    
    def _get_target_angle(self, env):
        """
        Calculate the target angle based on control mode and current state.
        
        Args:
            env: Environment object
            
        Returns:
            float: Target angle in radians
        """
        # Default fallback
        theta_reference = 0.0
        
        if hasattr(env, 'last_action') and env.last_action is not None:
            if hasattr(env, 'control_mode'):
                if env.control_mode == "position":
                    # Direct angle target
                    theta_reference = float(env.last_action[0])
                
                elif env.control_mode == "velocity":
                    # Integrate velocity to approximate position
                    theta_reference = env.arm_angle + \
                        float(env.last_action[0]) * env.time_step
                
                elif env.control_mode == "acceleration":
                    # Integrate acceleration twice
                    velocity_change = float(env.last_action[0]) * env.time_step
                    theta_reference = env.arm_angle + \
                        (env.angular_velocity + velocity_change/2) * env.time_step
                
                else:  # force control or other
                    # Use sinusoidal reference
                    theta_reference = self._calculate_sinusoidal_reference(env)
            else:
                theta_reference = self._calculate_sinusoidal_reference(env)
        else:
            theta_reference = self._calculate_sinusoidal_reference(env)
        
        return theta_reference
    
    def _calculate_sinusoidal_reference(self, env):
        """Calculate sinusoidal reference trajectory."""
        if hasattr(env, 'sinusoidal_magnitude') and \
           hasattr(env, 'sinusoidal_frequency') and \
           hasattr(env, 'step_count') and \
           hasattr(env, 'time_step'):
            return env.sinusoidal_magnitude * np.sin(
                2 * np.pi * env.sinusoidal_frequency * 
                env.step_count * env.time_step
            )
        return 0.0
    
    def _draw_torque_icon(self, env, scale, offset_x, offset_y):
        """
        Draw torque indicator using clockwise icon (if available).
        
        Returns:
            bool: True if icon was drawn, False otherwise
        """
        try:
            # Try to find clockwise icon
            filename = path.join(
                path.dirname(__file__), "assets", "clockwise.png"
            )
            
            if not path.exists(filename):
                return False
            
            image = pygame.image.load(filename).convert_alpha()
            
            if image is not None and env.last_torque_applied is not None:
                # Scale icon based on torque magnitude
                size = max(
                    1.0,
                    float(scale * min(
                        abs(env.last_torque_applied) / env.maximum_torque,
                        1.0
                    ) * 0.8)
                )
                size = int(size)
                
                if size > 0:
                    scaled_image = pygame.transform.smoothscale(
                        image, (size, size)
                    )
                    
                    # Flip for direction (positive torque = CW)
                    is_flip = bool(env.last_torque_applied > 0)
                    scaled_image = pygame.transform.flip(
                        scaled_image, is_flip, True
                    )
                    
                    env.surface.blit(
                        scaled_image,
                        (
                            offset_x - scaled_image.get_rect().centerx,
                            offset_y - scaled_image.get_rect().centery,
                        ),
                    )
                    return True
        except Exception:
            pass
        
        return False
    
    def _draw_torque_circle(self, env, scale, offset_x, connection_y):
        """
        Draw torque indicator as colored circle (fallback).
        
        Circle size indicates magnitude, color indicates direction.
        """
        radius = int(
            0.1 * scale + 
            0.3 * scale * min(
                abs(env.last_torque_applied) / (env.maximum_torque + 1e-8),
                1.0
            )
        )
        
        # Color based on direction
        # Red = Clockwise (positive), Blue = Counter-clockwise (negative)
        color = (220, 80, 80) if env.last_torque_applied >= 0 else (80, 120, 220)
        
        gfxdraw.aacircle(
            env.surface,
            int(offset_x),
            int(connection_y),
            max(2, radius),
            color
        )


# =====================================================
# USAGE EXAMPLE
# =====================================================
"""
# In your environment class:

class SoftRoboticEnv:
    def __init__(self):
        # Required attributes
        self.screen_dimension = 600
        self.pivot_y_fraction = 0.15
        self.arm_angle = 0.0
        self.render_mode = "human"  # or "rgb_array"
        self.maximum_torque = 2.0
        self.metadata = {"render_fps": 50}
        
        # Optional attributes for target marker
        self.last_action = None
        self.control_mode = "position"
        self.time_step = 0.02
        self.angular_velocity = 0.0
        self.sinusoidal_magnitude = 0.5
        self.sinusoidal_frequency = 0.5
        self.step_count = 0
        
        # Pygame objects (initialized by renderer)
        self.screen = None
        self.clock = None
        self.surface = None
        self.last_torque_applied = 0.0
        
        # Create renderer
        self.renderer = RobotRenderer()
    
    def render(self):
        return self.renderer.render(self)
    
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
"""
