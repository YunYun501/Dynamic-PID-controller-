from os import path
import numpy as np
from gymnasium.error import DependencyNotInstalled


class Rendering:
    """Handles rendering for the soft robotic arm."""
    
    def render(self, env):
        """
        Render the current state of the environment with T-shaped components.
        
        Returns:
            None or np.ndarray: RGB array if render_mode is "rgb_array"
        """
        if env.render_mode is None:
            if getattr(env, "spec", None) is not None:
                import gymnasium as gym
                gym.logger.warn(
                    "You are calling render method without specifying any render mode. "
                    "You can specify the render_mode at initialization, "
                    f'e.g. gym.make("{env.spec.id}", render_mode="rgb_array")'
                )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        # Initialize display
        if env.screen is None:
            pygame.init()
            if env.render_mode == "human":
                pygame.display.init()
                env.screen = pygame.display.set_mode((env.screen_dimension, env.screen_dimension))
            else:
                env.screen = pygame.Surface((env.screen_dimension, env.screen_dimension))
        if env.clock is None:
            env.clock = pygame.time.Clock()

        env.surface = pygame.Surface((env.screen_dimension, env.screen_dimension))
        env.surface.fill((255, 255, 255))

        # World -> screen mapping
        boundary = 2.2
        scale = env.screen_dimension / (boundary * 2)
        offset_x = env.screen_dimension // 2
        offset_y = int(env.screen_dimension * env.pivot_y_fraction)

        # Draw fixed upper T-shape (base)
        t_width = 0.3 * scale
        t_height = 0.4 * scale
        t_thickness = 0.1 * scale
        
        # Horizontal bar of upper T
        gfxdraw.aapolygon(env.surface, [
            (int(offset_x - t_width), int(offset_y - t_thickness)),
            (int(offset_x - t_width), int(offset_y + t_thickness)),
            (int(offset_x + t_width), int(offset_y + t_thickness)),
            (int(offset_x + t_width), int(offset_y - t_thickness))
        ], (100, 100, 100))
        gfxdraw.filled_polygon(env.surface, [
            (int(offset_x - t_width), int(offset_y - t_thickness)),
            (int(offset_x - t_width), int(offset_y + t_thickness)),
            (int(offset_x + t_width), int(offset_y + t_thickness)),
            (int(offset_x + t_width), int(offset_y - t_thickness))
        ], (100, 100, 100))
        
        # Vertical bar of upper T
        gfxdraw.aapolygon(env.surface, [
            (int(offset_x - t_thickness), int(offset_y)),
            (int(offset_x - t_thickness), int(offset_y + t_height)),
            (int(offset_x + t_thickness), int(offset_y + t_height)),
            (int(offset_x + t_thickness), int(offset_y))
        ], (100, 100, 100))
        gfxdraw.filled_polygon(env.surface, [
            (int(offset_x - t_thickness), int(offset_y)),
            (int(offset_x - t_thickness), int(offset_y + t_height)),
            (int(offset_x + t_thickness), int(offset_y + t_height)),
            (int(offset_x + t_thickness), int(offset_y))
        ], (100, 100, 100))

        # Draw dynamic lower T-shape (flipped along X-axis and connected at bottom tips)
        # Use same dimensions as upper T for consistency
        lower_t_width = t_width
        lower_t_height = t_height
        lower_t_thickness = t_thickness
        
        # Position of connection point (bottom of upper T's vertical bar)
        connection_y = offset_y + t_height
        
        # Create flipped T-shape and rotate it based on arm_angle
        # Horizontal bar of lower T (positioned below the connection point to connect at bottom tip)
        horizontal_coords = [
            pygame.math.Vector2(-lower_t_width, lower_t_height),  # Shifted down by lower_t_height
            pygame.math.Vector2(-lower_t_width, lower_t_height + lower_t_thickness),
            pygame.math.Vector2(lower_t_width, lower_t_height + lower_t_thickness),
            pygame.math.Vector2(lower_t_width, lower_t_height)
        ]
        
        # Vertical bar of lower T (pointing upward from connection point - flipped along X-axis)
        vertical_coords = [
            pygame.math.Vector2(-lower_t_thickness, lower_t_height),  # Shifted down by lower_t_height
            pygame.math.Vector2(-lower_t_thickness, 0),  # Now extends upward to connection point
            pygame.math.Vector2(lower_t_thickness, 0),   # Now extends upward to connection point
            pygame.math.Vector2(lower_t_thickness, lower_t_height)
        ]
        
        # Rotate and translate all coordinates
        rotated_horizontal = []
        rotated_vertical = []
        
        for coord in horizontal_coords:
            rotated_coord = coord.rotate_rad(env.arm_angle)
            rotated_coord.x += offset_x
            rotated_coord.y += connection_y
            rotated_horizontal.append((int(rotated_coord.x), int(rotated_coord.y)))
            
        for coord in vertical_coords:
            rotated_coord = coord.rotate_rad(env.arm_angle)
            rotated_coord.x += offset_x
            rotated_coord.y += connection_y
            rotated_vertical.append((int(rotated_coord.x), int(rotated_coord.y)))
        
        # Draw the rotated lower T components
        gfxdraw.aapolygon(env.surface, rotated_horizontal, (204, 77, 77))
        gfxdraw.filled_polygon(env.surface, rotated_horizontal, (204, 77, 77))
        gfxdraw.aapolygon(env.surface, rotated_vertical, (204, 77, 77))
        gfxdraw.filled_polygon(env.surface, rotated_vertical, (204, 77, 77))

# Target position marker
        if hasattr(env, 'last_action') and env.last_action is not None:
            # Use the control signal as target for position control mode
            if env.control_mode == "position":
                theta_reference = float(env.last_action[0])
            elif env.control_mode == "velocity":
                # For velocity control, integrate to get target position
                # This is a simplified approximation
                theta_reference = env.arm_angle + float(env.last_action[0]) * env.time_step
            elif env.control_mode == "acceleration":
                # For acceleration control, integrate twice to get target position
                # This is a simplified approximation
                velocity_change = float(env.last_action[0]) * env.time_step
                theta_reference = env.arm_angle + (env.angular_velocity + velocity_change/2) * env.time_step
            else:  # force control
                # For force control, use sinusoidal reference as fallback
                theta_reference = env.sinusoidal_magnitude * np.sin(2 * np.pi * env.sinusoidal_frequency * env.step_count * env.time_step)
        else:
            # Fallback to sinusoidal reference
            theta_reference = env.sinusoidal_magnitude * np.sin(2 * np.pi * env.sinusoidal_frequency * env.step_count * env.time_step)
            
        # Calculate target position of end of lower T (end of its vertical bar)
        # The end is now at the connection point (bottom tip of upper T)
        target_end_x = offset_x + t_height * np.sin(theta_reference)
        target_end_y = connection_y + t_height * np.cos(theta_reference)
        gfxdraw.aacircle(env.surface, int(target_end_x), int(target_end_y), max(2, int(0.07 * scale)), (0, 180, 0))
        gfxdraw.filled_circle(env.surface, int(target_end_x), int(target_end_y), max(2, int(0.07 * scale)), (0, 180, 0))

        # Torque indicator
        filename = path.join(path.dirname(__file__), "..", "assets", "clockwise.png")
        torque_drawn = False
        try:
            image = None
            if path.exists(filename):
                image = pygame.image.load(filename).convert_alpha()
            if image is not None and env.last_torque_applied is not None:
                size = max(1.0, float(scale * min(abs(env.last_torque_applied) / env.maximum_torque, 1.0) * 0.8))
                size = int(size)
                if size > 0:
                    scaled_image = pygame.transform.smoothscale(image, (size, size))
                    is_flip = bool(env.last_torque_applied > 0)
                    scaled_image = pygame.transform.flip(scaled_image, is_flip, True)
                    env.surface.blit(
                        scaled_image,
                        (
                            offset_x - scaled_image.get_rect().centerx,
                            offset_y - scaled_image.get_rect().centery,
                        ),
                    )
                    torque_drawn = True
        except Exception:
            torque_drawn = False

        if not torque_drawn and env.last_torque_applied is not None:
            radius = int(0.1 * scale + 0.3 * scale * min(abs(env.last_torque_applied) / (env.maximum_torque + 1e-8), 1.0))
            color = (220, 80, 80) if env.last_torque_applied >= 0 else (80, 120, 220)
            gfxdraw.aacircle(env.surface, int(offset_x), int(connection_y), max(2, int(radius)), color)

        # Connection point (bottom tip of upper T's vertical bar)
        gfxdraw.aacircle(env.surface, int(offset_x), int(connection_y), int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(env.surface, int(offset_x), int(connection_y), int(0.05 * scale), (0, 0, 0))

        # Blit to screen
        env.screen.blit(env.surface, (0, 0))
        if env.render_mode == "human":
            pygame.event.pump()
            env.clock.tick(env.metadata["render_fps"])
            pygame.display.flip()
        else:
            import numpy as _np
            return _np.transpose(_np.array(pygame.surfarray.pixels3d(env.screen)), axes=(1, 0, 2))