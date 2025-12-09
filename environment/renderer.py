"""Pygame renderer that follows the T-shaped geometry from RenderingBackup."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

try:
    import pygame
except Exception as exc:  # pragma: no cover - optional dependency
    pygame = None
    gfxdraw = None
    _pygame_import_error = exc
else:
    try:
        from pygame import gfxdraw  # type: ignore
    except Exception:
        gfxdraw = None
    _pygame_import_error = None


@dataclass
class _Colors:
    bg: Tuple[int, int, int] = (255, 255, 255)
    base: Tuple[int, int, int] = (100, 100, 100)
    arm: Tuple[int, int, int] = (204, 77, 77)
    pivot: Tuple[int, int, int] = (0, 0, 0)
    target: Tuple[int, int, int] = (0, 180, 0)
    text: Tuple[int, int, int] = (30, 30, 30)


class PendulumRenderer:
    """Pygame-based renderer supporting human and rgb_array modes."""

    def __init__(self, render_config: Any):
        if pygame is None:
            raise ImportError(f"pygame not available: {_pygame_import_error}")

        self.config = render_config
        self.colors = _Colors()
        self.screen_size = int(render_config.screen_size)
        self.pivot_y_fraction = float(render_config.pivot_y_fraction)
        self.mode = render_config.render_mode or 'human'
        self.clock = pygame.time.Clock()
        self.font = None
        self.window = None
        self.surface = None

        pygame.init()
        self._setup_surface()

    def _setup_surface(self) -> None:
        if self.mode == 'human':
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('Soft Robotic Arm')
            self.surface = self.window
            self.font = pygame.font.SysFont('arial', 16)
        else:
            self.surface = pygame.Surface((self.screen_size, self.screen_size))

    def _draw_polygon(self, points, color):
        """Antialiased polygon helper with a safe fallback."""
        if gfxdraw:
            gfxdraw.aapolygon(self.surface, points, color)
            gfxdraw.filled_polygon(self.surface, points, color)
        else:  # pragma: no cover - gfxdraw unavailable
            pygame.draw.polygon(self.surface, color, points)

    def _draw_circle(self, center, radius, color):
        if gfxdraw:
            gfxdraw.aacircle(self.surface, int(center[0]), int(center[1]), int(radius), color)
            gfxdraw.filled_circle(self.surface, int(center[0]), int(center[1]), int(radius), color)
        else:  # pragma: no cover - gfxdraw unavailable
            pygame.draw.circle(self.surface, color, (int(center[0]), int(center[1])), int(radius))

    def render(self, env_state: Any):
        if pygame is None:
            return None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        if self.surface is None:
            self._setup_surface()

        self.surface.fill(self.colors.bg)

        # Coordinate system aligned with RenderingBackup geometry
        boundary = 2.2
        scale = self.screen_size / (boundary * 2)
        offset_x = self.screen_size // 2
        offset_y = int(self.screen_size * self.pivot_y_fraction)

        t_width = 0.3 * scale
        t_height = 0.4 * scale
        t_thickness = 0.1 * scale

        # Draw fixed upper T (base)
        upper_horizontal = [
            (int(offset_x - t_width), int(offset_y - t_thickness)),
            (int(offset_x - t_width), int(offset_y + t_thickness)),
            (int(offset_x + t_width), int(offset_y + t_thickness)),
            (int(offset_x + t_width), int(offset_y - t_thickness)),
        ]
        upper_vertical = [
            (int(offset_x - t_thickness), int(offset_y)),
            (int(offset_x - t_thickness), int(offset_y + t_height)),
            (int(offset_x + t_thickness), int(offset_y + t_height)),
            (int(offset_x + t_thickness), int(offset_y)),
        ]
        self._draw_polygon(upper_horizontal, self.colors.base)
        self._draw_polygon(upper_vertical, self.colors.base)

        connection_y = offset_y + t_height
        angle = float(getattr(env_state, 'angle', 0.0))

        # Define the reverse T in local coordinates (horizontal bar at the bottom)
        horizontal_coords = [
            pygame.math.Vector2(-t_width, t_height),
            pygame.math.Vector2(-t_width, t_height + t_thickness),
            pygame.math.Vector2(t_width, t_height + t_thickness),
            pygame.math.Vector2(t_width, t_height),
        ]
        vertical_coords = [
            pygame.math.Vector2(-t_thickness, t_height),
            pygame.math.Vector2(-t_thickness, 0),
            pygame.math.Vector2(t_thickness, 0),
            pygame.math.Vector2(t_thickness, t_height),
        ]

        def rotate_and_translate(coord: pygame.math.Vector2):
            rotated = coord.rotate_rad(angle)
            rotated.x += offset_x
            rotated.y += connection_y
            return (int(rotated.x), int(rotated.y))

        lower_horizontal = [rotate_and_translate(c) for c in horizontal_coords]
        lower_vertical = [rotate_and_translate(c) for c in vertical_coords]
        self._draw_polygon(lower_horizontal, self.colors.arm)
        self._draw_polygon(lower_vertical, self.colors.arm)

        # Target indicator at the end of the swinging T
        target_angle = getattr(env_state, 'target', None)
        if target_angle is not None:
            target_end = (
                offset_x + t_height * np.sin(-target_angle),
                connection_y + t_height * np.cos(-target_angle),
            )
            self._draw_circle(target_end, max(2, int(0.07 * scale)), self.colors.target)

        # Pivot point
        self._draw_circle((offset_x, connection_y), max(2, int(0.05 * scale)), self.colors.pivot)

        # Overlay telemetry
        if self.font:
            text = (
                f"angle={getattr(env_state, 'angle', 0.0):+.2f} rad  "
                f"vel={getattr(env_state, 'velocity', 0.0):+.2f} rad/s  "
                f"torque={getattr(env_state, 'torque', 0.0):+.2f}"
            )
            txt_surf = self.font.render(text, True, self.colors.text)
            self.surface.blit(txt_surf, (10, self.screen_size - 25))

        if self.mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.config.fps)
            return None

        array = pygame.surfarray.array3d(self.surface)
        return np.transpose(array, (1, 0, 2))

    def close(self):
        if pygame is None:
            return None
        pygame.quit()
        self.window = None
        self.surface = None
        return None
