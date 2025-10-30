from __future__ import annotations

from typing import Tuple, Dict, Optional
from os import path

from gymnasium import spaces
import gymnasium as gym
import numpy as np
from gymnasium.error import DependencyNotInstalled


class SoftRobotic(gym.Env):
    """
    Description

    Coordinate system
    - x, y: Cartesian coordinates of the arm end-point. Right is +x, down is +y.
    - theta: Angle of the arm from the center (vertical) axis. Right/clockwise is +theta,
      left/counter-clockwise is -theta.
    - tau: Applied torque. Positive torque turns the arm to the right (clockwise).
    - The arm is a simple pendulum hanging from a pivot; gravity restores theta -> 0 (down).

    Stage 1 action space
    - Control left/right actuator forces: action = [F_left, F_right].
      Forces act in +y (down) and generate actuator torque
      tau_act = (F_right - F_left) * (actuator_offset * L).

    Stage 2 action space (optional)
    - Control PID gains: action = [Kp, Ki, Kd]. The torque is then computed internally as
      tau = clip(Kp*e + Ki*int_e + Kd*de/dt, [-max_tau, max_tau]), where e is tracking
      error against a reference trajectory.

    Observation space (float32 Box)
    - [0]  x_arm         End-effector x position (real)
    - [1]  y_arm         End-effector y position (real)
    - [2]  theta         Current angle
    - [3]  theta_dot     Angular velocity
    - [4]  theta_ddot    Angular acceleration (computed last step)
    - [5]  tau           Applied torque (after clipping)
    - [6]  x_target      Target x from control signal
    - [7]  y_target      Target y from control signal
    - [8]  theta_target  Target angle from control signal
    - [9]  ex            x error (x_target - x_arm)
    - [10] ey            y error (y_target - y_arm)
    - [11] etheta        theta error (theta_target - theta)
    - [12] p_error       PID proportional error (same as etheta)
    - [13] i_error       PID integral of theta error
    - [14] d_error       PID derivative of theta error

    Reward
    - Penalizes squared position error plus small control effort and velocity penalties:
      r = - (w_pos * ||p_err||^2 + w_tau * tau^2 + w_vel * theta_dot^2).

    Episode termination
    - Truncated after a fixed number of steps (max_steps). No hard-termination conditions
      by default, but can be extended.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        dt: float = 0.02,
        L: float = 1.0,
        I: float = 1.0,
        damping: float = 0.05,
        stiffness: float = 0.0,
        mass: float = 1.0,
        gravity: float = 9.81,
        com_ratio: float = 1.0,
        # Actuation
        max_force: float = 10.0,
        actuator_offset: float = 0.5,
        max_tau: float = 2.0,
        theta_limit: float = np.pi / 2,
        # Target/trajectory
        theta_amp: float = 0.5,
        num_cycles: int = 3,
        max_steps: int = 500,
        # Control mode: "forces", "torque" or "pid"
        control_mode: str = "forces",
        # If True, append [Kp, Ki, Kd] to the action to allow tuning
        action_includes_pid: bool = True,
        # PID action ranges (if control_mode == "pid")
        pid_range: Tuple[float, float, float, float, float, float] = (0.0, 20.0, 0.0, 10.0, 0.0, 5.0),
        random_start: bool = False,
        seed: Optional[int] = None,
        screen_dim: int = 600,
        pivot_y_frac: float = 0.15,
    ) -> None:
        super().__init__()

        assert control_mode in ("forces", "torque", "pid"), "control_mode must be 'forces', 'torque' or 'pid'"

        # Simulation params
        self.dt = float(dt)
        self.L = float(L)
        self.I = float(I)
        self.damping = float(damping)
        self.stiffness = float(stiffness)
        self.mass = float(mass)
        self.g = float(gravity)
        self.max_force = float(max_force)
        self.max_tau = float(max_tau)
        self.theta_limit = float(theta_limit)
        self.com_ratio = float(com_ratio)
        self.lc = self.L * self.com_ratio  # distance from pivot to CoM
        self.actuator_offset = float(actuator_offset)  # fraction of L used as lever arm
        self.action_includes_pid = bool(action_includes_pid)

        # Reference trajectory params
        self.theta_amp = float(theta_amp)
        self.num_cycles = int(num_cycles)
        self.max_steps = int(max_steps)

        # Control mode
        self.control_mode = control_mode
        self.pid_low = np.array([pid_range[0], pid_range[2], pid_range[4]], dtype=np.float32)
        self.pid_high = np.array([pid_range[1], pid_range[3], pid_range[5]], dtype=np.float32)
        # Default PID gains = midpoint of ranges
        self.Kp = float((self.pid_low[0] + self.pid_high[0]) / 2.0)
        self.Ki = float((self.pid_low[1] + self.pid_high[1]) / 2.0)
        self.Kd = float((self.pid_low[2] + self.pid_high[2]) / 2.0)

        # RNG
        self.np_random = np.random.default_rng(seed)
        self.random_start = bool(random_start)

        # Disturbance placeholders (not applied by default)
        self.force_left = 0.0
        self.force_right = 0.0
        self.x_force_left = -0.5 * self.L
        self.x_force_right = 0.5 * self.L

        # Render
        self.render_mode = render_mode
        self.screen_dim = int(screen_dim)
        self.pivot_y_frac = float(pivot_y_frac)
        self.screen = None
        self.clock = None
        self.surf = None
        self.isopen = True
        self.last_u = 0.0

        # Define action space
        if self.control_mode == "forces":
            if self.action_includes_pid:
                low = np.array([0.0, 0.0, *self.pid_low], dtype=np.float32)
                high = np.array([self.max_force, self.max_force, *self.pid_high], dtype=np.float32)
                self.action_space = spaces.Box(low=low, high=high, shape=(5,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([self.max_force, self.max_force], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                )
        elif self.control_mode == "torque":
            if self.action_includes_pid:
                low = np.array([-self.max_tau, *self.pid_low], dtype=np.float32)
                high = np.array([self.max_tau, *self.pid_high], dtype=np.float32)
                self.action_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(
                    low=np.array([-self.max_tau], dtype=np.float32),
                    high=np.array([self.max_tau], dtype=np.float32),
                    shape=(1,),
                    dtype=np.float32,
                )
        else:  # pid
            # Pure PID torque control using 3 parameters
            self.action_space = spaces.Box(
                low=self.pid_low,
                high=self.pid_high,
                shape=(3,),
                dtype=np.float32,
            )

        # Observation space bounds
        L = self.L
        th_lim = self.theta_limit
        th_dot_lim = 10.0
        th_ddot_lim = 100.0
        i_err_lim = 10.0
        obs_low = np.array(
            [
                -L,  # x
                0.0,  # y
                -th_lim,  # theta
                -th_dot_lim,  # theta_dot
                -th_ddot_lim,  # theta_ddot
                -self.max_tau,  # tau
                -L,  # x_target
                0.0,  # y_target
                -th_lim,  # theta_target
                -2 * L,  # ex
                -L,  # ey
                -2 * th_lim,  # etheta
                -2 * th_lim,  # p_error (theta error)
                -i_err_lim,   # i_error (integral of theta error)
                -th_dot_lim,  # d_error (derivative of theta error)
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                L,
                L,
                th_lim,
                th_dot_lim,
                th_ddot_lim,
                self.max_tau,
                L,
                L,
                th_lim,
                2 * L,
                L,
                2 * th_lim,
                2 * th_lim,
                i_err_lim,
                th_dot_lim,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Internal state
        self.theta = 0.0
        self.theta_dot = 0.0
        self.theta_ddot = 0.0
        self.tau = 0.0
        self.step_count = 0

        # PID internal memory (used only if control_mode == 'pid')
        self._int_e = 0.0
        self._prev_e = 0.0

    # ----- Helpers -----
    def _reference(self, step: int) -> Tuple[float, float, float]:
        """Compute reference targets (theta, x, y) for the given step.

        Uses a sinusoid in theta over the episode length.
        """
        # Frequency in cycles per episode
        w = 2.0 * np.pi * self.num_cycles / max(1, self.max_steps)
        theta_target = self.theta_amp * np.sin(w * step)
        x_target = self.L * np.sin(theta_target)
        y_target = self.L * np.cos(theta_target)
        return float(theta_target), float(x_target), float(y_target)

    def _kinematics(self, theta: float) -> Tuple[float, float]:
        """Return (x, y) end-effector position from angle theta."""
        x = self.L * np.sin(theta)
        y = self.L * np.cos(theta)
        return float(x), float(y)

    def _dynamics(self, tau: float) -> Tuple[float, float, float]:
        """Integrate one step of rotational dynamics using semi-implicit Euler.

        theta_ddot = (tau - damping*theta_dot - stiffness*theta) / I
        """
        # Gravitational torque tends to restore theta -> 0 (downward)
        tau_g = - self.mass * self.g * self.lc * np.sin(self.theta)
        theta_ddot = (tau + tau_g - self.damping * self.theta_dot - self.stiffness * self.theta) / self.I
        theta_dot = self.theta_dot + self.dt * theta_ddot
        theta = self.theta + self.dt * theta_dot

        # Clamp angle and velocities to limits
        theta = float(np.clip(theta, -self.theta_limit, self.theta_limit))
        theta_dot = float(np.clip(theta_dot, -10.0, 10.0))
        theta_ddot = float(np.clip(theta_ddot, -100.0, 100.0))
        return theta, theta_dot, theta_ddot

    def _get_obs(self) -> np.ndarray:
        th_ref, x_ref, y_ref = self._reference(self.step_count)
        x, y = self._kinematics(self.theta)
        ex = x_ref - x
        ey = y_ref - y
        et = th_ref - self.theta
        # PID error terms
        p_err = et
        d_err = (p_err - self._prev_e) / self.dt
        i_err = self._int_e
        obs = np.array(
            [
                x,
                y,
                self.theta,
                self.theta_dot,
                self.theta_ddot,
                self.tau,
                x_ref,
                y_ref,
                th_ref,
                ex,
                ey,
                et,
                p_err,
                i_err,
                d_err,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> Dict[str, float]:
        x, y = self._kinematics(self.theta)
        th_ref, x_ref, y_ref = self._reference(self.step_count)
        return {
            "step": float(self.step_count),
            "theta": float(self.theta),
            "theta_dot": float(self.theta_dot),
            "theta_ddot": float(self.theta_ddot),
            "tau": float(self.tau),
            "x": float(x),
            "y": float(y),
            "theta_target": float(th_ref),
            "x_target": float(x_ref),
            "y_target": float(y_ref),
            # Force application points/disturbances (not active by default)
            "force_left": float(self.force_left),
            "force_right": float(self.force_right),
            "x_force_left": float(self.x_force_left),
            "x_force_right": float(self.x_force_right),
            # PID gains for logging
            "Kp": float(self.Kp),
            "Ki": float(self.Ki),
            "Kd": float(self.Kd),
        }

    # ----- Gymnasium API -----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        if self.random_start:
            self.theta = float(self.np_random.uniform(-0.3, 0.3))
            self.theta_dot = float(self.np_random.uniform(-0.1, 0.1))
        else:
            self.theta = 0.0
            self.theta_dot = 0.0
        self.theta_ddot = 0.0
        self.tau = 0.0

        # Reset PID memory
        self._int_e = 0.0
        self._prev_e = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        # Compute previous PID errors (before action)
        th_ref_before, _, _ = self._reference(self.step_count)
        e_before = float(th_ref_before - self.theta)
        int_before = float(self._int_e)
        de_before = float((e_before - self._prev_e) / self.dt)

        # Determine actuator torque based on control mode
        if self.control_mode == "forces":
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if self.action_includes_pid and arr.size >= 5:
                F_left_raw, F_right_raw, Kp, Ki, Kd = arr[:5]
                self.Kp = float(np.clip(Kp, self.pid_low[0], self.pid_high[0]))
                self.Ki = float(np.clip(Ki, self.pid_low[1], self.pid_high[1]))
                self.Kd = float(np.clip(Kd, self.pid_low[2], self.pid_high[2]))
            else:
                assert arr.size >= 2, "Forces control expects [F_left, F_right] (+ optional Kp,Ki,Kd)"
                F_left_raw, F_right_raw = arr[:2]
            F_left = float(np.clip(F_left_raw, 0.0, self.max_force))
            F_right = float(np.clip(F_right_raw, 0.0, self.max_force))
            self.force_left = F_left
            self.force_right = F_right
            tau = (F_right - F_left) * (self.actuator_offset * self.L)
            tau = float(np.clip(tau, -self.max_tau, self.max_tau))
        elif self.control_mode == "torque":
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if self.action_includes_pid and arr.size >= 4:
                tau_cmd, Kp, Ki, Kd = arr[:4]
                self.Kp = float(np.clip(Kp, self.pid_low[0], self.pid_high[0]))
                self.Ki = float(np.clip(Ki, self.pid_low[1], self.pid_high[1]))
                self.Kd = float(np.clip(Kd, self.pid_low[2], self.pid_high[2]))
            else:
                tau_cmd = arr[0]
            tau = float(np.clip(float(tau_cmd), -self.max_tau, self.max_tau))
            self.force_left = 0.0
            self.force_right = 0.0
        else:
            # PID control: action = [Kp, Ki, Kd]
            gains = np.asarray(action, dtype=np.float32).reshape(-1)
            assert gains.size == 3, "PID control expects 3 parameters [Kp, Ki, Kd]"
            Kp, Ki, Kd = np.clip(gains, self.pid_low, self.pid_high)
            self.Kp, self.Ki, self.Kd = float(Kp), float(Ki), float(Kd)

            # Compute PID torque using pre-step errors
            int_tmp = int_before + e_before * self.dt
            tau = float(np.clip(self.Kp * e_before + self.Ki * int_tmp + self.Kd * de_before, -self.max_tau, self.max_tau))
            self.force_left = 0.0
            self.force_right = 0.0

        # Integrate dynamics
        self.tau = tau
        self.last_u = tau
        theta, theta_dot, theta_ddot = self._dynamics(tau)
        self.theta, self.theta_dot, self.theta_ddot = theta, theta_dot, theta_ddot
        self.step_count += 1

        # Update PID errors after step
        th_ref_after, x_ref, y_ref = self._reference(self.step_count)
        e_after = float(th_ref_after - self.theta)
        de_after = float((e_after - e_before) / self.dt)
        int_after = float(int_before + 0.5 * (e_before + e_after) * self.dt)
        # Commit integrator and last error
        self._int_e = int_after
        self._prev_e = e_after

        # Reward: improvement in absolute PID errors
        w_p, w_i, w_d = 1.0, 0.01, 0.1
        improv_p = abs(e_before) - abs(e_after)
        improv_i = abs(int_before) - abs(int_after)
        improv_d = abs(de_before) - abs(de_after)
        reward = float(w_p * improv_p + w_i * improv_i + w_d * improv_d)
        # Small effort penalty
        w_eff = 0.001
        reward -= float(w_eff * ((self.force_left / (self.max_force + 1e-8)) ** 2 + (self.force_right / (self.max_force + 1e-8)) ** 2))

        # Episode termination/truncation
        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            if getattr(self, "spec", None) is not None:
                gym.logger.warn(
                    "You are calling render method without specifying any render mode. "
                    "You can specify the render_mode at initialization, "
                    f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
                )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        # Initialize display/surfaces
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        # World -> screen mapping
        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset_x = self.screen_dim // 2
        offset_y = int(self.screen_dim * self.pivot_y_frac)

        # Draw arm as a rotated rectangle (rod)
        rod_length = self.L * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        angle = self.theta + np.pi / 2
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(angle)
            c = (c[0] + offset_x, c[1] + offset_y)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        # Base axle
        gfxdraw.aacircle(self.surf, offset_x, offset_y, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset_x, offset_y, int(rod_width / 2), (204, 77, 77))

        # End-effector point
        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(angle)
        rod_end = (int(rod_end[0] + offset_x), int(rod_end[1] + offset_y))
        gfxdraw.aacircle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))

        # Target end-effector marker
        th_ref, _, _ = self._reference(self.step_count)
        tgt_end = pygame.math.Vector2((rod_length, 0)).rotate_rad(th_ref + np.pi / 2)
        tgt_end = (int(tgt_end[0] + offset_x), int(tgt_end[1] + offset_y))
        gfxdraw.aacircle(self.surf, tgt_end[0], tgt_end[1], max(2, int(0.07 * scale)), (0, 180, 0))
        gfxdraw.filled_circle(self.surf, tgt_end[0], tgt_end[1], max(2, int(0.07 * scale)), (0, 180, 0))

        # Torque indicator (image if available; otherwise draw magnitude circle)
        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        torque_drawn = False
        try:
            img = None
            if path.exists(fname):
                img = pygame.image.load(fname).convert_alpha()
            if img is not None and self.last_u is not None:
                size = max(1.0, float(scale * min(abs(self.last_u) / self.max_tau, 1.0) * 0.8))
                size = int(size)
                if size > 0:
                    scale_img = pygame.transform.smoothscale(img, (size, size))
                    is_flip = bool(self.last_u > 0)
                    scale_img = pygame.transform.flip(scale_img, is_flip, True)
                    self.surf.blit(
                        scale_img,
                        (
                            offset_x - scale_img.get_rect().centerx,
                            offset_y - scale_img.get_rect().centery,
                        ),
                    )
                    torque_drawn = True
        except Exception:
            torque_drawn = False

        if not torque_drawn and self.last_u is not None:
            # Fallback: draw a circle around the axle with radius representing torque magnitude
            radius = int(0.1 * scale + 0.3 * scale * min(abs(self.last_u) / (self.max_tau + 1e-8), 1.0))
            color = (220, 80, 80) if self.last_u >= 0 else (80, 120, 220)
            gfxdraw.aacircle(self.surf, offset_x, offset_y, max(2, radius), color)

        # Axle center (on top)
        gfxdraw.aacircle(self.surf, offset_x, offset_y, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset_x, offset_y, int(0.05 * scale), (0, 0, 0))

        # Blit to screen
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:  # rgb_array
            import numpy as _np  # avoid shadowing top-level np
            return _np.transpose(_np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
