"""Unit tests for physics utilities."""
import numpy as np
from core.physics import PendulumDynamics, PendulumKinematics
from config.base_config import PhysicsConfig


def test_acceleration_signs():
    config = PhysicsConfig()
    dynamics = PendulumDynamics(config)
    accel = dynamics.compute_acceleration(angle=0.1, velocity=0.0, torque=0.0)
    assert np.isfinite(accel)


def test_forward_kinematics_range():
    kin = PendulumKinematics(arm_length=1.0)
    x, y = kin.get_end_effector_position(np.pi / 2)
    assert np.isclose(x, 1.0, atol=1e-6)
    assert np.isclose(y, 0.0, atol=1e-6)


def test_base_mass_changes_dynamics():
    """Adding base mass should introduce restoring acceleration from gravity."""
    cfg_with_base = PhysicsConfig(arm_mass=0.0, base_mass=1.0, moment_of_inertia=1.0, arm_length=1.0)
    cfg_without_base = PhysicsConfig(arm_mass=0.0, base_mass=0.0, moment_of_inertia=1.0, arm_length=1.0)
    dyn_with = PendulumDynamics(cfg_with_base)
    dyn_without = PendulumDynamics(cfg_without_base)

    accel_with = dyn_with.compute_acceleration(angle=0.2, velocity=0.0, torque=0.0)
    accel_without = dyn_without.compute_acceleration(angle=0.2, velocity=0.0, torque=0.0)

    assert accel_with < accel_without  # base mass creates restoring gravity torque
