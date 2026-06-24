import numpy as np

from src.navigation import apply_speed_scale, compute_velocity_command
from src.uncertainty_aware_controller import NavigationMode, UncertaintyAwareController


def test_velocity_command_norm():
    cmd = compute_velocity_command(np.array([0.0, 0.0]), np.array([1.0, 0.0]), max_speed=0.5)
    assert np.isclose(np.linalg.norm(cmd), 0.5)


def test_speed_scale():
    cmd = apply_speed_scale(np.array([1.0, 0.0]), 0.5)
    assert np.allclose(cmd, np.array([0.5, 0.0]))


def test_controller_recovery_mode():
    controller = UncertaintyAwareController()
    assert controller.update(1.0) == NavigationMode.RECOVERY
