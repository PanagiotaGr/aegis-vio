import numpy as np

from src.evaluator import ate_position_rmse


def test_ate_position_rmse_zero_for_identical_trajectories():
    trajectory = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert ate_position_rmse(trajectory, trajectory) == 0.0
