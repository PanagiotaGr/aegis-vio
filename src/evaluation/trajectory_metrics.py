"""Trajectory evaluation metrics for AegisVIO.

The first version implements simple position-only ATE/RMSE utilities.
Full SE(3) alignment and RPE will be added after the VIO estimator is running.
"""

from __future__ import annotations

import numpy as np


def rmse(errors: np.ndarray) -> float:
    """Root mean square error over a vector or matrix of errors."""
    errors = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(np.square(errors))))


def position_errors(estimated_xyz: np.ndarray, ground_truth_xyz: np.ndarray) -> np.ndarray:
    """Euclidean position error per timestamp.

    Both arrays must have shape (N, 3) and be already synchronized/aligned.
    """
    estimated_xyz = np.asarray(estimated_xyz, dtype=float)
    ground_truth_xyz = np.asarray(ground_truth_xyz, dtype=float)
    if estimated_xyz.shape != ground_truth_xyz.shape:
        raise ValueError("estimated and ground-truth trajectories must have the same shape")
    if estimated_xyz.ndim != 2 or estimated_xyz.shape[1] != 3:
        raise ValueError("trajectories must have shape (N, 3)")
    return np.linalg.norm(estimated_xyz - ground_truth_xyz, axis=1)


def ate_rmse(estimated_xyz: np.ndarray, ground_truth_xyz: np.ndarray) -> float:
    """Position-only Absolute Trajectory Error RMSE."""
    return rmse(position_errors(estimated_xyz, ground_truth_xyz))


def align_by_translation(estimated_xyz: np.ndarray, ground_truth_xyz: np.ndarray) -> np.ndarray:
    """Simple translation alignment using the first pose.

    This is not a full Umeyama alignment. It is useful for early debugging.
    """
    estimated_xyz = np.asarray(estimated_xyz, dtype=float)
    ground_truth_xyz = np.asarray(ground_truth_xyz, dtype=float)
    offset = ground_truth_xyz[0] - estimated_xyz[0]
    return estimated_xyz + offset
