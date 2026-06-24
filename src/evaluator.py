"""Evaluation metrics for AegisVIO."""

from __future__ import annotations

import numpy as np


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def position_errors(estimated_xyz: np.ndarray, ground_truth_xyz: np.ndarray) -> np.ndarray:
    estimated_xyz = np.asarray(estimated_xyz, dtype=float)
    ground_truth_xyz = np.asarray(ground_truth_xyz, dtype=float)
    if estimated_xyz.shape != ground_truth_xyz.shape:
        raise ValueError("estimated and ground-truth trajectories must have the same shape")
    return np.linalg.norm(estimated_xyz - ground_truth_xyz, axis=1)


def ate_position_rmse(estimated_xyz: np.ndarray, ground_truth_xyz: np.ndarray) -> float:
    return rmse(position_errors(estimated_xyz, ground_truth_xyz))
