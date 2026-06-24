"""Uncertainty metrics for state-estimation covariance matrices."""

from __future__ import annotations

import numpy as np


def _as_square_matrix(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    return covariance


def covariance_trace(covariance: np.ndarray) -> float:
    """Return trace(P), the sum of marginal variances."""
    covariance = _as_square_matrix(covariance)
    return float(np.trace(covariance))


def covariance_determinant(covariance: np.ndarray, jitter: float = 1e-12) -> float:
    """Return det(P), with a small diagonal jitter for numerical stability."""
    covariance = _as_square_matrix(covariance)
    stable = covariance + jitter * np.eye(covariance.shape[0])
    return float(np.linalg.det(stable))


def gaussian_entropy(covariance: np.ndarray, jitter: float = 1e-12) -> float:
    """Differential entropy of an n-dimensional Gaussian with covariance P."""
    covariance = _as_square_matrix(covariance)
    n = covariance.shape[0]
    det = covariance_determinant(covariance, jitter=jitter)
    return float(0.5 * np.log(((2.0 * np.pi * np.e) ** n) * det))


def mahalanobis_distance_squared(error: np.ndarray, covariance: np.ndarray, jitter: float = 1e-9) -> float:
    """Return e^T P^-1 e."""
    covariance = _as_square_matrix(covariance)
    error = np.asarray(error, dtype=float).reshape(-1)
    if error.shape[0] != covariance.shape[0]:
        raise ValueError("error vector dimension must match covariance size")
    stable = covariance + jitter * np.eye(covariance.shape[0])
    return float(error.T @ np.linalg.solve(stable, error))


def position_covariance(full_covariance: np.ndarray) -> np.ndarray:
    """Extract the 3x3 position covariance block from a state covariance."""
    covariance = _as_square_matrix(full_covariance)
    if covariance.shape[0] < 3:
        raise ValueError("full covariance must contain at least a 3D position block")
    return covariance[:3, :3]


def uncertainty_score(position_cov: np.ndarray, trace_weight: float = 1.0, entropy_weight: float = 0.1) -> float:
    """Scalar risk proxy from position covariance.

    The score is intentionally simple for the first research milestone.
    Later, this can be replaced by learned or task-aware risk functions.
    """
    p_cov = _as_square_matrix(position_cov)
    return trace_weight * covariance_trace(p_cov) + entropy_weight * gaussian_entropy(p_cov)
