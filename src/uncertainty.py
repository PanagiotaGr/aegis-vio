"""Uncertainty metrics for AegisVIO."""

from __future__ import annotations

import numpy as np


def covariance_trace(covariance: np.ndarray) -> float:
    covariance = np.asarray(covariance, dtype=float)
    return float(np.trace(covariance))


def covariance_entropy(covariance: np.ndarray, jitter: float = 1e-12) -> float:
    covariance = np.asarray(covariance, dtype=float)
    n = covariance.shape[0]
    stable = covariance + jitter * np.eye(n)
    det = max(float(np.linalg.det(stable)), jitter)
    return float(0.5 * np.log(((2.0 * np.pi * np.e) ** n) * det))


def mahalanobis_distance_squared(error: np.ndarray, covariance: np.ndarray, jitter: float = 1e-9) -> float:
    error = np.asarray(error, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    stable = covariance + jitter * np.eye(covariance.shape[0])
    return float(error.T @ np.linalg.solve(stable, error))


def uncertainty_score(covariance: np.ndarray, quality: float = 1.0) -> float:
    quality = max(float(quality), 1e-3)
    return covariance_trace(covariance) / quality
