"""Minimal Extended Kalman Filter utilities for AegisVIO."""

from __future__ import annotations

import numpy as np


class ExtendedKalmanFilter:
    """Small EKF core used as a foundation for later VIO experiments."""

    def __init__(self, state_dim: int, covariance_scale: float = 1.0) -> None:
        self.x = np.zeros(state_dim, dtype=float)
        self.P = np.eye(state_dim, dtype=float) * covariance_scale

    def predict(self, F: np.ndarray, Q: np.ndarray, u: np.ndarray | None = None, B: np.ndarray | None = None) -> None:
        if u is not None and B is not None:
            self.x = F @ self.x + B @ u
        else:
            self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        z = np.asarray(z, dtype=float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
