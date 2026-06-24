"""Initial AegisVIO estimator scaffold."""

from __future__ import annotations

import numpy as np

from src.ekf import ExtendedKalmanFilter
from src.uncertainty import uncertainty_score


class AegisVIOEstimator:
    """Small estimator wrapper combining state, covariance, and uncertainty.

    This is not yet a full visual-inertial odometry system. It provides the
    stable interface that later modules and ROS2 nodes will use.
    """

    def __init__(self) -> None:
        self.filter = ExtendedKalmanFilter(state_dim=6, covariance_scale=0.1)

    @property
    def state(self) -> np.ndarray:
        return self.filter.x

    @property
    def covariance(self) -> np.ndarray:
        return self.filter.P

    def predict_constant_velocity(self, dt: float, process_noise: float = 1e-3) -> None:
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        Q = np.eye(6) * process_noise
        self.filter.predict(F, Q)

    def update_position(self, position_xyz: np.ndarray, measurement_noise: float = 0.05) -> None:
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        R = np.eye(3) * measurement_noise**2
        self.filter.update(position_xyz, H, R)

    def uncertainty_score(self, quality: float = 1.0) -> float:
        return uncertainty_score(self.filter.P[:3, :3], quality=quality)
