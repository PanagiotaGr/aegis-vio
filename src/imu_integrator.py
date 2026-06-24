"""Simple IMU integration utilities for AegisVIO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IMUState:
    position: np.ndarray
    velocity: np.ndarray


class IMUIntegrator:
    """Minimal position/velocity propagation from acceleration.

    This is a lightweight starting point. Full quaternion preintegration can be
    added later for research-grade VIO.
    """

    def __init__(self, gravity: np.ndarray | None = None) -> None:
        self.gravity = np.array([0.0, 0.0, -9.81]) if gravity is None else gravity

    def propagate(self, state: IMUState, accel_world: np.ndarray, dt: float) -> IMUState:
        accel_world = np.asarray(accel_world, dtype=float)
        position = state.position + state.velocity * dt + 0.5 * accel_world * dt**2
        velocity = state.velocity + accel_world * dt
        return IMUState(position=position, velocity=velocity)
