"""Navigation utilities for AegisVIO."""

from __future__ import annotations

import numpy as np


def compute_velocity_command(current_xy: np.ndarray, target_xy: np.ndarray, max_speed: float = 1.0) -> np.ndarray:
    current_xy = np.asarray(current_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    direction = target_xy - current_xy
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.zeros_like(direction)
    return max_speed * direction / norm


def apply_speed_scale(command: np.ndarray, speed_scale: float) -> np.ndarray:
    return np.asarray(command, dtype=float) * float(speed_scale)
