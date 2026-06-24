"""Synthetic VIO uncertainty experiment.

This module creates a lightweight simulation that does not require the EuRoC dataset.
It is useful for validating the research idea behind AegisVIO:

    pose error grows under degraded perception,
    covariance should grow with it,
    and a controller can react to uncertainty.

The simulation is intentionally simple and interpretable. It is not a replacement
for a real VIO pipeline, but it gives the repository a runnable experiment from day one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticVIOResult:
    timestamps: np.ndarray
    ground_truth: np.ndarray
    estimate: np.ndarray
    covariance_trace: np.ndarray
    visual_quality: np.ndarray
    uncertainty_score: np.ndarray


def generate_ground_truth(num_steps: int = 600, dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Generate a smooth 2D robot trajectory with timestamps.

    The state is position only: [x, y].
    """
    t = np.arange(num_steps, dtype=float) * dt
    x = 0.04 * np.arange(num_steps) * dt * 20.0
    y = 1.5 * np.sin(0.35 * t) + 0.25 * np.sin(1.4 * t)
    return t, np.column_stack([x, y])


def visual_quality_profile(timestamps: np.ndarray) -> np.ndarray:
    """Create a synthetic perception-quality curve in [0, 1].

    Low values simulate blur, low texture, or low light.
    """
    quality = np.ones_like(timestamps)
    quality[(timestamps > 8.0) & (timestamps < 13.0)] = 0.35
    quality[(timestamps > 20.0) & (timestamps < 25.0)] = 0.18
    quality[(timestamps > 26.5) & (timestamps < 28.0)] = 0.08
    return quality


def run_synthetic_vio(
    num_steps: int = 600,
    dt: float = 0.05,
    process_noise: float = 0.002,
    nominal_measurement_noise: float = 0.02,
    seed: int = 7,
) -> SyntheticVIOResult:
    """Run a simple covariance-aware position estimation simulation.

    A constant-velocity predictor is corrected by noisy pseudo-visual measurements.
    When visual quality is poor, measurement noise increases and covariance grows.
    """
    rng = np.random.default_rng(seed)
    timestamps, gt = generate_ground_truth(num_steps=num_steps, dt=dt)
    quality = visual_quality_profile(timestamps)

    estimate = np.zeros_like(gt)
    estimate[0] = gt[0] + rng.normal(scale=0.03, size=2)

    covariance = np.eye(2) * 0.02
    covariance_trace = np.zeros(num_steps)
    uncertainty_score = np.zeros(num_steps)

    velocity = np.array([0.04 * 20.0 * dt, 0.0])
    q = np.eye(2) * process_noise

    for k in range(1, num_steps):
        # Predict.
        estimate[k] = estimate[k - 1] + velocity
        covariance = covariance + q

        # Quality-dependent visual update.
        measurement_std = nominal_measurement_noise / max(float(quality[k]), 0.05)
        z = gt[k] + rng.normal(scale=measurement_std, size=2)
        r = np.eye(2) * measurement_std**2

        innovation = z - estimate[k]
        s = covariance + r
        kalman_gain = covariance @ np.linalg.inv(s)
        estimate[k] = estimate[k] + kalman_gain @ innovation
        covariance = (np.eye(2) - kalman_gain) @ covariance

        # If perception is extremely poor, simulate unmodelled visual degradation.
        if quality[k] < 0.12:
            drift = rng.normal(loc=0.0, scale=0.04, size=2)
            estimate[k] += drift
            covariance += np.eye(2) * 0.04

        covariance_trace[k] = float(np.trace(covariance))
        uncertainty_score[k] = covariance_trace[k] / max(float(quality[k]), 0.05)

    covariance_trace[0] = float(np.trace(np.eye(2) * 0.02))
    uncertainty_score[0] = covariance_trace[0] / max(float(quality[0]), 0.05)

    return SyntheticVIOResult(
        timestamps=timestamps,
        ground_truth=gt,
        estimate=estimate,
        covariance_trace=covariance_trace,
        visual_quality=quality,
        uncertainty_score=uncertainty_score,
    )


def compute_position_error(result: SyntheticVIOResult) -> np.ndarray:
    """Euclidean position error over time."""
    return np.linalg.norm(result.estimate - result.ground_truth, axis=1)
