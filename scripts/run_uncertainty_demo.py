"""Run a synthetic uncertainty-aware navigation demo.

This script produces CSV results and plots without requiring a robotics dataset.
It is the first reproducible experiment for the AegisVIO repository.

Usage:
    python scripts/run_uncertainty_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulation.synthetic_vio import compute_position_error, run_synthetic_vio


RESULTS_DIR = Path("results/synthetic_uncertainty_demo")
PLOTS_DIR = Path("plots/synthetic_uncertainty_demo")


def classify_modes(uncertainty_score: np.ndarray) -> list[str]:
    """Simple uncertainty-to-mode policy for the demo."""
    modes: list[str] = []
    for score in uncertainty_score:
        if score < 0.08:
            modes.append("nominal")
        elif score < 0.35:
            modes.append("cautious")
        else:
            modes.append("recovery")
    return modes


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    result = run_synthetic_vio()
    error = compute_position_error(result)
    modes = classify_modes(result.uncertainty_score)

    df = pd.DataFrame(
        {
            "timestamp": result.timestamps,
            "gt_x": result.ground_truth[:, 0],
            "gt_y": result.ground_truth[:, 1],
            "est_x": result.estimate[:, 0],
            "est_y": result.estimate[:, 1],
            "position_error": error,
            "covariance_trace": result.covariance_trace,
            "visual_quality": result.visual_quality,
            "uncertainty_score": result.uncertainty_score,
            "mode": modes,
        }
    )
    csv_path = RESULTS_DIR / "synthetic_uncertainty_results.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(result.ground_truth[:, 0], result.ground_truth[:, 1], label="Ground truth")
    plt.plot(result.estimate[:, 0], result.estimate[:, 1], label="Estimate")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Synthetic trajectory")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "trajectory.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(result.timestamps, error, label="Position error")
    plt.plot(result.timestamps, result.uncertainty_score, label="Uncertainty score")
    plt.xlabel("time [s]")
    plt.title("Error and uncertainty over time")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "error_uncertainty.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(result.timestamps, result.visual_quality, label="Visual quality")
    plt.xlabel("time [s]")
    plt.ylabel("quality")
    plt.title("Synthetic visual degradation profile")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "visual_quality.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Synthetic uncertainty demo completed.")
    print(f"Results: {csv_path}")
    print(f"Plots:   {PLOTS_DIR}")
    print(f"Mean error: {np.mean(error):.4f}")
    print(f"Max uncertainty score: {np.max(result.uncertainty_score):.4f}")


if __name__ == "__main__":
    main()
