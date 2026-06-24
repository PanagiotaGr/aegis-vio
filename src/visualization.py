"""Visualization utilities for AegisVIO."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(estimated_xyz: np.ndarray, ground_truth_xyz: np.ndarray | None = None, output_path: str | Path | None = None) -> None:
    estimated_xyz = np.asarray(estimated_xyz, dtype=float)
    plt.figure(figsize=(7, 6))
    plt.plot(estimated_xyz[:, 0], estimated_xyz[:, 1], label="Estimated")
    if ground_truth_xyz is not None:
        ground_truth_xyz = np.asarray(ground_truth_xyz, dtype=float)
        plt.plot(ground_truth_xyz[:, 0], ground_truth_xyz[:, 1], label="Ground truth")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_uncertainty(timestamps: np.ndarray, scores: np.ndarray, output_path: str | Path | None = None) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(timestamps, scores, label="Uncertainty score")
    plt.xlabel("time [s]")
    plt.ylabel("score")
    plt.title("Uncertainty over time")
    plt.grid(True)
    plt.legend()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
