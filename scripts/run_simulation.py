"""Run a synthetic AEGIS-VIO uncertainty-aware navigation simulation.

The simulation does not require a real dataset. It generates a 2-D reference
trajectory, injects odometry drift and environment-dependent visual quality
loss, converts covariance into a risk score, and maps risk to simple navigation
modes.

Example:
    python scripts/run_simulation.py --scenario mixed --output_dir results/simulation
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from src.uncertainty import covariance_entropy, uncertainty_score


@dataclass(frozen=True)
class ModePolicy:
    """Threshold policy that maps risk to robot behavior."""

    cautious: float = 0.35
    recovery: float = 0.65
    halt: float = 0.9

    def classify(self, risk: float) -> tuple[str, float, float]:
        """Return mode, speed scale, and safety-margin multiplier."""
        if risk >= self.halt:
            return "HALT", 0.0, 2.5
        if risk >= self.recovery:
            return "RECOVERY", 0.25, 2.0
        if risk >= self.cautious:
            return "CAUTIOUS", 0.6, 1.5
        return "NORMAL", 1.0, 1.0


def make_reference_trajectory(steps: int, dt: float) -> np.ndarray:
    """Create a smooth 2-D ground-truth trajectory."""
    t = np.arange(steps, dtype=float) * dt
    x = 0.08 * t
    y = 0.8 * np.sin(0.08 * t)
    return np.column_stack([x, y])


def visual_quality_profile(steps: int, scenario: str) -> np.ndarray:
    """Return quality in [0, 1], where lower means harder visual tracking."""
    quality = np.full(steps, 0.95, dtype=float)

    if scenario in {"blur", "mixed"}:
        quality[steps // 4 : steps // 4 + steps // 8] *= 0.45
    if scenario in {"low_light", "mixed"}:
        quality[steps // 2 : steps // 2 + steps // 7] *= 0.25
    if scenario in {"feature_poor", "mixed"}:
        quality[(2 * steps) // 3 : (2 * steps) // 3 + steps // 6] *= 0.18

    return np.clip(quality, 0.05, 1.0)


def simulate(
    steps: int,
    dt: float,
    scenario: str,
    seed: int,
    policy: ModePolicy,
) -> list[dict[str, float | str]]:
    """Run the synthetic uncertainty simulation."""
    rng = np.random.default_rng(seed)
    truth = make_reference_trajectory(steps, dt)
    quality = visual_quality_profile(steps, scenario)

    estimate = truth[0].copy()
    covariance = np.eye(2) * 0.02
    rows: list[dict[str, float | str]] = []

    max_expected_score = 2.8
    for k in range(1, steps):
        nominal_delta = truth[k] - truth[k - 1]
        process_noise = 0.004 + 0.025 * (1.0 - quality[k])
        drift = rng.normal(0.0, process_noise, size=2)
        estimate = estimate + nominal_delta + drift

        covariance += np.eye(2) * (0.002 + 0.04 * (1.0 - quality[k]))
        if quality[k] > 0.7:
            covariance *= 0.96

        score = uncertainty_score(covariance, quality=quality[k])
        risk = float(np.clip(score / max_expected_score, 0.0, 1.0))
        mode, speed_scale, margin_scale = policy.classify(risk)
        position_error = float(np.linalg.norm(estimate - truth[k]))

        rows.append(
            {
                "step": k,
                "time_s": k * dt,
                "truth_x": truth[k, 0],
                "truth_y": truth[k, 1],
                "estimate_x": estimate[0],
                "estimate_y": estimate[1],
                "visual_quality": float(quality[k]),
                "cov_trace": float(np.trace(covariance)),
                "cov_entropy": covariance_entropy(covariance),
                "risk": risk,
                "mode": mode,
                "speed_scale": speed_scale,
                "safety_margin_scale": margin_scale,
                "position_error_m": position_error,
            }
        )

    return rows


def write_csv(rows: Iterable[dict[str, float | str]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("simulation produced no rows")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    time_s = np.array([float(row["time_s"]) for row in rows])
    risk = np.array([float(row["risk"]) for row in rows])
    quality = np.array([float(row["visual_quality"]) for row in rows])
    error = np.array([float(row["position_error_m"]) for row in rows])
    truth_x = np.array([float(row["truth_x"]) for row in rows])
    truth_y = np.array([float(row["truth_y"]) for row in rows])
    estimate_x = np.array([float(row["estimate_x"]) for row in rows])
    estimate_y = np.array([float(row["estimate_y"]) for row in rows])

    plt.figure(figsize=(8, 4))
    plt.plot(time_s, risk, label="risk")
    plt.plot(time_s, quality, label="visual quality")
    plt.xlabel("time [s]")
    plt.ylabel("normalized value")
    plt.title("AEGIS-VIO synthetic risk response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "risk_quality.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(time_s, error)
    plt.xlabel("time [s]")
    plt.ylabel("position error [m]")
    plt.title("Synthetic localization error")
    plt.tight_layout()
    plt.savefig(output_dir / "position_error.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(truth_x, truth_y, label="ground truth")
    plt.plot(estimate_x, estimate_y, label="estimated")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Synthetic 2-D trajectory")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["nominal", "blur", "low_light", "feature_poor", "mixed"],
        default="mixed",
        help="Environment degradation scenario to simulate.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=Path, default=Path("results/simulation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = simulate(
        steps=args.steps,
        dt=args.dt,
        scenario=args.scenario,
        seed=args.seed,
        policy=ModePolicy(),
    )
    csv_path = args.output_dir / "simulation_results.csv"
    write_csv(rows, csv_path)
    plot_results(rows, args.output_dir)

    mode_counts: dict[str, int] = {}
    for row in rows:
        mode = str(row["mode"])
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    max_risk = max(float(row["risk"]) for row in rows)
    mean_error = float(np.mean([float(row["position_error_m"]) for row in rows]))

    print("AEGIS-VIO synthetic simulation complete")
    print(f"scenario:        {args.scenario}")
    print(f"rows:            {len(rows)}")
    print(f"max risk:        {max_risk:.3f}")
    print(f"mean error [m]:  {mean_error:.3f}")
    print(f"mode counts:     {mode_counts}")
    print(f"csv:             {csv_path}")
    print(f"plots:           {args.output_dir}")


if __name__ == "__main__":
    main()
