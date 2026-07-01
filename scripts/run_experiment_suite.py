"""Run the full AEGIS-VIO synthetic experiment suite.

This script executes the synthetic simulator over multiple degradation scenarios
and random seeds, then exports:

- per-run CSV traces under results/experiment_suite/runs/
- summary_metrics.csv with aggregate statistics
- summary_report.md with a presentation-ready experiment summary

Example:
    python scripts/run_experiment_suite.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

from scripts.run_simulation import ModePolicy, plot_results, simulate, write_csv


SCENARIOS = ["nominal", "blur", "low_light", "feature_poor", "mixed"]


def summarize_rows(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    """Compute report-level metrics for one simulation run."""
    risks = [float(row["risk"]) for row in rows]
    errors = [float(row["position_error_m"]) for row in rows]
    qualities = [float(row["visual_quality"]) for row in rows]
    modes = Counter(str(row["mode"]) for row in rows)
    total = len(rows)

    high_risk_steps = sum(risk >= 0.65 for risk in risks)
    halted_steps = modes.get("HALT", 0)
    recovery_steps = modes.get("RECOVERY", 0)
    cautious_steps = modes.get("CAUTIOUS", 0)

    return {
        "steps": total,
        "mean_risk": mean(risks),
        "max_risk": max(risks),
        "mean_position_error_m": mean(errors),
        "max_position_error_m": max(errors),
        "mean_visual_quality": mean(qualities),
        "high_risk_ratio": high_risk_steps / total,
        "cautious_ratio": cautious_steps / total,
        "recovery_ratio": recovery_steps / total,
        "halt_ratio": halted_steps / total,
    }


def aggregate(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def write_summary_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        raise ValueError("no summary rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(summary_rows: list[dict[str, float | str]], path: Path) -> None:
    grouped: dict[str, list[dict[str, float | str]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row["scenario"]), []).append(row)

    lines = [
        "# AEGIS-VIO Synthetic Experiment Suite",
        "",
        "This report summarizes repeated synthetic stress tests for uncertainty-aware visual-inertial navigation.",
        "Each scenario is evaluated across multiple random seeds and reports localization error, risk response, and navigation-mode ratios.",
        "",
        "## Scenarios",
        "",
        "- `nominal`: high visual quality baseline.",
        "- `blur`: temporary visual degradation caused by motion blur.",
        "- `low_light`: reduced visual evidence from poor illumination.",
        "- `feature_poor`: texture-poor scene with weak feature tracking.",
        "- `mixed`: combined degradation stress test.",
        "",
        "## Aggregate results",
        "",
        "| Scenario | Mean risk | Max risk | Mean error [m] | High-risk ratio | Cautious | Recovery | Halt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for scenario in SCENARIOS:
        rows = grouped.get(scenario, [])
        if not rows:
            continue
        mean_risk, mean_risk_sd = aggregate(float(row["mean_risk"]) for row in rows)
        max_risk, _ = aggregate(float(row["max_risk"]) for row in rows)
        mean_error, mean_error_sd = aggregate(float(row["mean_position_error_m"]) for row in rows)
        high_risk, _ = aggregate(float(row["high_risk_ratio"]) for row in rows)
        cautious, _ = aggregate(float(row["cautious_ratio"]) for row in rows)
        recovery, _ = aggregate(float(row["recovery_ratio"]) for row in rows)
        halt, _ = aggregate(float(row["halt_ratio"]) for row in rows)
        lines.append(
            f"| {scenario} | {mean_risk:.3f} ± {mean_risk_sd:.3f} | {max_risk:.3f} | "
            f"{mean_error:.3f} ± {mean_error_sd:.3f} | {high_risk:.2%} | "
            f"{cautious:.2%} | {recovery:.2%} | {halt:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "A robust uncertainty-aware system should keep risk low in nominal conditions and increase caution/recovery behavior when visual quality degrades.",
            "For presentation, show the `mixed` scenario plots first because they demonstrate the complete risk-response loop.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python scripts/run_experiment_suite.py --seeds 0 1 2 3 4",
            "```",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiment_suite"))
    parser.add_argument("--plots", action="store_true", help="Also export plots for each run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str]] = []
    policy = ModePolicy()

    for scenario in args.scenarios:
        for seed in args.seeds:
            run_name = f"{scenario}_seed_{seed}"
            run_dir = runs_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            rows = simulate(
                steps=args.steps,
                dt=args.dt,
                scenario=scenario,
                seed=seed,
                policy=policy,
            )
            write_csv(rows, run_dir / "trace.csv")
            if args.plots:
                plot_results(rows, run_dir)

            metrics = summarize_rows(rows)
            summary_rows.append({"scenario": scenario, "seed": seed, **metrics})

    write_summary_csv(summary_rows, args.output_dir / "summary_metrics.csv")
    write_markdown_report(summary_rows, args.output_dir / "summary_report.md")

    print("AEGIS-VIO experiment suite complete")
    print(f"scenarios: {', '.join(args.scenarios)}")
    print(f"seeds:     {args.seeds}")
    print(f"summary:   {args.output_dir / 'summary_metrics.csv'}")
    print(f"report:    {args.output_dir / 'summary_report.md'}")


if __name__ == "__main__":
    main()
