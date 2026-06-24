"""Evaluate AegisVIO trajectory CSV outputs.

Expected CSV columns: est_x, est_y, est_z, gt_x, gt_y, gt_z
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.evaluator import ate_position_rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV file with estimated and ground-truth trajectory columns.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    estimated = df[["est_x", "est_y", "est_z"]].to_numpy()
    ground_truth = df[["gt_x", "gt_y", "gt_z"]].to_numpy()
    print(f"ATE position RMSE: {ate_position_rmse(estimated, ground_truth):.6f} m")


if __name__ == "__main__":
    main()
