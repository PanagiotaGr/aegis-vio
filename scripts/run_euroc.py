"""Run the first EuRoC loading check for AegisVIO.

Example:
    python scripts/run_euroc.py --sequence datasets/euroc/MH_01_easy
"""

from __future__ import annotations

import argparse

from src.dataset_loader import EuRoCDatasetLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, help="Path to EuRoC sequence folder containing mav0/.")
    args = parser.parse_args()

    loader = EuRoCDatasetLoader(args.sequence)
    summary = loader.summary()
    print("AegisVIO EuRoC loading check")
    print(f"cam0 frames:      {summary.cam0_frames}")
    print(f"cam1 frames:      {summary.cam1_frames}")
    print(f"IMU packets:      {summary.imu_packets}")
    print(f"ground truth:     {summary.has_ground_truth}")


if __name__ == "__main__":
    main()
