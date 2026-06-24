"""Run ORB feature tracking on a EuRoC sequence.

Example:
    python scripts/run_feature_tracking.py --sequence datasets/MH_01_easy --max-pairs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.data.euroc_loader import EuRoCLoader
from src.frontend.orb_tracker import ORBTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ORB matching on EuRoC camera frames.")
    parser.add_argument("--sequence", type=Path, required=True, help="Path to EuRoC sequence root containing mav0/.")
    parser.add_argument("--max-pairs", type=int, default=20, help="Maximum number of consecutive image pairs to process.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/feature_tracks"), help="Directory for match visualizations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader = EuRoCLoader(args.sequence, load_images=True)
    print("Dataset summary:", loader.summary())

    tracker = ORBTracker()
    frames = list(loader.iter_camera("cam0"))

    for i in range(min(args.max_pairs, len(frames) - 1)):
        f1 = frames[i]
        f2 = frames[i + 1]
        assert f1.image is not None and f2.image is not None

        result = tracker.match(f1.image, f2.image)
        vis = tracker.draw_matches(f1.image, f2.image, result, inliers_only=True)

        output_path = args.output_dir / f"matches_{i:04d}.png"
        cv2.imwrite(str(output_path), vis)
        print(
            f"pair={i:04d} matches={result.num_matches:4d} "
            f"inliers={result.num_inliers:4d} output={output_path}"
        )


if __name__ == "__main__":
    main()
