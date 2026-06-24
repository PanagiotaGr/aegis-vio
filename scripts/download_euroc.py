"""EuRoC MAV dataset setup helper.

This script intentionally does not commit or bundle datasets. It prints the
recommended local folder structure for AegisVIO experiments.
"""

from __future__ import annotations


def main() -> None:
    print("Download EuRoC MAV sequences from the official dataset page.")
    print("Recommended local path:")
    print("  datasets/euroc/MH_01_easy/mav0/...")
    print("Then run:")
    print("  python scripts/run_euroc.py --sequence datasets/euroc/MH_01_easy")


if __name__ == "__main__":
    main()
