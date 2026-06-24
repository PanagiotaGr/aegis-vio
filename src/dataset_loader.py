"""Dataset loading utilities for AegisVIO.

The first supported dataset is EuRoC MAV. The loader keeps the interface
simple so the rest of the pipeline can be tested before adding full VIO.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetSummary:
    cam0_frames: int
    cam1_frames: int
    imu_packets: int
    has_ground_truth: bool


class EuRoCDatasetLoader:
    """Load EuRoC MAV sequence metadata, camera timestamps, IMU, and GT."""

    def __init__(self, sequence_path: str | Path) -> None:
        self.sequence_path = Path(sequence_path)
        self.mav0 = self.sequence_path / "mav0"
        if not self.mav0.exists():
            raise FileNotFoundError(f"Expected EuRoC folder with mav0/: {self.sequence_path}")

    def load_camera_csv(self, camera: str = "cam0") -> pd.DataFrame:
        csv_path = self.mav0 / camera / "data.csv"
        image_dir = self.mav0 / camera / "data"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing camera CSV: {csv_path}")
        df = pd.read_csv(csv_path, comment="#", header=None, names=["timestamp_ns", "filename"])
        df["image_path"] = df["filename"].apply(lambda name: image_dir / str(name))
        return df

    def load_imu(self) -> pd.DataFrame:
        csv_path = self.mav0 / "imu0" / "data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing IMU CSV: {csv_path}")
        names = ["timestamp_ns", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"]
        return pd.read_csv(csv_path, comment="#", header=None, names=names)

    def load_ground_truth(self) -> pd.DataFrame | None:
        csv_path = self.mav0 / "state_groundtruth_estimate0" / "data.csv"
        if not csv_path.exists():
            return None
        names = [
            "timestamp_ns", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z",
            "v_x", "v_y", "v_z", "b_w_x", "b_w_y", "b_w_z", "b_a_x", "b_a_y", "b_a_z",
        ]
        return pd.read_csv(csv_path, comment="#", header=None, names=names)

    def summary(self) -> DatasetSummary:
        cam0 = self.load_camera_csv("cam0")
        cam1_path = self.mav0 / "cam1" / "data.csv"
        cam1_frames = len(self.load_camera_csv("cam1")) if cam1_path.exists() else 0
        imu = self.load_imu()
        gt = self.load_ground_truth()
        return DatasetSummary(len(cam0), cam1_frames, len(imu), gt is not None)
