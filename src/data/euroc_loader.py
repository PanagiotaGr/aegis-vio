"""EuRoC MAV dataset loader for AegisVIO.

This module loads the standard EuRoC MAV folder layout:

mav0/
  cam0/data.csv
  cam0/data/*.png
  cam1/data.csv
  cam1/data/*.png
  imu0/data.csv
  state_groundtruth_estimate0/data.csv

The first milestone is deliberately simple: reliable data loading,
timestamp parsing, and iteration over camera frames with nearby IMU packets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CameraFrame:
    """One camera frame with timestamp in nanoseconds."""

    timestamp_ns: int
    image_path: Path
    image: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ImuPacket:
    """One IMU measurement packet."""

    timestamp_ns: int
    gyro: np.ndarray  # rad/s, shape (3,)
    accel: np.ndarray  # m/s^2, shape (3,)


@dataclass(frozen=True)
class GroundTruthState:
    """Ground-truth state from EuRoC, when available."""

    timestamp_ns: int
    position: np.ndarray  # shape (3,)
    quaternion: np.ndarray  # w, x, y, z, shape (4,)
    velocity: np.ndarray  # shape (3,)
    gyro_bias: np.ndarray  # shape (3,)
    accel_bias: np.ndarray  # shape (3,)


class EuRoCLoader:
    """Utility class for loading and iterating through EuRoC MAV sequences."""

    def __init__(self, sequence_root: str | Path, load_images: bool = False) -> None:
        self.sequence_root = Path(sequence_root)
        self.mav0 = self.sequence_root / "mav0"
        self.load_images = load_images

        if not self.mav0.exists():
            raise FileNotFoundError(
                f"Expected EuRoC sequence root containing mav0/: {self.sequence_root}"
            )

        self.cam0 = self._load_camera_index("cam0")
        self.cam1 = self._load_camera_index("cam1") if (self.mav0 / "cam1").exists() else None
        self.imu = self._load_imu()
        self.ground_truth = self._load_ground_truth()

    def _load_camera_index(self, camera_name: str) -> pd.DataFrame:
        csv_path = self.mav0 / camera_name / "data.csv"
        image_dir = self.mav0 / camera_name / "data"

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing camera CSV: {csv_path}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing camera data directory: {image_dir}")

        df = pd.read_csv(csv_path, comment="#", header=None, names=["timestamp_ns", "filename"])
        df["timestamp_ns"] = df["timestamp_ns"].astype(np.int64)
        df["image_path"] = df["filename"].apply(lambda name: image_dir / str(name))
        return df

    def _load_imu(self) -> pd.DataFrame:
        csv_path = self.mav0 / "imu0" / "data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing IMU CSV: {csv_path}")

        names = [
            "timestamp_ns",
            "w_x", "w_y", "w_z",
            "a_x", "a_y", "a_z",
        ]
        df = pd.read_csv(csv_path, comment="#", header=None, names=names)
        df["timestamp_ns"] = df["timestamp_ns"].astype(np.int64)
        return df

    def _load_ground_truth(self) -> Optional[pd.DataFrame]:
        csv_path = self.mav0 / "state_groundtruth_estimate0" / "data.csv"
        if not csv_path.exists():
            return None

        names = [
            "timestamp_ns",
            "p_x", "p_y", "p_z",
            "q_w", "q_x", "q_y", "q_z",
            "v_x", "v_y", "v_z",
            "b_w_x", "b_w_y", "b_w_z",
            "b_a_x", "b_a_y", "b_a_z",
        ]
        df = pd.read_csv(csv_path, comment="#", header=None, names=names)
        df["timestamp_ns"] = df["timestamp_ns"].astype(np.int64)
        return df

    def iter_camera(self, camera_name: str = "cam0") -> Iterator[CameraFrame]:
        df = self.cam0 if camera_name == "cam0" else self.cam1
        if df is None:
            raise ValueError(f"Camera {camera_name} is not available in this sequence.")

        for row in df.itertuples(index=False):
            image = None
            if self.load_images:
                image = cv2.imread(str(row.image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(f"Could not read image: {row.image_path}")
            yield CameraFrame(int(row.timestamp_ns), Path(row.image_path), image)

    def imu_between(self, start_ns: int, end_ns: int) -> list[ImuPacket]:
        mask = (self.imu["timestamp_ns"] >= start_ns) & (self.imu["timestamp_ns"] < end_ns)
        packets: list[ImuPacket] = []
        for row in self.imu.loc[mask].itertuples(index=False):
            packets.append(
                ImuPacket(
                    timestamp_ns=int(row.timestamp_ns),
                    gyro=np.array([row.w_x, row.w_y, row.w_z], dtype=float),
                    accel=np.array([row.a_x, row.a_y, row.a_z], dtype=float),
                )
            )
        return packets

    def summary(self) -> dict[str, int | bool]:
        return {
            "cam0_frames": int(len(self.cam0)),
            "cam1_frames": int(len(self.cam1)) if self.cam1 is not None else 0,
            "imu_packets": int(len(self.imu)),
            "has_ground_truth": self.ground_truth is not None,
            "ground_truth_states": int(len(self.ground_truth)) if self.ground_truth is not None else 0,
        }
