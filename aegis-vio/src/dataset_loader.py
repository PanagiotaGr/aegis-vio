"""
EuRoC MAV Dataset Loader
Loads and parses the EuRoC MAV dataset including:
- Stereo camera images (cam0, cam1)
- IMU measurements (accelerometer, gyroscope)
- Ground truth poses
- Timestamps and synchronization
Reference: Burri et al., "The EuRoC micro aerial vehicle datasets", IJRR 2016
"""
import os
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Generator
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import yaml
@dataclass
class IMUMeasurement:
    """Single IMU measurement with timestamp."""
    timestamp: float  # nanoseconds
    gyroscope: np.ndarray  # rad/s [wx, wy, wz]
    accelerometer: np.ndarray  # m/s^2 [ax, ay, az]
    
    def __post_init__(self):
        self.gyroscope = np.asarray(self.gyroscope, dtype=np.float64)
        self.accelerometer = np.asarray(self.accelerometer, dtype=np.float64)
@dataclass
class CameraFrame:
    """Single camera frame with metadata."""
    timestamp: float  # nanoseconds
    image: np.ndarray  # grayscale or color image
    camera_id: int  # 0 for left, 1 for right
    filepath: str  # original file path
@dataclass
class GroundTruthPose:
    """Ground truth pose with full state."""
    timestamp: float  # nanoseconds
    position: np.ndarray  # [x, y, z] in world frame
    orientation: np.ndarray  # quaternion [qw, qx, qy, qz]
    velocity: np.ndarray  # [vx, vy, vz] in world frame
    angular_velocity: np.ndarray  # [wx, wy, wz] in body frame
    acceleration: np.ndarray  # [ax, ay, az] in body frame
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        self.acceleration = np.asarray(self.acceleration, dtype=np.float64)
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix from quaternion."""
        r = Rotation.from_quat([
            self.orientation[1],  # qx
            self.orientation[2],  # qy
            self.orientation[3],  # qz
            self.orientation[0],  # qw (scalar last for scipy)
        ])
        return r.as_matrix()
    
    @property
    def transformation_matrix(self) -> np.ndarray:
        """Get 4x4 homogeneous transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.position
        return T
@dataclass
class CameraCalibration:
    """Camera intrinsic and extrinsic calibration."""
    camera_id: int
    image_width: int
    image_height: int
    intrinsics: np.ndarray  # [fx, fy, cx, cy]
    distortion_model: str
    distortion_coeffs: np.ndarray
    T_BS: np.ndarray  # 4x4 transformation from sensor to body frame
    rate_hz: float
    
    @property
    def camera_matrix(self) -> np.ndarray:
        """Get 3x3 camera matrix K."""
        fx, fy, cx, cy = self.intrinsics
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
@dataclass
class IMUCalibration:
    """IMU calibration parameters."""
    gyroscope_noise_density: float  # rad/s/sqrt(Hz)
    gyroscope_random_walk: float  # rad/s^2/sqrt(Hz)
    accelerometer_noise_density: float  # m/s^2/sqrt(Hz)
    accelerometer_random_walk: float  # m/s^3/sqrt(Hz)
    rate_hz: float
    T_BS: np.ndarray  # 4x4 transformation from sensor to body frame
@dataclass
class EuRoCSequence:
    """Complete EuRoC sequence data."""
    name: str
    path: Path
    imu_measurements: List[IMUMeasurement] = field(default_factory=list)
    camera_frames: Dict[int, List[CameraFrame]] = field(default_factory=dict)
    ground_truth: List[GroundTruthPose] = field(default_factory=list)
    camera_calibrations: Dict[int, CameraCalibration] = field(default_factory=dict)
    imu_calibration: Optional[IMUCalibration] = None
    
class EuRoCDatasetLoader:
    """
    Complete loader for the EuRoC MAV dataset.
    
    Supports all machine hall (MH) and Vicon room (V1, V2) sequences.
    Handles stereo images, IMU data, and ground truth poses.
    
    Example usage:
        loader = EuRoCDatasetLoader("/path/to/euroc")
        sequence = loader.load_sequence("MH_01_easy")
        
        for imu in sequence.imu_measurements:
            process_imu(imu)
            
        for frame in loader.iterate_camera_frames(sequence, camera_id=0):
            process_frame(frame)
    """
    
    # Available sequences in EuRoC dataset
    SEQUENCES = {
        "MH_01_easy": "Machine Hall 01 - Easy",
        "MH_02_easy": "Machine Hall 02 - Easy",
        "MH_03_medium": "Machine Hall 03 - Medium",
        "MH_04_difficult": "Machine Hall 04 - Difficult",
        "MH_05_difficult": "Machine Hall 05 - Difficult",
        "V1_01_easy": "Vicon Room 1 01 - Easy",
        "V1_02_medium": "Vicon Room 1 02 - Medium",
        "V1_03_difficult": "Vicon Room 1 03 - Difficult",
        "V2_01_easy": "Vicon Room 2 01 - Easy",
        "V2_02_medium": "Vicon Room 2 02 - Medium",
        "V2_03_difficult": "Vicon Room 2 03 - Difficult",
    }
    
    def __init__(self, dataset_root: str, load_images: bool = True):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_root: Path to the EuRoC dataset root directory
            load_images: If True, load images into memory; if False, load on demand
        """
        self.dataset_root = Path(dataset_root)
        self.load_images = load_images
        
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
    
    def list_available_sequences(self) -> List[str]:
        """List all available sequences in the dataset directory."""
        available = []
        for seq_name in self.SEQUENCES:
            seq_path = self.dataset_root / seq_name
            if seq_path.exists():
                available.append(seq_name)
        return available
    
    def load_sequence(self, sequence_name: str) -> EuRoCSequence:
        """
        Load a complete EuRoC sequence.
        
        Args:
            sequence_name: Name of the sequence (e.g., "MH_01_easy")
            
        Returns:
            EuRoCSequence containing all data
        """
        seq_path = self.dataset_root / sequence_name
        mav_path = seq_path / "mav0"
        
        if not seq_path.exists():
            raise FileNotFoundError(f"Sequence not found: {seq_path}")
        
        sequence = EuRoCSequence(name=sequence_name, path=seq_path)
        
        # Load calibrations first
        sequence.camera_calibrations = self._load_camera_calibrations(mav_path)
        sequence.imu_calibration = self._load_imu_calibration(mav_path)
        
        # Load sensor data
        sequence.imu_measurements = self._load_imu_data(mav_path)
        sequence.ground_truth = self._load_ground_truth(mav_path)
        
        # Load camera frames (metadata, images loaded on demand if load_images=False)
        sequence.camera_frames = {
            0: self._load_camera_data(mav_path, camera_id=0),
            1: self._load_camera_data(mav_path, camera_id=1),
        }
        
        return sequence
    
    def _load_camera_calibrations(self, mav_path: Path) -> Dict[int, CameraCalibration]:
        """Load camera calibration from sensor.yaml files."""
        calibrations = {}
        
        for cam_id in [0, 1]:
            sensor_yaml = mav_path / f"cam{cam_id}" / "sensor.yaml"
            
            if not sensor_yaml.exists():
                continue
                
            with open(sensor_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            # Parse intrinsics
            intrinsics = np.array(data['intrinsics'], dtype=np.float64)
            
            # Parse distortion
            distortion_model = data.get('distortion_model', 'radtan')
            distortion_coeffs = np.array(
                data.get('distortion_coefficients', [0, 0, 0, 0]),
                dtype=np.float64
            )
            
            # Parse extrinsics (T_BS: transformation from sensor to body)
            T_BS_data = data.get('T_BS', {}).get('data', np.eye(4).flatten().tolist())
            T_BS = np.array(T_BS_data, dtype=np.float64).reshape(4, 4)
            
            calibrations[cam_id] = CameraCalibration(
                camera_id=cam_id,
                image_width=data.get('resolution', [752, 480])[0],
                image_height=data.get('resolution', [752, 480])[1],
                intrinsics=intrinsics,
                distortion_model=distortion_model,
                distortion_coeffs=distortion_coeffs,
                T_BS=T_BS,
                rate_hz=data.get('rate_hz', 20.0),
            )
        
        return calibrations
    
    def _load_imu_calibration(self, mav_path: Path) -> Optional[IMUCalibration]:
        """Load IMU calibration from sensor.yaml."""
        sensor_yaml = mav_path / "imu0" / "sensor.yaml"
        
        if not sensor_yaml.exists():
            # Return default calibration for ADIS16448 (EuRoC IMU)
            return IMUCalibration(
                gyroscope_noise_density=1.6968e-4,
                gyroscope_random_walk=1.9393e-5,
                accelerometer_noise_density=2.0e-3,
                accelerometer_random_walk=3.0e-3,
                rate_hz=200.0,
                T_BS=np.eye(4),
            )
        
        with open(sensor_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        T_BS_data = data.get('T_BS', {}).get('data', np.eye(4).flatten().tolist())
        T_BS = np.array(T_BS_data, dtype=np.float64).reshape(4, 4)
        
        return IMUCalibration(
            gyroscope_noise_density=data.get('gyroscope_noise_density', 1.6968e-4),
            gyroscope_random_walk=data.get('gyroscope_random_walk', 1.9393e-5),
            accelerometer_noise_density=data.get('accelerometer_noise_density', 2.0e-3),
            accelerometer_random_walk=data.get('accelerometer_random_walk', 3.0e-3),
            rate_hz=data.get('rate_hz', 200.0),
            T_BS=T_BS,
        )
    
    def _load_imu_data(self, mav_path: Path) -> List[IMUMeasurement]:
        """Load all IMU measurements from data.csv."""
        imu_csv = mav_path / "imu0" / "data.csv"
        
        if not imu_csv.exists():
            raise FileNotFoundError(f"IMU data not found: {imu_csv}")
        
        measurements = []
        
        with open(imu_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                if len(row) < 7:
                    continue
                    
                timestamp = float(row[0])
                gyro = np.array([float(row[1]), float(row[2]), float(row[3])])
                accel = np.array([float(row[4]), float(row[5]), float(row[6])])
                
                measurements.append(IMUMeasurement(
                    timestamp=timestamp,
                    gyroscope=gyro,
                    accelerometer=accel,
                ))
        
        return measurements
    
    def _load_camera_data(self, mav_path: Path, camera_id: int) -> List[CameraFrame]:
        """Load camera frame metadata (and optionally images)."""
        cam_path = mav_path / f"cam{camera_id}"
        data_csv = cam_path / "data.csv"
        data_dir = cam_path / "data"
        
        if not data_csv.exists():
            raise FileNotFoundError(f"Camera data not found: {data_csv}")
        
        frames = []
        
        with open(data_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                if len(row) < 2:
                    continue
                
                timestamp = float(row[0])
                filename = row[1].strip()
                filepath = str(data_dir / filename)
                
                if self.load_images:
                    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                else:
                    image = None
                
                frames.append(CameraFrame(
                    timestamp=timestamp,
                    image=image,
                    camera_id=camera_id,
                    filepath=filepath,
                ))
        
        return frames
    
    def _load_ground_truth(self, mav_path: Path) -> List[GroundTruthPose]:
        """Load ground truth poses from state_groundtruth_estimate0/data.csv."""
        # Try different possible ground truth file locations
        possible_paths = [
            mav_path / "state_groundtruth_estimate0" / "data.csv",
            mav_path / "ground_truth" / "data.csv",
            mav_path / "mocap0" / "data.csv",
        ]
        
        gt_csv = None
        for path in possible_paths:
            if path.exists():
                gt_csv = path
                break
        
        if gt_csv is None:
            print("Warning: Ground truth file not found")
            return []
        
        poses = []
        
        with open(gt_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                if len(row) < 17:
                    continue
                
                timestamp = float(row[0])
                position = np.array([float(row[1]), float(row[2]), float(row[3])])
                orientation = np.array([
                    float(row[4]),  # qw
                    float(row[5]),  # qx
                    float(row[6]),  # qy
                    float(row[7]),  # qz
                ])
                velocity = np.array([float(row[8]), float(row[9]), float(row[10])])
                
                # Angular velocity and acceleration (body frame)
                angular_vel = np.array([float(row[11]), float(row[12]), float(row[13])])
                acceleration = np.array([float(row[14]), float(row[15]), float(row[16])])
                
                poses.append(GroundTruthPose(
                    timestamp=timestamp,
                    position=position,
                    orientation=orientation,
                    velocity=velocity,
                    angular_velocity=angular_vel,
                    acceleration=acceleration,
                ))
        
        return poses
    
    def iterate_camera_frames(
        self,
        sequence: EuRoCSequence,
        camera_id: int = 0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Generator[CameraFrame, None, None]:
        """
        Iterate over camera frames with lazy loading.
        
        Args:
            sequence: Loaded EuRoC sequence
            camera_id: Camera to iterate (0 or 1)
            start_time: Optional start timestamp filter
            end_time: Optional end timestamp filter
            
        Yields:
            CameraFrame objects with loaded images
        """
        frames = sequence.camera_frames.get(camera_id, [])
        
        for frame in frames:
            if start_time is not None and frame.timestamp < start_time:
                continue
            if end_time is not None and frame.timestamp > end_time:
                break
            
            # Load image if not already loaded
            if frame.image is None:
                frame.image = cv2.imread(frame.filepath, cv2.IMREAD_GRAYSCALE)
            
            yield frame
    
    def get_synchronized_data(
        self,
        sequence: EuRoCSequence,
        camera_id: int = 0,
        max_time_diff_ns: float = 5e6,  # 5ms
    ) -> Generator[Tuple[CameraFrame, List[IMUMeasurement], Optional[GroundTruthPose]], None, None]:
        """
        Get synchronized camera, IMU, and ground truth data.
        
        For each camera frame, returns the IMU measurements between this frame
        and the previous frame, plus the closest ground truth pose.
        
        Args:
            sequence: Loaded EuRoC sequence
            camera_id: Camera to use for synchronization
            max_time_diff_ns: Maximum time difference for ground truth matching
            
        Yields:
            Tuple of (camera_frame, imu_measurements, ground_truth_pose)
        """
        frames = sequence.camera_frames.get(camera_id, [])
        imu_data = sequence.imu_measurements
        gt_data = sequence.ground_truth
        
        imu_idx = 0
        gt_idx = 0
        prev_timestamp = None
        
        for frame in frames:
            # Load image if needed
            if frame.image is None:
                frame.image = cv2.imread(frame.filepath, cv2.IMREAD_GRAYSCALE)
                if frame.image is None:
                    continue
            
            # Collect IMU measurements since last frame
            imu_between = []
            while imu_idx < len(imu_data):
                imu = imu_data[imu_idx]
                if prev_timestamp is not None and imu.timestamp < prev_timestamp:
                    imu_idx += 1
                    continue
                if imu.timestamp > frame.timestamp:
                    break
                imu_between.append(imu)
                imu_idx += 1
            
            # Find closest ground truth pose
            gt_pose = None
            while gt_idx < len(gt_data) - 1:
                if gt_data[gt_idx + 1].timestamp <= frame.timestamp:
                    gt_idx += 1
                else:
                    break
            
            if gt_idx < len(gt_data):
                time_diff = abs(gt_data[gt_idx].timestamp - frame.timestamp)
                if time_diff < max_time_diff_ns:
                    gt_pose = gt_data[gt_idx]
            
            prev_timestamp = frame.timestamp
            yield frame, imu_between, gt_pose
    
    def get_ground_truth_at_time(
        self,
        sequence: EuRoCSequence,
        timestamp: float,
        interpolate: bool = True,
    ) -> Optional[GroundTruthPose]:
        """
        Get ground truth pose at a specific timestamp.
        
        Args:
            sequence: Loaded EuRoC sequence
            timestamp: Query timestamp in nanoseconds
            interpolate: If True, interpolate between poses
            
        Returns:
            Ground truth pose or None if not available
        """
        gt_data = sequence.ground_truth
        
        if not gt_data:
            return None
        
        # Binary search for closest poses
        left, right = 0, len(gt_data) - 1
        
        while left < right:
            mid = (left + right) // 2
            if gt_data[mid].timestamp < timestamp:
                left = mid + 1
            else:
                right = mid
        
        if left == 0:
            return gt_data[0] if abs(gt_data[0].timestamp - timestamp) < 1e7 else None
        
        if left >= len(gt_data):
            return gt_data[-1] if abs(gt_data[-1].timestamp - timestamp) < 1e7 else None
        
        if not interpolate:
            # Return closest
            if abs(gt_data[left].timestamp - timestamp) < abs(gt_data[left-1].timestamp - timestamp):
                return gt_data[left]
            return gt_data[left - 1]
        
        # Interpolate between poses
        pose0 = gt_data[left - 1]
        pose1 = gt_data[left]
        
        t = (timestamp - pose0.timestamp) / (pose1.timestamp - pose0.timestamp)
        t = np.clip(t, 0, 1)
        
        # Linear interpolation for position and velocity
        position = (1 - t) * pose0.position + t * pose1.position
        velocity = (1 - t) * pose0.velocity + t * pose1.velocity
        
        # SLERP for orientation
        r0 = Rotation.from_quat([pose0.orientation[1], pose0.orientation[2],
                                  pose0.orientation[3], pose0.orientation[0]])
        r1 = Rotation.from_quat([pose1.orientation[1], pose1.orientation[2],
                                  pose1.orientation[3], pose1.orientation[0]])
        
        # Interpolate rotation
        key_rots = Rotation.concatenate([r0, r1])
        key_times = [0, 1]
        from scipy.spatial.transform import Slerp
        slerp = Slerp(key_times, key_rots)
        r_interp = slerp([t])[0]
        q_interp = r_interp.as_quat()  # [qx, qy, qz, qw]
        orientation = np.array([q_interp[3], q_interp[0], q_interp[1], q_interp[2]])
        
        # Linear interpolation for angular velocity and acceleration
        angular_velocity = (1 - t) * pose0.angular_velocity + t * pose1.angular_velocity
        acceleration = (1 - t) * pose0.acceleration + t * pose1.acceleration
        
        return GroundTruthPose(
            timestamp=timestamp,
            position=position,
            orientation=orientation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            acceleration=acceleration,
        )
def convert_timestamps_to_seconds(timestamps_ns: np.ndarray) -> np.ndarray:
    """Convert timestamps from nanoseconds to seconds."""
    return timestamps_ns * 1e-9
def create_timestamp_index(
    imu_measurements: List[IMUMeasurement],
    camera_frames: List[CameraFrame],
    ground_truth: List[GroundTruthPose],
) -> Dict[str, np.ndarray]:
    """Create timestamp arrays for all sensor modalities."""
    return {
        'imu': np.array([m.timestamp for m in imu_measurements]),
        'camera': np.array([f.timestamp for f in camera_frames]),
        'ground_truth': np.array([p.timestamp for p in ground_truth]),
    }
