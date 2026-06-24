"""
Visual-Inertial Odometry Estimator
Combines feature tracking, IMU integration, and state estimation
to produce full 6-DOF pose estimates with uncertainty.
This is the main VIO pipeline that orchestrates all components.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import cv2
from .dataset_loader import (
    EuRoCDatasetLoader, EuRoCSequence, IMUMeasurement, 
    CameraFrame, GroundTruthPose, CameraCalibration
)
from .feature_tracker import FeatureTracker, FrameFeatures, MatchResult
from .imu_integrator import IMUIntegrator, PreintegratedMeasurement, IMUBias
from .ekf import ExtendedKalmanFilter, EKFConfig, State
from .uncertainty import UncertaintyEstimator, UncertaintyMetrics
@dataclass
class VIOConfig:
    """Configuration for the VIO system."""
    # Feature tracking
    max_features: int = 300
    min_features: int = 50
    min_track_length: int = 3
    
    # Keyframe selection
    keyframe_interval: int = 10
    min_parallax: float = 30.0  # pixels
    min_tracked_ratio: float = 0.5
    
    # Camera parameters
    image_width: int = 752
    image_height: int = 480
    
    # EKF configuration
    ekf_config: EKFConfig = field(default_factory=EKFConfig)
    
    # Triangulation
    min_triangulation_angle: float = 2.0  # degrees
    max_triangulation_depth: float = 50.0  # meters
    
    # Outlier rejection
    reproj_threshold: float = 3.0  # pixels
@dataclass
class VIOResult:
    """Result from a single VIO processing step."""
    timestamp: float
    state: State
    covariance: np.ndarray
    uncertainty_metrics: UncertaintyMetrics
    is_keyframe: bool
    num_features: int
    num_inliers: int
    tracking_quality: float
@dataclass
class Keyframe:
    """Stored keyframe data."""
    frame_id: int
    timestamp: float
    state: State
    features: FrameFeatures
    track_ids: np.ndarray
    preintegrated: Optional[PreintegratedMeasurement] = None
class VIOEstimator:
    """
    Complete Visual-Inertial Odometry pipeline.
    
    Processes camera images and IMU data to estimate 6-DOF pose
    with uncertainty quantification.
    
    Example:
        vio = VIOEstimator(camera_calibration, config)
        vio.initialize(initial_pose)
        
        # Process data
        for image, imu_data, gt in dataset:
            result = vio.process_frame(image, imu_data, timestamp)
            print(f"Position: {result.state.position}")
            print(f"Uncertainty: {result.uncertainty_metrics.position_uncertainty}")
    """
    
    def __init__(
        self,
        camera_calibration: CameraCalibration,
        config: Optional[VIOConfig] = None,
    ):
        """
        Initialize the VIO estimator.
        
        Args:
            camera_calibration: Camera intrinsic and extrinsic calibration
            config: VIO configuration parameters
        """
        self.camera_calibration = camera_calibration
        self.config = config if config else VIOConfig()
        
        # Camera matrix
        self.K = camera_calibration.camera_matrix
        
        # Feature tracker
        self.tracker = FeatureTracker(
            max_features=self.config.max_features,
            min_distance=15,
        )
        
        # IMU integrator
        self.imu_integrator = IMUIntegrator()
        
        # Extended Kalman Filter
        self.ekf = ExtendedKalmanFilter(self.config.ekf_config)
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator()
        
        # State management
        self.initialized = False
        self.frame_count = 0
        self.keyframe_count = 0
        
        # Keyframe management
        self.keyframes: List[Keyframe] = []
        self.last_keyframe: Optional[Keyframe] = None
        
        # Current tracking state
        self.current_track_ids: Optional[np.ndarray] = None
        self.current_positions: Optional[np.ndarray] = None
        
        # Landmark management (for triangulation)
        self.landmarks: Dict[int, np.ndarray] = {}  # track_id -> 3D position
        
        # Previous image for optical flow
        self.prev_image: Optional[np.ndarray] = None
        
        # Results history
        self.trajectory: List[VIOResult] = []
    
    def initialize(
        self,
        initial_pose: GroundTruthPose,
        initial_bias: Optional[IMUBias] = None,
    ):
        """
        Initialize VIO with known initial pose.
        
        Args:
            initial_pose: Initial pose (typically from ground truth)
            initial_bias: Initial IMU bias estimate
        """
        # Create initial state
        initial_state = State(
            position=initial_pose.position.copy(),
            velocity=initial_pose.velocity.copy(),
            rotation=initial_pose.rotation_matrix.copy(),
            gyro_bias=initial_bias.gyroscope if initial_bias else np.zeros(3),
            accel_bias=initial_bias.accelerometer if initial_bias else np.zeros(3),
            timestamp=initial_pose.timestamp,
        )
        
        # Initialize EKF
        self.ekf.initialize(initial_state)
        
        # Initialize IMU integrator
        self.imu_integrator.reset(initial_state.bias)
        
        self.initialized = True
    
    def process_imu(self, imu: IMUMeasurement):
        """
        Process a single IMU measurement.
        
        Updates the state estimate using IMU prediction.
        
        Args:
            imu: IMU measurement
        """
        if not self.initialized:
            return
        
        # Add to pre-integration
        self.imu_integrator.add_measurement(imu)
        
        # Update EKF prediction
        self.ekf.predict_imu(imu)
    
    def process_frame(
        self,
        image: np.ndarray,
        timestamp: float,
        imu_data: Optional[List[IMUMeasurement]] = None,
    ) -> VIOResult:
        """
        Process a camera frame with optional IMU data.
        
        Args:
            image: Grayscale camera image
            timestamp: Frame timestamp (nanoseconds)
            imu_data: List of IMU measurements since last frame
            
        Returns:
            VIOResult with current state and uncertainty
        """
        if not self.initialized:
            raise RuntimeError("VIO not initialized")
        
        # Process IMU measurements
        if imu_data:
            for imu in imu_data:
                self.process_imu(imu)
        
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Feature tracking
        points, track_ids, track_lengths = self.tracker.detect_and_track(
            image, self.frame_count, timestamp
        )
        
        # Check if this should be a keyframe
        is_keyframe = self._check_keyframe_condition(
            points, track_ids, track_lengths
        )
        
        # Visual update
        num_inliers = 0
        if self.last_keyframe is not None and len(points) > 0:
            num_inliers = self._visual_update(
                points, track_ids, timestamp
            )
        
        # Create keyframe if needed
        if is_keyframe:
            self._create_keyframe(
                image, points, track_ids, timestamp
            )
        
        # Compute tracking quality
        tracking_quality = self._compute_tracking_quality(
            len(points), num_inliers, track_lengths
        )
        
        # Get current state and covariance
        state = self.ekf.get_state()
        covariance = self.ekf.get_covariance()
        
        # Compute uncertainty metrics
        uncertainty_metrics = self.uncertainty_estimator.compute_metrics(
            covariance, state, tracking_quality
        )
        
        # Store result
        result = VIOResult(
            timestamp=timestamp,
            state=state,
            covariance=covariance,
            uncertainty_metrics=uncertainty_metrics,
            is_keyframe=is_keyframe,
            num_features=len(points),
            num_inliers=num_inliers,
            tracking_quality=tracking_quality,
        )
        
        self.trajectory.append(result)
        
        # Update tracking state
        self.current_track_ids = track_ids
        self.current_positions = points
        self.prev_image = image.copy()
        self.frame_count += 1
        
        return result
    
    def _check_keyframe_condition(
        self,
        points: np.ndarray,
        track_ids: np.ndarray,
        track_lengths: np.ndarray,
    ) -> bool:
        """
        Check if current frame should be a keyframe.
        
        Conditions:
        1. Minimum frame interval
        2. Minimum parallax
        3. Minimum tracked feature ratio
        """
        # First frame is always a keyframe
        if self.last_keyframe is None:
            return True
        
        # Check frame interval
        frames_since_keyframe = self.frame_count - self.last_keyframe.frame_id
        if frames_since_keyframe < self.config.keyframe_interval // 2:
            return False
        
        if frames_since_keyframe >= self.config.keyframe_interval:
            return True
        
        # Check number of tracked features
        if len(points) < self.config.min_features:
            return True
        
        # Check parallax
        if self.last_keyframe is not None:
            common_tracks = np.intersect1d(track_ids, self.last_keyframe.track_ids)
            if len(common_tracks) > 0:
                # Get corresponding points
                kf_mask = np.isin(self.last_keyframe.track_ids, common_tracks)
                curr_mask = np.isin(track_ids, common_tracks)
                
                if np.sum(kf_mask) > 0 and np.sum(curr_mask) > 0:
                    kf_points = self.last_keyframe.features.positions[kf_mask]
                    curr_points = points[curr_mask]
                    
                    # Sort by track_id to align
                    kf_order = np.argsort(self.last_keyframe.track_ids[kf_mask])
                    curr_order = np.argsort(track_ids[curr_mask])
                    
                    if len(kf_order) == len(curr_order):
                        kf_points = kf_points[kf_order]
                        curr_points = curr_points[curr_order]
                        
                        parallax = np.linalg.norm(curr_points - kf_points, axis=1)
                        median_parallax = np.median(parallax)
                        
                        if median_parallax > self.config.min_parallax:
                            return True
        
        return False
    
    def _visual_update(
        self,
        points: np.ndarray,
        track_ids: np.ndarray,
        timestamp: float,
    ) -> int:
        """
        Perform visual measurement update.
        
        Returns:
            Number of inlier features used
        """
        if self.last_keyframe is None:
            return 0
        
        # Find common tracks with last keyframe
        common_tracks = np.intersect1d(track_ids, self.last_keyframe.track_ids)
        
        if len(common_tracks) < 8:
            return 0
        
        # Get corresponding points
        kf_mask = np.isin(self.last_keyframe.track_ids, common_tracks)
        curr_mask = np.isin(track_ids, common_tracks)
        
        kf_points = self.last_keyframe.features.positions[kf_mask]
        curr_points = points[curr_mask]
        
        # Sort by track_id to align
        kf_order = np.argsort(self.last_keyframe.track_ids[kf_mask])
        curr_order = np.argsort(track_ids[curr_mask])
        
        kf_points = kf_points[kf_order]
        curr_points = curr_points[curr_order]
        
        # Triangulate points
        landmark_positions = self._triangulate_points(
            kf_points, curr_points,
            self.last_keyframe.state, self.ekf.get_state()
        )
        
        # Predict feature positions from current state
        predicted_points = self._project_landmarks(
            landmark_positions, self.ekf.get_state()
        )
        
        if predicted_points is None:
            return 0
        
        # Visual update in EKF
        self.ekf.update_visual(
            curr_points,
            predicted_points,
            self.K,
            landmark_positions,
        )
        
        return len(common_tracks)
    
    def _triangulate_points(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        state1: State,
        state2: State,
    ) -> np.ndarray:
        """
        Triangulate 3D points from two views.
        
        Args:
            points1: Nx2 points in first view
            points2: Nx2 points in second view
            state1: State at first view
            state2: State at second view
            
        Returns:
            Nx3 triangulated 3D points in world frame
        """
        # Camera poses (world to camera)
        R1 = state1.rotation.T
        t1 = -R1 @ state1.position
        
        R2 = state2.rotation.T
        t2 = -R2 @ state2.position
        
        # Projection matrices
        P1 = self.K @ np.hstack([R1, t1.reshape(3, 1)])
        P2 = self.K @ np.hstack([R2, t2.reshape(3, 1)])
        
        # Triangulate
        points1_h = points1.T.astype(np.float64)
        points2_h = points2.T.astype(np.float64)
        
        points_4d = cv2.triangulatePoints(P1, P2, points1_h, points2_h)
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def _project_landmarks(
        self,
        landmarks: np.ndarray,
        state: State,
    ) -> Optional[np.ndarray]:
        """
        Project 3D landmarks to image plane.
        
        Args:
            landmarks: Nx3 world coordinates
            state: Current state
            
        Returns:
            Nx2 projected pixel coordinates or None
        """
        if len(landmarks) == 0:
            return None
        
        # Transform to camera frame
        R = state.rotation.T
        t = -R @ state.position
        
        points_cam = (R @ landmarks.T + t.reshape(3, 1)).T
        
        # Check if points are in front of camera
        valid = points_cam[:, 2] > 0.1
        if not np.any(valid):
            return None
        
        # Project
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        projected = np.zeros((len(landmarks), 2))
        projected[:, 0] = fx * points_cam[:, 0] / points_cam[:, 2] + cx
        projected[:, 1] = fy * points_cam[:, 1] / points_cam[:, 2] + cy
        
        # Mark invalid points
        projected[~valid] = -1
        
        return projected
    
    def _create_keyframe(
        self,
        image: np.ndarray,
        points: np.ndarray,
        track_ids: np.ndarray,
        timestamp: float,
    ):
        """Create and store a new keyframe."""
        state = self.ekf.get_state()
        
        # Get pre-integrated measurement since last keyframe
        preintegrated = self.imu_integrator.get_preintegrated()
        
        # Create keyframe
        keyframe = Keyframe(
            frame_id=self.frame_count,
            timestamp=timestamp,
            state=state.copy(),
            features=FrameFeatures(
                frame_id=self.frame_count,
                timestamp=timestamp,
                keypoints=[],
                descriptors=None,
                positions=points.copy(),
            ),
            track_ids=track_ids.copy(),
            preintegrated=preintegrated,
        )
        
        self.keyframes.append(keyframe)
        self.last_keyframe = keyframe
        self.keyframe_count += 1
        
        # Reset IMU integrator for next interval
        self.imu_integrator.reset(state.bias)
    
    def _compute_tracking_quality(
        self,
        num_features: int,
        num_inliers: int,
        track_lengths: np.ndarray,
    ) -> float:
        """
        Compute tracking quality metric.
        
        Returns value in [0, 1] where 1 is best quality.
        """
        # Feature count factor
        feature_factor = min(1.0, num_features / self.config.max_features)
        
        # Inlier ratio
        inlier_factor = num_inliers / max(1, num_features)
        
        # Track length factor
        if len(track_lengths) > 0:
            avg_track_length = np.mean(track_lengths)
            track_factor = min(1.0, avg_track_length / 10.0)
        else:
            track_factor = 0.0
        
        # Combined quality
        quality = 0.4 * feature_factor + 0.3 * inlier_factor + 0.3 * track_factor
        
        return quality
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get list of estimated positions."""
        return [r.state.position.copy() for r in self.trajectory]
    
    def get_trajectory_with_timestamps(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get trajectory as arrays of timestamps and positions."""
        timestamps = np.array([r.timestamp for r in self.trajectory])
        positions = np.array([r.state.position for r in self.trajectory])
        return timestamps, positions
    
    def get_covariance_history(self) -> List[np.ndarray]:
        """Get list of covariance matrices."""
        return [r.covariance.copy() for r in self.trajectory]
    
    def get_uncertainty_history(self) -> List[UncertaintyMetrics]:
        """Get list of uncertainty metrics."""
        return [r.uncertainty_metrics for r in self.trajectory]
    
    def reset(self):
        """Reset the VIO system."""
        self.initialized = False
        self.frame_count = 0
        self.keyframe_count = 0
        self.keyframes.clear()
        self.last_keyframe = None
        self.current_track_ids = None
        self.current_positions = None
        self.landmarks.clear()
        self.prev_image = None
        self.trajectory.clear()
        self.tracker.reset()
def run_vio_on_sequence(
    loader: EuRoCDatasetLoader,
    sequence_name: str,
    config: Optional[VIOConfig] = None,
) -> Tuple[VIOEstimator, List[VIOResult]]:
    """
    Run VIO on an entire EuRoC sequence.
    
    Args:
        loader: Dataset loader
        sequence_name: Name of sequence to process
        config: VIO configuration
        
    Returns:
        Tuple of (VIO estimator, list of results)
    """
    # Load sequence
    sequence = loader.load_sequence(sequence_name)
    
    # Get camera calibration
    cam_calib = sequence.camera_calibrations.get(0)
    if cam_calib is None:
        raise ValueError("Camera calibration not found")
    
    # Initialize VIO
    config = config if config else VIOConfig()
    vio = VIOEstimator(cam_calib, config)
    
    # Get initial pose from ground truth
    if sequence.ground_truth:
        initial_pose = sequence.ground_truth[0]
        vio.initialize(initial_pose)
    else:
        raise ValueError("Ground truth required for initialization")
    
    results = []
    
    # Process synchronized data
    for frame, imu_data, gt_pose in loader.get_synchronized_data(sequence):
        result = vio.process_frame(
            frame.image,
            frame.timestamp,
            imu_data,
        )
        results.append(result)
    
    return vio, results
