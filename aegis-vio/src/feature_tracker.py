"""
Feature Tracker for Visual-Inertial Odometry
Implements:
- ORB feature extraction
- Feature matching with descriptor matching and optical flow
- Outlier rejection using RANSAC and fundamental matrix
- Track management (new feature detection, track merging)
Reference: Rublee et al., "ORB: An efficient alternative to SIFT or SURF", ICCV 2011
"""
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
class MatchingMethod(Enum):
    """Feature matching methods."""
    DESCRIPTOR = "descriptor"
    OPTICAL_FLOW = "optical_flow"
    HYBRID = "hybrid"
@dataclass
class Feature:
    """Single tracked feature."""
    id: int
    position: np.ndarray  # [u, v] pixel coordinates
    descriptor: Optional[np.ndarray] = None
    track_length: int = 1
    last_seen_frame: int = 0
    response: float = 0.0  # Feature strength
    octave: int = 0
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
@dataclass
class FeatureTrack:
    """A track of features across multiple frames."""
    track_id: int
    observations: Dict[int, np.ndarray] = field(default_factory=dict)  # frame_id -> position
    
    @property
    def length(self) -> int:
        return len(self.observations)
    
    @property
    def first_frame(self) -> int:
        return min(self.observations.keys()) if self.observations else -1
    
    @property
    def last_frame(self) -> int:
        return max(self.observations.keys()) if self.observations else -1
@dataclass
class MatchResult:
    """Result of feature matching between two frames."""
    matches: List[Tuple[int, int]]  # (idx1, idx2) pairs
    inlier_mask: np.ndarray  # Boolean mask for inliers
    fundamental_matrix: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None
    num_inliers: int = 0
    
    @property
    def inlier_matches(self) -> List[Tuple[int, int]]:
        return [m for m, is_inlier in zip(self.matches, self.inlier_mask) if is_inlier]
@dataclass
class FrameFeatures:
    """All features detected in a single frame."""
    frame_id: int
    timestamp: float
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    positions: np.ndarray  # Nx2 array of [u, v] coordinates
    
    @property
    def num_features(self) -> int:
        return len(self.keypoints)
class FeatureTracker:
    """
    Complete feature tracker for VIO.
    
    Supports ORB feature detection and matching, with outlier rejection
    using RANSAC-based fundamental matrix estimation.
    
    Example:
        tracker = FeatureTracker()
        
        # Process first frame
        features1 = tracker.detect_features(image1, frame_id=0)
        
        # Process second frame and match
        features2 = tracker.detect_features(image2, frame_id=1)
        match_result = tracker.match_features(features1, features2)
        
        # Get inlier correspondences
        pts1, pts2 = tracker.get_matched_points(features1, features2, match_result)
    """
    
    def __init__(
        self,
        max_features: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        edge_threshold: int = 31,
        first_level: int = 0,
        wta_k: int = 2,
        patch_size: int = 31,
        fast_threshold: int = 20,
        matching_method: MatchingMethod = MatchingMethod.HYBRID,
        match_ratio_threshold: float = 0.75,
        ransac_threshold: float = 1.0,
        min_matches: int = 20,
        optical_flow_window: Tuple[int, int] = (21, 21),
        optical_flow_max_level: int = 3,
        grid_cell_size: int = 50,
        min_distance: int = 10,
    ):
        """
        Initialize the feature tracker.
        
        Args:
            max_features: Maximum number of features to detect
            scale_factor: Pyramid scale factor for ORB
            n_levels: Number of pyramid levels
            edge_threshold: Border size for feature rejection
            first_level: First pyramid level to use
            wta_k: Number of points for oriented BRIEF
            patch_size: Size of patch used for oriented BRIEF
            fast_threshold: Threshold for FAST corner detection
            matching_method: Method for feature matching
            match_ratio_threshold: Lowe's ratio test threshold
            ransac_threshold: RANSAC inlier threshold in pixels
            min_matches: Minimum matches required for valid result
            optical_flow_window: Window size for Lucas-Kanade optical flow
            optical_flow_max_level: Maximum pyramid level for optical flow
            grid_cell_size: Cell size for grid-based feature distribution
            min_distance: Minimum distance between features
        """
        self.max_features = max_features
        self.matching_method = matching_method
        self.match_ratio_threshold = match_ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches
        self.optical_flow_window = optical_flow_window
        self.optical_flow_max_level = optical_flow_max_level
        self.grid_cell_size = grid_cell_size
        self.min_distance = min_distance
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level,
            WTA_K=wta_k,
            patchSize=patch_size,
            fastThreshold=fast_threshold,
        )
        
        # Initialize descriptor matcher (Brute-Force with Hamming distance for ORB)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # FLANN-based matcher for faster matching with large feature sets
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1,
        )
        search_params = dict(checks=50)
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Track management
        self.next_track_id = 0
        self.active_tracks: Dict[int, FeatureTrack] = {}
        self.current_frame_id = -1
        self.prev_image: Optional[np.ndarray] = None
        self.prev_features: Optional[FrameFeatures] = None
        self.prev_track_ids: Optional[np.ndarray] = None
        
        # Feature ID management
        self.next_feature_id = 0
    
    def detect_features(
        self,
        image: np.ndarray,
        frame_id: int,
        timestamp: float = 0.0,
        mask: Optional[np.ndarray] = None,
    ) -> FrameFeatures:
        """
        Detect ORB features in an image.
        
        Args:
            image: Grayscale input image
            frame_id: Unique frame identifier
            timestamp: Frame timestamp
            mask: Optional mask for feature detection
            
        Returns:
            FrameFeatures containing detected keypoints and descriptors
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better feature detection in varying lighting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Create detection mask if needed
        detection_mask = self._create_detection_mask(image.shape, mask)
        
        # Detect ORB features
        keypoints, descriptors = self.orb.detectAndCompute(enhanced, detection_mask)
        
        # Apply non-maximum suppression with grid-based distribution
        keypoints, descriptors = self._distribute_features_in_grid(
            keypoints, descriptors, image.shape
        )
        
        # Convert keypoints to positions array
        if keypoints:
            positions = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        else:
            positions = np.empty((0, 2), dtype=np.float32)
        
        return FrameFeatures(
            frame_id=frame_id,
            timestamp=timestamp,
            keypoints=keypoints,
            descriptors=descriptors,
            positions=positions,
        )
    
    def _create_detection_mask(
        self,
        image_shape: Tuple[int, int],
        existing_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Create a detection mask, excluding image borders."""
        height, width = image_shape[:2]
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Exclude borders
        border = 20
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
        
        if existing_mask is not None:
            mask = cv2.bitwise_and(mask, existing_mask)
        
        return mask
    
    def _distribute_features_in_grid(
        self,
        keypoints: List[cv2.KeyPoint],
        descriptors: Optional[np.ndarray],
        image_shape: Tuple[int, int],
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Distribute features evenly across a grid to ensure good spatial coverage.
        """
        if not keypoints:
            return keypoints, descriptors
        
        height, width = image_shape[:2]
        n_cols = max(1, width // self.grid_cell_size)
        n_rows = max(1, height // self.grid_cell_size)
        
        # Target features per cell
        max_per_cell = max(1, self.max_features // (n_rows * n_cols))
        
        # Sort keypoints by response
        sorted_indices = sorted(
            range(len(keypoints)),
            key=lambda i: keypoints[i].response,
            reverse=True,
        )
        
        # Distribute across grid cells
        grid = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
        selected_indices = []
        
        for idx in sorted_indices:
            kp = keypoints[idx]
            col = min(int(kp.pt[0] / self.grid_cell_size), n_cols - 1)
            row = min(int(kp.pt[1] / self.grid_cell_size), n_rows - 1)
            
            if len(grid[row][col]) < max_per_cell:
                # Check minimum distance to existing features in cell
                too_close = False
                for existing_idx in grid[row][col]:
                    dist = np.linalg.norm(
                        np.array(kp.pt) - np.array(keypoints[existing_idx].pt)
                    )
                    if dist < self.min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    grid[row][col].append(idx)
                    selected_indices.append(idx)
                    
                    if len(selected_indices) >= self.max_features:
                        break
        
        # Build output
        selected_keypoints = [keypoints[i] for i in selected_indices]
        if descriptors is not None:
            selected_descriptors = descriptors[selected_indices]
        else:
            selected_descriptors = None
        
        return selected_keypoints, selected_descriptors
    
    def match_features(
        self,
        features1: FrameFeatures,
        features2: FrameFeatures,
        use_optical_flow: bool = True,
    ) -> MatchResult:
        """
        Match features between two frames.
        
        Args:
            features1: Features from first frame
            features2: Features from second frame
            use_optical_flow: Whether to use optical flow for matching
            
        Returns:
            MatchResult containing matches and inlier information
        """
        if features1.num_features < self.min_matches or features2.num_features < self.min_matches:
            return MatchResult(
                matches=[],
                inlier_mask=np.array([], dtype=bool),
                num_inliers=0,
            )
        
        if self.matching_method == MatchingMethod.DESCRIPTOR:
            matches = self._match_by_descriptor(features1, features2)
        elif self.matching_method == MatchingMethod.OPTICAL_FLOW:
            matches = self._match_by_optical_flow(features1, features2)
        else:  # HYBRID
            # Try optical flow first, fall back to descriptor matching
            if use_optical_flow and self.prev_image is not None:
                matches = self._match_by_optical_flow(features1, features2)
                if len(matches) < self.min_matches:
                    matches = self._match_by_descriptor(features1, features2)
            else:
                matches = self._match_by_descriptor(features1, features2)
        
        if len(matches) < self.min_matches:
            return MatchResult(
                matches=matches,
                inlier_mask=np.array([True] * len(matches)),
                num_inliers=len(matches),
            )
        
        # RANSAC outlier rejection
        return self._ransac_outlier_rejection(features1, features2, matches)
    
    def _match_by_descriptor(
        self,
        features1: FrameFeatures,
        features2: FrameFeatures,
    ) -> List[Tuple[int, int]]:
        """Match features using descriptor matching with ratio test."""
        if features1.descriptors is None or features2.descriptors is None:
            return []
        
        # KNN matching with k=2 for ratio test
        try:
            knn_matches = self.bf_matcher.knnMatch(
                features1.descriptors,
                features2.descriptors,
                k=2,
            )
        except cv2.error:
            return []
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < self.match_ratio_threshold * n.distance:
                good_matches.append((m.queryIdx, m.trainIdx))
        
        return good_matches
    
    def _match_by_optical_flow(
        self,
        features1: FrameFeatures,
        features2: FrameFeatures,
    ) -> List[Tuple[int, int]]:
        """Match features using Lucas-Kanade optical flow."""
        if self.prev_image is None:
            return []
        
        if features1.num_features == 0:
            return []
        
        # Track features using optical flow
        prev_pts = features1.positions.reshape(-1, 1, 2)
        
        # Note: This requires storing images, simplified here
        # In practice, you'd pass the actual images
        # For now, fall back to descriptor matching
        return self._match_by_descriptor(features1, features2)
    
    def track_features_optical_flow(
        self,
        prev_image: np.ndarray,
        curr_image: np.ndarray,
        prev_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track features between frames using optical flow.
        
        Args:
            prev_image: Previous grayscale image
            curr_image: Current grayscale image
            prev_points: Nx2 array of points in previous image
            
        Returns:
            Tuple of (tracked_points, tracked_mask, prev_points_filtered)
        """
        if prev_points.shape[0] == 0:
            return np.empty((0, 2)), np.array([]), np.empty((0, 2))
        
        prev_pts = prev_points.reshape(-1, 1, 2).astype(np.float32)
        
        # Forward optical flow
        curr_pts, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            prev_image,
            curr_image,
            prev_pts,
            None,
            winSize=self.optical_flow_window,
            maxLevel=self.optical_flow_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        # Backward optical flow for validation
        if curr_pts is not None:
            prev_pts_back, status_backward, _ = cv2.calcOpticalFlowPyrLK(
                curr_image,
                prev_image,
                curr_pts,
                None,
                winSize=self.optical_flow_window,
                maxLevel=self.optical_flow_max_level,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            
            # Check forward-backward consistency
            if prev_pts_back is not None:
                diff = np.abs(prev_pts - prev_pts_back).reshape(-1, 2).max(axis=1)
                status_fb = diff < 1.0
            else:
                status_fb = np.ones(len(status_forward), dtype=bool)
        else:
            return np.empty((0, 2)), np.array([]), np.empty((0, 2))
        
        # Combine status
        status = (status_forward.flatten() == 1) & (status_backward.flatten() == 1) & status_fb
        
        tracked_points = curr_pts.reshape(-1, 2)[status]
        prev_points_valid = prev_points[status]
        
        return tracked_points, status, prev_points_valid
    
    def _ransac_outlier_rejection(
        self,
        features1: FrameFeatures,
        features2: FrameFeatures,
        matches: List[Tuple[int, int]],
    ) -> MatchResult:
        """
        Reject outliers using RANSAC with fundamental matrix estimation.
        """
        if len(matches) < 8:  # Minimum for fundamental matrix
            return MatchResult(
                matches=matches,
                inlier_mask=np.ones(len(matches), dtype=bool),
                num_inliers=len(matches),
            )
        
        # Extract matched point coordinates
        pts1 = np.array([features1.positions[m[0]] for m in matches], dtype=np.float64)
        pts2 = np.array([features2.positions[m[1]] for m in matches], dtype=np.float64)
        
        # Estimate fundamental matrix with RANSAC
        F, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            cv2.FM_RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=0.999,
        )
        
        if mask is None:
            mask = np.ones(len(matches), dtype=np.uint8)
        
        inlier_mask = mask.flatten().astype(bool)
        num_inliers = np.sum(inlier_mask)
        
        return MatchResult(
            matches=matches,
            inlier_mask=inlier_mask,
            fundamental_matrix=F,
            num_inliers=num_inliers,
        )
    
    def get_matched_points(
        self,
        features1: FrameFeatures,
        features2: FrameFeatures,
        match_result: MatchResult,
        inliers_only: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get matched point coordinates.
        
        Args:
            features1: First frame features
            features2: Second frame features  
            match_result: Matching result
            inliers_only: If True, return only inlier matches
            
        Returns:
            Tuple of (points1, points2) as Nx2 arrays
        """
        if inliers_only:
            matches = match_result.inlier_matches
        else:
            matches = match_result.matches
        
        if not matches:
            return np.empty((0, 2)), np.empty((0, 2))
        
        pts1 = np.array([features1.positions[m[0]] for m in matches])
        pts2 = np.array([features2.positions[m[1]] for m in matches])
        
        return pts1, pts2
    
    def detect_and_track(
        self,
        image: np.ndarray,
        frame_id: int,
        timestamp: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect features and track from previous frame.
        
        Args:
            image: Current grayscale image
            frame_id: Frame identifier
            timestamp: Frame timestamp
            
        Returns:
            Tuple of (current_points, track_ids, track_lengths)
        """
        # Detect features in current frame
        current_features = self.detect_features(image, frame_id, timestamp)
        
        if self.prev_image is None or self.prev_features is None:
            # First frame
            n = current_features.num_features
            track_ids = np.arange(self.next_track_id, self.next_track_id + n)
            self.next_track_id += n
            track_lengths = np.ones(n, dtype=np.int32)
            
            self.prev_image = image.copy()
            self.prev_features = current_features
            self.prev_track_ids = track_ids
            
            return current_features.positions, track_ids, track_lengths
        
        # Track features from previous frame
        tracked_pts, status, prev_pts_valid = self.track_features_optical_flow(
            self.prev_image,
            image,
            self.prev_features.positions,
        )
        
        # Update track IDs for tracked features
        valid_track_ids = self.prev_track_ids[status]
        
        # Create mask for new feature detection
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        for pt in tracked_pts:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), self.min_distance, 0, -1)
        
        # Detect new features in areas without tracks
        new_features = self.detect_features(image, frame_id, timestamp, mask)
        
        # Combine tracked and new features
        if new_features.num_features > 0:
            all_points = np.vstack([tracked_pts, new_features.positions])
            n_new = new_features.num_features
            new_track_ids = np.arange(self.next_track_id, self.next_track_id + n_new)
            self.next_track_id += n_new
            all_track_ids = np.concatenate([valid_track_ids, new_track_ids])
            
            # Update track lengths
            track_lengths = np.ones(len(all_track_ids), dtype=np.int32)
            for i, tid in enumerate(valid_track_ids):
                if tid in self.active_tracks:
                    track_lengths[i] = self.active_tracks[tid].length + 1
        else:
            all_points = tracked_pts
            all_track_ids = valid_track_ids
            track_lengths = np.ones(len(all_track_ids), dtype=np.int32)
        
        # Update active tracks
        for i, tid in enumerate(all_track_ids):
            if tid not in self.active_tracks:
                self.active_tracks[tid] = FeatureTrack(track_id=tid)
            self.active_tracks[tid].observations[frame_id] = all_points[i]
        
        # Store for next iteration
        self.prev_image = image.copy()
        self.prev_features = FrameFeatures(
            frame_id=frame_id,
            timestamp=timestamp,
            keypoints=[],
            descriptors=None,
            positions=all_points,
        )
        self.prev_track_ids = all_track_ids
        self.current_frame_id = frame_id
        
        return all_points, all_track_ids, track_lengths
    
    def get_tracks_for_triangulation(
        self,
        min_length: int = 3,
        max_tracks: int = 100,
    ) -> List[FeatureTrack]:
        """
        Get tracks suitable for triangulation.
        
        Args:
            min_length: Minimum track length
            max_tracks: Maximum number of tracks to return
            
        Returns:
            List of FeatureTrack objects
        """
        valid_tracks = [
            track for track in self.active_tracks.values()
            if track.length >= min_length
        ]
        
        # Sort by length (prefer longer tracks)
        valid_tracks.sort(key=lambda t: t.length, reverse=True)
        
        return valid_tracks[:max_tracks]
    
    def reset(self):
        """Reset the tracker state."""
        self.next_track_id = 0
        self.active_tracks.clear()
        self.current_frame_id = -1
        self.prev_image = None
        self.prev_features = None
        self.prev_track_ids = None
        self.next_feature_id = 0
def compute_essential_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    ransac_threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute essential matrix from point correspondences.
    
    Args:
        pts1: Nx2 points in first image
        pts2: Nx2 points in second image
        K: 3x3 camera intrinsic matrix
        ransac_threshold: RANSAC inlier threshold
        
    Returns:
        Tuple of (essential_matrix, inlier_mask)
    """
    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold,
    )
    
    if mask is None:
        mask = np.ones(len(pts1), dtype=np.uint8)
    
    return E, mask.flatten().astype(bool)
def recover_pose_from_essential(
    E: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover relative pose from essential matrix.
    
    Args:
        E: 3x3 essential matrix
        pts1: Nx2 points in first image
        pts2: Nx2 points in second image
        K: 3x3 camera intrinsic matrix
        mask: Optional inlier mask
        
    Returns:
        Tuple of (rotation_matrix, translation_vector, triangulated_points_mask)
    """
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    return R, t.flatten(), mask_pose.flatten().astype(bool)
