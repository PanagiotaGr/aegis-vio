"""
Trajectory Evaluation Module
Implements standard robotics evaluation metrics:
- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE)
- Root Mean Square Error (RMSE)
- Normalized Estimation Error Squared (NEES)
- Normalized Innovation Squared (NIS)
Reference: Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012
"""
import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import json
from pathlib import Path
@dataclass
class TrajectoryError:
    """Single error between estimated and ground truth pose."""
    timestamp: float
    position_error: np.ndarray  # 3D position error
    rotation_error: np.ndarray  # Rotation error as axis-angle
    
    @property
    def translation_error(self) -> float:
        """Translation error magnitude."""
        return np.linalg.norm(self.position_error)
    
    @property
    def rotation_error_rad(self) -> float:
        """Rotation error magnitude in radians."""
        return np.linalg.norm(self.rotation_error)
    
    @property
    def rotation_error_deg(self) -> float:
        """Rotation error magnitude in degrees."""
        return np.degrees(self.rotation_error_rad)
@dataclass
class ATEResult:
    """Absolute Trajectory Error results."""
    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    errors: List[float]
    
    # Separate translation and rotation
    rmse_translation: float
    rmse_rotation: float  # degrees
    
    # Per-axis errors
    rmse_x: float
    rmse_y: float
    rmse_z: float
@dataclass
class RPEResult:
    """Relative Pose Error results."""
    rmse_translation: float
    rmse_rotation: float  # degrees
    mean_translation: float
    mean_rotation: float
    std_translation: float
    std_rotation: float
    
    # Detailed errors
    translation_errors: List[float]
    rotation_errors: List[float]
@dataclass
class ConsistencyResult:
    """Covariance consistency metrics."""
    avg_nees: float
    expected_nees: float
    nees_values: List[float]
    is_consistent: bool
    chi2_lower: float
    chi2_upper: float
    
    avg_nis: float
    nis_values: List[float]
@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    ate: ATEResult
    rpe: RPEResult
    consistency: Optional[ConsistencyResult]
    
    # Summary statistics
    sequence_name: str
    sequence_length: float  # meters
    duration: float  # seconds
    num_poses: int
    
    # Timing
    avg_processing_time: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'sequence_name': self.sequence_name,
            'sequence_length': self.sequence_length,
            'duration': self.duration,
            'num_poses': self.num_poses,
            'ate': {
                'rmse': self.ate.rmse,
                'mean': self.ate.mean,
                'median': self.ate.median,
                'std': self.ate.std,
                'min': self.ate.min,
                'max': self.ate.max,
                'rmse_translation': self.ate.rmse_translation,
                'rmse_rotation_deg': self.ate.rmse_rotation,
                'rmse_x': self.ate.rmse_x,
                'rmse_y': self.ate.rmse_y,
                'rmse_z': self.ate.rmse_z,
            },
            'rpe': {
                'rmse_translation': self.rpe.rmse_translation,
                'rmse_rotation_deg': self.rpe.rmse_rotation,
                'mean_translation': self.rpe.mean_translation,
                'mean_rotation_deg': self.rpe.mean_rotation,
            },
            'consistency': {
                'avg_nees': self.consistency.avg_nees if self.consistency else None,
                'is_consistent': self.consistency.is_consistent if self.consistency else None,
            } if self.consistency else None,
            'avg_processing_time_ms': self.avg_processing_time * 1000,
        }
    
    def save(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
class TrajectoryEvaluator:
    """
    Evaluator for VIO trajectory accuracy.
    
    Computes standard metrics used in the robotics community for
    evaluating odometry and SLAM systems.
    
    Example:
        evaluator = TrajectoryEvaluator()
        
        # Align trajectories
        evaluator.set_trajectories(
            estimated_positions=positions_est,
            estimated_timestamps=timestamps_est,
            ground_truth_positions=positions_gt,
            ground_truth_timestamps=timestamps_gt,
        )
        
        # Compute metrics
        ate = evaluator.compute_ate()
        rpe = evaluator.compute_rpe(delta=1.0)
        
        print(f"ATE RMSE: {ate.rmse:.3f} m")
        print(f"RPE RMSE: {rpe.rmse_translation:.3f} m")
    """
    
    def __init__(self, align_trajectories: bool = True):
        """
        Initialize evaluator.
        
        Args:
            align_trajectories: If True, align trajectories using Umeyama alignment
        """
        self.align_trajectories = align_trajectories
        
        # Trajectories
        self.est_positions: Optional[np.ndarray] = None
        self.est_rotations: Optional[List[np.ndarray]] = None
        self.est_timestamps: Optional[np.ndarray] = None
        
        self.gt_positions: Optional[np.ndarray] = None
        self.gt_rotations: Optional[List[np.ndarray]] = None
        self.gt_timestamps: Optional[np.ndarray] = None
        
        # Aligned trajectories
        self.aligned_est_positions: Optional[np.ndarray] = None
        self.alignment_transform: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    
    def set_trajectories(
        self,
        estimated_positions: np.ndarray,
        estimated_timestamps: np.ndarray,
        ground_truth_positions: np.ndarray,
        ground_truth_timestamps: np.ndarray,
        estimated_rotations: Optional[List[np.ndarray]] = None,
        ground_truth_rotations: Optional[List[np.ndarray]] = None,
    ):
        """
        Set estimated and ground truth trajectories.
        
        Args:
            estimated_positions: Nx3 estimated positions
            estimated_timestamps: N timestamps
            ground_truth_positions: Mx3 ground truth positions
            ground_truth_timestamps: M timestamps
            estimated_rotations: N rotation matrices (optional)
            ground_truth_rotations: M rotation matrices (optional)
        """
        self.est_positions = np.asarray(estimated_positions)
        self.est_timestamps = np.asarray(estimated_timestamps)
        self.est_rotations = estimated_rotations
        
        self.gt_positions = np.asarray(ground_truth_positions)
        self.gt_timestamps = np.asarray(ground_truth_timestamps)
        self.gt_rotations = ground_truth_rotations
        
        # Associate and align
        self._associate_trajectories()
        
        if self.align_trajectories:
            self._align_trajectories()
        else:
            self.aligned_est_positions = self.est_positions
    
    def _associate_trajectories(self, max_diff: float = 0.02):
        """
        Associate estimated and ground truth poses by timestamp.
        
        Args:
            max_diff: Maximum time difference for association (seconds)
        """
        # Find closest ground truth for each estimate
        associated_gt = []
        associated_est_idx = []
        
        for i, t_est in enumerate(self.est_timestamps):
            # Convert to same units if needed
            t_est_s = t_est * 1e-9 if t_est > 1e15 else t_est
            gt_times_s = self.gt_timestamps * 1e-9 if self.gt_timestamps[0] > 1e15 else self.gt_timestamps
            
            idx = np.argmin(np.abs(gt_times_s - t_est_s))
            time_diff = np.abs(gt_times_s[idx] - t_est_s)
            
            if time_diff < max_diff:
                associated_gt.append(idx)
                associated_est_idx.append(i)
        
        # Keep only associated poses
        if associated_est_idx:
            self.est_positions = self.est_positions[associated_est_idx]
            self.est_timestamps = self.est_timestamps[associated_est_idx]
            self.gt_positions = self.gt_positions[associated_gt]
            self.gt_timestamps = self.gt_timestamps[associated_gt]
            
            if self.est_rotations:
                self.est_rotations = [self.est_rotations[i] for i in associated_est_idx]
            if self.gt_rotations:
                self.gt_rotations = [self.gt_rotations[i] for i in associated_gt]
    
    def _align_trajectories(self):
        """Align estimated trajectory to ground truth using Umeyama algorithm."""
        R, t, s = self._umeyama_alignment(
            self.est_positions.T,
            self.gt_positions.T,
        )
        
        self.alignment_transform = (R, t, s)
        
        # Apply alignment
        self.aligned_est_positions = (
            s * (R @ self.est_positions.T).T + t
        )
    
    def _umeyama_alignment(
        self,
        source: np.ndarray,
        target: np.ndarray,
        with_scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Umeyama alignment algorithm.
        
        Finds rotation R, translation t, and scale s such that:
        target ≈ s * R @ source + t
        
        Args:
            source: 3xN source points
            target: 3xN target points
            with_scale: If True, compute scale factor
            
        Returns:
            Tuple of (rotation_matrix, translation_vector, scale)
        """
        # Centroids
        mu_s = source.mean(axis=1, keepdims=True)
        mu_t = target.mean(axis=1, keepdims=True)
        
        # Centered points
        source_c = source - mu_s
        target_c = target - mu_t
        
        # Covariance
        n = source.shape[1]
        cov = (target_c @ source_c.T) / n
        
        # SVD
        U, D, Vt = np.linalg.svd(cov)
        
        # Rotation
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
        R = U @ S @ Vt
        
        # Scale
        if with_scale:
            var_s = np.sum(source_c ** 2) / n
            s = np.trace(np.diag(D) @ S) / var_s
        else:
            s = 1.0
        
        # Translation
        t = mu_t.flatten() - s * R @ mu_s.flatten()
        
        return R, t, s
    
    def compute_ate(self) -> ATEResult:
        """
        Compute Absolute Trajectory Error.
        
        ATE measures the global consistency of the trajectory.
        
        Returns:
            ATEResult with all ATE metrics
        """
        if self.aligned_est_positions is None:
            raise ValueError("Trajectories not set")
        
        # Position errors
        errors_xyz = self.aligned_est_positions - self.gt_positions
        errors_norm = np.linalg.norm(errors_xyz, axis=1)
        
        # Rotation errors
        rotation_errors = []
        if self.est_rotations and self.gt_rotations:
            for R_est, R_gt in zip(self.est_rotations, self.gt_rotations):
                R_err = R_est @ R_gt.T
                angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
                rotation_errors.append(np.degrees(angle))
        
        rotation_errors = np.array(rotation_errors) if rotation_errors else np.zeros(len(errors_norm))
        
        return ATEResult(
            rmse=np.sqrt(np.mean(errors_norm ** 2)),
            mean=np.mean(errors_norm),
            median=np.median(errors_norm),
            std=np.std(errors_norm),
            min=np.min(errors_norm),
            max=np.max(errors_norm),
            errors=errors_norm.tolist(),
            rmse_translation=np.sqrt(np.mean(errors_norm ** 2)),
            rmse_rotation=np.sqrt(np.mean(rotation_errors ** 2)),
            rmse_x=np.sqrt(np.mean(errors_xyz[:, 0] ** 2)),
            rmse_y=np.sqrt(np.mean(errors_xyz[:, 1] ** 2)),
            rmse_z=np.sqrt(np.mean(errors_xyz[:, 2] ** 2)),
        )
    
    def compute_rpe(
        self,
        delta: float = 1.0,
        delta_unit: str = 'seconds',
    ) -> RPEResult:
        """
        Compute Relative Pose Error.
        
        RPE measures the local consistency (drift) of the trajectory.
        
        Args:
            delta: Delta for relative pose computation
            delta_unit: 'seconds', 'meters', or 'frames'
            
        Returns:
            RPEResult with all RPE metrics
        """
        if self.aligned_est_positions is None:
            raise ValueError("Trajectories not set")
        
        translation_errors = []
        rotation_errors = []
        
        n = len(self.aligned_est_positions)
        
        # Find pairs based on delta
        for i in range(n):
            # Find j based on delta_unit
            if delta_unit == 'frames':
                j = i + int(delta)
            elif delta_unit == 'seconds':
                t_i = self.est_timestamps[i] * 1e-9 if self.est_timestamps[i] > 1e15 else self.est_timestamps[i]
                j = i + 1
                while j < n:
                    t_j = self.est_timestamps[j] * 1e-9 if self.est_timestamps[j] > 1e15 else self.est_timestamps[j]
                    if t_j - t_i >= delta:
                        break
                    j += 1
            else:  # meters
                j = i + 1
                dist = 0
                while j < n and dist < delta:
                    dist += np.linalg.norm(
                        self.aligned_est_positions[j] - self.aligned_est_positions[j-1]
                    )
                    j += 1
            
            if j >= n:
                continue
            
            # Relative pose from estimated
            est_rel = self.aligned_est_positions[j] - self.aligned_est_positions[i]
            
            # Relative pose from ground truth
            gt_rel = self.gt_positions[j] - self.gt_positions[i]
            
            # Translation error
            trans_err = np.linalg.norm(est_rel - gt_rel)
            translation_errors.append(trans_err)
            
            # Rotation error
            if self.est_rotations and self.gt_rotations:
                R_est_i, R_est_j = self.est_rotations[i], self.est_rotations[j]
                R_gt_i, R_gt_j = self.gt_rotations[i], self.gt_rotations[j]
                
                R_est_rel = R_est_j @ R_est_i.T
                R_gt_rel = R_gt_j @ R_gt_i.T
                R_err = R_est_rel @ R_gt_rel.T
                
                angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
                rotation_errors.append(np.degrees(angle))
            else:
                rotation_errors.append(0.0)
        
        translation_errors = np.array(translation_errors)
        rotation_errors = np.array(rotation_errors)
        
        return RPEResult(
            rmse_translation=np.sqrt(np.mean(translation_errors ** 2)),
            rmse_rotation=np.sqrt(np.mean(rotation_errors ** 2)),
            mean_translation=np.mean(translation_errors),
            mean_rotation=np.mean(rotation_errors),
            std_translation=np.std(translation_errors),
            std_rotation=np.std(rotation_errors),
            translation_errors=translation_errors.tolist(),
            rotation_errors=rotation_errors.tolist(),
        )
    
    def compute_consistency(
        self,
        covariances: List[np.ndarray],
        state_dim: int = 6,
    ) -> ConsistencyResult:
        """
        Compute covariance consistency metrics (NEES).
        
        Args:
            covariances: List of covariance matrices
            state_dim: Dimension of state for pose (typically 6)
            
        Returns:
            ConsistencyResult with NEES analysis
        """
        if self.aligned_est_positions is None:
            raise ValueError("Trajectories not set")
        
        from scipy.stats import chi2
        
        nees_values = []
        
        for i, cov in enumerate(covariances):
            if i >= len(self.aligned_est_positions):
                break
            
            # Position error
            error_pos = self.aligned_est_positions[i] - self.gt_positions[i]
            
            # Extract position covariance (first 3x3 block)
            P_pos = cov[:3, :3]
            
            try:
                P_inv = np.linalg.inv(P_pos)
                nees = error_pos.T @ P_inv @ error_pos
                nees_values.append(nees)
            except np.linalg.LinAlgError:
                continue
        
        if not nees_values:
            return ConsistencyResult(
                avg_nees=float('inf'),
                expected_nees=state_dim,
                nees_values=[],
                is_consistent=False,
                chi2_lower=0,
                chi2_upper=float('inf'),
                avg_nis=0,
                nis_values=[],
            )
        
        nees_values = np.array(nees_values)
        avg_nees = np.mean(nees_values)
        
        # Chi-squared bounds (95% confidence)
        n = len(nees_values)
        alpha = 0.05
        chi2_lower = chi2.ppf(alpha / 2, df=3 * n) / n
        chi2_upper = chi2.ppf(1 - alpha / 2, df=3 * n) / n
        
        is_consistent = chi2_lower <= avg_nees <= chi2_upper
        
        return ConsistencyResult(
            avg_nees=avg_nees,
            expected_nees=3.0,  # 3 DOF for position
            nees_values=nees_values.tolist(),
            is_consistent=is_consistent,
            chi2_lower=chi2_lower,
            chi2_upper=chi2_upper,
            avg_nis=0.0,  # Would need innovation data
            nis_values=[],
        )
    
    def compute_full_evaluation(
        self,
        sequence_name: str,
        covariances: Optional[List[np.ndarray]] = None,
        processing_times: Optional[List[float]] = None,
    ) -> EvaluationResult:
        """
        Compute complete evaluation with all metrics.
        
        Args:
            sequence_name: Name of the sequence
            covariances: Optional list of covariance matrices
            processing_times: Optional list of processing times
            
        Returns:
            Complete EvaluationResult
        """
        ate = self.compute_ate()
        rpe = self.compute_rpe(delta=1.0, delta_unit='seconds')
        
        consistency = None
        if covariances:
            consistency = self.compute_consistency(covariances)
        
        # Compute trajectory length
        trajectory_length = np.sum(np.linalg.norm(
            np.diff(self.gt_positions, axis=0), axis=1
        ))
        
        # Duration
        t_start = self.est_timestamps[0]
        t_end = self.est_timestamps[-1]
        if t_start > 1e15:  # nanoseconds
            duration = (t_end - t_start) * 1e-9
        else:
            duration = t_end - t_start
        
        # Average processing time
        avg_time = np.mean(processing_times) if processing_times else 0.0
        
        return EvaluationResult(
            ate=ate,
            rpe=rpe,
            consistency=consistency,
            sequence_name=sequence_name,
            sequence_length=trajectory_length,
            duration=duration,
            num_poses=len(self.est_positions),
            avg_processing_time=avg_time,
        )
def compare_methods(
    results: Dict[str, EvaluationResult],
) -> str:
    """
    Generate comparison table for multiple methods.
    
    Args:
        results: Dictionary mapping method name to results
        
    Returns:
        Formatted comparison table as string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Method':<20} {'ATE RMSE (m)':<15} {'RPE RMSE (m)':<15} {'Rot RMSE (deg)':<15}")
    lines.append("=" * 80)
    
    for name, result in results.items():
        lines.append(
            f"{name:<20} {result.ate.rmse:<15.4f} "
            f"{result.rpe.rmse_translation:<15.4f} {result.rpe.rmse_rotation:<15.4f}"
        )
    
    lines.append("=" * 80)
    
    return "\n".join(lines)
