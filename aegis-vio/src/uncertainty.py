"""
Uncertainty Estimation and Quantification
Implements uncertainty metrics for VIO:
- Covariance analysis
- Shannon entropy
- Mahalanobis distance
- Confidence ellipsoids
- Uncertainty detection and thresholds
Reference: Bar-Shalom et al., "Estimation with Applications to Tracking
and Navigation", 2001
"""
import numpy as np
from scipy import linalg
from scipy.stats import chi2
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
class UncertaintyLevel(Enum):
    """Uncertainty level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
@dataclass
class UncertaintyMetrics:
    """
    Complete uncertainty metrics for a state estimate.
    """
    # Position uncertainty (meters)
    position_uncertainty: float
    position_covariance: np.ndarray
    
    # Velocity uncertainty (m/s)
    velocity_uncertainty: float
    velocity_covariance: np.ndarray
    
    # Rotation uncertainty (radians)
    rotation_uncertainty: float
    rotation_covariance: np.ndarray
    
    # Combined pose uncertainty
    pose_uncertainty: float
    pose_covariance: np.ndarray
    
    # Full state uncertainty
    total_uncertainty: float
    
    # Information metrics
    entropy: float
    information: float
    
    # Confidence metrics
    confidence_level: float  # 0-1
    uncertainty_level: UncertaintyLevel
    
    # Ellipsoid parameters (for visualization)
    position_ellipsoid: Tuple[np.ndarray, np.ndarray]  # (axes, rotation)
    
    # Tracking quality influence
    tracking_quality: float
    
    # Detection flags
    high_uncertainty_detected: bool
    
    def __post_init__(self):
        self.position_covariance = np.asarray(self.position_covariance)
        self.velocity_covariance = np.asarray(self.velocity_covariance)
        self.rotation_covariance = np.asarray(self.rotation_covariance)
        self.pose_covariance = np.asarray(self.pose_covariance)
@dataclass
class UncertaintyThresholds:
    """Thresholds for uncertainty classification."""
    position_low: float = 0.05      # meters
    position_medium: float = 0.15   # meters
    position_high: float = 0.5      # meters
    
    velocity_low: float = 0.1       # m/s
    velocity_medium: float = 0.3    # m/s
    velocity_high: float = 1.0      # m/s
    
    rotation_low: float = 0.02      # radians (~1 degree)
    rotation_medium: float = 0.05   # radians (~3 degrees)
    rotation_high: float = 0.15     # radians (~9 degrees)
    
    entropy_low: float = 5.0
    entropy_medium: float = 10.0
    entropy_high: float = 20.0
    
    tracking_quality_threshold: float = 0.5
class UncertaintyEstimator:
    """
    Uncertainty estimator for VIO state estimates.
    
    Computes various uncertainty metrics from the state covariance
    and provides classification and detection capabilities.
    
    Example:
        estimator = UncertaintyEstimator()
        metrics = estimator.compute_metrics(covariance, state, tracking_quality)
        
        if metrics.high_uncertainty_detected:
            print("Warning: High uncertainty detected!")
    """
    
    def __init__(
        self,
        thresholds: Optional[UncertaintyThresholds] = None,
        confidence_probability: float = 0.95,
    ):
        """
        Initialize the uncertainty estimator.
        
        Args:
            thresholds: Uncertainty classification thresholds
            confidence_probability: Probability for confidence ellipsoids
        """
        self.thresholds = thresholds if thresholds else UncertaintyThresholds()
        self.confidence_probability = confidence_probability
        
        # Chi-squared value for confidence ellipsoid (3 DOF for 3D)
        self.chi2_3dof = chi2.ppf(confidence_probability, df=3)
        self.chi2_6dof = chi2.ppf(confidence_probability, df=6)
    
    def compute_metrics(
        self,
        covariance: np.ndarray,
        state,  # State object
        tracking_quality: float = 1.0,
    ) -> UncertaintyMetrics:
        """
        Compute all uncertainty metrics from state covariance.
        
        Args:
            covariance: 15x15 error state covariance matrix
            state: Current state estimate
            tracking_quality: Feature tracking quality (0-1)
            
        Returns:
            UncertaintyMetrics with all computed values
        """
        # Extract sub-covariances
        P_pos = covariance[0:3, 0:3]
        P_vel = covariance[3:6, 3:6]
        P_rot = covariance[6:9, 6:9]
        
        # 6x6 pose covariance [position, rotation]
        pose_idx = np.array([0, 1, 2, 6, 7, 8])
        P_pose = covariance[np.ix_(pose_idx, pose_idx)]
        
        # Compute uncertainties (standard deviations)
        position_uncertainty = self._compute_uncertainty_norm(P_pos)
        velocity_uncertainty = self._compute_uncertainty_norm(P_vel)
        rotation_uncertainty = self._compute_uncertainty_norm(P_rot)
        pose_uncertainty = self._compute_uncertainty_norm(P_pose)
        total_uncertainty = self._compute_uncertainty_norm(covariance)
        
        # Compute entropy
        entropy = self._compute_entropy(covariance)
        information = self._compute_information(covariance)
        
        # Compute confidence ellipsoid
        position_ellipsoid = self._compute_confidence_ellipsoid(P_pos)
        
        # Classify uncertainty level
        uncertainty_level = self._classify_uncertainty(
            position_uncertainty,
            velocity_uncertainty,
            rotation_uncertainty,
            entropy,
        )
        
        # Compute confidence level
        confidence_level = self._compute_confidence_level(
            position_uncertainty,
            velocity_uncertainty,
            rotation_uncertainty,
            tracking_quality,
        )
        
        # Detect high uncertainty
        high_uncertainty = self._detect_high_uncertainty(
            uncertainty_level, tracking_quality
        )
        
        return UncertaintyMetrics(
            position_uncertainty=position_uncertainty,
            position_covariance=P_pos,
            velocity_uncertainty=velocity_uncertainty,
            velocity_covariance=P_vel,
            rotation_uncertainty=rotation_uncertainty,
            rotation_covariance=P_rot,
            pose_uncertainty=pose_uncertainty,
            pose_covariance=P_pose,
            total_uncertainty=total_uncertainty,
            entropy=entropy,
            information=information,
            confidence_level=confidence_level,
            uncertainty_level=uncertainty_level,
            position_ellipsoid=position_ellipsoid,
            tracking_quality=tracking_quality,
            high_uncertainty_detected=high_uncertainty,
        )
    
    def _compute_uncertainty_norm(self, covariance: np.ndarray) -> float:
        """
        Compute scalar uncertainty from covariance matrix.
        
        Uses the trace (sum of variances) which corresponds to
        the expected squared error.
        """
        return np.sqrt(np.trace(covariance))
    
    def _compute_entropy(self, covariance: np.ndarray) -> float:
        """
        Compute differential entropy of Gaussian distribution.
        
        H(x) = 0.5 * ln((2*pi*e)^n * det(P))
        
        Higher entropy = more uncertainty.
        """
        n = covariance.shape[0]
        
        # Regularize for numerical stability
        P_reg = covariance + 1e-10 * np.eye(n)
        
        try:
            log_det = np.linalg.slogdet(P_reg)[1]
            entropy = 0.5 * (n * np.log(2 * np.pi * np.e) + log_det)
        except np.linalg.LinAlgError:
            entropy = float('inf')
        
        return entropy
    
    def _compute_information(self, covariance: np.ndarray) -> float:
        """
        Compute Fisher information (inverse of entropy).
        
        Higher information = less uncertainty.
        """
        entropy = self._compute_entropy(covariance)
        if entropy <= 0:
            return float('inf')
        return 1.0 / entropy
    
    def _compute_confidence_ellipsoid(
        self,
        covariance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence ellipsoid parameters.
        
        Returns:
            Tuple of (semi-axes lengths, rotation matrix)
        """
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            return np.ones(3), np.eye(3)
        
        # Ensure positive eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        # Semi-axes scaled by chi-squared value
        axes = np.sqrt(eigenvalues * self.chi2_3dof)
        
        # Rotation matrix (eigenvectors)
        rotation = eigenvectors
        
        return axes, rotation
    
    def _classify_uncertainty(
        self,
        pos_unc: float,
        vel_unc: float,
        rot_unc: float,
        entropy: float,
    ) -> UncertaintyLevel:
        """Classify overall uncertainty level."""
        # Check position
        if pos_unc > self.thresholds.position_high:
            return UncertaintyLevel.CRITICAL
        if pos_unc > self.thresholds.position_medium:
            pos_level = 2
        elif pos_unc > self.thresholds.position_low:
            pos_level = 1
        else:
            pos_level = 0
        
        # Check velocity
        if vel_unc > self.thresholds.velocity_high:
            return UncertaintyLevel.CRITICAL
        if vel_unc > self.thresholds.velocity_medium:
            vel_level = 2
        elif vel_unc > self.thresholds.velocity_low:
            vel_level = 1
        else:
            vel_level = 0
        
        # Check rotation
        if rot_unc > self.thresholds.rotation_high:
            return UncertaintyLevel.CRITICAL
        if rot_unc > self.thresholds.rotation_medium:
            rot_level = 2
        elif rot_unc > self.thresholds.rotation_low:
            rot_level = 1
        else:
            rot_level = 0
        
        # Combined level
        max_level = max(pos_level, vel_level, rot_level)
        
        if max_level == 0:
            return UncertaintyLevel.LOW
        elif max_level == 1:
            return UncertaintyLevel.MEDIUM
        else:
            return UncertaintyLevel.HIGH
    
    def _compute_confidence_level(
        self,
        pos_unc: float,
        vel_unc: float,
        rot_unc: float,
        tracking_quality: float,
    ) -> float:
        """
        Compute overall confidence level (0-1).
        
        1 = high confidence, 0 = low confidence.
        """
        # Normalize uncertainties to [0, 1] using thresholds
        pos_conf = 1.0 - min(1.0, pos_unc / self.thresholds.position_high)
        vel_conf = 1.0 - min(1.0, vel_unc / self.thresholds.velocity_high)
        rot_conf = 1.0 - min(1.0, rot_unc / self.thresholds.rotation_high)
        
        # Combined confidence (weighted average)
        confidence = (
            0.4 * pos_conf +
            0.2 * vel_conf +
            0.2 * rot_conf +
            0.2 * tracking_quality
        )
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _detect_high_uncertainty(
        self,
        level: UncertaintyLevel,
        tracking_quality: float,
    ) -> bool:
        """Detect if uncertainty is critically high."""
        if level == UncertaintyLevel.CRITICAL:
            return True
        
        if level == UncertaintyLevel.HIGH:
            if tracking_quality < self.thresholds.tracking_quality_threshold:
                return True
        
        return False
    
    def compute_mahalanobis_distance(
        self,
        state_estimate: np.ndarray,
        state_true: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """
        Compute Mahalanobis distance between estimate and truth.
        
        d = sqrt((x - x_true)^T * P^-1 * (x - x_true))
        
        Args:
            state_estimate: Estimated state vector
            state_true: True state vector
            covariance: State covariance matrix
            
        Returns:
            Mahalanobis distance
        """
        error = state_estimate - state_true
        
        try:
            P_inv = np.linalg.inv(covariance)
            d_squared = error.T @ P_inv @ error
            return np.sqrt(d_squared)
        except np.linalg.LinAlgError:
            return float('inf')
    
    def compute_covariance_consistency(
        self,
        errors: List[np.ndarray],
        covariances: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute covariance consistency metrics.
        
        Checks if the estimated covariance matches the actual errors.
        
        Args:
            errors: List of state errors
            covariances: List of corresponding covariances
            
        Returns:
            Dictionary with NEES, chi-squared bounds, etc.
        """
        if len(errors) == 0 or len(covariances) == 0:
            return {}
        
        n = errors[0].shape[0]
        N = len(errors)
        
        # Compute NEES for each sample
        nees_values = []
        for error, cov in zip(errors, covariances):
            try:
                P_inv = np.linalg.inv(cov)
                nees = error.T @ P_inv @ error
                nees_values.append(nees)
            except np.linalg.LinAlgError:
                continue
        
        if not nees_values:
            return {}
        
        nees_values = np.array(nees_values)
        
        # Average NEES
        avg_nees = np.mean(nees_values)
        
        # Chi-squared bounds (95% confidence)
        alpha = 0.05
        chi2_lower = chi2.ppf(alpha / 2, df=n * N) / N
        chi2_upper = chi2.ppf(1 - alpha / 2, df=n * N) / N
        
        # Check consistency
        is_consistent = chi2_lower <= avg_nees <= chi2_upper
        
        return {
            'avg_nees': avg_nees,
            'expected_nees': float(n),
            'chi2_lower_bound': chi2_lower,
            'chi2_upper_bound': chi2_upper,
            'is_consistent': is_consistent,
            'nees_ratio': avg_nees / n,  # Should be ~1 for consistent estimator
        }
def compute_position_uncertainty_bounds(
    covariance_3x3: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute position uncertainty bounds for each axis.
    
    Args:
        covariance_3x3: 3x3 position covariance matrix
        confidence: Confidence level (default 95%)
        
    Returns:
        Tuple of (x_bound, y_bound, z_bound) in meters
    """
    # Chi-squared value for 1 DOF
    k = chi2.ppf(confidence, df=1)
    
    # Standard deviations
    std = np.sqrt(np.diag(covariance_3x3))
    
    # Bounds
    bounds = std * np.sqrt(k)
    
    return tuple(bounds)
def compute_rotation_uncertainty_degrees(
    covariance_3x3: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute rotation uncertainty in degrees for each axis.
    
    Args:
        covariance_3x3: 3x3 rotation (axis-angle) covariance matrix
        
    Returns:
        Tuple of (roll_std, pitch_std, yaw_std) in degrees
    """
    std_rad = np.sqrt(np.diag(covariance_3x3))
    std_deg = np.degrees(std_rad)
    return tuple(std_deg)
def propagate_uncertainty_monte_carlo(
    mean: np.ndarray,
    covariance: np.ndarray,
    transform_fn,
    n_samples: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate uncertainty through nonlinear transform using Monte Carlo.
    
    Args:
        mean: Input mean vector
        covariance: Input covariance matrix
        transform_fn: Function that transforms the state
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Tuple of (output_mean, output_covariance)
    """
    # Sample from input distribution
    samples = np.random.multivariate_normal(mean, covariance, n_samples)
    
    # Transform each sample
    transformed = np.array([transform_fn(s) for s in samples])
    
    # Compute output statistics
    output_mean = np.mean(transformed, axis=0)
    output_cov = np.cov(transformed.T)
    
    return output_mean, output_cov
def compute_observability_metric(
    jacobian_history: List[np.ndarray],
    state_dim: int,
) -> float:
    """
    Compute observability metric from measurement Jacobian history.
    
    Uses the smallest singular value of the stacked observability matrix.
    
    Args:
        jacobian_history: List of measurement Jacobians
        state_dim: Dimension of state vector
        
    Returns:
        Observability metric (higher = more observable)
    """
    if len(jacobian_history) == 0:
        return 0.0
    
    # Stack Jacobians
    O = np.vstack(jacobian_history)
    
    # Compute singular values
    try:
        _, s, _ = np.linalg.svd(O, full_matrices=False)
        
        # Smallest singular value indicates observability
        # Normalize by number of measurements
        return s.min() / len(jacobian_history)
    except np.linalg.LinAlgError:
        return 0.0
