"""
IMU Pre-integration for Visual-Inertial Odometry
Implements IMU pre-integration between keyframes using the
on-manifold approach from Forster et al.
Reference: Forster et al., "On-Manifold Preintegration for Real-Time 
Visual-Inertial Odometry", TRO 2017
"""
import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .dataset_loader import IMUMeasurement
@dataclass
class IMUBias:
    """IMU bias values."""
    gyroscope: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accelerometer: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def __post_init__(self):
        self.gyroscope = np.asarray(self.gyroscope, dtype=np.float64)
        self.accelerometer = np.asarray(self.accelerometer, dtype=np.float64)
    
    def copy(self) -> 'IMUBias':
        return IMUBias(
            gyroscope=self.gyroscope.copy(),
            accelerometer=self.accelerometer.copy(),
        )
@dataclass
class PreintegratedMeasurement:
    """
    Pre-integrated IMU measurement between two keyframes.
    
    Contains:
    - Delta rotation (as rotation matrix)
    - Delta velocity
    - Delta position
    - Covariance matrix
    - Jacobians with respect to biases
    """
    delta_R: np.ndarray  # 3x3 rotation matrix
    delta_v: np.ndarray  # 3D velocity change
    delta_p: np.ndarray  # 3D position change
    delta_t: float  # Total integration time
    
    covariance: np.ndarray  # 9x9 or 15x15 covariance matrix
    
    # Jacobians for bias correction
    dR_dbg: np.ndarray  # 3x3 Jacobian of delta_R w.r.t. gyro bias
    dv_dbg: np.ndarray  # 3x3 Jacobian of delta_v w.r.t. gyro bias
    dv_dba: np.ndarray  # 3x3 Jacobian of delta_v w.r.t. accel bias
    dp_dbg: np.ndarray  # 3x3 Jacobian of delta_p w.r.t. gyro bias
    dp_dba: np.ndarray  # 3x3 Jacobian of delta_p w.r.t. accel bias
    
    # Reference bias used during integration
    bias_ref: IMUBias = field(default_factory=IMUBias)
    
    # Number of integrated measurements
    num_measurements: int = 0
    
    def correct_for_bias_change(
        self,
        new_bias: IMUBias,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Correct pre-integrated measurements for a change in bias estimate.
        
        First-order correction using stored Jacobians.
        
        Args:
            new_bias: Updated bias estimate
            
        Returns:
            Tuple of (corrected_delta_R, corrected_delta_v, corrected_delta_p)
        """
        dbg = new_bias.gyroscope - self.bias_ref.gyroscope
        dba = new_bias.accelerometer - self.bias_ref.accelerometer
        
        # Correct rotation using exponential map
        delta_theta = self.dR_dbg @ dbg
        delta_R_correction = exp_so3(delta_theta)
        corrected_delta_R = self.delta_R @ delta_R_correction
        
        # Correct velocity and position
        corrected_delta_v = self.delta_v + self.dv_dbg @ dbg + self.dv_dba @ dba
        corrected_delta_p = self.delta_p + self.dp_dbg @ dbg + self.dp_dba @ dba
        
        return corrected_delta_R, corrected_delta_v, corrected_delta_p
class IMUIntegrator:
    """
    IMU pre-integrator for VIO.
    
    Integrates IMU measurements between keyframes and computes
    the pre-integrated measurement with covariance.
    
    Example:
        integrator = IMUIntegrator(
            gyro_noise=1e-4,
            accel_noise=2e-3,
            gyro_walk=1e-5,
            accel_walk=3e-3,
        )
        
        # Add measurements
        for imu in imu_measurements:
            integrator.add_measurement(imu)
        
        # Get pre-integrated measurement
        preint = integrator.get_preintegrated()
        
        # Reset for next keyframe interval
        integrator.reset(new_bias)
    """
    
    GRAVITY = np.array([0, 0, -9.81])  # Gravity in world frame
    
    def __init__(
        self,
        gyro_noise: float = 1.6968e-4,
        accel_noise: float = 2.0e-3,
        gyro_walk: float = 1.9393e-5,
        accel_walk: float = 3.0e-3,
        initial_bias: Optional[IMUBias] = None,
    ):
        """
        Initialize the IMU integrator.
        
        Args:
            gyro_noise: Gyroscope noise density (rad/s/sqrt(Hz))
            accel_noise: Accelerometer noise density (m/s^2/sqrt(Hz))
            gyro_walk: Gyroscope random walk (rad/s^2/sqrt(Hz))
            accel_walk: Accelerometer random walk (m/s^3/sqrt(Hz))
            initial_bias: Initial bias estimate
        """
        self.gyro_noise = gyro_noise
        self.accel_noise = accel_noise
        self.gyro_walk = gyro_walk
        self.accel_walk = accel_walk
        
        self.bias = initial_bias if initial_bias else IMUBias()
        
        # Pre-integrated values
        self._reset_integration()
    
    def _reset_integration(self):
        """Reset integration state."""
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.delta_t = 0.0
        
        # Covariance matrix [delta_theta, delta_v, delta_p]
        self.covariance = np.zeros((9, 9))
        
        # Jacobians w.r.t. bias
        self.dR_dbg = np.zeros((3, 3))
        self.dv_dbg = np.zeros((3, 3))
        self.dv_dba = np.zeros((3, 3))
        self.dp_dbg = np.zeros((3, 3))
        self.dp_dba = np.zeros((3, 3))
        
        self.num_measurements = 0
        self.last_timestamp = None
    
    def reset(self, new_bias: Optional[IMUBias] = None):
        """
        Reset for new integration interval.
        
        Args:
            new_bias: New bias estimate to use
        """
        if new_bias:
            self.bias = new_bias.copy()
        self._reset_integration()
    
    def add_measurement(self, imu: IMUMeasurement, dt: Optional[float] = None):
        """
        Add an IMU measurement to the pre-integration.
        
        Args:
            imu: IMU measurement
            dt: Time delta (if None, computed from timestamps)
        """
        if dt is None:
            if self.last_timestamp is None:
                self.last_timestamp = imu.timestamp
                return
            dt = (imu.timestamp - self.last_timestamp) * 1e-9  # Convert ns to s
            self.last_timestamp = imu.timestamp
        
        if dt <= 0 or dt > 1.0:  # Skip invalid time deltas
            return
        
        # Bias-corrected measurements
        omega = imu.gyroscope - self.bias.gyroscope
        accel = imu.accelerometer - self.bias.accelerometer
        
        # Integration using mid-point method
        self._integrate_midpoint(omega, accel, dt)
        
        # Update covariance
        self._propagate_covariance(omega, accel, dt)
        
        self.num_measurements += 1
    
    def _integrate_midpoint(
        self,
        omega: np.ndarray,
        accel: np.ndarray,
        dt: float,
    ):
        """
        Integrate using mid-point method.
        
        Updates delta_R, delta_v, delta_p.
        """
        # Rotation increment
        theta = omega * dt
        dR = exp_so3(theta)
        
        # Rotation at mid-point
        R_mid = self.delta_R @ exp_so3(theta / 2)
        
        # Rotated acceleration at mid-point
        accel_world = R_mid @ accel
        
        # Update position (before velocity for mid-point)
        self.delta_p = (
            self.delta_p +
            self.delta_v * dt +
            0.5 * accel_world * dt * dt
        )
        
        # Update velocity
        self.delta_v = self.delta_v + accel_world * dt
        
        # Update rotation
        self.delta_R = self.delta_R @ dR
        
        # Ensure rotation matrix stays valid
        self.delta_R = normalize_rotation(self.delta_R)
        
        # Update total time
        self.delta_t += dt
        
        # Update Jacobians w.r.t. bias
        self._update_jacobians(omega, accel, dt, dR)
    
    def _update_jacobians(
        self,
        omega: np.ndarray,
        accel: np.ndarray,
        dt: float,
        dR: np.ndarray,
    ):
        """Update Jacobians with respect to bias."""
        # Jacobian of rotation w.r.t. gyro bias
        # dR_dbg(k+1) = dR_k^T * dR_dbg(k) - Jr(omega * dt) * dt
        Jr = right_jacobian_so3(omega * dt)
        self.dR_dbg = dR.T @ self.dR_dbg - Jr * dt
        
        # Jacobian of velocity w.r.t. gyro bias
        self.dv_dbg = self.dv_dbg - self.delta_R @ skew(accel) @ self.dR_dbg * dt
        
        # Jacobian of velocity w.r.t. accel bias
        self.dv_dba = self.dv_dba - self.delta_R * dt
        
        # Jacobian of position w.r.t. gyro bias
        self.dp_dbg = (
            self.dp_dbg +
            self.dv_dbg * dt -
            0.5 * self.delta_R @ skew(accel) @ self.dR_dbg * dt * dt
        )
        
        # Jacobian of position w.r.t. accel bias
        self.dp_dba = self.dp_dba + self.dv_dba * dt - 0.5 * self.delta_R * dt * dt
    
    def _propagate_covariance(
        self,
        omega: np.ndarray,
        accel: np.ndarray,
        dt: float,
    ):
        """
        Propagate covariance matrix.
        
        State: [delta_theta, delta_v, delta_p]
        """
        # Noise covariance matrix
        sigma_g = self.gyro_noise ** 2
        sigma_a = self.accel_noise ** 2
        
        # Discrete-time noise covariance
        Q = np.diag([
            sigma_g, sigma_g, sigma_g,
            sigma_a, sigma_a, sigma_a,
        ]) * dt
        
        # State transition Jacobian
        F = np.eye(9)
        
        # dtheta/dtheta
        F[0:3, 0:3] = exp_so3(-omega * dt)
        
        # dv/dtheta
        F[3:6, 0:3] = -self.delta_R @ skew(accel) * dt
        
        # dp/dtheta
        F[6:9, 0:3] = -0.5 * self.delta_R @ skew(accel) * dt * dt
        
        # dp/dv
        F[6:9, 3:6] = np.eye(3) * dt
        
        # Noise Jacobian
        G = np.zeros((9, 6))
        
        # Effect of gyro noise on rotation
        Jr = right_jacobian_so3(omega * dt)
        G[0:3, 0:3] = Jr * dt
        
        # Effect of accel noise on velocity
        G[3:6, 3:6] = self.delta_R * dt
        
        # Effect of accel noise on position
        G[6:9, 3:6] = 0.5 * self.delta_R * dt * dt
        
        # Propagate covariance
        self.covariance = F @ self.covariance @ F.T + G @ Q @ G.T
    
    def get_preintegrated(self) -> PreintegratedMeasurement:
        """
        Get the pre-integrated measurement.
        
        Returns:
            PreintegratedMeasurement with all integration results
        """
        return PreintegratedMeasurement(
            delta_R=self.delta_R.copy(),
            delta_v=self.delta_v.copy(),
            delta_p=self.delta_p.copy(),
            delta_t=self.delta_t,
            covariance=self.covariance.copy(),
            dR_dbg=self.dR_dbg.copy(),
            dv_dbg=self.dv_dbg.copy(),
            dv_dba=self.dv_dba.copy(),
            dp_dbg=self.dp_dbg.copy(),
            dp_dba=self.dp_dba.copy(),
            bias_ref=self.bias.copy(),
            num_measurements=self.num_measurements,
        )
    
    def predict_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        rotation: np.ndarray,
        preint: Optional[PreintegratedMeasurement] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict state at end of integration interval.
        
        Args:
            position: Initial position (3D)
            velocity: Initial velocity (3D)
            rotation: Initial rotation matrix (3x3)
            preint: Pre-integrated measurement (uses current if None)
            
        Returns:
            Tuple of (predicted_position, predicted_velocity, predicted_rotation)
        """
        if preint is None:
            preint = self.get_preintegrated()
        
        # Predicted rotation
        R_pred = rotation @ preint.delta_R
        R_pred = normalize_rotation(R_pred)
        
        # Predicted velocity (accounting for gravity)
        v_pred = (
            velocity +
            self.GRAVITY * preint.delta_t +
            rotation @ preint.delta_v
        )
        
        # Predicted position
        p_pred = (
            position +
            velocity * preint.delta_t +
            0.5 * self.GRAVITY * preint.delta_t ** 2 +
            rotation @ preint.delta_p
        )
        
        return p_pred, v_pred, R_pred
# =====================================================================
# Lie Group Utility Functions
# =====================================================================
def skew(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3D vector.
    
    skew(v) * u = v x u (cross product)
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ], dtype=np.float64)
def exp_so3(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map from so(3) to SO(3).
    
    Converts rotation vector (axis-angle) to rotation matrix.
    Uses Rodrigues' formula with small angle approximation.
    """
    theta = np.linalg.norm(omega)
    
    if theta < 1e-10:
        # First-order approximation for small angles
        return np.eye(3) + skew(omega)
    
    axis = omega / theta
    K = skew(axis)
    
    # Rodrigues' formula
    return (
        np.eye(3) +
        np.sin(theta) * K +
        (1 - np.cos(theta)) * K @ K
    )
def log_so3(R: np.ndarray) -> np.ndarray:
    """
    Logarithmic map from SO(3) to so(3).
    
    Converts rotation matrix to rotation vector.
    """
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    
    if theta < 1e-10:
        # Small angle approximation
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) / 2
    
    if abs(theta - np.pi) < 1e-10:
        # Near pi, extract axis from diagonal
        diag = np.diag(R)
        idx = np.argmax(diag)
        axis = np.zeros(3)
        axis[idx] = np.sqrt((diag[idx] + 1) / 2)
        for i in range(3):
            if i != idx:
                axis[i] = R[idx, i] / (2 * axis[idx])
        return axis * theta
    
    return theta / (2 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
def right_jacobian_so3(omega: np.ndarray) -> np.ndarray:
    """
    Right Jacobian of SO(3).
    
    Jr(omega) such that exp(omega + delta) ≈ exp(omega) * exp(Jr * delta)
    """
    theta = np.linalg.norm(omega)
    
    if theta < 1e-10:
        return np.eye(3) - 0.5 * skew(omega)
    
    axis = omega / theta
    K = skew(axis)
    
    return (
        np.eye(3) -
        (1 - np.cos(theta)) / theta * K +
        (theta - np.sin(theta)) / theta * K @ K
    )
def left_jacobian_so3(omega: np.ndarray) -> np.ndarray:
    """
    Left Jacobian of SO(3).
    
    Jl(omega) = Jr(-omega)
    """
    return right_jacobian_so3(-omega)
def normalize_rotation(R: np.ndarray) -> np.ndarray:
    """
    Project matrix to SO(3) using SVD.
    
    Ensures the result is a valid rotation matrix.
    """
    U, _, Vt = np.linalg.svd(R)
    R_normalized = U @ Vt
    
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R_normalized) < 0:
        U[:, -1] *= -1
        R_normalized = U @ Vt
    
    return R_normalized
def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qw, qx, qy, qz] to rotation matrix.
    """
    qw, qx, qy, qz = q
    
    # Normalize
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-10:
        return np.eye(3)
    qw, qx, qy, qz = q / n
    
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])
def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [qw, qx, qy, qz].
    """
    r = Rotation.from_matrix(R)
    q = r.as_quat()  # [qx, qy, qz, qw]
    return np.array([q[3], q[0], q[1], q[2]])
File 7: src/ekf.py
Path: aegis-vio/src/ekf.py

python


"""
Extended Kalman Filter for Visual-Inertial Odometry
Implements error-state EKF (ESKF) for VIO with:
- IMU-driven prediction
- Visual measurement updates
- Covariance propagation
- Jacobian computations
State vector: [position, velocity, rotation, gyro_bias, accel_bias]
Error state: [δp, δv, δθ, δbg, δba] (15-dimensional)
Reference: Sola, "Quaternion kinematics for the error-state Kalman filter", 2017
"""
import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from .imu_integrator import (
    IMUIntegrator, PreintegratedMeasurement, IMUBias,
    skew, exp_so3, log_so3, quaternion_to_rotation, rotation_to_quaternion,
    normalize_rotation
)
from .dataset_loader import IMUMeasurement
@dataclass
class State:
    """
    Full VIO state.
    
    Includes position, velocity, orientation, and IMU biases.
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    timestamp: float = 0.0
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=np.float64)
        self.accel_bias = np.asarray(self.accel_bias, dtype=np.float64)
    
    @property
    def quaternion(self) -> np.ndarray:
        """Get orientation as quaternion [qw, qx, qy, qz]."""
        return rotation_to_quaternion(self.rotation)
    
    @property
    def bias(self) -> IMUBias:
        """Get IMU bias object."""
        return IMUBias(
            gyroscope=self.gyro_bias.copy(),
            accelerometer=self.accel_bias.copy(),
        )
    
    def copy(self) -> 'State':
        return State(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            rotation=self.rotation.copy(),
            gyro_bias=self.gyro_bias.copy(),
            accel_bias=self.accel_bias.copy(),
            timestamp=self.timestamp,
        )
    
    def to_vector(self) -> np.ndarray:
        """Convert state to vector [p, v, q, bg, ba]."""
        q = self.quaternion
        return np.concatenate([
            self.position,
            self.velocity,
            q,
            self.gyro_bias,
            self.accel_bias,
        ])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, timestamp: float = 0.0) -> 'State':
        """Create state from vector."""
        p = vec[0:3]
        v = vec[3:6]
        q = vec[6:10]
        bg = vec[10:13]
        ba = vec[13:16]
        
        R = quaternion_to_rotation(q)
        
        return cls(
            position=p,
            velocity=v,
            rotation=R,
            gyro_bias=bg,
            accel_bias=ba,
            timestamp=timestamp,
        )
@dataclass
class EKFConfig:
    """Configuration for the Extended Kalman Filter."""
    # IMU noise parameters
    gyro_noise: float = 1.6968e-4  # rad/s/sqrt(Hz)
    accel_noise: float = 2.0e-3    # m/s^2/sqrt(Hz)
    gyro_walk: float = 1.9393e-5   # rad/s^2/sqrt(Hz)
    accel_walk: float = 3.0e-3     # m/s^3/sqrt(Hz)
    
    # Visual measurement noise
    pixel_noise: float = 1.0       # pixels
    
    # Initial covariance
    init_position_std: float = 0.1     # m
    init_velocity_std: float = 0.1     # m/s
    init_rotation_std: float = 0.1     # rad
    init_gyro_bias_std: float = 0.01   # rad/s
    init_accel_bias_std: float = 0.1   # m/s^2
    
    # Gravity
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    
    # Keyframe interval
    keyframe_interval: int = 10
class ExtendedKalmanFilter:
    """
    Error-State Extended Kalman Filter for VIO.
    
    The filter maintains the nominal state and error state covariance.
    IMU measurements drive the prediction step, and visual measurements
    provide updates.
    
    Error state: δx = [δp, δv, δθ, δbg, δba] ∈ R^15
    
    Example:
        ekf = ExtendedKalmanFilter(config)
        ekf.initialize(initial_state)
        
        # IMU prediction
        ekf.predict_imu(imu_measurement, dt)
        
        # Visual update
        ekf.update_visual(observed_points, predicted_points, camera_matrix)
        
        # Get current state
        state = ekf.get_state()
        covariance = ekf.get_covariance()
    """
    
    # Error state indices
    IDX_P = slice(0, 3)    # Position
    IDX_V = slice(3, 6)    # Velocity
    IDX_R = slice(6, 9)    # Rotation (axis-angle error)
    IDX_BG = slice(9, 12)  # Gyro bias
    IDX_BA = slice(12, 15) # Accel bias
    
    STATE_DIM = 15
    
    def __init__(self, config: Optional[EKFConfig] = None):
        """
        Initialize the EKF.
        
        Args:
            config: EKF configuration parameters
        """
        self.config = config if config else EKFConfig()
        
        # Nominal state
        self.state = State()
        
        # Error state covariance (15x15)
        self.P = np.eye(self.STATE_DIM)
        
        # Process noise covariance (continuous)
        self.Q_continuous = self._build_process_noise()
        
        # IMU integrator for prediction
        self.imu_integrator = IMUIntegrator(
            gyro_noise=self.config.gyro_noise,
            accel_noise=self.config.accel_noise,
            gyro_walk=self.config.gyro_walk,
            accel_walk=self.config.accel_walk,
        )
        
        # Track last IMU timestamp
        self.last_imu_timestamp: Optional[float] = None
        
        # Initialized flag
        self.initialized = False
    
    def _build_process_noise(self) -> np.ndarray:
        """Build continuous-time process noise covariance matrix."""
        Q = np.zeros((12, 12))
        
        # Gyroscope noise (affects velocity and rotation)
        Q[0:3, 0:3] = np.eye(3) * self.config.gyro_noise ** 2
        
        # Accelerometer noise (affects position and velocity)
        Q[3:6, 3:6] = np.eye(3) * self.config.accel_noise ** 2
        
        # Gyroscope bias random walk
        Q[6:9, 6:9] = np.eye(3) * self.config.gyro_walk ** 2
        
        # Accelerometer bias random walk
        Q[9:12, 9:12] = np.eye(3) * self.config.accel_walk ** 2
        
        return Q
    
    def initialize(
        self,
        initial_state: State,
        initial_covariance: Optional[np.ndarray] = None,
    ):
        """
        Initialize the filter with a known state.
        
        Args:
            initial_state: Initial state estimate
            initial_covariance: Initial error covariance (uses defaults if None)
        """
        self.state = initial_state.copy()
        
        if initial_covariance is not None:
            self.P = initial_covariance.copy()
        else:
            self.P = np.diag([
                # Position
                self.config.init_position_std ** 2,
                self.config.init_position_std ** 2,
                self.config.init_position_std ** 2,
                # Velocity
                self.config.init_velocity_std ** 2,
                self.config.init_velocity_std ** 2,
                self.config.init_velocity_std ** 2,
                # Rotation
                self.config.init_rotation_std ** 2,
                self.config.init_rotation_std ** 2,
                self.config.init_rotation_std ** 2,
                # Gyro bias
                self.config.init_gyro_bias_std ** 2,
                self.config.init_gyro_bias_std ** 2,
                self.config.init_gyro_bias_std ** 2,
                # Accel bias
                self.config.init_accel_bias_std ** 2,
                self.config.init_accel_bias_std ** 2,
                self.config.init_accel_bias_std ** 2,
            ])
        
        # Initialize IMU integrator with current bias
        self.imu_integrator.reset(self.state.bias)
        
        self.initialized = True
    
    def predict_imu(self, imu: IMUMeasurement, dt: Optional[float] = None):
        """
        Prediction step using IMU measurement.
        
        Updates the nominal state using IMU kinematics and
        propagates the error state covariance.
        
        Args:
            imu: IMU measurement (gyroscope and accelerometer)
            dt: Time step (computed from timestamps if None)
        """
        if not self.initialized:
            raise RuntimeError("EKF not initialized")
        
        # Compute dt from timestamps
        if dt is None:
            if self.last_imu_timestamp is None:
                self.last_imu_timestamp = imu.timestamp
                return
            dt = (imu.timestamp - self.last_imu_timestamp) * 1e-9
            self.last_imu_timestamp = imu.timestamp
        
        if dt <= 0 or dt > 1.0:
            return
        
        # Bias-corrected measurements
        omega = imu.gyroscope - self.state.gyro_bias
        accel = imu.accelerometer - self.state.accel_bias
        
        # Current rotation
        R = self.state.rotation
        
        # Rotation increment
        dtheta = omega * dt
        dR = exp_so3(dtheta)
        
        # State prediction (nominal state update)
        # Position
        self.state.position = (
            self.state.position +
            self.state.velocity * dt +
            0.5 * (R @ accel + self.config.gravity) * dt ** 2
        )
        
        # Velocity
        self.state.velocity = (
            self.state.velocity +
            (R @ accel + self.config.gravity) * dt
        )
        
        # Rotation
        self.state.rotation = normalize_rotation(R @ dR)
        
        # Update timestamp
        self.state.timestamp = imu.timestamp
        
        # Covariance propagation
        self._propagate_covariance(omega, accel, dt)
    
    def _propagate_covariance(
        self,
        omega: np.ndarray,
        accel: np.ndarray,
        dt: float,
    ):
        """
        Propagate error state covariance.
        
        Computes F (state transition) and G (noise) Jacobians
        and updates P using discrete-time propagation.
        """
        R = self.state.rotation
        
        # State transition Jacobian F (15x15)
        F = np.eye(self.STATE_DIM)
        
        # δp_new = δp + δv * dt
        F[self.IDX_P, self.IDX_V] = np.eye(3) * dt
        
        # δv_new = δv - R * [a]_x * δθ * dt - R * δba * dt
        F[self.IDX_V, self.IDX_R] = -R @ skew(accel) * dt
        F[self.IDX_V, self.IDX_BA] = -R * dt
        
        # δθ_new = exp(-ω * dt) * δθ - δbg * dt
        F[self.IDX_R, self.IDX_R] = exp_so3(-omega * dt)
        F[self.IDX_R, self.IDX_BG] = -np.eye(3) * dt
        
        # Biases: random walk (no coupling)
        # F[self.IDX_BG, self.IDX_BG] = I (already set)
        # F[self.IDX_BA, self.IDX_BA] = I (already set)
        
        # Noise Jacobian G (15x12)
        G = np.zeros((self.STATE_DIM, 12))
        
        # Gyro noise affects rotation
        G[self.IDX_R, 0:3] = -np.eye(3) * dt
        
        # Accel noise affects velocity
        G[self.IDX_V, 3:6] = -R * dt
        
        # Gyro bias random walk
        G[self.IDX_BG, 6:9] = np.eye(3) * dt
        
        # Accel bias random walk
        G[self.IDX_BA, 9:12] = np.eye(3) * dt
        
        # Discrete-time process noise
        Q_d = G @ self.Q_continuous @ G.T
        
        # Propagate covariance
        self.P = F @ self.P @ F.T + Q_d
        
        # Ensure symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
    
    def predict_preintegrated(self, preint: PreintegratedMeasurement):
        """
        Prediction step using pre-integrated IMU measurements.
        
        Used for keyframe-to-keyframe prediction.
        
        Args:
            preint: Pre-integrated IMU measurement
        """
        if not self.initialized:
            raise RuntimeError("EKF not initialized")
        
        # Correct pre-integration for current bias estimate
        delta_R, delta_v, delta_p = preint.correct_for_bias_change(self.state.bias)
        
        R = self.state.rotation
        dt = preint.delta_t
        g = self.config.gravity
        
        # State prediction
        self.state.position = (
            self.state.position +
            self.state.velocity * dt +
            0.5 * g * dt ** 2 +
            R @ delta_p
        )
        
        self.state.velocity = (
            self.state.velocity +
            g * dt +
            R @ delta_v
        )
        
        self.state.rotation = normalize_rotation(R @ delta_R)
        
        # Covariance propagation using pre-integrated covariance
        self._propagate_covariance_preintegrated(preint, delta_R)
    
    def _propagate_covariance_preintegrated(
        self,
        preint: PreintegratedMeasurement,
        delta_R: np.ndarray,
    ):
        """Propagate covariance using pre-integrated measurement covariance."""
        R = self.state.rotation
        dt = preint.delta_t
        
        # State transition Jacobian
        F = np.eye(self.STATE_DIM)
        
        # Position
        F[self.IDX_P, self.IDX_V] = np.eye(3) * dt
        F[self.IDX_P, self.IDX_R] = -R @ skew(preint.delta_p)
        
        # Velocity
        F[self.IDX_V, self.IDX_R] = -R @ skew(preint.delta_v)
        
        # Rotation
        F[self.IDX_R, self.IDX_R] = delta_R.T
        
        # Jacobians w.r.t. bias (from pre-integration)
        F[self.IDX_P, self.IDX_BG] = R @ preint.dp_dbg
        F[self.IDX_P, self.IDX_BA] = R @ preint.dp_dba
        F[self.IDX_V, self.IDX_BG] = R @ preint.dv_dbg
        F[self.IDX_V, self.IDX_BA] = R @ preint.dv_dba
        F[self.IDX_R, self.IDX_BG] = delta_R.T @ preint.dR_dbg
        
        # Transform pre-integrated covariance to state space
        # Pre-integrated cov is [δθ, δv, δp]
        G_preint = np.zeros((self.STATE_DIM, 9))
        G_preint[self.IDX_R, 0:3] = np.eye(3)
        G_preint[self.IDX_V, 3:6] = R
        G_preint[self.IDX_P, 6:9] = R
        
        Q_preint = G_preint @ preint.covariance @ G_preint.T
        
        # Add bias random walk noise
        Q_bias = np.zeros((self.STATE_DIM, self.STATE_DIM))
        Q_bias[self.IDX_BG, self.IDX_BG] = np.eye(3) * self.config.gyro_walk ** 2 * dt
        Q_bias[self.IDX_BA, self.IDX_BA] = np.eye(3) * self.config.accel_walk ** 2 * dt
        
        # Propagate covariance
        self.P = F @ self.P @ F.T + Q_preint + Q_bias
        self.P = 0.5 * (self.P + self.P.T)
    
    def update_visual(
        self,
        observations: np.ndarray,
        predictions: np.ndarray,
        camera_matrix: np.ndarray,
        landmark_positions: Optional[np.ndarray] = None,
    ) -> float:
        """
        Update step using visual measurements.
        
        Uses reprojection error as the measurement model.
        
        Args:
            observations: Nx2 observed feature positions (pixels)
            predictions: Nx2 predicted feature positions (pixels)
            camera_matrix: 3x3 camera intrinsic matrix
            landmark_positions: Nx3 landmark positions in world frame (optional)
            
        Returns:
            Normalized innovation squared (NIS) for consistency check
        """
        if not self.initialized:
            raise RuntimeError("EKF not initialized")
        
        n_features = len(observations)
        if n_features == 0:
            return 0.0
        
        # Innovation (measurement residual)
        z = (observations - predictions).flatten()  # 2N x 1
        
        # Measurement noise covariance
        R_meas = np.eye(2 * n_features) * self.config.pixel_noise ** 2
        
        # Measurement Jacobian
        H = self._compute_visual_jacobian(
            observations, camera_matrix, landmark_positions
        )
        
        if H is None:
            # Fallback: simple update using pixel error
            return self._simple_visual_update(z, R_meas)
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_meas
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix, skip update
            return 0.0
        
        # Error state update
        dx = K @ z
        
        # Apply correction to nominal state
        self._apply_error_state_correction(dx)
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        # Compute NIS for consistency check
        nis = z.T @ np.linalg.inv(S) @ z / len(z)
        
        return nis
    
    def _compute_visual_jacobian(
        self,
        observations: np.ndarray,
        camera_matrix: np.ndarray,
        landmark_positions: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Compute measurement Jacobian for visual features.
        
        The Jacobian relates error state to pixel measurement.
        """
        if landmark_positions is None:
            return None
        
        n_features = len(observations)
        H = np.zeros((2 * n_features, self.STATE_DIM))
        
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        
        R = self.state.rotation
        p = self.state.position
        
        for i, pw in enumerate(landmark_positions):
            # Transform to camera frame
            pc = R.T @ (pw - p)
            
            if pc[2] < 0.1:
                continue
            
            x, y, z = pc
            z_inv = 1.0 / z
            z_inv2 = z_inv ** 2
            
            # Jacobian of projection w.r.t. camera-frame point
            J_proj = np.array([
                [fx * z_inv, 0, -fx * x * z_inv2],
                [0, fy * z_inv, -fy * y * z_inv2],
            ])
            
            # Jacobian of camera-frame point w.r.t. error state
            # pc = R^T (pw - p)
            
            # w.r.t. position: -R^T
            J_p = -R.T
            
            # w.r.t. rotation: [R^T (pw - p)]_x = [pc]_x
            J_theta = skew(pc)
            
            # Combined Jacobian
            H[2*i:2*i+2, self.IDX_P] = J_proj @ J_p
            H[2*i:2*i+2, self.IDX_R] = J_proj @ J_theta
        
        return H
    
    def _simple_visual_update(
        self,
        innovation: np.ndarray,
        R_meas: np.ndarray,
    ) -> float:
        """
        Simple visual update without landmark positions.
        
        Uses a simplified measurement model that only updates
        the rotation based on pixel errors.
        """
        # Simplified: assume measurement mostly affects rotation
        n_features = len(innovation) // 2
        
        # Approximate Jacobian (identity mapping to rotation)
        H = np.zeros((len(innovation), self.STATE_DIM))
        for i in range(n_features):
            H[2*i, self.IDX_R[0]] = 1.0
            H[2*i+1, self.IDX_R[1]] = 1.0
        
        # Scale factor for pixel to rotation
        scale = 1e-4
        H *= scale
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_meas
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return 0.0
        
        # Error state update
        dx = K @ innovation
        
        # Limit correction magnitude
        dx = np.clip(dx, -0.1, 0.1)
        
        # Apply correction
        self._apply_error_state_correction(dx)
        
        # Covariance update
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        return np.mean(innovation ** 2)
    
    def _apply_error_state_correction(self, dx: np.ndarray):
        """
        Apply error state correction to nominal state.
        
        The error state is then reset to zero.
        """
        # Position correction
        self.state.position = self.state.position + dx[self.IDX_P]
        
        # Velocity correction
        self.state.velocity = self.state.velocity + dx[self.IDX_V]
        
        # Rotation correction (multiplicative)
        dtheta = dx[self.IDX_R]
        dR = exp_so3(dtheta)
        self.state.rotation = normalize_rotation(self.state.rotation @ dR)
        
        # Bias corrections (additive)
        self.state.gyro_bias = self.state.gyro_bias + dx[self.IDX_BG]
        self.state.accel_bias = self.state.accel_bias + dx[self.IDX_BA]
        
        # Update IMU integrator bias
        self.imu_integrator.bias = self.state.bias
    
    def get_state(self) -> State:
        """Get current state estimate."""
        return self.state.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get current error state covariance."""
        return self.P.copy()
    
    def get_position_covariance(self) -> np.ndarray:
        """Get 3x3 position covariance."""
        return self.P[self.IDX_P, self.IDX_P].copy()
    
    def get_velocity_covariance(self) -> np.ndarray:
        """Get 3x3 velocity covariance."""
        return self.P[self.IDX_V, self.IDX_V].copy()
    
    def get_rotation_covariance(self) -> np.ndarray:
        """Get 3x3 rotation (axis-angle) covariance."""
        return self.P[self.IDX_R, self.IDX_R].copy()
    
    def get_pose_covariance(self) -> np.ndarray:
        """Get 6x6 pose (position + rotation) covariance."""
        indices = np.concatenate([
            np.arange(self.IDX_P.start, self.IDX_P.stop),
            np.arange(self.IDX_R.start, self.IDX_R.stop),
        ])
        return self.P[np.ix_(indices, indices)].copy()
    
    def compute_nis(
        self,
        innovation: np.ndarray,
        innovation_covariance: np.ndarray,
    ) -> float:
        """
        Compute Normalized Innovation Squared (NIS).
        
        NIS should follow chi-squared distribution with
        degrees of freedom equal to measurement dimension.
        
        Args:
            innovation: Measurement residual
            innovation_covariance: Innovation covariance matrix
            
        Returns:
            NIS value
        """
        try:
            nis = innovation.T @ np.linalg.inv(innovation_covariance) @ innovation
            return nis / len(innovation)
        except np.linalg.LinAlgError:
            return float('inf')
    
    def compute_nees(self, true_state: State) -> float:
        """
        Compute Normalized Estimation Error Squared (NEES).
        
        NEES should follow chi-squared distribution with
        degrees of freedom equal to state dimension.
        
        Args:
            true_state: Ground truth state
            
        Returns:
            NEES value
        """
        # Compute error between estimated and true state
        error = np.zeros(self.STATE_DIM)
        
        # Position error
        error[self.IDX_P] = self.state.position - true_state.position
        
        # Velocity error
        error[self.IDX_V] = self.state.velocity - true_state.velocity
        
        # Rotation error (in tangent space)
        R_err = self.state.rotation @ true_state.rotation.T
        error[self.IDX_R] = log_so3(R_err)
        
        # Bias errors
        error[self.IDX_BG] = self.state.gyro_bias - true_state.gyro_bias
        error[self.IDX_BA] = self.state.accel_bias - true_state.accel_bias
        
        # Compute NEES
        try:
            nees = error.T @ np.linalg.inv(self.P) @ error
            return nees / self.STATE_DIM
        except np.linalg.LinAlgError:
            return float('inf')
def initialize_from_ground_truth(
    gt_pose,
    config: Optional[EKFConfig] = None,
) -> Tuple[ExtendedKalmanFilter, State]:
    """
    Initialize EKF from ground truth pose.
    
    Args:
        gt_pose: Ground truth pose from dataset
        config: EKF configuration
        
    Returns:
        Tuple of (initialized EKF, initial state)
    """
    config = config if config else EKFConfig()
    ekf = ExtendedKalmanFilter(config)
    
    initial_state = State(
        position=gt_pose.position.copy(),
        velocity=gt_pose.velocity.copy(),
        rotation=gt_pose.rotation_matrix.copy(),
        gyro_bias=np.zeros(3),
        accel_bias=np.zeros(3),
        timestamp=gt_pose.timestamp,
    )
    
    ekf.initialize(initial_state)
    
    return ekf, initial_state
