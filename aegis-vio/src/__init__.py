"""
AEGIS-VIO: Uncertainty-Aware Visual-Inertial Navigation System
A research-grade implementation of Visual-Inertial Odometry with
uncertainty quantification and uncertainty-aware navigation control.
Author: Panagiota Gr
Institution: Research Project for PhD Application
Target Lab: Vision for Robotics Lab (V4RL), ETH Zurich
"""
__version__ = "1.0.0"
__author__ = "Panagiota Gr"
from .dataset_loader import EuRoCDatasetLoader
from .feature_tracker import FeatureTracker
from .imu_integrator import IMUIntegrator
from .vio_estimator import VIOEstimator
from .ekf import ExtendedKalmanFilter
from .uncertainty import UncertaintyEstimator
from .navigation import NavigationController
from .uncertainty_aware_controller import UncertaintyAwareController
from .evaluator import TrajectoryEvaluator
from .visualization import Visualizer
__all__ = [
    "EuRoCDatasetLoader",
    "FeatureTracker",
    "IMUIntegrator",
    "VIOEstimator",
    "ExtendedKalmanFilter",
    "UncertaintyEstimator",
    "NavigationController",
    "UncertaintyAwareController",
    "TrajectoryEvaluator",
    "Visualizer",
]
