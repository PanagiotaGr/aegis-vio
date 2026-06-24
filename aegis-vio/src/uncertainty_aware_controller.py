"""
Uncertainty-Aware Navigation Controller
NOVEL RESEARCH CONTRIBUTION
This module implements adaptive navigation that modifies behavior
based on state estimation uncertainty:
1. Speed Reduction: Reduce velocity when uncertainty is high
2. Safety Margin Increase: Expand collision avoidance margins
3. Observability Seeking: Prefer trajectories that improve state observability
4. Conservative Planning: Switch to more conservative paths under uncertainty
Reference: Novel contribution for uncertainty-aware autonomous navigation
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
from .navigation import (
    NavigationController, NavigationConfig, NavigationCommand,
    NavigationState, Waypoint
)
from .uncertainty import UncertaintyMetrics, UncertaintyLevel, UncertaintyEstimator
from .ekf import State
class AdaptationMode(Enum):
    """Uncertainty adaptation modes."""
    NONE = "none"              # No adaptation
    SPEED_ONLY = "speed"       # Only adapt speed
    MARGIN_ONLY = "margin"     # Only adapt safety margins
    FULL = "full"              # Full adaptation (speed + margin + observability)
@dataclass
class UncertaintyAdaptationConfig:
    """Configuration for uncertainty-aware adaptation."""
    # Speed adaptation
    min_speed_factor: float = 0.2       # Minimum speed as fraction of nominal
    speed_reduction_rate: float = 2.0   # How aggressively to reduce speed
    
    # Safety margin adaptation
    base_safety_margin: float = 0.5     # Base safety margin (m)
    max_safety_margin: float = 2.0      # Maximum safety margin (m)
    margin_scale_factor: float = 3.0    # How much to scale margin with uncertainty
    
    # Observability preferences
    prefer_textured_regions: bool = True
    observability_weight: float = 0.3
    
    # Uncertainty thresholds
    speed_reduction_threshold: float = 0.1   # Start reducing speed above this uncertainty
    emergency_threshold: float = 0.5         # Trigger emergency behavior above this
    recovery_threshold: float = 0.08         # Resume normal behavior below this
    
    # Adaptation rate
    adaptation_smoothing: float = 0.9   # Exponential smoothing factor
@dataclass
class AdaptationState:
    """Current state of uncertainty adaptation."""
    speed_factor: float = 1.0
    safety_margin: float = 0.5
    in_recovery_mode: bool = False
    adaptation_active: bool = False
    observability_score: float = 1.0
    
    # History for analysis
    uncertainty_history: List[float] = field(default_factory=list)
    speed_factor_history: List[float] = field(default_factory=list)
class UncertaintyAwareController:
    """
    Uncertainty-aware navigation controller.
    
    Extends basic navigation with adaptive behavior based on
    state estimation uncertainty.
    
    Key Features:
    1. Adaptive Speed: Reduces speed when position uncertainty is high
    2. Dynamic Safety Margins: Increases obstacle clearance under uncertainty
    3. Observability Seeking: Steers toward feature-rich regions
    4. Recovery Mode: Special handling for high-uncertainty situations
    
    Example:
        controller = UncertaintyAwareController(
            nav_config=NavigationConfig(),
            adaptation_config=UncertaintyAdaptationConfig(),
        )
        
        controller.set_waypoints(waypoints)
        
        while controller.has_waypoints():
            cmd = controller.compute_adaptive_command(
                state=current_state,
                uncertainty=uncertainty_metrics,
            )
            apply_command(cmd)
    """
    
    def __init__(
        self,
        nav_config: Optional[NavigationConfig] = None,
        adaptation_config: Optional[UncertaintyAdaptationConfig] = None,
        mode: AdaptationMode = AdaptationMode.FULL,
    ):
        """
        Initialize uncertainty-aware controller.
        
        Args:
            nav_config: Base navigation configuration
            adaptation_config: Uncertainty adaptation configuration
            mode: Adaptation mode
        """
        self.nav_config = nav_config if nav_config else NavigationConfig()
        self.adaptation_config = adaptation_config if adaptation_config else UncertaintyAdaptationConfig()
        self.mode = mode
        
        # Base navigation controller
        self.nav_controller = NavigationController(self.nav_config)
        
        # Adaptation state
        self.adaptation_state = AdaptationState(
            safety_margin=self.adaptation_config.base_safety_margin
        )
        
        # Uncertainty estimator for additional analysis
        self.uncertainty_estimator = UncertaintyEstimator()
    
    def set_waypoints(self, waypoints: List[Waypoint]):
        """Set navigation waypoints."""
        self.nav_controller.set_waypoints(waypoints)
        self.reset_adaptation()
    
    def add_waypoint(self, waypoint: Waypoint):
        """Add a waypoint."""
        self.nav_controller.add_waypoint(waypoint)
    
    def has_waypoints(self) -> bool:
        """Check if there are remaining waypoints."""
        return self.nav_controller.has_waypoints()
    
    def reset_adaptation(self):
        """Reset adaptation state."""
        self.adaptation_state = AdaptationState(
            safety_margin=self.adaptation_config.base_safety_margin
        )
    
    def compute_adaptive_command(
        self,
        state: State,
        uncertainty: UncertaintyMetrics,
        obstacles: Optional[List[np.ndarray]] = None,
    ) -> NavigationCommand:
        """
        Compute navigation command with uncertainty adaptation.
        
        Args:
            state: Current state estimate
            uncertainty: Current uncertainty metrics
            obstacles: Optional list of obstacle positions
            
        Returns:
            Adapted navigation command
        """
        # Update adaptation parameters
        self._update_adaptation(uncertainty)
        
        # Get base command
        base_cmd = self.nav_controller.compute_command(
            position=state.position,
            velocity=state.velocity,
            yaw=self._get_yaw_from_rotation(state.rotation),
        )
        
        if base_cmd.is_completed:
            return base_cmd
        
        # Apply adaptations
        adapted_velocity = self._adapt_velocity(
            base_cmd.velocity_command,
            uncertainty,
        )
        
        # Check obstacles with adapted safety margin
        if obstacles:
            adapted_velocity = self._apply_obstacle_avoidance(
                state.position,
                adapted_velocity,
                obstacles,
            )
        
        # Apply observability seeking if enabled
        if self.mode == AdaptationMode.FULL:
            adapted_velocity = self._apply_observability_preference(
                state.position,
                adapted_velocity,
                uncertainty,
            )
        
        return NavigationCommand(
            velocity_command=adapted_velocity,
            yaw_rate_command=base_cmd.yaw_rate_command,
            is_completed=base_cmd.is_completed,
            distance_to_target=base_cmd.distance_to_target,
        )
    
    def _update_adaptation(self, uncertainty: UncertaintyMetrics):
        """Update adaptation parameters based on uncertainty."""
        pos_unc = uncertainty.position_uncertainty
        
        # Update history
        self.adaptation_state.uncertainty_history.append(pos_unc)
        if len(self.adaptation_state.uncertainty_history) > 100:
            self.adaptation_state.uncertainty_history.pop(0)
        
        # Check for high uncertainty
        if pos_unc > self.adaptation_config.emergency_threshold:
            self.adaptation_state.in_recovery_mode = True
            self.adaptation_state.adaptation_active = True
        elif pos_unc < self.adaptation_config.recovery_threshold:
            self.adaptation_state.in_recovery_mode = False
            if pos_unc < self.adaptation_config.speed_reduction_threshold / 2:
                self.adaptation_state.adaptation_active = False
        else:
            self.adaptation_state.adaptation_active = (
                pos_unc > self.adaptation_config.speed_reduction_threshold
            )
        
        # Compute speed factor
        if self.mode in [AdaptationMode.SPEED_ONLY, AdaptationMode.FULL]:
            target_speed_factor = self._compute_speed_factor(pos_unc)
        else:
            target_speed_factor = 1.0
        
        # Smooth speed factor changes
        alpha = self.adaptation_config.adaptation_smoothing
        self.adaptation_state.speed_factor = (
            alpha * self.adaptation_state.speed_factor +
            (1 - alpha) * target_speed_factor
        )
        
        self.adaptation_state.speed_factor_history.append(
            self.adaptation_state.speed_factor
        )
        if len(self.adaptation_state.speed_factor_history) > 100:
            self.adaptation_state.speed_factor_history.pop(0)
        
        # Compute safety margin
        if self.mode in [AdaptationMode.MARGIN_ONLY, AdaptationMode.FULL]:
            self.adaptation_state.safety_margin = self._compute_safety_margin(pos_unc)
        
        # Update observability score
        self.adaptation_state.observability_score = uncertainty.tracking_quality
    
    def _compute_speed_factor(self, position_uncertainty: float) -> float:
        """
        Compute speed reduction factor based on uncertainty.
        
        Uses an exponential reduction curve.
        """
        threshold = self.adaptation_config.speed_reduction_threshold
        rate = self.adaptation_config.speed_reduction_rate
        min_factor = self.adaptation_config.min_speed_factor
        
        if position_uncertainty < threshold:
            return 1.0
        
        # Exponential reduction
        excess = position_uncertainty - threshold
        factor = np.exp(-rate * excess)
        
        return max(min_factor, factor)
    
    def _compute_safety_margin(self, position_uncertainty: float) -> float:
        """
        Compute adaptive safety margin based on uncertainty.
        
        Margin increases linearly with uncertainty.
        """
        base = self.adaptation_config.base_safety_margin
        max_margin = self.adaptation_config.max_safety_margin
        scale = self.adaptation_config.margin_scale_factor
        
        margin = base + scale * position_uncertainty
        
        return min(max_margin, margin)
    
    def _adapt_velocity(
        self,
        velocity: np.ndarray,
        uncertainty: UncertaintyMetrics,
    ) -> np.ndarray:
        """Apply speed factor to velocity command."""
        speed = np.linalg.norm(velocity)
        
        if speed < 1e-6:
            return velocity
        
        # Apply speed factor
        adapted_speed = speed * self.adaptation_state.speed_factor
        
        # Additional reduction in recovery mode
        if self.adaptation_state.in_recovery_mode:
            adapted_speed *= 0.5
        
        return velocity / speed * adapted_speed
    
    def _apply_obstacle_avoidance(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        obstacles: List[np.ndarray],
    ) -> np.ndarray:
        """
        Apply obstacle avoidance with adaptive safety margin.
        
        Uses potential field method.
        """
        repulsion = np.zeros(3)
        margin = self.adaptation_state.safety_margin
        
        for obs in obstacles:
            diff = position - obs
            dist = np.linalg.norm(diff)
            
            if dist < margin * 2:
                # Repulsive force
                if dist > 0.1:
                    direction = diff / dist
                else:
                    direction = np.array([1, 0, 0])
                
                # Force magnitude increases as we get closer
                strength = (margin * 2 - dist) / margin
                repulsion += direction * strength
        
        # Combine with desired velocity
        adapted = velocity + repulsion
        
        # Maintain speed limit
        speed = np.linalg.norm(adapted)
        max_speed = np.linalg.norm(velocity)
        
        if speed > max_speed and speed > 0:
            adapted = adapted / speed * max_speed
        
        return adapted
    
    def _apply_observability_preference(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        uncertainty: UncertaintyMetrics,
    ) -> np.ndarray:
        """
        Bias velocity toward directions with better observability.
        
        This is a simplified heuristic. A full implementation would
        use feature density maps or learned observability predictions.
        """
        if not self.adaptation_config.prefer_textured_regions:
            return velocity
        
        # Only apply when tracking quality is low
        if uncertainty.tracking_quality > 0.7:
            return velocity
        
        weight = self.adaptation_config.observability_weight
        
        # Heuristic: prefer motion that doesn't point directly up or down
        # (cameras generally see more features looking horizontally)
        vertical_component = velocity[2]
        
        if abs(vertical_component) > np.linalg.norm(velocity[:2]):
            # Reduce vertical motion
            velocity_adapted = velocity.copy()
            velocity_adapted[2] *= (1 - weight)
            
            # Maintain speed
            speed_orig = np.linalg.norm(velocity)
            speed_new = np.linalg.norm(velocity_adapted)
            
            if speed_new > 0.1:
                velocity_adapted = velocity_adapted / speed_new * speed_orig
            
            return velocity_adapted
        
        return velocity
    
    def _get_yaw_from_rotation(self, R: np.ndarray) -> float:
        """Extract yaw angle from rotation matrix."""
        return np.arctan2(R[1, 0], R[0, 0])
    
    def get_adaptation_state(self) -> AdaptationState:
        """Get current adaptation state."""
        return self.adaptation_state
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        return {
            'mode': self.mode.value,
            'speed_factor': self.adaptation_state.speed_factor,
            'safety_margin': self.adaptation_state.safety_margin,
            'in_recovery_mode': self.adaptation_state.in_recovery_mode,
            'adaptation_active': self.adaptation_state.adaptation_active,
            'observability_score': self.adaptation_state.observability_score,
            'avg_uncertainty': (
                np.mean(self.adaptation_state.uncertainty_history)
                if self.adaptation_state.uncertainty_history else 0.0
            ),
            'avg_speed_factor': (
                np.mean(self.adaptation_state.speed_factor_history)
                if self.adaptation_state.speed_factor_history else 1.0
            ),
        }
    
    def emergency_stop(self):
        """Trigger emergency stop."""
        self.nav_controller.emergency_stop()
        self.adaptation_state.in_recovery_mode = True
        self.adaptation_state.speed_factor = 0.0
    
    def compute_risk_adjusted_path(
        self,
        current_position: np.ndarray,
        target_position: np.ndarray,
        position_uncertainty: float,
        obstacles: Optional[List[np.ndarray]] = None,
    ) -> List[Waypoint]:
        """
        Generate a risk-adjusted path to target.
        
        Creates intermediate waypoints that account for uncertainty.
        
        Args:
            current_position: Current position
            target_position: Target position
            position_uncertainty: Current position uncertainty
            obstacles: Optional obstacle positions
            
        Returns:
            List of waypoints
        """
        distance = np.linalg.norm(target_position - current_position)
        
        # More waypoints when uncertainty is high
        base_spacing = 2.0  # meters
        uncertainty_factor = max(1.0, position_uncertainty * 10)
        spacing = base_spacing / uncertainty_factor
        
        n_waypoints = max(2, int(distance / spacing))
        
        waypoints = []
        
        for i in range(1, n_waypoints + 1):
            t = i / n_waypoints
            position = (1 - t) * current_position + t * target_position
            
            # Add lateral offset if near obstacles
            if obstacles:
                margin = self._compute_safety_margin(position_uncertainty)
                offset = self._compute_obstacle_offset(position, obstacles, margin)
                position = position + offset
            
            # Reduce velocity based on uncertainty
            velocity = self.nav_config.max_velocity * self._compute_speed_factor(
                position_uncertainty
            )
            
            waypoints.append(Waypoint(
                position=position,
                velocity=velocity,
                tolerance=max(0.3, position_uncertainty * 2),
            ))
        
        return waypoints
    
    def _compute_obstacle_offset(
        self,
        position: np.ndarray,
        obstacles: List[np.ndarray],
        margin: float,
    ) -> np.ndarray:
        """Compute position offset to avoid obstacles."""
        offset = np.zeros(3)
        
        for obs in obstacles:
            diff = position - obs
            dist = np.linalg.norm(diff)
            
            if dist < margin * 1.5 and dist > 0.1:
                direction = diff / dist
                push = (margin * 1.5 - dist) * direction
                offset += push
        
        return offset
def evaluate_path_uncertainty(
    path: List[Waypoint],
    uncertainty_predictions: List[float],
) -> Dict:
    """
    Evaluate path quality considering predicted uncertainty.
    
    Args:
        path: List of waypoints
        uncertainty_predictions: Predicted uncertainty at each waypoint
        
    Returns:
        Path quality metrics
    """
    if len(path) != len(uncertainty_predictions):
        raise ValueError("Path and predictions must have same length")
    
    uncertainties = np.array(uncertainty_predictions)
    
    return {
        'total_length': sum(
            np.linalg.norm(path[i+1].position - path[i].position)
            for i in range(len(path) - 1)
        ),
        'max_uncertainty': np.max(uncertainties),
        'mean_uncertainty': np.mean(uncertainties),
        'uncertainty_integral': np.sum(uncertainties),
        'uncertainty_variance': np.var(uncertainties),
        'high_uncertainty_segments': np.sum(uncertainties > 0.3),
    }
