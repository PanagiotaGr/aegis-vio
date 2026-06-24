"""
Navigation Controller for Autonomous Drones
Implements waypoint following and basic navigation with
hooks for uncertainty-aware decision making.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
class NavigationState(Enum):
    """Navigation state machine states."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    HOVERING = "hovering"
    LANDING = "landing"
    EMERGENCY = "emergency"
@dataclass
class Waypoint:
    """Navigation waypoint."""
    position: np.ndarray  # [x, y, z] in world frame
    heading: Optional[float] = None  # Desired yaw angle (radians)
    velocity: float = 1.0  # Desired velocity (m/s)
    tolerance: float = 0.5  # Position tolerance (m)
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
@dataclass
class NavigationCommand:
    """Command output from navigation controller."""
    velocity_command: np.ndarray  # Desired velocity [vx, vy, vz]
    yaw_rate_command: float  # Desired yaw rate (rad/s)
    is_completed: bool  # True if current waypoint reached
    distance_to_target: float
    
    def __post_init__(self):
        self.velocity_command = np.asarray(self.velocity_command, dtype=np.float64)
@dataclass
class NavigationConfig:
    """Navigation controller configuration."""
    max_velocity: float = 2.0  # m/s
    max_acceleration: float = 1.0  # m/s^2
    max_yaw_rate: float = 1.0  # rad/s
    
    position_gain: float = 1.0
    velocity_gain: float = 0.5
    yaw_gain: float = 1.0
    
    waypoint_tolerance: float = 0.5  # m
    heading_tolerance: float = 0.1  # rad
    
    slowdown_distance: float = 3.0  # Start slowing down at this distance
    min_velocity: float = 0.3  # Minimum velocity when approaching waypoint
class NavigationController:
    """
    Basic navigation controller for waypoint following.
    
    Implements simple proportional control with velocity limits.
    Designed to be extended by UncertaintyAwareController.
    
    Example:
        nav = NavigationController()
        nav.set_waypoints([
            Waypoint([0, 0, 1]),
            Waypoint([5, 0, 1]),
            Waypoint([5, 5, 1]),
        ])
        
        while nav.has_waypoints():
            cmd = nav.compute_command(current_position, current_velocity)
            send_to_drone(cmd)
    """
    
    def __init__(self, config: Optional[NavigationConfig] = None):
        """
        Initialize navigation controller.
        
        Args:
            config: Navigation configuration
        """
        self.config = config if config else NavigationConfig()
        
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx: int = 0
        self.state: NavigationState = NavigationState.IDLE
        
        # For velocity smoothing
        self.last_velocity_command: np.ndarray = np.zeros(3)
    
    def set_waypoints(self, waypoints: List[Waypoint]):
        """
        Set navigation waypoints.
        
        Args:
            waypoints: List of waypoints to follow
        """
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.state = NavigationState.NAVIGATING if waypoints else NavigationState.IDLE
    
    def add_waypoint(self, waypoint: Waypoint):
        """Add a single waypoint to the end of the list."""
        self.waypoints.append(waypoint)
        if self.state == NavigationState.IDLE:
            self.state = NavigationState.NAVIGATING
    
    def has_waypoints(self) -> bool:
        """Check if there are remaining waypoints."""
        return self.current_waypoint_idx < len(self.waypoints)
    
    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Get the current target waypoint."""
        if self.has_waypoints():
            return self.waypoints[self.current_waypoint_idx]
        return None
    
    def compute_command(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        yaw: float = 0.0,
    ) -> NavigationCommand:
        """
        Compute navigation command.
        
        Args:
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            yaw: Current yaw angle (radians)
            
        Returns:
            NavigationCommand with desired velocities
        """
        if not self.has_waypoints() or self.state != NavigationState.NAVIGATING:
            return NavigationCommand(
                velocity_command=np.zeros(3),
                yaw_rate_command=0.0,
                is_completed=True,
                distance_to_target=0.0,
            )
        
        waypoint = self.waypoints[self.current_waypoint_idx]
        
        # Position error
        position_error = waypoint.position - position
        distance = np.linalg.norm(position_error)
        
        # Check if waypoint reached
        if distance < waypoint.tolerance:
            self.current_waypoint_idx += 1
            if not self.has_waypoints():
                self.state = NavigationState.HOVERING
            return NavigationCommand(
                velocity_command=np.zeros(3),
                yaw_rate_command=0.0,
                is_completed=True,
                distance_to_target=distance,
            )
        
        # Compute desired velocity direction
        direction = position_error / distance
        
        # Compute desired speed (with slowdown near waypoint)
        max_speed = min(waypoint.velocity, self.config.max_velocity)
        
        if distance < self.config.slowdown_distance:
            # Linear slowdown
            speed_factor = distance / self.config.slowdown_distance
            desired_speed = max(
                self.config.min_velocity,
                max_speed * speed_factor
            )
        else:
            desired_speed = max_speed
        
        # Desired velocity
        desired_velocity = direction * desired_speed
        
        # Velocity error for damping
        velocity_error = desired_velocity - velocity
        
        # Apply gains
        velocity_command = (
            self.config.position_gain * position_error +
            self.config.velocity_gain * velocity_error
        )
        
        # Limit velocity
        speed = np.linalg.norm(velocity_command)
        if speed > self.config.max_velocity:
            velocity_command = velocity_command / speed * self.config.max_velocity
        
        # Limit acceleration (smooth commands)
        velocity_diff = velocity_command - self.last_velocity_command
        accel = np.linalg.norm(velocity_diff)
        dt = 0.02  # Assume 50Hz control rate
        if accel / dt > self.config.max_acceleration:
            velocity_diff = velocity_diff / accel * self.config.max_acceleration * dt
            velocity_command = self.last_velocity_command + velocity_diff
        
        self.last_velocity_command = velocity_command.copy()
        
        # Yaw control
        yaw_rate_command = 0.0
        if waypoint.heading is not None:
            yaw_error = self._wrap_angle(waypoint.heading - yaw)
            yaw_rate_command = self.config.yaw_gain * yaw_error
            yaw_rate_command = np.clip(
                yaw_rate_command,
                -self.config.max_yaw_rate,
                self.config.max_yaw_rate
            )
        
        return NavigationCommand(
            velocity_command=velocity_command,
            yaw_rate_command=yaw_rate_command,
            is_completed=False,
            distance_to_target=distance,
        )
    
    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def emergency_stop(self):
        """Trigger emergency stop."""
        self.state = NavigationState.EMERGENCY
        self.last_velocity_command = np.zeros(3)
    
    def reset(self):
        """Reset the navigation controller."""
        self.waypoints.clear()
        self.current_waypoint_idx = 0
        self.state = NavigationState.IDLE
        self.last_velocity_command = np.zeros(3)
    
    def get_remaining_distance(self, position: np.ndarray) -> float:
        """
        Get total remaining distance to final waypoint.
        
        Args:
            position: Current position
            
        Returns:
            Total remaining distance
        """
        if not self.has_waypoints():
            return 0.0
        
        total = 0.0
        current_pos = position.copy()
        
        for i in range(self.current_waypoint_idx, len(self.waypoints)):
            wp = self.waypoints[i]
            total += np.linalg.norm(wp.position - current_pos)
            current_pos = wp.position
        
        return total
    
    def compute_risk_score(
        self,
        position: np.ndarray,
        position_uncertainty: float,
    ) -> float:
        """
        Compute navigation risk score based on uncertainty.
        
        Args:
            position: Current position
            position_uncertainty: Position uncertainty (std dev)
            
        Returns:
            Risk score (0 = safe, 1 = high risk)
        """
        if not self.has_waypoints():
            return 0.0
        
        waypoint = self.waypoints[self.current_waypoint_idx]
        distance = np.linalg.norm(waypoint.position - position)
        
        # Risk increases when uncertainty is large relative to distance
        if distance < 0.1:
            return 0.0
        
        uncertainty_ratio = position_uncertainty / distance
        
        # Sigmoid-like mapping
        risk = 2.0 / (1.0 + np.exp(-5 * (uncertainty_ratio - 0.3))) - 1.0
        
        return np.clip(risk, 0.0, 1.0)
def generate_circular_trajectory(
    center: np.ndarray,
    radius: float,
    altitude: float,
    n_waypoints: int = 8,
    velocity: float = 1.0,
) -> List[Waypoint]:
    """
    Generate circular trajectory waypoints.
    
    Args:
        center: Center of circle [x, y]
        radius: Circle radius
        altitude: Flight altitude
        n_waypoints: Number of waypoints
        velocity: Desired velocity
        
    Returns:
        List of waypoints
    """
    waypoints = []
    angles = np.linspace(0, 2 * np.pi, n_waypoints, endpoint=False)
    
    for theta in angles:
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        
        # Heading tangent to circle
        heading = theta + np.pi / 2
        
        waypoints.append(Waypoint(
            position=np.array([x, y, altitude]),
            heading=heading,
            velocity=velocity,
        ))
    
    return waypoints
def generate_lawnmower_pattern(
    start: np.ndarray,
    width: float,
    length: float,
    altitude: float,
    spacing: float,
    velocity: float = 1.0,
) -> List[Waypoint]:
    """
    Generate lawnmower (survey) pattern waypoints.
    
    Args:
        start: Starting corner [x, y]
        width: Survey width
        length: Survey length
        altitude: Flight altitude
        spacing: Line spacing
        velocity: Desired velocity
        
    Returns:
        List of waypoints
    """
    waypoints = []
    n_lines = int(np.ceil(width / spacing))
    
    for i in range(n_lines):
        y = start[1] + i * spacing
        
        if i % 2 == 0:
            # Forward pass
            x1 = start[0]
            x2 = start[0] + length
        else:
            # Backward pass
            x1 = start[0] + length
            x2 = start[0]
        
        waypoints.append(Waypoint(
            position=np.array([x1, y, altitude]),
            velocity=velocity,
        ))
        waypoints.append(Waypoint(
            position=np.array([x2, y, altitude]),
            velocity=velocity,
        ))
    
    return waypoints
