"""Uncertainty-aware navigation controller.

This module is the research core of AegisVIO: it converts estimator
uncertainty into a discrete navigation mode that can later be used by a robot
or ROS2 node.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NavigationMode(str, Enum):
    NOMINAL = "nominal"
    CAUTIOUS = "cautious"
    RECOVERY = "recovery"


@dataclass
class ControllerConfig:
    cautious_threshold: float = 0.10
    recovery_threshold: float = 0.35
    hysteresis: float = 0.03
    nominal_speed: float = 1.0
    cautious_speed: float = 0.5
    recovery_speed: float = 0.2


class UncertaintyAwareController:
    """Threshold-and-hysteresis controller for uncertainty-aware autonomy."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self.mode = NavigationMode.NOMINAL

    def update(self, uncertainty_score: float) -> NavigationMode:
        c = self.config
        if self.mode == NavigationMode.NOMINAL:
            if uncertainty_score >= c.recovery_threshold:
                self.mode = NavigationMode.RECOVERY
            elif uncertainty_score >= c.cautious_threshold:
                self.mode = NavigationMode.CAUTIOUS
        elif self.mode == NavigationMode.CAUTIOUS:
            if uncertainty_score >= c.recovery_threshold:
                self.mode = NavigationMode.RECOVERY
            elif uncertainty_score < c.cautious_threshold - c.hysteresis:
                self.mode = NavigationMode.NOMINAL
        else:
            if uncertainty_score < c.cautious_threshold - c.hysteresis:
                self.mode = NavigationMode.NOMINAL
            elif uncertainty_score < c.recovery_threshold - c.hysteresis:
                self.mode = NavigationMode.CAUTIOUS
        return self.mode

    def speed_scale(self) -> float:
        if self.mode == NavigationMode.NOMINAL:
            return self.config.nominal_speed
        if self.mode == NavigationMode.CAUTIOUS:
            return self.config.cautious_speed
        return self.config.recovery_speed
