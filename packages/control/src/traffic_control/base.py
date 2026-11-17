"""Base controller interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from traffic_domain.schemas import (
    TrafficState,
    ControllerRecommendation,
    SignalPhase,
    ControllerAction,
    IntersectionConfig,
)
from traffic_config.settings import ControlSettings


class BaseController(ABC):
    """Abstract base class for traffic signal controllers."""

    def __init__(
        self,
        intersection_config: IntersectionConfig,
        settings: ControlSettings | None = None,
    ):
        """Initialize controller.

        Args:
            intersection_config: Intersection configuration
            settings: Control settings
        """
        self.config = intersection_config
        self.settings = settings
        self._current_phase: SignalPhase | None = None
        self._phase_start_time: datetime | None = None
        self._phase_elapsed: float = 0.0

    @property
    def current_phase(self) -> SignalPhase | None:
        return self._current_phase

    @property
    def phase_elapsed(self) -> float:
        return self._phase_elapsed

    def set_phase(self, phase: SignalPhase, timestamp: datetime | None = None) -> None:
        """Set current phase (called when phase actually changes)."""
        self._current_phase = phase
        self._phase_start_time = timestamp or datetime.now(timezone.utc)
        self._phase_elapsed = 0.0

    def update_phase_elapsed(self, delta_seconds: float) -> None:
        """Update elapsed time in current phase."""
        self._phase_elapsed += delta_seconds

    @abstractmethod
    def compute_action(self, state: TrafficState) -> ControllerRecommendation:
        """Compute control action for given traffic state.

        Args:
            state: Current traffic state

        Returns:
            Controller recommendation
        """
        pass

    def _enforce_constraints(
        self,
        action: ControllerAction,
        duration: float,
        current_phase: SignalPhase,
    ) -> tuple[ControllerAction, float]:
        """Enforce safety constraints on controller action.

        Args:
            action: Proposed action
            duration: Proposed duration
            current_phase: Current signal phase

        Returns:
            Tuple of (constrained_action, constrained_duration)
        """
        if not self.settings:
            return action, duration

        min_green = self.settings.min_green_time
        max_green = self.settings.max_green_time

        # If extending green, ensure we don't exceed max
        if action == ControllerAction.EXTEND_GREEN:
            duration = min(duration, max_green - self._phase_elapsed)
            if duration <= 0:
                # Can't extend, must switch
                action = ControllerAction.SWITCH_PHASE
                duration = min_green

        # If switching phase, ensure min green was met
        if action == ControllerAction.SWITCH_PHASE:
            if self._phase_elapsed < min_green:
                # Not enough green time yet, hold instead
                action = ControllerAction.HOLD_PHASE
                duration = min_green - self._phase_elapsed

        return action, max(0.0, duration)

    def _check_pedestrian_conflict(self, phase: SignalPhase) -> bool:
        """Check if phase conflicts with pedestrian crossing."""
        # Simplified: assume pedestrian phase is separate
        return False

    def _check_emergency_preempt(self, state: TrafficState) -> bool:
        """Check if emergency vehicle preemption is needed."""
        # Would check for emergency vehicles in state
        return False