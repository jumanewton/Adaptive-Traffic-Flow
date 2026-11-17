"""Fixed-time traffic signal controller."""

from __future__ import annotations

from datetime import datetime, timezone

from traffic_domain.schemas import (
    TrafficState,
    ControllerRecommendation,
    SignalPhase,
    ControllerAction,
    IntersectionConfig,
)
from traffic_config.settings import ControlSettings
from traffic_control.base import BaseController


class FixedTimeController(BaseController):
    """Fixed-time signal controller with predefined phase splits."""

    def __init__(
        self,
        intersection_config: IntersectionConfig,
        settings: ControlSettings | None = None,
    ):
        super().__init__(intersection_config, settings)

        # Default phase sequence for 4-phase intersection
        self._phase_sequence = [
            SignalPhase.NORTH_SOUTH_GREEN,
            SignalPhase.NORTH_SOUTH_YELLOW,
            SignalPhase.EAST_WEST_GREEN,
            SignalPhase.EAST_WEST_YELLOW,
        ]

        # Phase durations (seconds)
        self._phase_durations = {
            SignalPhase.NORTH_SOUTH_GREEN: 50,
            SignalPhase.NORTH_SOUTH_YELLOW: 3,
            SignalPhase.EAST_WEST_GREEN: 50,
            SignalPhase.EAST_WEST_YELLOW: 3,
        }

        # Override from settings if available
        if settings and settings.fixed_time_phase_splits:
            splits = settings.fixed_time_phase_splits
            cycle = settings.fixed_time_cycle_length
            ns_green = splits.get("north_south", cycle // 2 - 5)
            ew_green = splits.get("east_west", cycle // 2 - 5)
            self._phase_durations[SignalPhase.NORTH_SOUTH_GREEN] = ns_green
            self._phase_durations[SignalPhase.EAST_WEST_GREEN] = ew_green

        self._phase_index = 0

    def compute_action(self, state: TrafficState) -> ControllerRecommendation:
        """Compute fixed-time action based on cycle timing."""
        current_phase = self._current_phase or self._phase_sequence[0]

        # Get current phase duration
        phase_duration = self._phase_durations.get(current_phase, 30)

        # Time remaining in current phase
        time_remaining = phase_duration - self._phase_elapsed

        if time_remaining <= 0:
            # Switch to next phase
            self._phase_index = (self._phase_index + 1) % len(self._phase_sequence)
            next_phase = self._phase_sequence[self._phase_index]

            action = ControllerAction.SWITCH_PHASE
            duration = self._phase_durations.get(next_phase, 30)
            reason = f"Fixed-time cycle: switching from {current_phase.value} to {next_phase.value}"

            # Apply constraints
            action, duration = self._enforce_constraints(action, duration, current_phase)

            return ControllerRecommendation(
                intersection_id=state.intersection_id,
                timestamp=datetime.now(timezone.utc),
                current_phase=current_phase,
                action=action,
                duration_seconds=duration,
                reason=reason,
                controller_type="fixed_time",
                simulation_only=True,
            )
        else:
            # Hold current phase - update elapsed time
            if self.settings:
                self.update_phase_elapsed(self.settings.control_loop_interval_s)
            else:
                self.update_phase_elapsed(1.0)

            action = ControllerAction.HOLD_PHASE
            duration = time_remaining
            reason = f"Fixed-time: holding {current_phase.value} for {time_remaining:.1f}s"

            action, duration = self._enforce_constraints(action, duration, current_phase)

            return ControllerRecommendation(
                intersection_id=state.intersection_id,
                timestamp=datetime.now(timezone.utc),
                current_phase=current_phase,
                action=action,
                duration_seconds=duration,
                reason=reason,
                controller_type="fixed_time",
                simulation_only=True,
            )

    def reset(self) -> None:
        """Reset controller to initial state."""
        self._phase_index = 0
        self._current_phase = None
        self._phase_elapsed = 0.0