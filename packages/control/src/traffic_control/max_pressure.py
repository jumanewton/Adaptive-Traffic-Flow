"""Max-pressure traffic signal controller.

Implements the max-pressure control algorithm:
- Pressure = sum of upstream queue lengths - sum of downstream queue lengths
- Select phase with maximum pressure
- Considers downstream congestion
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from traffic_domain.schemas import (
    TrafficState,
    ControllerRecommendation,
    SignalPhase,
    ControllerAction,
    IntersectionConfig,
    ApproachState,
)
from traffic_config.settings import ControlSettings
from traffic_control.base import BaseController


class MaxPressureController(BaseController):
    """Max-pressure adaptive signal controller.

    Pressure for a phase = sum of queue lengths on approaches getting green
                          - sum of queue lengths on conflicting approaches
    """

    def __init__(
        self,
        intersection_config: IntersectionConfig,
        settings: ControlSettings | None = None,
    ):
        super().__init__(intersection_config, settings)

        # Phase to approach mapping (which approaches get green in each phase)
        self._phase_approaches: dict[SignalPhase, list[str]] = {
            SignalPhase.NORTH_SOUTH_GREEN: ["northbound", "southbound"],
            SignalPhase.EAST_WEST_GREEN: ["eastbound", "westbound"],
        }

        # Conflicting approaches for each phase
        self._phase_conflicts: dict[SignalPhase, list[str]] = {
            SignalPhase.NORTH_SOUTH_GREEN: ["eastbound", "westbound"],
            SignalPhase.EAST_WEST_GREEN: ["northbound", "southbound"],
        }

        # Downstream approach mapping (for downstream pressure)
        self._downstream: dict[str, str | None] = {}

        # Build downstream mapping from config
        for approach in intersection_config.approaches:
            # Simplified: assume straight-through goes to opposite approach
            if "north" in approach.approach_id:
                self._downstream[approach.approach_id] = "southbound"
            elif "south" in approach.approach_id:
                self._downstream[approach.approach_id] = "northbound"
            elif "east" in approach.approach_id:
                self._downstream[approach.approach_id] = "westbound"
            elif "west" in approach.approach_id:
                self._downstream[approach.approach_id] = "eastbound"
            else:
                self._downstream[approach.approach_id] = None

        # Controller parameters
        self._pressure_weight = 1.0
        self._downstream_weight = 0.5

        if settings:
            self._pressure_weight = settings.max_pressure_pressure_weight
            self._downstream_weight = settings.max_pressure_downstream_weight

    def compute_action(self, state: TrafficState) -> ControllerRecommendation:
        """Compute max-pressure action."""
        current_phase = self._current_phase or SignalPhase.NORTH_SOUTH_GREEN

        # Calculate pressure for each phase
        pressures = {}
        for phase in [SignalPhase.NORTH_SOUTH_GREEN, SignalPhase.EAST_WEST_GREEN]:
            pressures[phase] = self._calculate_pressure(state, phase)

        # Find phase with maximum pressure
        best_phase = max(pressures, key=pressures.get)
        max_pressure = pressures[best_phase]

        # Decide action
        if best_phase == current_phase:
            # Extend current green
            action = ControllerAction.EXTEND_GREEN
            duration = self._calculate_extension_duration(state, current_phase)
            reason = f"Max-pressure: staying on {current_phase.value} (pressure={max_pressure:.1f})"
        else:
            # Switch phase
            action = ControllerAction.SWITCH_PHASE
            duration = self._settings.min_green_time if self._settings else 10
            reason = (
                f"Max-pressure: switching from {current_phase.value} to {best_phase.value} "
                f"(pressure: {pressures[current_phase]:.1f} -> {max_pressure:.1f})"
            )

        # Apply safety constraints
        action, duration = self._enforce_constraints(action, duration, current_phase)

        return ControllerRecommendation(
            intersection_id=state.intersection_id,
            timestamp=datetime.now(timezone.utc),
            current_phase=current_phase,
            action=action,
            duration_seconds=duration,
            reason=reason,
            controller_type="max_pressure",
            simulation_only=True,
            confidence=min(1.0, max_pressure / 100.0),  # Normalize confidence
        )

    def _calculate_pressure(self, state: TrafficState, phase: SignalPhase) -> float:
        """Calculate pressure for a given phase."""
        # Upstream pressure: sum of queue lengths on approaches getting green
        upstream_approaches = self._phase_approaches.get(phase, [])
        upstream_pressure = sum(
            state.get_approach(appr).queue_length_m
            for appr in upstream_approaches
            if state.get_approach(appr)
        )

        # Downstream pressure: sum of queue lengths on conflicting approaches
        conflicting_approaches = self._phase_conflicts.get(phase, [])
        downstream_pressure = 0.0

        for appr in conflicting_approaches:
            approach_state = state.get_approach(appr)
            if approach_state:
                # Add downstream pressure if there's a downstream approach
                downstream = self._downstream.get(appr)
                if downstream:
                    downstream_state = state.get_approach(downstream)
                    if downstream_state:
                        downstream_pressure += downstream_state.queue_length_m * self._downstream_weight

                # Add conflicting approach queue length
                downstream_pressure += approach_state.queue_length_m

        # Total pressure = upstream - downstream
        total_pressure = upstream_pressure * self._pressure_weight - downstream_pressure

        return total_pressure

    def _calculate_extension_duration(
        self,
        state: TrafficState,
        current_phase: SignalPhase,
    ) -> float:
        """Calculate how long to extend current green."""
        if not self._settings:
            return 5.0

        min_green = self._settings.max_pressure_min_green
        max_green = self._settings.max_pressure_max_green

        # Base extension
        extension = 5.0

        # Extend more if pressure is high
        pressure = self._calculate_pressure(state, current_phase)
        if pressure > 50:
            extension = 10.0
        elif pressure > 20:
            extension = 8.0

        # Ensure within bounds
        remaining = max_green - self._phase_elapsed
        return min(extension, remaining, max_green - min_green)

    def reset(self) -> None:
        """Reset controller state."""
        self._current_phase = None
        self._phase_elapsed = 0.0


# Add helper method to TrafficState for easier access
def _get_approach(self, approach_id: str) -> ApproachState | None:
    for appr in self.approaches:
        if appr.approach_id == approach_id:
            return appr
    return None


# Monkey-patch for convenience
TrafficState.get_approach = _get_approach