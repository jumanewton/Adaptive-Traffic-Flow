"""Model Predictive Control (MPC) for traffic signal optimization."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any
from collections import deque

from traffic_domain.schemas import (
    TrafficState,
    ControllerRecommendation,
    ControllerAction,
    SignalPhase,
    IntersectionConfig,
    ApproachConfig,
    LaneConfig,
)
from traffic_config.settings import ControlSettings
from traffic_control.base import BaseController
from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry


logger = get_logger(__name__)


@dataclass
class MPCPrediction:
    """MPC prediction result."""

    phase_sequence: list[SignalPhase]
    durations: list[float]
    predicted_queue: np.ndarray  # [horizon, approaches]
    predicted_wait: np.ndarray   # [horizon, approaches]
    total_cost: float


class TrafficFlowModel:
    """Simplified traffic flow model for MPC prediction."""

    def __init__(
        self,
        intersection_config: IntersectionConfig,
        dt: float = 1.0,
        saturation_flow: float = 1800.0,  # veh/h/lane
    ):
        self._config = intersection_config
        self._dt = dt
        self._saturation_flow = saturation_flow / 3600.0  # veh/s/lane
        self._approach_ids = [a.approach_id for a in intersection_config.approaches]
        self._n_approaches = len(self._approach_ids)
        self._phase_to_approaches = self._build_phase_mapping()

    def _build_phase_mapping(self) -> dict[SignalPhase, list[int]]:
        """Map each phase to the approach indices it serves."""
        mapping = {}
        for phase in self._config.phases:
            served = []
            for i, approach in enumerate(self._config.approaches):
                if approach.signal_group in phase.value:
                    served.append(i)
            mapping[phase] = served
        return mapping

    def predict(
        self,
        current_state: TrafficState,
        phase_sequence: list[SignalPhase],
        durations: list[float],
        horizon: int = 30,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict queue lengths and waiting times over horizon.

        Args:
            current_state: Current traffic state
            phase_sequence: Sequence of phases to simulate
            durations: Duration for each phase
            horizon: Prediction horizon in seconds

        Returns:
            Tuple of (queue_predictions, wait_predictions) each [horizon, n_approaches]
        """
        # Current queue lengths per approach (meters)
        queue = np.array([
            next((a.queue_length_m for a in current_state.approaches if a.approach_id == aid), 0.0)
            for aid in self._approach_ids
        ])

        # Current waiting times per approach (seconds)
        wait = np.array([
            next((a.avg_wait_time_s for a in current_state.approaches if a.approach_id == aid), 0.0)
            for aid in self._approach_ids
        ])

        # Arrival rates per approach (veh/s)
        arrivals = np.array([
            next((a.vehicle_count / max(a.occupancy * 10, 1) * 0.1, 0.1)  # rough estimate
            for a in current_state.approaches if a.approach_id == aid)
            for aid in self._approach_ids
        ])

        # Simplified: use average flow if no data
        if np.all(arrivals == 0):
            arrivals = np.full(self._n_approaches, 0.1)

        queue_pred = np.zeros((horizon, self._n_approaches))
        wait_pred = np.zeros((horizon, self._n_approaches))

        t = 0
        phase_idx = 0
        phase_elapsed = 0.0

        for step in range(horizon):
            # Current phase
            if phase_idx < len(phase_sequence):
                current_phase = phase_sequence[phase_idx]
                served = self._phase_to_approaches.get(current_phase, [])
            else:
                served = []

            # Departure rate for served approaches
            departure = np.zeros(self._n_approaches)
            for i in served:
                # Number of lanes served in this approach
                n_lanes = len([
                    l for l in self._config.approaches[i].lanes
                    if not l.is_turning or l.turn_direction == "left"
                ])
                departure[i] = n_lanes * self._saturation_flow

            # Update queue (vehicles)
            queue_veh = queue / 7.5  # Convert meters to vehicles (7.5m per vehicle)
            queue_veh = np.maximum(0, queue_veh + arrivals * self._dt - departure * self._dt)
            queue = queue_veh * 7.5

            # Update waiting time
            wait = wait + self._dt
            for i in served:
                if queue[i] > 0:
                    wait[i] = np.maximum(0, wait[i] - self._dt * 0.5)  # Waiting decreases when moving

            queue_pred[step] = queue
            wait_pred[step] = wait

            # Phase transition
            phase_elapsed += self._dt
            if phase_idx < len(durations) and phase_elapsed >= durations[phase_idx]:
                phase_idx += 1
                phase_elapsed = 0.0

        return queue_pred, wait_pred


class MPCController(BaseController):
    """Model Predictive Control for adaptive traffic signals."""

    def __init__(
        self,
        intersection_config: IntersectionConfig,
        settings: ControlSettings,
        horizon: int = 30,
        dt: float = 1.0,
        cost_weights: dict[str, float] | None = None,
    ):
        super().__init__(intersection_config, settings)
        self._horizon = horizon
        self._dt = dt
        self._flow_model = TrafficFlowModel(intersection_config, dt)
        self._cost_weights = cost_weights or {
            "queue": 1.0,
            "wait": 2.0,
            "switch": 10.0,  # Penalty for phase changes
            "starvation": 50.0,  # Penalty for max green violation
        }
        self._metrics = get_metrics_registry()
        self._prediction_history: deque[MPCPrediction] = deque(maxlen=100)

    def compute_action(self, state: TrafficState) -> ControllerRecommendation:
        """Compute optimal phase sequence using MPC."""
        # Generate candidate phase sequences
        candidates = self._generate_candidates(state)

        # Evaluate each candidate
        best_candidate = None
        best_cost = float('inf')

        for phase_seq, durations in candidates:
            cost, pred = self._evaluate_candidate(state, phase_seq, durations)
            if cost < best_cost:
                best_cost = cost
                best_candidate = (phase_seq, durations, pred)

        if best_candidate is None:
            # Fallback to current phase
            return ControllerRecommendation(
                action=ControllerAction.HOLD,
                current_phase=self._current_phase,
                duration_seconds=self._settings.min_green_time,
                confidence=0.5,
                reason="MPC: no valid candidates, holding",
                controller_type="mpc",
            )

        phase_seq, durations, pred = best_candidate

        # Determine action
        if phase_seq[0] != self._current_phase:
            action = ControllerAction.SWITCH_PHASE
            next_phase = phase_seq[0]
            duration = durations[0]
        else:
            action = ControllerAction.EXTEND
            next_phase = self._current_phase
            duration = durations[0] - self._phase_elapsed

        # Store prediction for monitoring
        self._prediction_history.append(pred)

        # Record metrics
        self._metrics.record_histogram(
            "mpc_optimization_cost",
            best_cost,
            labels={"intersection": self._intersection_config.intersection_id},
        )
        self._metrics.record_histogram(
            "mpc_horizon_queue",
            float(np.mean(pred.predicted_queue)),
            labels={"intersection": self._intersection_config.intersection_id},
        )

        return ControllerRecommendation(
            action=action,
            current_phase=next_phase,
            duration_seconds=max(duration, self._settings.min_green_time),
            confidence=0.8,
            reason=f"MPC: optimal cost={best_cost:.1f}, horizon={self._horizon}s",
            controller_type="mpc",
        )

    def _generate_candidates(
        self,
        state: TrafficState,
    ) -> list[tuple[list[SignalPhase], list[float]]]:
        """Generate candidate phase sequences."""
        candidates = []
        phases = self._intersection_config.phases

        if not phases:
            return candidates

        current_idx = phases.index(self._current_phase) if self._current_phase in phases else 0

        # Candidate 1: Stay in current phase
        candidates.append((
            [self._current_phase] * 3,
            [self._settings.max_green_time] * 3,
        ))

        # Candidate 2: Switch to next phase immediately
        next_idx = (current_idx + 1) % len(phases)
        next_phase = phases[next_idx]
        candidates.append((
            [next_phase] + [phases[(next_idx + 1) % len(phases)]] * 2,
            [self._settings.min_green_time, self._settings.max_green_time, self._settings.max_green_time],
        ))

        # Candidate 3: Skip to phase after next (if pressure high)
        skip_idx = (current_idx + 2) % len(phases)
        skip_phase = phases[skip_idx]
        candidates.append((
            [skip_phase] + [phases[(skip_idx + 1) % len(phases)]] * 2,
            [self._settings.min_green_time, self._settings.max_green_time, self._settings.max_green_time],
        ))

        # Candidate 4: Full cycle from current
        full_cycle = phases[current_idx:] + phases[:current_idx]
        cycle_durations = [self._settings.max_green_time] * len(phases)
        candidates.append((full_cycle, cycle_durations))

        # Candidate 5: Short cycle for high pressure
        if self._estimate_pressure(state) > 0.7:
            short_cycle = phases[current_idx:] + phases[:current_idx]
            short_durations = [self._settings.min_green_time] * len(phases)
            candidates.append((short_cycle, short_durations))

        return candidates

    def _estimate_pressure(self, state: TrafficState) -> float:
        """Estimate intersection pressure (0-1)."""
        total_queue = sum(a.queue_length_m for a in state.approaches)
        max_possible = len(state.approaches) * 100  # 100m max queue per approach
        return min(total_queue / max_possible, 1.0)

    def _evaluate_candidate(
        self,
        state: TrafficState,
        phase_seq: list[SignalPhase],
        durations: list[float],
    ) -> tuple[float, MPCPrediction]:
        """Evaluate a candidate phase sequence."""
        # Predict queues and waits
        pred_queue, pred_wait = self._flow_model.predict(state, phase_seq, durations, self._horizon)

        # Cost components
        queue_cost = self._cost_weights["queue"] * np.sum(pred_queue)
        wait_cost = self._cost_weights["wait"] * np.sum(pred_wait)

        # Switch penalty
        switch_cost = 0
        if phase_seq[0] != self._current_phase:
            switch_cost = self._cost_weights["switch"]

        # Starvation penalty (approaches not served)
        served_approaches = set()
        for phase in phase_seq:
            served_approaches.update(self._flow_model._phase_to_approaches.get(phase, []))

        all_approaches = set(range(len(self._flow_model._approach_ids)))
        starved = all_approaches - served_approaches
        starvation_cost = self._cost_weights["starvation"] * len(starved)

        total_cost = queue_cost + wait_cost + switch_cost + starvation_cost

        pred = MPCPrediction(
            phase_sequence=phase_seq,
            durations=durations,
            predicted_queue=pred_queue,
            predicted_wait=pred_wait,
            total_cost=total_cost,
        )

        return total_cost, pred

    def get_predictions(self) -> list[MPCPrediction]:
        """Get recent predictions for monitoring."""
        return list(self._prediction_history)


def create_mpc_controller(
    intersection_config: IntersectionConfig,
    settings: ControlSettings,
    **kwargs,
) -> MPCController:
    """Factory function to create MPC controller."""
    return MPCController(intersection_config, settings, **kwargs)