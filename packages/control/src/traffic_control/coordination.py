"""Multi-intersection coordination for corridor/arterial control."""

from __future__ import annotations

import asyncio
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from traffic_domain.schemas import (
    TrafficState,
    ControllerRecommendation,
    ControllerAction,
    SignalPhase,
    IntersectionConfig,
)
from traffic_config.settings import ControlSettings
from traffic_control.base import BaseController
from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry
from traffic_streaming.producer import SyncEventProducer
from traffic_streaming.consumer import EventConsumer, EventHandler
from traffic_domain.schemas import TrafficEvent, EventType


logger = get_logger(__name__)


class CoordinationStrategy(str, Enum):
    """Coordination strategies for multi-intersection control."""

    FIXED_OFFSET = "fixed_offset"      # Fixed time offsets between intersections
    MAX_PRESSURE = "max_pressure"      # Distributed max pressure
    GREEN_WAVE = "green_wave"          # Green wave progression
    ADAPTIVE = "adaptive"              # Adaptive coordination based on traffic
    RHYTHMIC = "rhythmic"              # Rhythmic coordination (same cycle length)


@dataclass
class IntersectionLink:
    """Link between two intersections."""

    upstream_id: str
    downstream_id: str
    distance_m: float
    travel_time_s: float
    approach_upstream: str  # Approach at upstream that feeds downstream
    approach_downstream: str  # Approach at downstream that receives flow


@dataclass
class CoordinationPlan:
    """Coordination plan for a corridor."""

    cycle_length: float
    offsets: dict[str, float]  # intersection_id -> offset (seconds)
    strategy: CoordinationStrategy
    bandwidth: float  # Green bandwidth efficiency (0-1)


class MultiIntersectionCoordinator:
    """Coordinates multiple intersections along a corridor."""

    def __init__(
        self,
        intersections: dict[str, IntersectionConfig],
        links: list[IntersectionLink],
        settings: ControlSettings,
        strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE,
    ):
        self._intersections = intersections
        self._links = links
        self._settings = settings
        self._strategy = strategy
        self._metrics = get_metrics_registry()

        # Build adjacency
        self._upstream: dict[str, list[IntersectionLink]] = defaultdict(list)
        self._downstream: dict[str, list[IntersectionLink]] = defaultdict(list)
        for link in links:
            self._upstream[link.downstream_id].append(link)
            self._downstream[link.upstream_id].append(link)

        # Current state
        self._intersection_states: dict[str, TrafficState] = {}
        self._local_controllers: dict[str, BaseController] = {}
        self._coordination_plan: CoordinationPlan | None = None

        # Communication
        self._producer: SyncEventProducer | None = None
        self._consumer: EventConsumer | None = None

    def register_controller(self, intersection_id: str, controller: BaseController) -> None:
        """Register a local controller for an intersection."""
        self._local_controllers[intersection_id] = controller

    def update_state(self, intersection_id: str, state: TrafficState) -> None:
        """Update traffic state for an intersection."""
        self._intersection_states[intersection_id] = state

    def compute_coordination_plan(self) -> CoordinationPlan:
        """Compute optimal coordination plan based on current strategy."""
        if self._strategy == CoordinationStrategy.FIXED_OFFSET:
            return self._compute_fixed_offset()
        elif self._strategy == CoordinationStrategy.GREEN_WAVE:
            return self._compute_green_wave()
        elif self._strategy == CoordinationStrategy.MAX_PRESSURE:
            return self._compute_max_pressure_coordination()
        elif self._strategy == CoordinationStrategy.RHYTHMIC:
            return self._compute_rhythmic()
        else:
            return self._compute_adaptive()

    def _get_cycle_length(self) -> float:
        """Get cycle length from settings."""
        return getattr(self._settings, 'fixed_time_cycle_length',
                       getattr(self._settings, 'cycle_length', 120))

    def _compute_fixed_offset(self) -> CoordinationPlan:
        """Fixed offset plan based on link travel times."""
        cycle = self._get_cycle_length()
        offsets = {}

        # Sort intersections topologically (simplified)
        sorted_ids = self._topological_sort()

        for i, int_id in enumerate(sorted_ids):
            if i == 0:
                offsets[int_id] = 0.0
            else:
                # Find link from previous
                prev_id = sorted_ids[i - 1]
                link = next((l for l in self._links if l.upstream_id == prev_id and l.downstream_id == int_id), None)
                if link:
                    offsets[int_id] = (offsets[prev_id] + link.travel_time_s) % cycle
                else:
                    offsets[int_id] = 0.0

        return CoordinationPlan(
            cycle_length=cycle,
            offsets=offsets,
            strategy=CoordinationStrategy.FIXED_OFFSET,
            bandwidth=0.7,
        )

    def _compute_green_wave(self) -> CoordinationPlan:
        """Green wave progression based on platoon speeds."""
        cycle = self._get_cycle_length()
        offsets = {}

        # Estimate platoon speed from upstream flows
        sorted_ids = self._topological_sort()
        platoon_speed = 13.9  # 50 km/h default

        for i, int_id in enumerate(sorted_ids):
            if i == 0:
                offsets[int_id] = 0.0
            else:
                prev_id = sorted_ids[i - 1]
                link = next((l for l in self._links if l.upstream_id == prev_id and l.downstream_id == int_id), None)
                if link:
                    # Offset = travel time at platoon speed
                    travel_time = link.distance_m / platoon_speed
                    offsets[int_id] = (offsets[prev_id] + travel_time) % cycle
                else:
                    offsets[int_id] = 0.0

        # Calculate bandwidth
        bandwidth = self._calculate_bandwidth(offsets, cycle)

        return CoordinationPlan(
            cycle_length=cycle,
            offsets=offsets,
            strategy=CoordinationStrategy.GREEN_WAVE,
            bandwidth=bandwidth,
        )

    def _compute_max_pressure_coordination(self) -> CoordinationPlan:
        """Distributed max pressure with coordination."""
        # Each intersection runs max pressure locally
        # Coordination adjusts cycle length based on system pressure
        total_pressure = 0
        n_intersections = len(self._intersections)

        for int_id, state in self._intersection_states.items():
            pressure = self._calculate_intersection_pressure(state)
            total_pressure += pressure

        avg_pressure = total_pressure / max(n_intersections, 1)

        base_cycle = self._get_cycle_length()
        # Adjust cycle length based on pressure
        if avg_pressure > 0.8:
            cycle = base_cycle * 1.2  # Longer cycle
        elif avg_pressure < 0.3:
            cycle = base_cycle * 0.8  # Shorter cycle
        else:
            cycle = base_cycle

        # Simple fixed offsets
        offsets = {int_id: 0.0 for int_id in self._intersections}

        return CoordinationPlan(
            cycle_length=cycle,
            offsets=offsets,
            strategy=CoordinationStrategy.MAX_PRESSURE,
            bandwidth=0.5,
        )

    def _compute_rhythmic(self) -> CoordinationPlan:
        """Rhythmic coordination - all intersections same cycle, optimized offsets."""
        cycle = self._get_cycle_length()

        # Optimize offsets using MILP or heuristic (simplified here)
        offsets = self._optimize_offsets_heuristic(cycle)
        bandwidth = self._calculate_bandwidth(offsets, cycle)

        return CoordinationPlan(
            cycle_length=cycle,
            offsets=offsets,
            strategy=CoordinationStrategy.RHYTHMIC,
            bandwidth=bandwidth,
        )

    def _compute_adaptive(self) -> CoordinationPlan:
        """Adaptive coordination based on real-time traffic."""
        # Use green wave if flows are directional
        # Use max pressure if balanced
        directional_score = self._calculate_directional_score()

        if directional_score > 0.6:
            return self._compute_green_wave()
        else:
            return self._compute_max_pressure_coordination()

    def _calculate_directional_score(self) -> float:
        """Calculate how directional the flow is (0=balanced, 1=single direction)."""
        if not self._intersection_states:
            return 0.5

        total_ns = 0
        total_ew = 0

        for state in self._intersection_states.values():
            for approach in state.approaches:
                if approach.approach_id in ("northbound", "southbound"):
                    total_ns += approach.vehicle_count
                else:
                    total_ew += approach.vehicle_count

        total = total_ns + total_ew
        if total == 0:
            return 0.5

        return abs(total_ns - total_ew) / total

    def _topological_sort(self) -> list[str]:
        """Simple topological sort of intersections."""
        # Simplified: sort by number of upstream links
        in_degree = {int_id: 0 for int_id in self._intersections}
        for link in self._links:
            in_degree[link.downstream_id] += 1

        result = []
        queue = [int_id for int_id, deg in in_degree.items() if deg == 0]

        while queue:
            int_id = queue.pop(0)
            result.append(int_id)
            for link in self._downstream.get(int_id, []):
                in_degree[link.downstream_id] -= 1
                if in_degree[link.downstream_id] == 0:
                    queue.append(link.downstream_id)

        # Add any remaining (cycles)
        for int_id in self._intersections:
            if int_id not in result:
                result.append(int_id)

        return result

    def _optimize_offsets_heuristic(self, cycle: float) -> dict[str, float]:
        """Heuristic offset optimization."""
        offsets = {}
        sorted_ids = self._topological_sort()

        for i, int_id in enumerate(sorted_ids):
            if i == 0:
                offsets[int_id] = 0.0
            else:
                best_offset = 0.0
                best_score = -1

                # Try multiple offset candidates
                for candidate in np.linspace(0, cycle, 12):
                    test_offsets = {**offsets, int_id: candidate}
                    score = self._evaluate_offsets(test_offsets, cycle)
                    if score > best_score:
                        best_score = score
                        best_offset = candidate

                offsets[int_id] = best_offset

        return offsets

    def _evaluate_offsets(self, offsets: dict[str, float], cycle: float) -> float:
        """Evaluate offset quality (simplified)."""
        score = 0
        for link in self._links:
            upstream_offset = offsets.get(link.upstream_id, 0)
            downstream_offset = offsets.get(link.downstream_id, 0)

            # Ideal: downstream green starts when platoon arrives
            ideal_offset = (upstream_offset + link.travel_time_s) % cycle
            diff = abs(downstream_offset - ideal_offset)
            diff = min(diff, cycle - diff)  # Circular distance

            score += 1.0 - diff / (cycle / 2)

        return score / max(len(self._links), 1)

    def _calculate_bandwidth(self, offsets: dict[str, float], cycle: float) -> float:
        """Calculate green bandwidth efficiency."""
        if not self._links:
            return 0.0

        bandwidth_sum = 0
        for link in self._links:
            upstream_offset = offsets.get(link.upstream_id, 0)
            downstream_offset = offsets.get(link.downstream_id, 0)

            # Time when platoon arrives at downstream
            arrival = (upstream_offset + link.travel_time_s) % cycle

            # Green window at downstream (simplified: 50% of cycle)
            green_start = downstream_offset
            green_end = (downstream_offset + cycle * 0.5) % cycle

            # Check if arrival falls in green window
            if green_start <= green_end:
                in_green = green_start <= arrival <= green_end
            else:
                in_green = arrival >= green_start or arrival <= green_end

            bandwidth_sum += 1.0 if in_green else 0.0

        return bandwidth_sum / len(self._links)

    def _calculate_intersection_pressure(self, state: TrafficState) -> float:
        """Calculate pressure at intersection."""
        total_queue = sum(a.queue_length_m for a in state.approaches)
        max_queue = len(state.approaches) * 100
        return min(total_queue / max_queue, 1.0)

    def get_coordination_recommendations(self) -> dict[str, ControllerRecommendation]:
        """Get coordinated recommendations for all intersections."""
        if self._coordination_plan is None:
            self._coordination_plan = self.compute_coordination_plan()

        recommendations = {}

        for int_id, controller in self._local_controllers.items():
            state = self._intersection_states.get(int_id)
            if state is None:
                continue

            # Get local recommendation
            local_rec = controller.compute_action(state)

            # Adjust based on coordination plan
            offset = self._coordination_plan.offsets.get(int_id, 0)
            cycle = self._coordination_plan.cycle_length

            # If local recommendation wants to switch but offset says wait, extend
            if local_rec.action == ControllerAction.SWITCH_PHASE:
                # Check if we're at the right offset
                # This is simplified - real implementation would track cycle position
                pass

            recommendations[int_id] = local_rec

        return recommendations

    async def start_communication(self, bootstrap_servers: str, group_id: str) -> None:
        """Start Kafka communication for distributed coordination."""
        self._producer = SyncEventProducer(
            bootstrap_servers=bootstrap_servers,
            client_id=f"{group_id}-coordinator",
        )

        self._consumer = EventConsumer(
            topics=["traffic.state", "coordination.plan"],
            group_id=group_id,
        )

        # Add handler for state updates
        self._consumer.add_handler(CoordinationStateHandler(self))

        await self._consumer.start()

    async def stop_communication(self) -> None:
        """Stop communication."""
        if self._consumer:
            await self._consumer.stop()
        if self._producer:
            self._producer.close()

    def publish_coordination_plan(self, topic: str = "coordination.plan") -> None:
        """Publish coordination plan to Kafka."""
        if self._producer and self._coordination_plan:
            event = TrafficEvent(
                event_type=EventType.COORDINATION_PLAN,
                intersection_id="coordinator",
                payload={
                    "cycle_length": self._coordination_plan.cycle_length,
                    "offsets": self._coordination_plan.offsets,
                    "strategy": self._coordination_plan.strategy.value,
                    "bandwidth": self._coordination_plan.bandwidth,
                },
                source_service="coordinator",
            )
            self._producer.publish(topic, event)


class CoordinationStateHandler(EventHandler):
    """Handle traffic state events for coordination."""

    def __init__(self, coordinator: MultiIntersectionCoordinator):
        self._coordinator = coordinator

    @property
    def event_types(self) -> list[EventType]:
        return [EventType.TRAFFIC_STATE]

    async def handle(self, event: TrafficEvent) -> None:
        try:
            state = TrafficState.model_validate(event.payload)
            self._coordinator.update_state(event.intersection_id, state)
        except Exception as e:
            logger.error("Failed to process coordination state", error=str(e))


@dataclass
class CorridorConfig:
    """Configuration for a traffic corridor."""

    corridor_id: str
    name: str
    intersections: list[IntersectionConfig]
    links: list[IntersectionLink]
    strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE


def create_corridor_coordinator(
    corridor: CorridorConfig,
    settings: ControlSettings,
    local_controllers: dict[str, BaseController],
) -> MultiIntersectionCoordinator:
    """Factory function to create corridor coordinator."""
    intersections = {ic.intersection_id: ic for ic in corridor.intersections}

    coordinator = MultiIntersectionCoordinator(
        intersections=intersections,
        links=corridor.links,
        settings=settings,
        strategy=corridor.strategy,
    )

    for int_id, controller in local_controllers.items():
        coordinator.register_controller(int_id, controller)

    return coordinator