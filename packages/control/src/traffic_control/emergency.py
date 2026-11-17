"""Emergency vehicle priority and preemption for traffic signals."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections import deque

from traffic_domain.schemas import (
    TrafficState,
    ControllerRecommendation,
    ControllerAction,
    SignalPhase,
    IntersectionConfig,
    TrafficEvent,
    EventType,
)
from traffic_config.settings import ControlSettings
from traffic_control.base import BaseController
from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry


logger = get_logger(__name__)


class EmergencyVehicleType(str, Enum):
    """Types of emergency vehicles."""

    AMBULANCE = "ambulance"
    FIRE_TRUCK = "fire_truck"
    POLICE = "police"
    OTHER = "other"


class PreemptionStatus(str, Enum):
    """Preemption status."""

    INACTIVE = "inactive"
    REQUESTED = "requested"
    GRANTED = "granted"
    ACTIVE = "active"
    RELEASING = "releasing"


@dataclass
class EmergencyVehicle:
    """Emergency vehicle detection."""

    vehicle_id: str
    vehicle_type: EmergencyVehicleType
    intersection_id: str
    approach_id: str
    distance_m: float
    speed_kmh: float
    eta_seconds: float
    siren_active: bool = True
    detected_at: float = field(default_factory=time.time)
    priority: int = 10  # Higher = more urgent


@dataclass
class PreemptionRequest:
    """Preemption request for an intersection."""

    request_id: str
    vehicle: EmergencyVehicle
    target_phase: SignalPhase
    status: PreemptionStatus = PreemptionStatus.REQUESTED
    granted_at: float | None = None
    released_at: float | None = None


class EmergencyVehicleDetector:
    """Detect emergency vehicles from various sources."""

    def __init__(self, settings: ControlSettings):
        self._settings = settings
        self._active_vehicles: dict[str, EmergencyVehicle] = {}
        self._metrics = get_metrics_registry()

    def process_detection(
        self,
        vehicle_id: str,
        vehicle_type: EmergencyVehicleType,
        intersection_id: str,
        approach_id: str,
        distance_m: float,
        speed_kmh: float,
        siren_active: bool = True,
    ) -> EmergencyVehicle:
        """Process emergency vehicle detection."""
        eta = (distance_m / 1000.0) / max(speed_kmh / 3600.0, 0.1)

        vehicle = EmergencyVehicle(
            vehicle_id=vehicle_id,
            vehicle_type=vehicle_type,
            intersection_id=intersection_id,
            approach_id=approach_id,
            distance_m=distance_m,
            speed_kmh=speed_kmh,
            eta_seconds=eta,
            siren_active=siren_active,
            priority=self._get_priority(vehicle_type),
        )

        self._active_vehicles[vehicle_id] = vehicle

        self._metrics.inc_counter(
            "emergency_vehicle_detected_total",
            labels={"type": vehicle_type.value, "intersection": intersection_id},
        )

        logger.info("Emergency vehicle detected",
                    vehicle_id=vehicle_id,
                    type=vehicle_type.value,
                    intersection=intersection_id,
                    eta=eta)

        return vehicle

    def process_event(self, event: TrafficEvent) -> EmergencyVehicle | None:
        """Process emergency vehicle event from Kafka."""
        if event.event_type != EventType.EMERGENCY_VEHICLE:
            return None

        payload = event.payload
        return self.process_detection(
            vehicle_id=payload.get("vehicle_id", "unknown"),
            vehicle_type=EmergencyVehicleType(payload.get("vehicle_type", "other")),
            intersection_id=event.intersection_id,
            approach_id=payload.get("approach_id", ""),
            distance_m=payload.get("distance_m", 0),
            speed_kmh=payload.get("speed_kmh", 50),
            siren_active=payload.get("siren_active", True),
        )

    def update_vehicle(self, vehicle_id: str, distance_m: float, speed_kmh: float) -> EmergencyVehicle | None:
        """Update vehicle position."""
        if vehicle_id not in self._active_vehicles:
            return None

        vehicle = self._active_vehicles[vehicle_id]
        vehicle.distance_m = distance_m
        vehicle.speed_kmh = speed_kmh
        vehicle.eta_seconds = (distance_m / 1000.0) / max(speed_kmh / 3600.0, 0.1)
        return vehicle

    def remove_vehicle(self, vehicle_id: str) -> bool:
        """Remove vehicle (passed intersection or cancelled)."""
        if vehicle_id in self._active_vehicles:
            del self._active_vehicles[vehicle_id]
            self._metrics.inc_counter("emergency_vehicle_cleared_total")
            return True
        return False

    def get_vehicles_for_intersection(self, intersection_id: str) -> list[EmergencyVehicle]:
        """Get all active emergency vehicles for an intersection."""
        return [
            v for v in self._active_vehicles.values()
            if v.intersection_id == intersection_id
        ]

    def get_highest_priority_vehicle(self, intersection_id: str) -> EmergencyVehicle | None:
        """Get highest priority vehicle for intersection."""
        vehicles = self.get_vehicles_for_intersection(intersection_id)
        if not vehicles:
            return None
        return max(vehicles, key=lambda v: v.priority)

    def _get_priority(self, vehicle_type: EmergencyVehicleType) -> int:
        """Get priority level for vehicle type."""
        priorities = {
            EmergencyVehicleType.AMBULANCE: 10,
            EmergencyVehicleType.FIRE_TRUCK: 9,
            EmergencyVehicleType.POLICE: 8,
            EmergencyVehicleType.OTHER: 5,
        }
        return priorities.get(vehicle_type, 5)


class PreemptionManager:
    """Manage signal preemption for emergency vehicles."""

    def __init__(
        self,
        intersection_config: IntersectionConfig,
        settings: ControlSettings,
        detector: EmergencyVehicleDetector,
    ):
        self._config = intersection_config
        self._settings = settings
        self._detector = detector
        self._metrics = get_metrics_registry()

        # Active preemption
        self._current_preemption: PreemptionRequest | None = None
        self._preemption_history: deque[PreemptionRequest] = deque(maxlen=100)

        # Preemption phases (typically all-red -> emergency green)
        self._preemption_phase = SignalPhase.ALL_RED
        self._recovery_phase: SignalPhase | None = None

    def check_preemption(self, state: TrafficState) -> tuple[bool, ControllerRecommendation | None]:
        """Check if preemption should be activated."""
        vehicles = self._detector.get_vehicles_for_intersection(self._config.intersection_id)

        if not vehicles:
            # No emergency vehicles - check if releasing preemption
            if self._current_preemption and self._current_preemption.status == PreemptionStatus.ACTIVE:
                return self._release_preemption(state)
            return False, None

        # Get highest priority vehicle
        vehicle = max(vehicles, key=lambda v: v.priority)

        # Check if vehicle is close enough to warrant preemption
        if vehicle.eta_seconds > self._settings.emergency_preempt_time:
            return False, None

        # Determine target phase
        target_phase = self._get_preemption_phase(vehicle.approach_id)

        # Create or update preemption request
        if self._current_preemption is None or self._current_preemption.vehicle.vehicle_id != vehicle.vehicle_id:
            self._current_preemption = PreemptionRequest(
                request_id=f"preempt-{vehicle.vehicle_id}",
                vehicle=vehicle,
                target_phase=target_phase,
                status=PreemptionStatus.REQUESTED,
            )

        # Grant preemption if not already granted
        if self._current_preemption.status == PreemptionStatus.REQUESTED:
            return self._grant_preemption(state, vehicle, target_phase)

        # Already active
        if self._current_preemption.status == PreemptionStatus.ACTIVE:
            return True, self._create_preemption_recommendation(vehicle, target_phase)

        return False, None

    def _grant_preemption(
        self,
        state: TrafficState,
        vehicle: EmergencyVehicle,
        target_phase: SignalPhase,
    ) -> tuple[bool, ControllerRecommendation]:
        """Grant preemption and create recommendation."""
        self._current_preemption.status = PreemptionStatus.GRANTED
        self._current_preemption.granted_at = time.time()

        # Save current phase for recovery
        self._recovery_phase = self._get_current_phase_for_approach(vehicle.approach_id)

        # Move to active
        self._current_preemption.status = PreemptionStatus.ACTIVE

        self._metrics.inc_counter(
            "preemption_granted_total",
            labels={"intersection": self._config.intersection_id, "vehicle_type": vehicle.vehicle_type.value},
        )

        logger.warning("PREEMPTION GRANTED",
                       intersection=self._config.intersection_id,
                       vehicle=vehicle.vehicle_id,
                       target_phase=target_phase.value)

        return True, self._create_preemption_recommendation(vehicle, target_phase)

    def _release_preemption(self, state: TrafficState) -> tuple[bool, ControllerRecommendation]:
        """Release preemption and return to normal operation."""
        if self._current_preemption:
            self._current_preemption.status = PreemptionStatus.RELEASING
            self._current_preemption.released_at = time.time()
            self._preemption_history.append(self._current_preemption)

            self._metrics.inc_counter(
                "preemption_released_total",
                labels={"intersection": self._config.intersection_id},
            )

            logger.info("PREEMPTION RELEASED",
                        intersection=self._config.intersection_id,
                        vehicle=self._current_preemption.vehicle.vehicle_id)

        # Create recovery recommendation
        recovery = ControllerRecommendation(
            action=ControllerAction.SWITCH_PHASE,
            current_phase=self._recovery_phase or SignalPhase.NORTH_SOUTH_GREEN,
            duration_seconds=self._settings.min_green_time,
            confidence=1.0,
            reason="Emergency preemption released, recovering normal operation",
            controller_type="emergency_recovery",
        )

        self._current_preemption = None
        self._recovery_phase = None

        return True, recovery

    def _create_preemption_recommendation(
        self,
        vehicle: EmergencyVehicle,
        target_phase: SignalPhase,
    ) -> ControllerRecommendation:
        """Create controller recommendation for preemption."""
        # Stay in preemption phase until vehicle passes
        remaining_time = max(vehicle.eta_seconds + 5, self._settings.min_green_time)  # 5s buffer

        return ControllerRecommendation(
            action=ControllerAction.SWITCH_PHASE if target_phase != self._preemption_phase else ControllerAction.EXTEND,
            current_phase=target_phase,
            duration_seconds=remaining_time,
            confidence=1.0,
            reason=f"Emergency preemption for {vehicle.vehicle_type.value} (ETA: {vehicle.eta_seconds:.1f}s)",
            controller_type="emergency_preemption",
        )

    def _get_preemption_phase(self, approach_id: str) -> SignalPhase:
        """Get preemption phase for approach."""
        # Map approach to signal phase
        approach_to_phase = {
            "northbound": SignalPhase.NORTH_SOUTH_GREEN,
            "southbound": SignalPhase.NORTH_SOUTH_GREEN,
            "eastbound": SignalPhase.EAST_WEST_GREEN,
            "westbound": SignalPhase.EAST_WEST_GREEN,
        }
        return approach_to_phase.get(approach_id, SignalPhase.NORTH_SOUTH_GREEN)

    def _get_current_phase_for_approach(self, approach_id: str) -> SignalPhase:
        """Get the phase that serves the given approach."""
        for phase in self._config.phases:
            for approach in self._config.approaches:
                if approach.approach_id == approach_id:
                    if approach.signal_group in phase.value:
                        return phase
        return SignalPhase.NORTH_SOUTH_GREEN

    def is_preempted(self) -> bool:
        """Check if currently preempted."""
        return self._current_preemption is not None and self._current_preemption.status == PreemptionStatus.ACTIVE

    def get_preemption_status(self) -> dict[str, Any]:
        """Get current preemption status."""
        if self._current_preemption is None:
            return {"status": "inactive"}

        return {
            "status": self._current_preemption.status.value,
            "vehicle_id": self._current_preemption.vehicle.vehicle_id,
            "vehicle_type": self._current_preemption.vehicle.vehicle_type.value,
            "target_phase": self._current_preemption.target_phase.value,
            "eta_seconds": self._current_preemption.vehicle.eta_seconds,
            "granted_at": self._current_preemption.granted_at,
        }


class EmergencyPriorityController(BaseController):
    """Controller wrapper that adds emergency vehicle priority."""

    def __init__(
        self,
        base_controller: BaseController,
        intersection_config: IntersectionConfig,
        settings: ControlSettings,
    ):
        super().__init__(intersection_config, settings)
        self._base_controller = base_controller
        self._detector = EmergencyVehicleDetector(settings)
        self._preemption_manager = PreemptionManager(intersection_config, settings, self._detector)

    def compute_action(self, state: TrafficState) -> ControllerRecommendation:
        """Compute action with emergency priority."""
        # Check for preemption
        preempted, preempt_rec = self._preemption_manager.check_preemption(state)

        if preempted and preempt_rec:
            return preempt_rec

        # Normal operation
        return self._base_controller.compute_action(state)

    def process_emergency_event(self, event: TrafficEvent) -> EmergencyVehicle | None:
        """Process emergency vehicle event."""
        return self._detector.process_event(event)

    def get_preemption_status(self) -> dict[str, Any]:
        """Get preemption status."""
        return self._preemption_manager.get_preemption_status()


def create_emergency_controller(
    base_controller: BaseController,
    intersection_config: IntersectionConfig,
    settings: ControlSettings,
) -> EmergencyPriorityController:
    """Factory function to create emergency priority controller."""
    return EmergencyPriorityController(base_controller, intersection_config, settings)