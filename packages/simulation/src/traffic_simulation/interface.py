"""Simulator-independent interface for traffic control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from traffic_domain.schemas import (
    TrafficState,
    ControllerRecommendation,
    SignalPhase,
    IntersectionConfig,
)


@dataclass
class SimulatorState:
    """Current state of the simulation."""
    step: int
    time: float
    intersections: dict[str, TrafficState]
    vehicles: dict[str, Any]  # Vehicle states
    metrics: dict[str, float]


class SimulatorInterface(ABC):
    """Abstract interface for traffic simulators."""

    @abstractmethod
    def start(self, config: IntersectionConfig) -> None:
        """Start the simulation."""
        pass

    @abstractmethod
    def step(self, actions: dict[str, ControllerRecommendation]) -> SimulatorState:
        """Advance simulation by one step with control actions.

        Args:
            actions: Control actions per intersection

        Returns:
            Updated simulator state
        """
        pass

    @abstractmethod
    def get_state(self) -> SimulatorState:
        """Get current simulation state."""
        pass

    @abstractmethod
    def set_signal(self, intersection_id: str, phase: SignalPhase) -> None:
        """Set signal phase for an intersection."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the simulation."""
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if simulation is running."""
        pass