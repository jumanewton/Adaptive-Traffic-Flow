"""Shared domain schemas and contracts for Adaptive Traffic Flow Lab."""

from traffic_domain.schemas import (
    # Core entities
    Detection,
    Track,
    TrackStatus,
    BoundingBox,
    Point,
    # Traffic state
    TrafficState,
    ApproachState,
    LaneState,
    # Signal control
    SignalPhase,
    PhaseState,
    ControllerAction,
    ControllerRecommendation,
    # Events
    TrafficEvent,
    EventType,
    # Experiment
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    # Intersection
    Intersection,
    IntersectionConfig,
    ApproachConfig,
    LaneConfig,
    # Calibration
    CameraCalibration,
    HomographyMatrix,
)

__all__ = [
    "Detection",
    "Track",
    "TrackStatus",
    "BoundingBox",
    "Point",
    "TrafficState",
    "ApproachState",
    "LaneState",
    "SignalPhase",
    "PhaseState",
    "ControllerAction",
    "ControllerRecommendation",
    "TrafficEvent",
    "EventType",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentStatus",
    "Intersection",
    "IntersectionConfig",
    "ApproachConfig",
    "LaneConfig",
    "CameraCalibration",
    "HomographyMatrix",
]