"""Domain schemas and contracts for Adaptive Traffic Flow Lab.

All schemas are versioned and designed for serialization across service boundaries.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SchemaVersion(int, Enum):
    """Schema version for forward compatibility."""
    V1 = 1


class TrackStatus(str, Enum):
    """Lifecycle status of a track."""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


class EventType(str, Enum):
    """Types of traffic events published to the event stream."""
    TRAFFIC_STATE = "traffic_state"
    CONTROLLER_RECOMMENDATION = "controller_recommendation"
    PHASE_CHANGE = "phase_change"
    INCIDENT = "incident"
    EMERGENCY_VEHICLE = "emergency_vehicle"
    COORDINATION_PLAN = "coordination_plan"
    HEARTBEAT = "heartbeat"


class ExperimentStatus(str, Enum):
    """Status of an experiment run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SignalPhase(str, Enum):
    """Named signal phases for a standard 4-approach intersection."""
    NORTH_SOUTH_GREEN = "north_south_green"
    NORTH_SOUTH_YELLOW = "north_south_yellow"
    EAST_WEST_GREEN = "east_west_green"
    EAST_WEST_YELLOW = "east_west_yellow"
    ALL_RED = "all_red"
    PEDESTRIAN = "pedestrian"
    EMERGENCY = "emergency"


class PhaseState(str, Enum):
    """Current state of a signal phase."""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    FLASHING_YELLOW = "flashing_yellow"
    FLASHING_RED = "flashing_red"


class ControllerAction(str, Enum):
    """Actions a controller can recommend."""
    EXTEND_GREEN = "extend_green"
    SHORTEN_GREEN = "shorten_green"
    SWITCH_PHASE = "switch_phase"
    HOLD_PHASE = "hold_phase"
    EMERGENCY_PREEMPT = "emergency_preempt"
    PEDESTRIAN_PHASE = "pedestrian_phase"


# --- Geometry primitives ---


class Point(BaseModel):
    """2D point in pixel or world coordinates."""
    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""
    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Point:
        return Point(x=(self.x1 + self.x2) / 2, y=(self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height


class HomographyMatrix(BaseModel):
    """3x3 homography matrix for perspective transform (pixel <-> world)."""
    model_config = ConfigDict(frozen=True)

    matrix: Annotated[list[list[float]], Field(min_length=3, max_length=3)]

    @field_validator("matrix")
    @classmethod
    def _validate_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("Homography matrix must be 3x3")
        return v


# --- Camera calibration ---


class CameraCalibration(BaseModel):
    """Camera calibration including intrinsics and extrinsics."""
    model_config = ConfigDict(frozen=True)

    camera_id: str
    image_width: int
    image_height: int
    # Intrinsics
    fx: float
    fy: float
    cx: float
    cy: float
    # Distortion coefficients (OpenCV order: k1, k2, p1, p2, k3)
    distortion: Annotated[list[float], Field(min_length=5, max_length=5)] = [0.0, 0.0, 0.0, 0.0, 0.0]
    # Extrinsics (world to camera)
    rotation: Annotated[list[list[float]], Field(min_length=3, max_length=3)]  # 3x3 rotation matrix
    translation: Annotated[list[float], Field(min_length=3, max_length=3)]  # 3x1 translation vector
    # Optional homography for ground plane (pixel -> world meters)
    ground_homography: HomographyMatrix | None = None


# --- Perception ---


class Detection(BaseModel):
    """Single object detection from a frame."""
    model_config = ConfigDict(frozen=True)

    detection_id: UUID = Field(default_factory=uuid4)
    frame_id: UUID
    timestamp: datetime
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    # Optional 3D position if calibration available
    world_position: Point | None = None
    # Optional tracking association
    track_id: UUID | None = None


class Track(BaseModel):
    """Persistent track across frames."""
    model_config = ConfigDict(frozen=True)

    track_id: UUID = Field(default_factory=uuid4)
    class_id: int
    class_name: str
    status: TrackStatus = TrackStatus.TENTATIVE
    # History of detections (most recent last)
    detections: list[Detection] = Field(default_factory=list)
    # Kalman filter state (position, velocity in world coords)
    world_position: Point | None = None
    world_velocity: Point | None = None
    # Age in frames
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    @property
    def latest_detection(self) -> Detection | None:
        return self.detections[-1] if self.detections else None

    @property
    def latest_bbox(self) -> BoundingBox | None:
        det = self.latest_detection
        return det.bbox if det else None


# --- Intersection topology ---


class LaneConfig(BaseModel):
    """Configuration for a single lane."""
    model_config = ConfigDict(frozen=True)

    lane_id: str
    approach_id: str
    # Lane geometry in world coordinates (meters)
    centerline: list[Point] = Field(default_factory=list)  # Polyline from stop line upstream
    width: float = 3.5  # meters
    length: float | None = None  # meters, if known
    # Lane type
    is_turning: bool = False
    turn_direction: Literal["left", "right", "straight"] | None = None
    # Stop line position (world coords)
    stop_line: Point | None = None
    # Detector region (polygon in world coords)
    detector_region: list[Point] | None = None


class ApproachConfig(BaseModel):
    """Configuration for an intersection approach (arm)."""
    model_config = ConfigDict(frozen=True)

    approach_id: str
    name: str  # e.g., "northbound"
    # Lanes belonging to this approach
    lanes: list[LaneConfig] = Field(default_factory=list)
    # Approach geometry
    centerline: list[Point] = Field(default_factory=list)
    # Signal group this approach belongs to
    signal_group: str
    # Pedestrian crossing
    has_crosswalk: bool = True
    crosswalk_region: list[Point] | None = None


class IntersectionConfig(BaseModel):
    """Complete intersection configuration."""
    model_config = ConfigDict(frozen=True)

    intersection_id: str
    name: str
    # Approaches (typically 4: N, S, E, W)
    approaches: list[ApproachConfig] = Field(default_factory=list)
    # Signal phase definitions
    phases: list[SignalPhase] = Field(default_factory=list)
    # Phase timing constraints (seconds)
    min_green: dict[str, float] = Field(default_factory=dict)
    max_green: dict[str, float] = Field(default_factory=dict)
    yellow_time: float = 3.0
    all_red_time: float = 2.0
    # Pedestrian timing
    pedestrian_walk_time: float = 7.0
    pedestrian_clearance_time: float = 10.0
    # Location
    center: Point | None = None
    # Coordinate reference system
    crs: str = "EPSG:3857"  # Web Mercator default


class Intersection(BaseModel):
    """Runtime intersection state."""
    model_config = ConfigDict(frozen=True)

    config: IntersectionConfig
    current_phase: SignalPhase
    phase_start_time: datetime
    phase_elapsed: float  # seconds
    # Per-approach state
    approach_states: dict[str, ApproachState] = Field(default_factory=dict)


# --- Traffic state ---


class LaneState(BaseModel):
    """Estimated state for a single lane."""
    model_config = ConfigDict(frozen=True)

    lane_id: str
    vehicle_count: int = 0
    queue_length_m: float = 0.0
    occupancy: float = 0.0  # 0.0 to 1.0
    average_speed_kmh: float = 0.0
    # Vehicle types breakdown
    vehicle_types: dict[str, int] = Field(default_factory=dict)
    # Timestamp of this measurement
    measured_at: datetime


class ApproachState(BaseModel):
    """Aggregated state for an approach."""
    model_config = ConfigDict(frozen=True)

    approach_id: str
    vehicle_count: int = 0
    queue_length_m: float = 0.0
    average_speed_kmh: float = 0.0
    pedestrian_count: int = 0
    # Lane-level breakdown
    lanes: dict[str, LaneState] = Field(default_factory=dict)
    measured_at: datetime


class TrafficState(BaseModel):
    """Complete traffic state for an intersection at a point in time.

    This is the core contract between perception and control.
    """
    model_config = ConfigDict(frozen=True)

    schema_version: SchemaVersion = SchemaVersion.V1
    intersection_id: str
    observed_at: datetime
    approaches: list[ApproachState] = Field(default_factory=list)
    # Overall intersection metrics
    total_vehicles: int = 0
    total_pedestrians: int = 0
    # Data quality indicators
    perception_latency_ms: float | None = None
    tracking_quality: float | None = None  # 0.0 to 1.0
    # Source info
    source: Literal["camera", "simulation", "synthetic", "hybrid"] = "camera"
    camera_ids: list[str] = Field(default_factory=list)


# --- Control ---


class ControllerRecommendation(BaseModel):
    """Recommendation from a traffic signal controller."""
    model_config = ConfigDict(frozen=True)

    schema_version: SchemaVersion = SchemaVersion.V1
    intersection_id: str
    timestamp: datetime
    current_phase: SignalPhase
    action: ControllerAction
    duration_seconds: Annotated[float, Field(ge=0.0)]
    reason: str
    # Safety/constraints
    respects_min_green: bool = True
    respects_max_green: bool = True
    respects_pedestrian: bool = True
    respects_emergency: bool = True
    # Metadata
    controller_type: str  # e.g., "fixed_time", "max_pressure", "mpc", "rl"
    simulation_only: bool = True
    # Confidence/quality
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


# --- Events ---


class TrafficEvent(BaseModel):
    """Versioned event for the streaming layer."""
    model_config = ConfigDict(frozen=True)

    schema_version: SchemaVersion = SchemaVersion.V1
    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    intersection_id: str
    timestamp: datetime
    # Payload varies by event_type
    payload: dict
    # Metadata
    source_service: str
    trace_id: UUID = Field(default_factory=uuid4)


# --- Experiments ---


class ExperimentConfig(BaseModel):
    """Configuration for a simulation experiment."""
    model_config = ConfigDict(frozen=True)

    experiment_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    # Intersection and scenario
    intersection_id: str
    scenario_config: dict  # SUMO route/demand config
    # Controllers to compare
    controllers: list[str] = Field(default_factory=list)  # Controller type names
    # Duration
    simulation_steps: int = 3600  # seconds
    step_length: float = 1.0
    # Random seed for reproducibility
    seed: int = 42
    # Output
    output_dir: str = "experiments/results"
    # Metrics to collect
    metrics: list[str] = Field(default_factory=lambda: [
        "waiting_time",
        "queue_length",
        "throughput",
        "stops",
        "travel_time",
        "emissions",
    ])


class ExperimentResult(BaseModel):
    """Results from a completed experiment."""
    model_config = ConfigDict(frozen=True)

    experiment_id: UUID
    experiment_name: str
    controller_type: str
    status: ExperimentStatus
    started_at: datetime
    completed_at: datetime | None = None
    # Aggregated metrics
    avg_waiting_time: float | None = None
    max_queue_length: float | None = None
    throughput_veh_per_hour: float | None = None
    avg_stops_per_veh: float | None = None
    avg_travel_time: float | None = None
    total_emissions: float | None = None
    # Per-approach breakdown
    approach_metrics: dict[str, dict] = Field(default_factory=dict)
    # Time series (optional, can be large)
    time_series: dict[str, list[float]] | None = None
    # Error info if failed
    error: str | None = None