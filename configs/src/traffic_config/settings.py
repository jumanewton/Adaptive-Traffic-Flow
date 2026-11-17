"""Configuration settings using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VisionSettings(BaseSettings):
    """Vision/perception settings."""

    model_config = SettingsConfigDict(env_prefix="VISION_")

    # Model
    model_name: str = "yolov8n.pt"
    model_path: Path | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    half_precision: bool = False

    # Inference
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300
    img_size: int = 640

    # Classes to detect (COCO classes)
    classes: list[int] = Field(default_factory=lambda: [2, 3, 5, 7])  # car, motorcycle, bus, truck
    # Add pedestrian class
    pedestrian_class: int = 0

    # Camera
    camera_id: str = "cam_01"
    camera_fps: int = 30
    camera_resolution: tuple[int, int] = (1920, 1080)

    # Calibration
    calibration_file: Path | None = None
    enable_calibration: bool = True

    @field_validator("confidence_threshold")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("iou_threshold")
    @classmethod
    def _validate_iou(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        return v


class TrackingSettings(BaseSettings):
    """ByteTrack tracking settings."""

    model_config = SettingsConfigDict(env_prefix="TRACKING_")

    # ByteTrack parameters
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    min_box_area: float = 10.0
    mot20: bool = False

    # Track management
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3

    # Kalman filter
    use_kalman: bool = True


class TrafficSettings(BaseSettings):
    """Traffic state estimation settings."""

    model_config = SettingsConfigDict(env_prefix="TRAFFIC_")

    # Queue estimation
    queue_detection_distance_m: float = 10.0  # Distance from stop line to consider queued
    min_speed_queued_kmh: float = 5.0  # Below this speed = queued
    occupancy_time_window_s: float = 5.0  # Time window for occupancy calculation

    # Speed estimation
    speed_smoothing_alpha: float = 0.3  # Exponential smoothing factor
    min_track_length_for_speed: int = 5  # Minimum track points for speed estimate

    # Lane mapping
    lane_assignment_tolerance_m: float = 1.5  # Distance tolerance for lane assignment

    # Update interval
    state_update_interval_s: float = 1.0


class ControlSettings(BaseSettings):
    """Traffic signal control settings."""

    model_config = SettingsConfigDict(env_prefix="CONTROL_")

    # Controller type
    controller_type: Literal["fixed_time", "max_pressure", "mpc", "rl"] = "max_pressure"

    # Fixed-time
    fixed_time_cycle_length: int = 120  # seconds
    fixed_time_phase_splits: dict[str, int] = Field(
        default_factory=lambda: {"north_south": 50, "east_west": 50}
    )

    # Max-pressure
    max_pressure_min_green: int = 10
    max_pressure_max_green: int = 60
    max_pressure_pressure_weight: float = 1.0
    max_pressure_downstream_weight: float = 0.5

    # Safety constraints (always enforced)
    min_green_time: int = 7
    max_green_time: int = 120
    yellow_time: int = 3
    all_red_time: int = 2
    pedestrian_walk_time: int = 7
    pedestrian_clearance_time: int = 10

    # Emergency vehicle
    emergency_preempt_enabled: bool = True
    emergency_clearance_time: int = 5

    # Control loop
    control_loop_interval_s: float = 1.0

    @field_validator("min_green_time", "max_green_time", "yellow_time", "all_red_time",
                     "pedestrian_walk_time", "pedestrian_clearance_time",
                     "emergency_clearance_time", "max_pressure_min_green", "max_pressure_max_green",
                     "fixed_time_cycle_length")
    @classmethod
    def _validate_positive_time(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Time values must be non-negative")
        return v

    @field_validator("min_green_time")
    @classmethod
    def _validate_min_green(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_green_time must be at least 1 second")
        return v


class SimulationSettings(BaseSettings):
    """SUMO simulation settings."""

    model_config = SettingsConfigDict(env_prefix="SIM_")

    # SUMO paths
    sumo_home: Path | None = None
    sumo_binary: str = "sumo"
    sumo_gui_binary: str = "sumo-gui"

    # Network
    network_file: Path | None = None
    route_file: Path | None = None
    additional_files: list[Path] = Field(default_factory=list)

    # Simulation
    step_length: float = 1.0
    simulation_duration: int = 3600  # seconds
    seed: int = 42

    # TraCI
    traci_port: int = 8813
    traci_host: str = "localhost"

    # Output
    output_dir: Path = Path("simulation/output")
    emit_fcd: bool = False  # Floating car data
    emit_queue: bool = True
    emit_emissions: bool = True


class StreamingSettings(BaseSettings):
    """Event streaming (Kafka/Redpanda) settings."""

    model_config = SettingsConfigDict(env_prefix="STREAM_")

    # Bootstrap
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "traffic-lab"

    # Topics
    traffic_state_topic: str = "traffic.state"
    controller_action_topic: str = "traffic.control.action"
    phase_change_topic: str = "traffic.phase.change"
    incident_topic: str = "traffic.incident"

    # Producer
    acks: Literal["all", "1", "0"] = "all"
    retries: int = 3
    linger_ms: int = 5
    compression_type: Literal["gzip", "snappy", "lz4", "zstd", "none"] = "gzip"

    # Consumer
    group_id: str = "traffic-lab-consumer"
    auto_offset_reset: Literal["earliest", "latest"] = "latest"
    enable_auto_commit: bool = True

    # Schema registry (optional)
    schema_registry_url: str | None = None


class StorageSettings(BaseSettings):
    """Data storage settings."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    # Database
    database_url: str = "postgresql://traffic:traffic@localhost:5432/traffic_lab"
    clickhouse_url: str | None = None
    timeseries_url: str | None = None

    # Object storage (MinIO/S3)
    s3_endpoint: str = "localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "traffic-lab"
    s3_region: str = "us-east-1"
    s3_secure: bool = False

    # Local paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    exports_dir: Path = Path("models/exports")


class ObservabilitySettings(BaseSettings):
    """Observability settings."""

    model_config = SettingsConfigDict(env_prefix="OBS_")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False
    log_file: Path | None = None

    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"

    # Tracing
    tracing_enabled: bool = True
    otlp_endpoint: str | None = None
    console_export: bool = False

    # Service
    service_name: str = "traffic-lab"
    environment: Literal["development", "staging", "production"] = "development"


class Settings(BaseSettings):
    """Root settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Sub-settings (loaded from env with prefixes)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    traffic: TrafficSettings = Field(default_factory=TrafficSettings)
    control: ControlSettings = Field(default_factory=ControlSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    streaming: StreamingSettings = Field(default_factory=StreamingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    # Global
    project_root: Path = Path(__file__).parents[3]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()