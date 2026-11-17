"""Tests for configuration settings."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from traffic_config.settings import (
    Settings,
    VisionSettings,
    TrackingSettings,
    TrafficSettings,
    ControlSettings,
    SimulationSettings,
    StreamingSettings,
    StorageSettings,
    ObservabilitySettings,
    get_settings,
)


class TestVisionSettings:
    """Tests for vision settings."""

    def test_defaults(self) -> None:
        s = VisionSettings()
        assert s.model_name == "yolov8n.pt"
        assert s.device == "auto"
        assert s.confidence_threshold == 0.25
        assert s.classes == [2, 3, 5, 7]

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VISION_MODEL_NAME", "yolov8s.pt")
        monkeypatch.setenv("VISION_CONFIDENCE_THRESHOLD", "0.5")
        monkeypatch.setenv("VISION_DEVICE", "cuda")

        s = VisionSettings()
        assert s.model_name == "yolov8s.pt"
        assert s.confidence_threshold == 0.5
        assert s.device == "cuda"


class TestTrackingSettings:
    """Tests for tracking settings."""

    def test_defaults(self) -> None:
        s = TrackingSettings()
        assert s.track_thresh == 0.5
        assert s.track_buffer == 30
        assert s.match_thresh == 0.8

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRACKING_TRACK_THRESH", "0.6")
        monkeypatch.setenv("TRACKING_TRACK_BUFFER", "60")

        s = TrackingSettings()
        assert s.track_thresh == 0.6
        assert s.track_buffer == 60


class TestControlSettings:
    """Tests for control settings."""

    def test_defaults(self) -> None:
        s = ControlSettings()
        assert s.controller_type == "max_pressure"
        assert s.min_green_time == 7
        assert s.max_green_time == 120
        assert s.yellow_time == 3

    def test_invalid_controller_type(self) -> None:
        with pytest.raises(ValidationError):
            ControlSettings(controller_type="invalid_type")

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONTROL_CONTROLLER_TYPE", "fixed_time")
        monkeypatch.setenv("CONTROL_MIN_GREEN_TIME", "10")

        s = ControlSettings()
        assert s.controller_type == "fixed_time"
        assert s.min_green_time == 10


class TestSimulationSettings:
    """Tests for simulation settings."""

    def test_defaults(self) -> None:
        s = SimulationSettings()
        assert s.sumo_binary == "sumo"
        assert s.step_length == 1.0
        assert s.simulation_duration == 3600
        assert s.seed == 42


class TestStreamingSettings:
    """Tests for streaming settings."""

    def test_defaults(self) -> None:
        s = StreamingSettings()
        assert s.bootstrap_servers == "localhost:9092"
        assert s.traffic_state_topic == "traffic.state"
        assert s.acks == "all"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STREAM_BOOTSTRAP_SERVERS", "kafka:9092")
        monkeypatch.setenv("STREAM_TRAFFIC_STATE_TOPIC", "custom.traffic")

        s = StreamingSettings()
        assert s.bootstrap_servers == "kafka:9092"
        assert s.traffic_state_topic == "custom.traffic"


class TestStorageSettings:
    """Tests for storage settings."""

    def test_defaults(self) -> None:
        s = StorageSettings()
        assert "postgresql" in s.database_url
        assert s.s3_bucket == "traffic-lab"


class TestObservabilitySettings:
    """Tests for observability settings."""

    def test_defaults(self) -> None:
        s = ObservabilitySettings()
        assert s.log_level == "INFO"
        assert s.metrics_enabled is True
        assert s.tracing_enabled is True


class TestRootSettings:
    """Tests for root settings aggregation."""

    def test_all_subsettings_present(self) -> None:
        s = Settings()
        assert isinstance(s.vision, VisionSettings)
        assert isinstance(s.tracking, TrackingSettings)
        assert isinstance(s.traffic, TrafficSettings)
        assert isinstance(s.control, ControlSettings)
        assert isinstance(s.simulation, SimulationSettings)
        assert isinstance(s.streaming, StreamingSettings)
        assert isinstance(s.storage, StorageSettings)
        assert isinstance(s.observability, ObservabilitySettings)

    def test_project_root_path(self) -> None:
        s = Settings()
        assert s.project_root.name == "yoholo"

    def test_env_file_loading(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("VISION_MODEL_NAME=yolov8m.pt\nCONTROL_CONTROLLER_TYPE=fixed_time\n")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VISION_MODEL_NAME", "yolov8m.pt")
        monkeypatch.setenv("CONTROL_CONTROLLER_TYPE", "fixed_time")

        # Clear cache and create new settings
        from traffic_config.settings import get_settings
        get_settings.cache_clear()

        s = get_settings()
        assert s.vision.model_name == "yolov8m.pt"
        assert s.control.controller_type == "fixed_time"

    def test_get_settings_cached(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2  # Same instance due to lru_cache


class TestSettingsValidation:
    """Tests for validation edge cases."""

    def test_vision_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            VisionSettings(confidence_threshold=1.5)

        with pytest.raises(ValidationError):
            VisionSettings(confidence_threshold=-0.1)

    def test_control_timing_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ControlSettings(min_green_time=0)

        with pytest.raises(ValidationError):
            ControlSettings(yellow_time=-1)

    def test_streaming_acks_validation(self) -> None:
        with pytest.raises(ValidationError):
            StreamingSettings(acks="invalid")

    def test_observability_log_level(self) -> None:
        with pytest.raises(ValidationError):
            ObservabilitySettings(log_level="INVALID")