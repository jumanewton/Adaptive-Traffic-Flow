"""Contract tests for domain schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from traffic_domain.schemas import (
    # Core
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
    SchemaVersion,
)


class TestGeometryPrimitives:
    """Tests for geometry primitive schemas."""

    def test_point_creation(self) -> None:
        p = Point(x=10.5, y=20.3)
        assert p.x == 10.5
        assert p.y == 20.3

    def test_bounding_box_properties(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.center == Point(x=50, y=25)
        assert bbox.area == 5000

    def test_bounding_box_frozen(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        with pytest.raises(Exception):
            bbox.x1 = 5  # type: ignore

    def test_homography_matrix_valid(self) -> None:
        matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        h = HomographyMatrix(matrix=matrix)
        assert h.matrix == matrix

    def test_homography_matrix_invalid_shape(self) -> None:
        with pytest.raises(ValueError):
            HomographyMatrix(matrix=[[1, 0], [0, 1]])

        with pytest.raises(ValueError):
            HomographyMatrix(matrix=[[1, 0, 0], [0, 1, 0]])


class TestCameraCalibration:
    """Tests for camera calibration schema."""

    def test_minimal_calibration(self) -> None:
        calib = CameraCalibration(
            camera_id="cam_01",
            image_width=1920,
            image_height=1080,
            fx=1000.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0,
            rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation=[0, 0, 5],
        )
        assert calib.camera_id == "cam_01"
        assert calib.distortion == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert calib.ground_homography is None


class TestPerceptionSchemas:
    """Tests for detection and tracking schemas."""

    def test_detection_creation(self) -> None:
        det = Detection(
            frame_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            class_id=2,
            class_name="car",
            confidence=0.95,
        )
        assert det.class_name == "car"
        assert det.confidence == 0.95
        assert det.track_id is None

    def test_detection_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            Detection(
                frame_id=uuid4(),
                timestamp=datetime.now(timezone.utc),
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                class_id=2,
                class_name="car",
                confidence=1.5,  # Invalid > 1
            )

    def test_track_creation(self) -> None:
        track = Track(
            track_id=uuid4(),
            class_id=2,
            class_name="car",
            status=TrackStatus.CONFIRMED,
        )
        assert track.status == TrackStatus.CONFIRMED
        assert track.age == 0
        assert track.hits == 0

    def track_properties(self) -> None:
        det = Detection(
            frame_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            class_id=2,
            class_name="car",
            confidence=0.9,
        )
        track = Track(
            track_id=uuid4(),
            class_id=2,
            class_name="car",
            detections=[det],
        )
        assert track.latest_detection == det
        assert track.latest_bbox == det.bbox


class TestIntersectionTopology:
    """Tests for intersection configuration schemas."""

    def test_lane_config(self) -> None:
        lane = LaneConfig(
            lane_id="lane_1",
            approach_id="northbound",
            centerline=[Point(x=0, y=0), Point(x=0, y=50)],
            width=3.5,
            is_turning=False,
        )
        assert lane.lane_id == "lane_1"

    def test_approach_config(self) -> None:
        approach = ApproachConfig(
            approach_id="northbound",
            name="Northbound",
            lanes=[
                LaneConfig(lane_id="lane_1", approach_id="northbound"),
                LaneConfig(lane_id="lane_2", approach_id="northbound"),
            ],
            signal_group="NS",
        )
        assert len(approach.lanes) == 2

    def test_intersection_config(self) -> None:
        config = IntersectionConfig(
            intersection_id="int_01",
            name="Main & 1st",
            approaches=[
                ApproachConfig(
                    approach_id="northbound",
                    name="Northbound",
                    signal_group="NS",
                ),
                ApproachConfig(
                    approach_id="southbound",
                    name="Southbound",
                    signal_group="NS",
                ),
            ],
            phases=[
                SignalPhase.NORTH_SOUTH_GREEN,
                SignalPhase.NORTH_SOUTH_YELLOW,
                SignalPhase.EAST_WEST_GREEN,
                SignalPhase.EAST_WEST_YELLOW,
            ],
        )
        assert config.intersection_id == "int_01"
        assert len(config.approaches) == 2
        assert len(config.phases) == 4


class TestTrafficState:
    """Tests for traffic state schemas."""

    def test_lane_state(self) -> None:
        lane_state = LaneState(
            lane_id="lane_1",
            vehicle_count=5,
            queue_length_m=25.0,
            occupancy=0.3,
            average_speed_kmh=15.0,
            vehicle_types={"car": 4, "truck": 1},
            measured_at=datetime.now(timezone.utc),
        )
        assert lane_state.vehicle_count == 5

    def test_approach_state(self) -> None:
        approach = ApproachState(
            approach_id="northbound",
            vehicle_count=10,
            queue_length_m=50.0,
            average_speed_kmh=12.0,
            pedestrian_count=2,
            lanes={
                "lane_1": LaneState(lane_id="lane_1", vehicle_count=5, measured_at=datetime.now(timezone.utc)),
                "lane_2": LaneState(lane_id="lane_2", vehicle_count=5, measured_at=datetime.now(timezone.utc)),
            },
            measured_at=datetime.now(timezone.utc),
        )
        assert approach.vehicle_count == 10

    def test_traffic_state_complete(self) -> None:
        state = TrafficState(
            intersection_id="int_01",
            observed_at=datetime.now(timezone.utc),
            approaches=[
                ApproachState(
                    approach_id="northbound",
                    vehicle_count=18,
                    queue_length_m=74.0,
                    average_speed_kmh=8.0,
                    pedestrian_count=3,
                    measured_at=datetime.now(timezone.utc),
                )
            ],
            total_vehicles=18,
            total_pedestrians=3,
            source="camera",
        )
        assert state.schema_version == SchemaVersion.V1
        assert state.intersection_id == "int_01"
        assert len(state.approaches) == 1


class TestControlSchemas:
    """Tests for control schemas."""

    def test_controller_recommendation(self) -> None:
        rec = ControllerRecommendation(
            intersection_id="int_01",
            timestamp=datetime.now(timezone.utc),
            current_phase=SignalPhase.NORTH_SOUTH_GREEN,
            action=ControllerAction.EXTEND_GREEN,
            duration_seconds=8.0,
            reason="high_northbound_pressure",
            controller_type="max_pressure",
            simulation_only=True,
        )
        assert rec.action == ControllerAction.EXTEND_GREEN
        assert rec.confidence == 1.0

    def test_controller_recommendation_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            ControllerRecommendation(
                intersection_id="int_01",
                timestamp=datetime.now(timezone.utc),
                current_phase=SignalPhase.NORTH_SOUTH_GREEN,
                action=ControllerAction.HOLD_PHASE,
                duration_seconds=5.0,
                reason="test",
                controller_type="test",
                confidence=1.5,  # Invalid > 1
            )


class TestEventSchemas:
    """Tests for event streaming schemas."""

    def test_traffic_event(self) -> None:
        event = TrafficEvent(
            event_type=EventType.TRAFFIC_STATE,
            intersection_id="int_01",
            timestamp=datetime.now(timezone.utc),
            payload={"vehicle_count": 10},
            source_service="vision_worker",
        )
        assert event.schema_version == SchemaVersion.V1
        assert event.event_type == EventType.TRAFFIC_STATE
        assert event.trace_id is not None


class TestExperimentSchemas:
    """Tests for experiment schemas."""

    def test_experiment_config(self) -> None:
        config = ExperimentConfig(
            name="test_experiment",
            intersection_id="int_01",
            scenario_config={"demand": "peak"},
            controllers=["fixed_time", "max_pressure"],
            simulation_steps=3600,
        )
        assert config.experiment_id is not None
        assert len(config.controllers) == 2

    def test_experiment_result(self) -> None:
        result = ExperimentResult(
            experiment_id=uuid4(),
            experiment_name="test",
            controller_type="max_pressure",
            status=ExperimentStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            avg_waiting_time=25.5,
            throughput_veh_per_hour=1200,
        )
        assert result.status == ExperimentStatus.COMPLETED
        assert result.avg_waiting_time == 25.5


class TestSerialization:
    """Tests for JSON serialization/deserialization."""

    def test_traffic_state_json_roundtrip(self) -> None:
        state = TrafficState(
            intersection_id="int_01",
            observed_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            approaches=[
                ApproachState(
                    approach_id="northbound",
                    vehicle_count=18,
                    queue_length_m=74.0,
                    average_speed_kmh=8.0,
                    pedestrian_count=3,
                    measured_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                )
            ],
        )

        # Serialize
        json_str = state.model_dump_json()
        data = json.loads(json_str)

        # Verify key fields
        assert data["intersection_id"] == "int_01"
        assert data["approaches"][0]["vehicle_count"] == 18
        assert data["schema_version"] == 1

        # Deserialize
        restored = TrafficState.model_validate_json(json_str)
        assert restored.intersection_id == state.intersection_id
        assert restored.approaches[0].vehicle_count == state.approaches[0].vehicle_count

    def test_controller_recommendation_json_roundtrip(self) -> None:
        rec = ControllerRecommendation(
            intersection_id="int_01",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            current_phase=SignalPhase.NORTH_SOUTH_GREEN,
            action=ControllerAction.EXTEND_GREEN,
            duration_seconds=8.0,
            reason="high_northbound_pressure",
            controller_type="max_pressure",
            simulation_only=True,
        )

        json_str = rec.model_dump_json()
        restored = ControllerRecommendation.model_validate_json(json_str)

        assert restored.action == ControllerAction.EXTEND_GREEN
        assert restored.duration_seconds == 8.0
        assert restored.simulation_only is True


class TestSchemaVersioning:
    """Tests for schema version handling."""

    def test_schema_version_default(self) -> None:
        state = TrafficState(
            intersection_id="int_01",
            observed_at=datetime.now(timezone.utc),
        )
        assert state.schema_version == SchemaVersion.V1

    def test_event_schema_version(self) -> None:
        event = TrafficEvent(
            event_type=EventType.TRAFFIC_STATE,
            intersection_id="int_01",
            timestamp=datetime.now(timezone.utc),
            payload={},
            source_service="test",
        )
        assert event.schema_version == SchemaVersion.V1