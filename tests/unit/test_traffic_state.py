"""Tests for traffic state estimation."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from traffic_domain.schemas import (
    Track,
    TrackStatus,
    BoundingBox,
    Point,
    IntersectionConfig,
    ApproachConfig,
    LaneConfig,
    Detection,
)
from traffic_traffic.lane_mapper import LaneMapper
from traffic_traffic.state_estimator import TrafficStateEstimator


def create_test_tracks(num_vehicles: int = 5, approach_id: str = "northbound") -> list[Track]:
    """Create test tracks for a specific approach."""
    tracks = []
    for i in range(num_vehicles):
        # Create detections with world positions along the approach
        detections = []
        for j in range(3):
            x = 0.0 if "north" in approach_id or "south" in approach_id else (i - 2) * 3.5
            y = -50.0 + j * 20.0 if "north" in approach_id else 50.0 - j * 20.0
            if "east" in approach_id:
                x = 50.0 - j * 20.0
                y = (i - 2) * 3.5
            elif "west" in approach_id:
                x = -50.0 + j * 20.0
                y = (i - 2) * 3.5

            detections.append(Detection(
                frame_id=uuid4(),
                timestamp=datetime.now(timezone.utc),
                bbox=BoundingBox(x1=x-2, y1=y-2, x2=x+2, y2=y+2),
                class_id=2,
                class_name="car",
                confidence=0.9,
            ))

        track = Track(
            track_id=uuid4(),
            class_id=2,
            class_name="car",
            status=TrackStatus.CONFIRMED,
            detections=detections,
            world_position=Point(x=x, y=y),
            world_velocity=Point(x=0.1, y=0.5) if "north" in approach_id else Point(x=0.1, y=-0.5),
            age=3,
            hits=3,
            time_since_update=0,
        )
        tracks.append(track)
    return tracks


class TestLaneMapper:
    """Tests for LaneMapper."""

    def test_assign_track_to_approach(self):
        """Test track-to-approach assignment."""
        config = IntersectionConfig(
            intersection_id="int_01",
            name="Test",
            approaches=[
                ApproachConfig(
                    approach_id="northbound",
                    name="Northbound",
                    lanes=[LaneConfig(lane_id="nb_1", approach_id="northbound")],
                    signal_group="NS",
                    centerline=[Point(x=0, y=-100), Point(x=0, y=0)],
                ),
                ApproachConfig(
                    approach_id="southbound",
                    name="Southbound",
                    lanes=[LaneConfig(lane_id="sb_1", approach_id="southbound")],
                    signal_group="NS",
                    centerline=[Point(x=0, y=100), Point(x=0, y=0)],
                ),
            ],
        )
        mapper = LaneMapper(config)

        # Track in northbound area
        track = Track(
            track_id=uuid4(),
            class_id=2,
            class_name="car",
            status=TrackStatus.CONFIRMED,
            world_position=Point(x=0, y=-50),
        )
        approach = mapper.assign_track_to_approach(track)
        assert approach == "northbound"

        # Track in southbound area - create new track with different position
        track = Track(
            track_id=uuid4(),
            class_id=2,
            class_name="car",
            status=TrackStatus.CONFIRMED,
            world_position=Point(x=0, y=50),
        )
        approach = mapper.assign_track_to_approach(track)
        assert approach == "southbound"

    def test_assign_track_to_lane(self):
        """Test track-to-lane assignment."""
        config = IntersectionConfig(
            intersection_id="int_01",
            name="Test",
            approaches=[
                ApproachConfig(
                    approach_id="northbound",
                    name="Northbound",
                    lanes=[
                        LaneConfig(lane_id="nb_left", approach_id="northbound", centerline=[
                            Point(x=-3.5, y=-100), Point(x=-3.5, y=0)
                        ]),
                        LaneConfig(lane_id="nb_thru", approach_id="northbound", centerline=[
                            Point(x=0, y=-100), Point(x=0, y=0)
                        ]),
                    ],
                    signal_group="NS",
                    centerline=[Point(x=0, y=-100), Point(x=0, y=0)],
                ),
            ],
        )
        mapper = LaneMapper(config, tolerance_m=1.0)  # Smaller tolerance to avoid overlap

        # Track in left lane
        track = Track(
            track_id=uuid4(),
            class_id=2,
            class_name="car",
            status=TrackStatus.CONFIRMED,
            world_position=Point(x=-3.5, y=-50),
        )
        lane = mapper.assign_track_to_lane(track)
        assert lane == "nb_left"

        # Track in through lane - create new track, position clearly in through lane (x=2.0)
        track = Track(
            track_id=uuid4(),
            class_id=2,
            class_name="car",
            status=TrackStatus.CONFIRMED,
            world_position=Point(x=2.0, y=-50),
        )
        lane = mapper.assign_track_to_lane(track)
        assert lane == "nb_thru"

    def test_is_queued(self):
        """Test queue detection."""
        config = IntersectionConfig(
            intersection_id="int_01",
            name="Test",
            approaches=[
                ApproachConfig(
                    approach_id="northbound",
                    name="Northbound",
                    lanes=[
                        LaneConfig(
                            lane_id="nb_1",
                            approach_id="northbound",
                            centerline=[Point(x=0, y=-100), Point(x=0, y=0)],
                            stop_line=Point(x=0, y=0),
                            length=100,
                        ),
                    ],
                    signal_group="NS",
                    centerline=[Point(x=0, y=-100), Point(x=0, y=0)],
                ),
            ],
        )
        mapper = LaneMapper(config)

        # Slow track near stop line = queued
        track = Track(
            track_id=uuid4(),
            class_id=2,
            class_name="car",
            status=TrackStatus.CONFIRMED,
            world_position=Point(x=0, y=-5),
            world_velocity=Point(x=0, y=0.5),  # ~1.8 km/h
        )
        assert mapper.is_queued(track, "nb_1", min_speed_kmh=5.0, queue_distance_m=10.0)

        # Fast track near stop line = not queued
        track = Track(
            track_id=uuid4(),
            class_id=2,
            class_name="car",
            status=TrackStatus.CONFIRMED,
            world_position=Point(x=0, y=-5),
            world_velocity=Point(x=0, y=5.0),  # ~18 km/h
        )
        assert not mapper.is_queued(track, "nb_1", min_speed_kmh=5.0, queue_distance_m=10.0)

        # Slow track far from stop line = not queued
        track = Track(
            track_id=uuid4(),
            class_id=2,
            class_name="car",
            status=TrackStatus.CONFIRMED,
            world_position=Point(x=0, y=-50),
            world_velocity=Point(x=0, y=0.5)
        )
        assert not mapper.is_queued(track, "nb_1", min_speed_kmh=5.0, queue_distance_m=10.0)


class TestTrafficStateEstimator:
    """Tests for TrafficStateEstimator."""

    def test_estimate_basic(self):
        """Test basic traffic state estimation."""
        config = IntersectionConfig(
            intersection_id="int_01",
            name="Test",
            approaches=[
                ApproachConfig(
                    approach_id="northbound",
                    name="Northbound",
                    lanes=[LaneConfig(lane_id="nb_1", approach_id="northbound", length=100)],
                    signal_group="NS",
                    centerline=[Point(x=0, y=-100), Point(x=0, y=0)],
                ),
                ApproachConfig(
                    approach_id="southbound",
                    name="Southbound",
                    lanes=[LaneConfig(lane_id="sb_1", approach_id="southbound", length=100)],
                    signal_group="NS",
                    centerline=[Point(x=0, y=100), Point(x=0, y=0)],
                ),
            ],
        )
        estimator = TrafficStateEstimator(config)

        # Create tracks for both approaches
        tracks = create_test_tracks(5, "northbound") + create_test_tracks(3, "southbound")

        state = estimator.estimate(tracks)

        assert state.intersection_id == "int_01"
        assert state.total_vehicles == 8
        assert len(state.approaches) == 2

        nb = next(a for a in state.approaches if a.approach_id == "northbound")
        sb = next(a for a in state.approaches if a.approach_id == "southbound")

        assert nb.vehicle_count == 5
        assert sb.vehicle_count == 3

    def test_estimate_with_pedestrians(self):
        """Test estimation includes pedestrian counting."""
        config = IntersectionConfig(
            intersection_id="int_01",
            name="Test",
            approaches=[
                ApproachConfig(
                    approach_id="northbound",
                    name="Northbound",
                    lanes=[LaneConfig(lane_id="nb_1", approach_id="northbound")],
                    signal_group="NS",
                    centerline=[Point(x=0, y=-100), Point(x=0, y=0)],
                ),
            ],
        )
        estimator = TrafficStateEstimator(config)

        # Create pedestrian tracks
        tracks = []
        for i in range(3):
            track = Track(
                track_id=uuid4(),
                class_id=0,
                class_name="person",
                status=TrackStatus.CONFIRMED,
                world_position=Point(x=0, y=-30 - i * 5),
            )
            tracks.append(track)

        state = estimator.estimate(tracks)

        assert state.total_pedestrians == 3
        assert state.total_vehicles == 0

    def test_lane_level_estimates(self):
        """Test lane-level metrics are computed."""
        config = IntersectionConfig(
            intersection_id="int_01",
            name="Test",
            approaches=[
                ApproachConfig(
                    approach_id="northbound",
                    name="Northbound",
                    lanes=[
                        LaneConfig(lane_id="nb_left", approach_id="northbound", length=100),
                        LaneConfig(lane_id="nb_thru", approach_id="northbound", length=100),
                    ],
                    signal_group="NS",
                    centerline=[Point(x=0, y=-100), Point(x=0, y=0)],
                ),
            ],
        )
        estimator = TrafficStateEstimator(config)

        # Create tracks distributed across lanes - need to provide lane geometries
        # for lane assignment to work properly
        config = IntersectionConfig(
            intersection_id="int_01",
            name="Test",
            approaches=[
                ApproachConfig(
                    approach_id="northbound",
                    name="Northbound",
                    lanes=[
                        LaneConfig(
                            lane_id="nb_left",
                            approach_id="northbound",
                            length=100,
                            centerline=[Point(x=-3.5, y=-100), Point(x=-3.5, y=0)]
                        ),
                        LaneConfig(
                            lane_id="nb_thru",
                            approach_id="northbound",
                            length=100,
                            centerline=[Point(x=0, y=-100), Point(x=0, y=0)]
                        ),
                    ],
                    signal_group="NS",
                    centerline=[Point(x=0, y=-100), Point(x=0, y=0)],
                ),
            ],
        )
        estimator = TrafficStateEstimator(config)

        # Create tracks distributed across lanes
        tracks = []
        for i in range(3):
            track = Track(
                track_id=uuid4(),
                class_id=2,
                class_name="car",
                status=TrackStatus.CONFIRMED,
                world_position=Point(x=-3.5, y=-50 - i * 10),  # Left lane
            )
            tracks.append(track)

        for i in range(2):
            track = Track(
                track_id=uuid4(),
                class_id=2,
                class_name="car",
                status=TrackStatus.CONFIRMED,
                world_position=Point(x=0, y=-50 - i * 10),  # Through lane
            )
            tracks.append(track)

        state = estimator.estimate(tracks)

        nb = next(a for a in state.approaches if a.approach_id == "northbound")
        assert "nb_left" in nb.lanes
        assert "nb_thru" in nb.lanes
        assert nb.lanes["nb_left"].vehicle_count == 3
        assert nb.lanes["nb_thru"].vehicle_count == 2


class TestQueueEstimation:
    """Tests for queue length estimation."""

    def test_estimate_queue_length(self):
        """Test queue length estimation."""
        config = IntersectionConfig(
            intersection_id="int_01",
            name="Test",
            approaches=[
                ApproachConfig(
                    approach_id="northbound",
                    name="Northbound",
                    lanes=[
                        LaneConfig(
                            lane_id="nb_1",
                            approach_id="northbound",
                            centerline=[Point(x=0, y=-100), Point(x=0, y=0)],
                            stop_line=Point(x=0, y=0),
                            length=100,
                        ),
                    ],
                    signal_group="NS",
                    centerline=[Point(x=0, y=-100), Point(x=0, y=0)],
                ),
            ],
        )
        mapper = LaneMapper(config)

        # Create queued tracks at different distances
        tracks = []
        for dist in [2, 5, 8, 15]:  # meters from stop line
            track = Track(
                track_id=uuid4(),
                class_id=2,
                class_name="car",
                status=TrackStatus.CONFIRMED,
                world_position=Point(x=0, y=-dist),
                world_velocity=Point(x=0, y=0.5),  # Slow = queued
            )
            tracks.append(track)

        queue_len = mapper.get_distance_to_stop_line(tracks[0], "nb_1")
        assert queue_len is not None
        # Should estimate max queue from furthest queued vehicle