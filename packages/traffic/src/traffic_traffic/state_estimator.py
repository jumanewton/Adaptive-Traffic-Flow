"""Traffic state estimator - aggregates tracks into traffic state."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from traffic_domain.schemas import (
    Track,
    TrafficState,
    ApproachState,
    LaneState,
    IntersectionConfig,
    ApproachConfig,
)
from traffic_traffic.lane_mapper import (
    LaneMapper,
    estimate_queue_length,
    estimate_occupancy,
    estimate_average_speed,
)
from traffic_observability.logging import get_logger


logger = get_logger(__name__)


class TrafficStateEstimator:
    """Estimates traffic state from tracks."""

    def __init__(
        self,
        intersection_config: IntersectionConfig,
        queue_distance_m: float = 10.0,
        min_speed_kmh: float = 5.0,
        occupancy_time_window_s: float = 5.0,
    ):
        """Initialize state estimator.

        Args:
            intersection_config: Intersection configuration
            queue_distance_m: Distance from stop line to consider queued
            min_speed_kmh: Speed threshold for queued vehicles
            occupancy_time_window_s: Time window for occupancy calculation
        """
        self.config = intersection_config
        self.lane_mapper = LaneMapper(intersection_config)
        self.queue_distance_m = queue_distance_m
        self.min_speed_kmh = min_speed_kmh
        self.occupancy_time_window_s = occupancy_time_window_s

    def estimate(self, tracks: list[Track], timestamp: datetime | None = None) -> TrafficState:
        """Estimate traffic state from tracks.

        Args:
            tracks: List of active tracks
            timestamp: Observation timestamp (defaults to now)

        Returns:
            TrafficState with per-approach and per-lane metrics
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Group tracks by approach and lane
        approach_tracks: dict[str, list[Track]] = defaultdict(list)
        lane_tracks: dict[str, list[Track]] = defaultdict(list)

        for track in tracks:
            approach_id = self.lane_mapper.assign_track_to_approach(track)
            if approach_id:
                approach_tracks[approach_id].append(track)

            lane_id = self.lane_mapper.assign_track_to_lane(track)
            if lane_id:
                lane_tracks[lane_id].append(track)

        # Build approach states
        approaches = []
        total_vehicles = 0
        total_pedestrians = 0

        for approach_config in self.config.approaches:
            approach_id = approach_config.approach_id
            tracks_in_approach = approach_tracks.get(approach_id, [])

            # Count vehicles and pedestrians
            vehicle_count = sum(1 for t in tracks_in_approach if t.class_name != "person")
            pedestrian_count = sum(1 for t in tracks_in_approach if t.class_name == "person")

            total_vehicles += vehicle_count
            total_pedestrians += pedestrian_count

            # Vehicle types breakdown
            vehicle_types = defaultdict(int)
            for t in tracks_in_approach:
                if t.class_name != "person":
                    vehicle_types[t.class_name] += 1

            # Estimate queue length
            queue_length = 0.0
            for lane in approach_config.lanes:
                lane_queue = estimate_queue_length(
                    tracks_in_approach,
                    self.lane_mapper,
                    lane.lane_id,
                    self.queue_distance_m,
                    self.min_speed_kmh,
                )
                queue_length = max(queue_length, lane_queue)

            # Estimate average speed
            speeds = []
            for track in tracks_in_approach:
                if track.world_velocity:
                    speed = (track.world_velocity.x ** 2 + track.world_velocity.y ** 2) ** 0.5
                    speeds.append(speed * 3.6)
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

            # Lane states
            lane_states = {}
            for lane in approach_config.lanes:
                lane_tracks_list = lane_tracks.get(lane.lane_id, [])
                lane_vehicle_count = sum(1 for t in lane_tracks_list if t.class_name != "person")

                lane_queue = estimate_queue_length(
                    lane_tracks_list,
                    self.lane_mapper,
                    lane.lane_id,
                    self.queue_distance_m,
                    self.min_speed_kmh,
                )
                lane_occupancy = estimate_occupancy(
                    lane_tracks_list,
                    self.lane_mapper,
                    lane.lane_id,
                    self.occupancy_time_window_s,
                )
                lane_speed = estimate_average_speed(lane_tracks_list, self.lane_mapper, lane.lane_id)

                lane_states[lane.lane_id] = LaneState(
                    lane_id=lane.lane_id,
                    vehicle_count=lane_vehicle_count,
                    queue_length_m=lane_queue,
                    occupancy=lane_occupancy,
                    average_speed_kmh=lane_speed,
                    vehicle_types=dict(
                        (t.class_name, sum(1 for tt in lane_tracks_list if tt.class_name == t.class_name))
                        for t in lane_tracks_list if t.class_name != "person"
                    ),
                    measured_at=timestamp,
                )

            approach_state = ApproachState(
                approach_id=approach_id,
                vehicle_count=vehicle_count,
                queue_length_m=queue_length,
                average_speed_kmh=avg_speed,
                pedestrian_count=pedestrian_count,
                lanes=lane_states,
                measured_at=timestamp,
            )
            approaches.append(approach_state)

        return TrafficState(
            intersection_id=self.config.intersection_id,
            observed_at=timestamp,
            approaches=approaches,
            total_vehicles=total_vehicles,
            total_pedestrians=total_pedestrians,
            source="camera",
        )