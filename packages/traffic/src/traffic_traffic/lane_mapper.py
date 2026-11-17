"""Lane mapping - assign tracks to lanes and approaches."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from shapely.geometry import Point as ShapelyPoint, Polygon, LineString

from traffic_domain.schemas import (
    Track,
    LaneConfig,
    ApproachConfig,
    IntersectionConfig,
    Point,
)
from traffic_observability.logging import get_logger


logger = get_logger(__name__)


class LaneMapper:
    """Maps tracks to lanes and approaches based on geometry."""

    def __init__(self, intersection_config: IntersectionConfig, tolerance_m: float = 1.5):
        """Initialize lane mapper.

        Args:
            intersection_config: Intersection configuration with lane geometries
            tolerance_m: Distance tolerance for lane assignment (meters)
        """
        self.config = intersection_config
        self.tolerance = tolerance_m

        # Pre-compute lane polygons for fast point-in-polygon tests
        self._lane_polygons: dict[str, Polygon] = {}
        self._approach_polygons: dict[str, Polygon] = {}
        self._stop_lines: dict[str, LineString] = {}
        self._build_geometries()

    def _build_geometries(self) -> None:
        """Build Shapely geometries from config."""
        for approach in self.config.approaches:
            # Build approach polygon from centerline
            if len(approach.centerline) >= 2:
                coords = [(p.x, p.y) for p in approach.centerline]
                # Create buffer around centerline
                line = LineString(coords)
                self._approach_polygons[approach.approach_id] = line.buffer(10.0)  # 10m buffer

            # Build lane polygons
            for lane in approach.lanes:
                if len(lane.centerline) >= 2:
                    coords = [(p.x, p.y) for p in lane.centerline]
                    line = LineString(coords)
                    self._lane_polygons[lane.lane_id] = line.buffer(lane.width / 2 + self.tolerance)

                # Stop line
                if lane.stop_line:
                    # Create perpendicular line at stop line
                    pass  # Simplified

    def assign_track_to_lane(self, track: Track) -> str | None:
        """Assign a track to a lane based on its position.

        Args:
            track: Track with world position

        Returns:
            Lane ID if assigned, None otherwise
        """
        if not track.world_position:
            return None

        point = ShapelyPoint(track.world_position.x, track.world_position.y)

        # Check lane polygons
        for lane_id, polygon in self._lane_polygons.items():
            if polygon.contains(point):
                return lane_id

        # Fallback: check approach polygons
        for approach_id, polygon in self._approach_polygons.items():
            if polygon.contains(point):
                # Find first lane in this approach
                for lane in self.config.approaches:
                    if lane.approach_id == approach_id and lane.lanes:
                        return lane.lanes[0].lane_id

        return None

    def assign_track_to_approach(self, track: Track) -> str | None:
        """Assign a track to an approach."""
        if not track.world_position:
            return None

        point = ShapelyPoint(track.world_position.x, track.world_position.y)

        for approach_id, polygon in self._approach_polygons.items():
            if polygon.contains(point):
                return approach_id

        return None

    def get_distance_to_stop_line(self, track: Track, lane_id: str) -> float | None:
        """Get distance from track to stop line along lane."""
        if not track.world_position:
            return None

        lane_config = self._get_lane_config(lane_id)
        if not lane_config or not lane_config.stop_line:
            return None

        # Project track position onto lane centerline
        point = ShapelyPoint(track.world_position.x, track.world_position.y)
        line = LineString([(p.x, p.y) for p in lane_config.centerline])

        # Distance along line
        projected = line.project(point)
        stop_line_proj = line.project(ShapelyPoint(lane_config.stop_line.x, lane_config.stop_line.y))

        return abs(stop_line_proj - projected)

    def is_queued(self, track: Track, lane_id: str, min_speed_kmh: float = 5.0, queue_distance_m: float = 10.0) -> bool:
        """Determine if a track is queued (stopped or slow near stop line)."""
        # Check speed
        if track.world_velocity:
            speed = (track.world_velocity.x ** 2 + track.world_velocity.y ** 2) ** 0.5
            speed_kmh = speed * 3.6
            if speed_kmh > min_speed_kmh:
                return False

        # Check distance to stop line
        dist = self.get_distance_to_stop_line(track, lane_id)
        if dist is not None and dist <= queue_distance_m:
            return True

        return False

    def _get_lane_config(self, lane_id: str) -> LaneConfig | None:
        """Get lane config by ID."""
        for approach in self.config.approaches:
            for lane in approach.lanes:
                if lane.lane_id == lane_id:
                    return lane
        return None


def estimate_queue_length(
    tracks: list[Track],
    lane_mapper: LaneMapper,
    lane_id: str,
    queue_distance_m: float = 10.0,
    min_speed_kmh: float = 5.0,
) -> float:
    """Estimate queue length for a lane."""
    queued_tracks = [
        t for t in tracks
        if lane_mapper.assign_track_to_lane(t) == lane_id
        and lane_mapper.is_queued(t, lane_id, min_speed_kmh, queue_distance_m)
    ]

    if not queued_tracks:
        return 0.0

    # Find furthest queued vehicle from stop line
    max_dist = 0.0
    for track in queued_tracks:
        dist = lane_mapper.get_distance_to_stop_line(track, lane_id)
        if dist is not None:
            max_dist = max(max_dist, dist)

    return max_dist


def estimate_occupancy(
    tracks: list[Track],
    lane_mapper: LaneMapper,
    lane_id: str,
    time_window_s: float = 5.0,
) -> float:
    """Estimate lane occupancy (0.0 to 1.0)."""
    lane_config = lane_mapper._get_lane_config(lane_id)
    if not lane_config:
        return 0.0

    lane_tracks = [t for t in tracks if lane_mapper.assign_track_to_lane(t) == lane_id]

    if not lane_tracks:
        return 0.0

    # Simple occupancy: fraction of lane length occupied by vehicles
    # Assuming average vehicle length of 5m
    vehicle_length = 5.0
    occupied_length = len(lane_tracks) * vehicle_length
    lane_length = lane_config.length or 100.0

    occupancy = min(occupied_length / lane_length, 1.0)
    return occupancy


def estimate_average_speed(
    tracks: list[Track],
    lane_mapper: LaneMapper,
    lane_id: str,
) -> float:
    """Estimate average speed for a lane (km/h)."""
    lane_tracks = [t for t in tracks if lane_mapper.assign_track_to_lane(t) == lane_id]

    speeds = []
    for track in lane_tracks:
        if track.world_velocity:
            speed = (track.world_velocity.x ** 2 + track.world_velocity.y ** 2) ** 0.5
            speeds.append(speed * 3.6)  # m/s to km/h

    if not speeds:
        return 0.0

    return sum(speeds) / len(speeds)