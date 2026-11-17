#!/usr/bin/env python
"""Demo script showing domain schema usage."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

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
    ControllerAction,
    ControllerRecommendation,
    # Events
    TrafficEvent,
    EventType,
    # Intersection
    IntersectionConfig,
    ApproachConfig,
    LaneConfig,
)


def demo_detection() -> Detection:
    """Create a sample detection."""
    return Detection(
        frame_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        bbox=BoundingBox(x1=100, y1=200, x2=300, y2=400),
        class_id=2,
        class_name="car",
        confidence=0.92,
        world_position=Point(x=15.5, y=42.3),
    )


def demo_track() -> Track:
    """Create a sample track with history."""
    detections = [
        Detection(
            frame_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            bbox=BoundingBox(x1=100, y1=200, x2=300, y2=400),
            class_id=2,
            class_name="car",
            confidence=0.9,
        ),
        Detection(
            frame_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            bbox=BoundingBox(x1=105, y1=205, x2=305, y2=405),
            class_id=2,
            class_name="car",
            confidence=0.92,
        ),
    ]
    return Track(
        track_id=uuid4(),
        class_id=2,
        class_name="car",
        status=TrackStatus.CONFIRMED,
        detections=detections,
        world_position=Point(x=15.5, y=42.3),
        world_velocity=Point(x=0.5, y=0.2),
        age=2,
        hits=2,
        time_since_update=0,
    )


def demo_traffic_state() -> TrafficState:
    """Create a sample traffic state."""
    return TrafficState(
        intersection_id="int_01",
        observed_at=datetime.now(timezone.utc),
        approaches=[
            ApproachState(
                approach_id="northbound",
                vehicle_count=18,
                queue_length_m=74.0,
                average_speed_kmh=8.0,
                pedestrian_count=3,
                lanes={
                    "lane_1": LaneState(
                        lane_id="lane_1",
                        vehicle_count=10,
                        queue_length_m=40.0,
                        occupancy=0.4,
                        average_speed_kmh=7.0,
                        measured_at=datetime.now(timezone.utc),
                    ),
                    "lane_2": LaneState(
                        lane_id="lane_2",
                        vehicle_count=8,
                        queue_length_m=34.0,
                        occupancy=0.3,
                        average_speed_kmh=9.0,
                        measured_at=datetime.now(timezone.utc),
                    ),
                },
                measured_at=datetime.now(timezone.utc),
            ),
            ApproachState(
                approach_id="southbound",
                vehicle_count=12,
                queue_length_m=45.0,
                average_speed_kmh=15.0,
                pedestrian_count=1,
                measured_at=datetime.now(timezone.utc),
            ),
        ],
        total_vehicles=30,
        total_pedestrians=4,
        perception_latency_ms=45.2,
        tracking_quality=0.94,
        source="camera",
        camera_ids=["cam_01"],
    )


def demo_controller_recommendation() -> ControllerRecommendation:
    """Create a sample controller recommendation."""
    return ControllerRecommendation(
        intersection_id="int_01",
        timestamp=datetime.now(timezone.utc),
        current_phase=SignalPhase.NORTH_SOUTH_GREEN,
        action=ControllerAction.EXTEND_GREEN,
        duration_seconds=8.0,
        reason="High northbound pressure (18 vehicles, 74m queue) vs southbound (12 vehicles, 45m queue)",
        controller_type="max_pressure",
        simulation_only=True,
        confidence=0.85,
    )


def demo_event() -> TrafficEvent:
    """Create a sample traffic event."""
    state = demo_traffic_state()
    return TrafficEvent(
        event_type=EventType.TRAFFIC_STATE,
        intersection_id="int_01",
        timestamp=datetime.now(timezone.utc),
        payload=state.model_dump(mode="json"),
        source_service="state_service",
    )


def demo_intersection_config() -> IntersectionConfig:
    """Create a sample intersection configuration."""
    return IntersectionConfig(
        intersection_id="int_01",
        name="Main St & 1st Ave",
        approaches=[
            ApproachConfig(
                approach_id="northbound",
                name="Northbound Main St",
                lanes=[
                    LaneConfig(lane_id="nb_left", approach_id="northbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="nb_thru", approach_id="northbound"),
                    LaneConfig(lane_id="nb_right", approach_id="northbound", is_turning=True, turn_direction="right"),
                ],
                signal_group="NS",
                has_crosswalk=True,
            ),
            ApproachConfig(
                approach_id="southbound",
                name="Southbound Main St",
                lanes=[
                    LaneConfig(lane_id="sb_left", approach_id="southbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="sb_thru", approach_id="southbound"),
                    LaneConfig(lane_id="sb_right", approach_id="southbound", is_turning=True, turn_direction="right"),
                ],
                signal_group="NS",
                has_crosswalk=True,
            ),
            ApproachConfig(
                approach_id="eastbound",
                name="Eastbound 1st Ave",
                lanes=[
                    LaneConfig(lane_id="eb_left", approach_id="eastbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="eb_thru", approach_id="eastbound"),
                ],
                signal_group="EW",
                has_crosswalk=True,
            ),
            ApproachConfig(
                approach_id="westbound",
                name="Westbound 1st Ave",
                lanes=[
                    LaneConfig(lane_id="wb_left", approach_id="westbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="wb_thru", approach_id="westbound"),
                ],
                signal_group="EW",
                has_crosswalk=True,
            ),
        ],
        phases=[
            SignalPhase.NORTH_SOUTH_GREEN,
            SignalPhase.NORTH_SOUTH_YELLOW,
            SignalPhase.EAST_WEST_GREEN,
            SignalPhase.EAST_WEST_YELLOW,
        ],
        min_green={"NS": 10, "EW": 10},
        max_green={"NS": 60, "EW": 60},
    )


def main() -> None:
    print("=" * 60)
    print("Adaptive Traffic Flow Lab - Domain Schema Demo")
    print("=" * 60)

    # Demo detection
    print("\n1. Detection:")
    det = demo_detection()
    print(f"   {det.class_name} @ {det.bbox.center} (conf={det.confidence:.2f})")

    # Demo track
    print("\n2. Track:")
    track = demo_track()
    print(f"   Track {track.track_id} ({track.class_name}) - {track.status.value}")
    print(f"   Age: {track.age} frames, Hits: {track.hits}")
    print(f"   Position: {track.world_position}, Velocity: {track.world_velocity}")

    # Demo traffic state
    print("\n3. Traffic State:")
    state = demo_traffic_state()
    print(f"   Intersection: {state.intersection_id}")
    print(f"   Total vehicles: {state.total_vehicles}, Pedestrians: {state.total_pedestrians}")
    for approach in state.approaches:
        print(f"   {approach.approach_id}: {approach.vehicle_count} veh, "
              f"{approach.queue_length_m:.1f}m queue, {approach.average_speed_kmh:.1f} km/h")

    # Demo controller recommendation
    print("\n4. Controller Recommendation:")
    rec = demo_controller_recommendation()
    print(f"   Current phase: {rec.current_phase.value}")
    print(f"   Action: {rec.action.value} for {rec.duration_seconds}s")
    print(f"   Reason: {rec.reason}")
    print(f"   Controller: {rec.controller_type}, Confidence: {rec.confidence:.2f}")

    # Demo event
    print("\n5. Traffic Event:")
    event = demo_event()
    print(f"   Event ID: {event.event_id}")
    print(f"   Type: {event.event_type.value}")
    print(f"   Source: {event.source_service}")

    # Demo intersection config
    print("\n6. Intersection Config:")
    config = demo_intersection_config()
    print(f"   Intersection: {config.name} ({config.intersection_id})")
    print(f"   Approaches: {len(config.approaches)}")
    for approach in config.approaches:
        print(f"   - {approach.name} ({len(approach.lanes)} lanes, signal group: {approach.signal_group})")

    # JSON serialization demo
    print("\n7. JSON Serialization:")
    json_str = state.model_dump_json(indent=2)
    print(f"   TrafficState JSON ({len(json_str)} chars):")
    print("   " + "\n   ".join(json_str.split("\n")[:10]) + "\n   ...")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()