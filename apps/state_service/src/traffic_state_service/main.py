"""State Service main entry point.

Consumes tracks/detections from vision worker, estimates traffic state,
and publishes TrafficState events to the event stream.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from traffic_config.settings import get_settings
from traffic_domain.schemas import (
    Detection,
    Track,
    TrafficEvent,
    EventType,
    TrafficState,
    ApproachState,
    LaneState,
    IntersectionConfig,
    ApproachConfig,
    LaneConfig,
    BoundingBox,
    Point,
)
from traffic_observability.logging import configure_logging, get_logger
from traffic_observability.metrics import get_metrics_registry, create_traffic_metrics
from traffic_streaming.consumer import SyncEventConsumer, EventHandler, EventConsumer
from traffic_streaming.producer import SyncEventProducer
from traffic_traffic.lane_mapper import LaneMapper
from traffic_traffic.state_estimator import TrafficStateEstimator
from traffic_vision.calibration import CameraCalibrator, load_calibration


logger = get_logger(__name__)

# Global for graceful shutdown
_shutdown = False


def signal_handler(signum: int, frame) -> None:
    global _shutdown
    logger.info("Shutdown signal received", signal=signum)
    _shutdown = True


class TrackEventHandler(EventHandler):
    """Handle track events from vision worker."""

    def __init__(self, state_estimator: TrafficStateEstimator, producer: SyncEventProducer, topic: str):
        self._estimator = state_estimator
        self._producer = producer
        self._topic = topic

    @property
    def event_types(self) -> list[EventType]:
        return [EventType.TRAFFIC_STATE]  # Vision worker publishes TrafficState events

    async def handle(self, event: TrafficEvent) -> None:
        """Process incoming traffic state event from vision worker."""
        # The vision worker publishes TrafficState directly
        # We can refine it with lane mapping or just forward
        try:
            # Re-publish with enrichment if needed
            self._producer.publish(self._topic, event)
        except Exception as e:
            logger.warning("Failed to republish event", error=str(e))


class TracksFromDetectionsHandler(EventHandler):
    """Handle raw detections and build tracks internally (alternative mode)."""

    def __init__(
        self,
        state_estimator: TrafficStateEstimator,
        producer: SyncEventProducer,
        topic: str,
        tracker,
    ):
        self._estimator = state_estimator
        self._producer = producer
        self._topic = topic
        self._tracker = tracker

    @property
    def event_types(self) -> list[EventType]:
        return [EventType.TRAFFIC_STATE]  # Could also handle raw detections

    async def handle(self, event: TrafficEvent) -> None:
        """Process incoming events."""
        try:
            self._producer.publish(self._topic, event)
        except Exception as e:
            logger.warning("Failed to republish event", error=str(e))


def create_default_intersection_config() -> IntersectionConfig:
    """Create a default 4-approach intersection configuration."""
    return IntersectionConfig(
        intersection_id="intersection-01",
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
            "north_south_green",
            "north_south_yellow",
            "east_west_green",
            "east_west_yellow",
        ],
    )


def create_lane_geometries(config: IntersectionConfig, image_width: int, image_height: int) -> IntersectionConfig:
    """Create lane geometries for a standard intersection layout.

    This creates a simple geometric layout for testing. In production,
    this would come from calibrated camera homography.
    """
    approach_length = min(image_width, image_height) // 3
    lane_width = 3.5  # meters (in world coords)

    # We'll create world coordinates (meters) with origin at intersection center
    # North is negative Y, South is positive Y, East is positive X, West is negative X

    new_approaches = []
    for approach in config.approaches:
        new_lanes = []
        new_centerline = []

        if "north" in approach.approach_id:
            # Northbound: vehicles moving south (positive Y direction)
            new_centerline = [
                Point(x=0, y=-approach_length),
                Point(x=0, y=-50),
                Point(x=0, y=-10),
            ]
            for i, lane in enumerate(approach.lanes):
                new_lanes.append(LaneConfig(
                    lane_id=lane.lane_id,
                    approach_id=lane.approach_id,
                    centerline=[
                        Point(x=(i - 1) * lane_width, y=-approach_length),
                        Point(x=(i - 1) * lane_width, y=-10),
                    ],
                    width=lane.width,
                    length=approach_length,
                    is_turning=lane.is_turning,
                    turn_direction=lane.turn_direction,
                    stop_line=Point(x=(i - 1) * lane_width, y=0),
                ))

        elif "south" in approach.approach_id:
            # Southbound: vehicles moving north (negative Y direction)
            new_centerline = [
                Point(x=0, y=approach_length),
                Point(x=0, y=50),
                Point(x=0, y=10),
            ]
            for i, lane in enumerate(approach.lanes):
                new_lanes.append(LaneConfig(
                    lane_id=lane.lane_id,
                    approach_id=lane.approach_id,
                    centerline=[
                        Point(x=(i - 1) * lane_width, y=approach_length),
                        Point(x=(i - 1) * lane_width, y=10),
                    ],
                    width=lane.width,
                    length=approach_length,
                    is_turning=lane.is_turning,
                    turn_direction=lane.turn_direction,
                    stop_line=Point(x=(i - 1) * lane_width, y=0),
                ))

        elif "east" in approach.approach_id:
            # Eastbound: vehicles moving west (negative X direction)
            new_centerline = [
                Point(x=approach_length, y=0),
                Point(x=50, y=0),
                Point(x=10, y=0),
            ]
            for i, lane in enumerate(approach.lanes):
                new_lanes.append(LaneConfig(
                    lane_id=lane.lane_id,
                    approach_id=lane.approach_id,
                    centerline=[
                        Point(x=approach_length, y=(i - 0.5) * lane_width),
                        Point(x=10, y=(i - 0.5) * lane_width),
                    ],
                    width=lane.width,
                    length=approach_length,
                    is_turning=lane.is_turning,
                    turn_direction=lane.turn_direction,
                    stop_line=Point(x=0, y=(i - 0.5) * lane_width),
                ))

        elif "west" in approach.approach_id:
            # Westbound: vehicles moving east (positive X direction)
            new_centerline = [
                Point(x=-approach_length, y=0),
                Point(x=-50, y=0),
                Point(x=-10, y=0),
            ]
            for i, lane in enumerate(approach.lanes):
                new_lanes.append(LaneConfig(
                    lane_id=lane.lane_id,
                    approach_id=lane.approach_id,
                    centerline=[
                        Point(x=-approach_length, y=(i - 0.5) * lane_width),
                        Point(x=-10, y=(i - 0.5) * lane_width),
                    ],
                    width=lane.width,
                    length=approach_length,
                    is_turning=lane.is_turning,
                    turn_direction=lane.turn_direction,
                    stop_line=Point(x=0, y=(i - 0.5) * lane_width),
                ))

        new_approaches.append(ApproachConfig(
            approach_id=approach.approach_id,
            name=approach.name,
            lanes=new_lanes,
            centerline=new_centerline,
            signal_group=approach.signal_group,
            has_crosswalk=approach.has_crosswalk,
            crosswalk_region=approach.crosswalk_region,
        ))

    return IntersectionConfig(
        intersection_id=config.intersection_id,
        name=config.name,
        approaches=new_approaches,
        phases=config.phases,
        min_green=config.min_green,
        max_green=config.max_green,
        yellow_time=config.yellow_time,
        all_red_time=config.all_red_time,
        pedestrian_walk_time=config.pedestrian_walk_time,
        pedestrian_clearance_time=config.pedestrian_clearance_time,
    )


def tracks_to_world_tracks(tracks: list[Track], calibrator: CameraCalibrator | None) -> list[Track]:
    """Convert pixel-space tracks to world-space tracks using calibration."""
    if not calibrator:
        return tracks

    world_tracks = []
    for track in tracks:
        if track.status.value not in ("confirmed", "tentative"):
            continue

        # Convert latest detection bbox center to world coordinates
        if track.latest_bbox:
            pixel_center = track.latest_bbox.center
            world_pos = calibrator.bbox_center_to_world(pixel_center)

            # Estimate velocity from track history
            world_vel = None
            if len(track.detections) >= 2:
                last_det = track.detections[-1]
                prev_det = track.detections[-2]
                if last_det.timestamp and prev_det.timestamp:
                    dt = (last_det.timestamp - prev_det.timestamp).total_seconds()
                    if dt > 0:
                        last_world = calibrator.bbox_center_to_world(last_det.bbox.center)
                        prev_world = calibrator.bbox_center_to_world(prev_det.bbox.center)
                        if last_world and prev_world:
                            world_vel = Point(
                                x=(last_world.x - prev_world.x) / dt,
                                y=(last_world.y - prev_world.y) / dt,
                            )

            # Create new track with world coordinates
            world_track = Track(
                track_id=track.track_id,
                class_id=track.class_id,
                class_name=track.class_name,
                status=track.status,
                detections=track.detections,
                world_position=world_pos,
                world_velocity=world_vel,
                age=track.age,
                hits=track.hits,
                time_since_update=track.time_since_update,
            )
            world_tracks.append(world_track)

    return world_tracks


def run_state_service(
    input_source: str | None = None,
    config_file: str | None = None,
    calibration_file: str | None = None,
    output_video: str | None = None,
) -> int:
    """Run the state estimation service.

    Args:
        input_source: Video file path or camera index (for standalone mode)
        config_file: Intersection config JSON file
        calibration_file: Camera calibration JSON file
        output_video: Optional output video with state visualization

    Returns:
        Exit code (0 = success)
    """
    global _shutdown

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    settings = get_settings()
    vision_cfg = settings.vision
    tracking_cfg = settings.tracking
    streaming_cfg = settings.streaming
    traffic_cfg = settings.traffic
    obs_cfg = settings.observability

    configure_logging(
        level=obs_cfg.log_level,
        json_output=obs_cfg.log_json,
        service_name="state_service",
    )

    # Initialize metrics
    metrics_registry = get_metrics_registry()
    traffic_metrics = create_traffic_metrics(metrics_registry, "state_service")

    # Initialize event producer (optional)
    producer = None
    try:
        producer = SyncEventProducer(
            bootstrap_servers=streaming_cfg.bootstrap_servers,
            client_id=f"{streaming_cfg.client_id}-state",
        )
        logger.info("Event producer connected", bootstrap_servers=streaming_cfg.bootstrap_servers)
    except Exception as e:
        logger.warning("Event producer unavailable, running without streaming", error=str(e))

    # Initialize event consumer (optional)
    consumer = None
    try:
        consumer = SyncEventConsumer(
            topics=[streaming_cfg.traffic_state_topic],
            group_id=f"{streaming_cfg.group_id}-state",
        )
        logger.info("Event consumer connected", topics=[streaming_cfg.traffic_state_topic])
    except Exception as e:
        logger.warning("Event consumer unavailable", error=str(e))

    # Load intersection configuration
    if config_file and Path(config_file).exists():
        # TODO: Load from file
        intersection_config = create_default_intersection_config()
    else:
        intersection_config = create_default_intersection_config()

    # Add lane geometries (simplified for testing)
    intersection_config = create_lane_geometries(intersection_config, vision_cfg.camera_resolution[0], vision_cfg.camera_resolution[1])

    # Load calibration
    calibrator = None
    if calibration_file and Path(calibration_file).exists():
        calibration = load_calibration(calibration_file)
        calibrator = CameraCalibrator(calibration)
        logger.info("Camera calibration loaded", file=calibration_file)

    # Initialize lane mapper and state estimator
    lane_mapper = LaneMapper(intersection_config, tolerance_m=traffic_cfg.lane_assignment_tolerance_m)
    state_estimator = TrafficStateEstimator(
        intersection_config,
        queue_distance_m=traffic_cfg.queue_detection_distance_m,
        min_speed_kmh=traffic_cfg.min_speed_queued_kmh,
        occupancy_time_window_s=traffic_cfg.occupancy_time_window_s,
    )

    # Initialize detector and tracker for standalone mode
    detector = None
    tracker = None
    cap = None
    writer = None

    if input_source:
        # Standalone mode: process video directly
        from traffic_vision.detector import YOLODetector
        from traffic_tracking.tracker import ByteTracker

        detector = YOLODetector(vision_cfg)
        detector.warmup()
        tracker = ByteTracker(tracking_cfg, camera_id=vision_cfg.camera_id)

        # Open video
        try:
            source: str | int = int(input_source)
        except ValueError:
            source = input_source

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Failed to open video source", source=source)
            return 1

        fps = cap.get(cv2.CAP_PROP_FPS) or vision_cfg.camera_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or vision_cfg.camera_resolution[0])
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or vision_cfg.camera_resolution[1])

        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    try:
        if input_source:
            # Standalone mode: process video frames
            logger.info("Starting standalone video processing", source=input_source)
            while not _shutdown:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream")
                    break

                frame_count += 1
                frame_timestamp = datetime.now(timezone.utc)

                # Detect
                detections = detector.detect(frame, frame_count)
                for det in detections:
                    det.frame_id = uuid4()
                    det.timestamp = frame_timestamp

                # Track
                tracks = tracker.update(detections)

                # Convert to world coordinates
                world_tracks = tracks_to_world_tracks(tracks, calibrator)

                # Estimate traffic state
                traffic_state = state_estimator.estimate(world_tracks, frame_timestamp)

                # Publish event
                event = TrafficEvent(
                    event_type=EventType.TRAFFIC_STATE,
                    intersection_id=intersection_config.intersection_id,
                    timestamp=frame_timestamp,
                    payload=traffic_state.model_dump(mode="json"),
                    source_service="state_service",
                )

                if producer:
                    try:
                        producer.publish(streaming_cfg.traffic_state_topic, event)
                    except Exception as e:
                        logger.warning("Failed to publish event", error=str(e))

                # Visualize
                if writer:
                    annotated = draw_state_visualization(frame, traffic_state, tracks, lane_mapper)
                    writer.write(annotated)

                # Log progress
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Processing",
                        frame=frame_count,
                        fps=f"{fps_actual:.1f}",
                        vehicles=traffic_state.total_vehicles,
                        pedestrians=traffic_state.total_pedestrians,
                    )

        else:
            # Streaming mode: consume from Kafka
            logger.info("Starting streaming mode")
            if consumer:
                consumer.run_forever()

    except Exception as e:
        logger.exception("Error in state service", error=str(e))
        return 1

    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        if producer:
            producer.close()
        if consumer:
            consumer.close()
        logger.info("State service stopped", frames_processed=frame_count)

    return 0


def draw_state_visualization(
    frame: np.ndarray,
    state: TrafficState,
    tracks: list[Track],
    lane_mapper: LaneMapper,
) -> np.ndarray:
    """Draw traffic state visualization on frame."""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Draw approach info
    y_offset = 30
    for approach in state.approaches:
        text = f"{approach.approach_id}: {approach.vehicle_count} veh, {approach.queue_length_m:.1f}m queue, {approach.average_speed_kmh:.1f} km/h"
        cv2.putText(annotated, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25

    # Draw lane info
    for approach in state.approaches:
        for lane_id, lane_state in approach.lanes.items():
            text = f"  {lane_id}: {lane_state.vehicle_count} veh, occ={lane_state.occupancy:.2f}, q={lane_state.queue_length_m:.1f}m"
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20

    # Draw tracks with lane assignments
    for track in tracks:
        if track.status.value not in ("confirmed", "tentative"):
            continue

        if track.latest_bbox:
            x1, y1, x2, y2 = map(int, [
                track.latest_bbox.x1, track.latest_bbox.y1,
                track.latest_bbox.x2, track.latest_bbox.y2
            ])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Show lane assignment
            lane_id = lane_mapper.assign_track_to_lane(track)
            approach_id = lane_mapper.assign_track_to_approach(track)
            label = f"ID:{track.track_id} {track.class_name}"
            if lane_id:
                label += f" [{lane_id}]"
            if approach_id:
                label += f" ({approach_id})"
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Summary
    cv2.putText(annotated, f"Total: {state.total_vehicles} veh, {state.total_pedestrians} ped",
               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description="Traffic State Service")
    parser.add_argument(
        "source",
        nargs="?",
        help="Video file path or camera index (standalone mode)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output annotated video path",
    )
    parser.add_argument(
        "--config",
        help="Intersection config JSON file",
    )
    parser.add_argument(
        "--calibration",
        help="Camera calibration JSON file",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Run in streaming mode (consume from Kafka)",
    )

    args = parser.parse_args()

    if args.streaming and not args.source:
        return run_state_service(config_file=args.config, calibration_file=args.calibration)
    elif args.source:
        return run_state_service(
            input_source=args.source,
            config_file=args.config,
            calibration_file=args.calibration,
            output_video=args.output,
        )
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())