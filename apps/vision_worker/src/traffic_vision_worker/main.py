"""Vision Worker main entry point.

Processes video streams:
1. Reads frames from video file or camera
2. Runs YOLO detection
3. Runs ByteTrack tracking
4. Publishes detections/tracks to event stream
5. Optionally writes annotated video output
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
    BoundingBox,
    Point,
)
from traffic_observability.logging import configure_logging, get_logger
from traffic_observability.metrics import get_metrics_registry, create_traffic_metrics, track_latency
from traffic_streaming.producer import SyncEventProducer
from traffic_vision.detector import YOLODetector
from traffic_tracking.tracker import ByteTracker


logger = get_logger(__name__)

# Global for graceful shutdown
_shutdown = False


def signal_handler(signum: int, frame) -> None:
    global _shutdown
    logger.info("Shutdown signal received", signal=signum)
    _shutdown = True


def draw_annotations(
    frame: np.ndarray,
    detections: list[Detection],
    tracks: list[Track],
    show_detections: bool = True,
    show_tracks: bool = True,
) -> np.ndarray:
    """Draw detection and tracking annotations on frame."""
    annotated = frame.copy()

    # Draw detections (green boxes)
    if show_detections:
        for det in detections:
            x1, y1, x2, y2 = map(int, [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw tracks (blue boxes with track ID)
    if show_tracks:
        for track in tracks:
            if track.latest_bbox:
                x1, y1, x2, y2 = map(int, [
                    track.latest_bbox.x1, track.latest_bbox.y1,
                    track.latest_bbox.x2, track.latest_bbox.y2
                ])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"ID:{track.track_id} {track.class_name}"
                if track.status.value == "confirmed":
                    label += " ✓"
                cv2.putText(annotated, label, (x1, y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Draw track history (last 30 positions)
                if len(track.detections) > 1:
                    pts = []
                    for det in track.detections[-30:]:
                        cx, cy = map(int, [det.bbox.center.x, det.bbox.center.y])
                        pts.append((cx, cy))
                    for i in range(1, len(pts)):
                        cv2.line(annotated, pts[i-1], pts[i], (255, 0, 0), 1)

    return annotated


def tracks_to_traffic_state(tracks: list[Track], intersection_id: str, camera_id: str) -> TrafficState:
    """Convert tracks to a basic TrafficState (simplified - no lane mapping)."""
    approaches = {}
    total_vehicles = 0
    total_pedestrians = 0

    for track in tracks:
        if track.status.value not in ("confirmed", "tentative"):
            continue

        # Simple approach assignment based on bbox center x position
        # This is a placeholder - real implementation uses lane_mapper
        if track.latest_bbox:
            cx = track.latest_bbox.center.x
            # Assume 1920 width, split into 4 approaches
            if cx < 480:
                approach_id = "westbound"
            elif cx < 960:
                approach_id = "southbound"
            elif cx < 1440:
                approach_id = "eastbound"
            else:
                approach_id = "northbound"
        else:
            approach_id = "unknown"

        if approach_id not in approaches:
            approaches[approach_id] = {"vehicles": 0, "pedestrians": 0, "tracks": []}

        if track.class_name == "person":
            approaches[approach_id]["pedestrians"] += 1
            total_pedestrians += 1
        else:
            approaches[approach_id]["vehicles"] += 1
            total_vehicles += 1

        approaches[approach_id]["tracks"].append(track)

    # Build approach states
    approach_states = []
    for approach_id, data in approaches.items():
        # Estimate queue length from track positions (simplified)
        queue_length = len(data["tracks"]) * 5.0  # Rough estimate

        # Estimate average speed
        speeds = []
        for t in data["tracks"]:
            if t.world_velocity:
                speed = (t.world_velocity.x ** 2 + t.world_velocity.y ** 2) ** 0.5
                speeds.append(speed * 3.6)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        approach_states.append(ApproachState(
            approach_id=approach_id,
            vehicle_count=data["vehicles"],
            queue_length_m=queue_length,
            average_speed_kmh=avg_speed,
            pedestrian_count=data["pedestrians"],
            measured_at=datetime.now(timezone.utc),
        ))

    return TrafficState(
        intersection_id=intersection_id,
        observed_at=datetime.now(timezone.utc),
        approaches=approach_states,
        total_vehicles=total_vehicles,
        total_pedestrians=total_pedestrians,
        source="camera",
        camera_ids=[camera_id],
    )


def run_vision_worker(
    video_source: str | int,
    output_video: str | None = None,
    config_override: dict | None = None,
) -> int:
    """Run the vision worker on a video source.

    Args:
        video_source: Path to video file or camera index (0, 1, ...)
        output_video: Optional path to write annotated output video
        config_override: Optional configuration overrides

    Returns:
        Exit code (0 = success)
    """
    global _shutdown

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load configuration
    settings = get_settings()
    vision_cfg = settings.vision
    tracking_cfg = settings.tracking
    streaming_cfg = settings.streaming
    obs_cfg = settings.observability

    # Override config if provided
    if config_override:
        for key, value in config_override.items():
            if hasattr(vision_cfg, key):
                setattr(vision_cfg, key, value)

    # Configure logging
    configure_logging(
        level=obs_cfg.log_level,
        json_output=obs_cfg.log_json,
        service_name="vision_worker",
    )

    # Initialize metrics
    metrics_registry = get_metrics_registry()
    traffic_metrics = create_traffic_metrics(metrics_registry, "vision_worker")

    # Pre-initialize metrics with labels to avoid runtime errors
    traffic_metrics["tracks_active"].labels(camera_id=vision_cfg.camera_id)
    traffic_metrics["track_age_frames"].labels(camera_id=vision_cfg.camera_id)

    # Initialize event producer (sync wrapper for simplicity) - optional
    producer = None
    try:
        producer = SyncEventProducer(
            bootstrap_servers=streaming_cfg.bootstrap_servers,
            client_id=f"{streaming_cfg.client_id}-vision",
        )
        logger.info("Event producer connected", bootstrap_servers=streaming_cfg.bootstrap_servers)
    except Exception as e:
        logger.warning("Event producer unavailable, running without streaming", error=str(e))

    # Initialize detector
    detector = YOLODetector(vision_cfg)
    detector.warmup()

    # Initialize tracker
    tracker = ByteTracker(tracking_cfg, camera_id=vision_cfg.camera_id)

    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error("Failed to open video source", source=video_source)
        return 1

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or vision_cfg.camera_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or vision_cfg.camera_resolution[0])
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or vision_cfg.camera_resolution[1])

    logger.info(
        "Video source opened",
        source=video_source,
        fps=fps,
        resolution=f"{width}x{height}",
    )

    # Setup output video writer if requested
    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        logger.info("Output video writer initialized", path=output_video)

    frame_count = 0
    start_time = time.time()

    try:
        while not _shutdown:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break

            frame_count += 1
            frame_timestamp = datetime.now(timezone.utc)

            # Run YOLO detection
            detections = detector.detect(frame, frame_count)

            # Update frame_id and timestamp for detections
            for det in detections:
                det.frame_id = uuid4()
                det.timestamp = frame_timestamp

            # Run ByteTrack tracking
            tracks = tracker.update(detections)

            # Convert to traffic state and publish (if producer available)
            traffic_state = tracks_to_traffic_state(
                tracks, "intersection-01", vision_cfg.camera_id
            )

            event = TrafficEvent(
                event_type=EventType.TRAFFIC_STATE,
                intersection_id="intersection-01",
                timestamp=frame_timestamp,
                payload=traffic_state.model_dump(mode="json"),
                source_service="vision_worker",
            )

            if producer:
                try:
                    producer.publish(streaming_cfg.traffic_state_topic, event)
                except Exception as e:
                    logger.warning("Failed to publish event", error=str(e))

            # Draw annotations
            annotated_frame = draw_annotations(frame, detections, tracks)

            # Write output frame
            if writer:
                writer.write(annotated_frame)

            # Log progress periodically
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                logger.info(
                    "Processing",
                    frame=frame_count,
                    fps=f"{fps_actual:.1f}",
                    detections=len(detections),
                    tracks=len([t for t in tracks if t.status.value == "confirmed"]),
                )

            # Throttle to target FPS if processing too fast
            loop_elapsed = time.time() - loop_start
            target_frame_time = 1.0 / fps
            if loop_elapsed < target_frame_time:
                time.sleep(target_frame_time - loop_elapsed)

    except Exception as e:
        logger.exception("Error processing video", error=str(e))
        return 1

    finally:
        cap.release()
        if writer:
            writer.release()
        if producer:
            producer.close()
        logger.info("Vision worker stopped", frames_processed=frame_count)

    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Traffic Vision Worker")
    parser.add_argument(
        "source",
        help="Video file path or camera index (e.g., 0 for webcam)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output annotated video path",
    )
    parser.add_argument(
        "--model",
        help="YOLO model path (overrides config)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device (overrides config)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        help="Confidence threshold (overrides config)",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable annotation drawing (faster processing)",
    )

    args = parser.parse_args()

    # Parse source (int for camera, str for file)
    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    config_override = {}
    if args.model:
        config_override["model_path"] = args.model
    if args.device:
        config_override["device"] = args.device
    if args.conf:
        config_override["confidence_threshold"] = args.conf

    return run_vision_worker(source, args.output, config_override)


if __name__ == "__main__":
    sys.exit(main())