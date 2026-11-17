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
import signal
import sys
from pathlib import Path

import cv2

from traffic_config.settings import get_settings
from traffic_domain.schemas import Detection, Track, TrafficEvent, EventType
from traffic_observability.logging import configure_logging, get_logger
from traffic_observability.metrics import get_metrics_registry, create_traffic_metrics
from traffic_streaming.producer import EventProducer


logger = get_logger(__name__)

# Global for graceful shutdown
_shutdown = False


def signal_handler(signum: int, frame) -> None:
    global _shutdown
    logger.info("Shutdown signal received", signal=signum)
    _shutdown = True


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

    # Initialize event producer
    producer = EventProducer(
        bootstrap_servers=streaming_cfg.bootstrap_servers,
        client_id=f"{streaming_cfg.client_id}-vision",
    )

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

    # TODO: Initialize YOLO model
    # model = YOLO(vision_cfg.model_path or vision_cfg.model_name)
    # model.to(vision_cfg.device)

    # TODO: Initialize ByteTrack tracker
    # tracker = ByteTrack(tracking_cfg)

    frame_count = 0

    try:
        while not _shutdown:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break

            frame_count += 1
            frame_id = frame_count  # Use frame number as ID for now

            # TODO: Run YOLO inference
            # results = model(frame, conf=vision_cfg.confidence_threshold, ...)
            # detections = convert_yolo_results(results, frame_id)

            # TODO: Run ByteTrack tracking
            # tracks = tracker.update(detections)

            # TODO: Publish detections/tracks to event stream
            # event = TrafficEvent(...)
            # producer.publish(streaming_cfg.traffic_state_topic, event)

            # TODO: Draw annotations on frame
            # annotated_frame = draw_annotations(frame, detections, tracks)

            # Write output frame
            if writer:
                writer.write(frame)  # Replace with annotated_frame when ready

            # Log progress periodically
            if frame_count % 100 == 0:
                logger.debug("Processing frame", frame=frame_count)

    except Exception as e:
        logger.exception("Error processing video", error=str(e))
        return 1

    finally:
        cap.release()
        if writer:
            writer.release()
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