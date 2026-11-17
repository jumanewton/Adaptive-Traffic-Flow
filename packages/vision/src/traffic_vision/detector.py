"""Detector interface and YOLO implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from ultralytics import YOLO

from traffic_config.settings import VisionSettings, get_settings
from traffic_domain.schemas import Detection, BoundingBox, Point
from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry, track_latency


logger = get_logger(__name__)


class Detector(ABC):
    """Abstract detector interface."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        """Run detection on a frame.

        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier

        Returns:
            List of detections
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Warm up the model (e.g., first inference)."""
        pass


class YOLODetector(Detector):
    """Ultralytics YOLO detector."""

    def __init__(self, settings: VisionSettings | None = None):
        """Initialize YOLO detector.

        Args:
            settings: Vision settings (defaults to global config)
        """
        self._settings = settings or get_settings().vision
        self._model: YOLO | None = None
        self._metrics = get_metrics_registry()
        self._class_names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
        }

    def warmup(self) -> None:
        """Warm up the model with a dummy inference."""
        if self._model is None:
            self._load_model()

        # Create dummy frame
        dummy = np.zeros((self._settings.img_size, self._settings.img_size, 3), dtype=np.uint8)
        _ = self._model(dummy, verbose=False)
        logger.info("YOLO model warmed up")

    def _load_model(self) -> None:
        """Load YOLO model."""
        model_path = self._settings.model_path or self._settings.model_name
        self._model = YOLO(model_path)

        # Set device
        if self._settings.device != "auto":
            self._model.to(self._settings.device)

        # Enable half precision if requested
        if self._settings.half_precision:
            self._model.model.half()

        logger.info("YOLO model loaded", model=model_path, device=self._settings.device)

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        """Run YOLO detection on a frame."""
        if self._model is None:
            self._load_model()

        # Run inference
        results = self._model(
            frame,
            conf=self._settings.confidence_threshold,
            iou=self._settings.iou_threshold,
            max_det=self._settings.max_det,
            imgsz=self._settings.img_size,
            classes=self._settings.classes,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Filter by class
                if cls_id not in self._settings.classes and cls_id != self._settings.pedestrian_class:
                    continue

                # Filter by confidence
                if conf < self._settings.confidence_threshold:
                    continue

                class_name = self._class_names.get(cls_id, f"class_{cls_id}")

                detection = Detection(
                    detection_id=uuid4(),
                    frame_id=uuid4(),  # Will be set by caller
                    timestamp=None,  # Will be set by caller
                    bbox=BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                )
                detections.append(detection)

                # Record metrics
                self._metrics.inc_counter(
                    "traffic_detections_total",
                    labels={"class_name": class_name, "camera_id": self._settings.camera_id},
                )

        return detections

    def detect_batch(self, frames: list[np.ndarray], frame_ids: list[int]) -> list[list[Detection]]:
        """Run detection on a batch of frames."""
        if self._model is None:
            self._load_model()

        results = self._model(
            frames,
            conf=self._settings.confidence_threshold,
            iou=self._settings.iou_threshold,
            max_det=self._settings.max_det,
            imgsz=self._settings.img_size,
            classes=self._settings.classes,
            verbose=False,
        )

        batch_detections = []
        for frame_idx, result in enumerate(results):
            detections = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    if cls_id not in self._settings.classes and cls_id != self._settings.pedestrian_class:
                        continue

                    if conf < self._settings.confidence_threshold:
                        continue

                    class_name = self._class_names.get(cls_id, f"class_{cls_id}")

                    detection = Detection(
                        detection_id=uuid4(),
                        frame_id=uuid4(),
                        timestamp=None,
                        bbox=BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        ),
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                    )
                    detections.append(detection)

                    self._metrics.inc_counter(
                        "traffic_detections_total",
                        labels={"class_name": class_name, "camera_id": self._settings.camera_id},
                    )

            batch_detections.append(detections)

        return batch_detections


def create_detector(detector_type: str = "yolo", settings: VisionSettings | None = None) -> Detector:
    """Factory function to create detector."""
    if detector_type.lower() == "yolo":
        return YOLODetector(settings)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")