"""BentoML service for YOLO model serving."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from ultralytics import YOLO

try:
    import bentoml
    from bentoml.io import JSON, Image, Multipart, NumpyNdarray
    _BENTOML_AVAILABLE = True
except ImportError:
    _BENTOML_AVAILABLE = False
    bentoml = None
    JSON = Image = Multipart = NumpyNdarray = None

from traffic_config.settings import get_settings
from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry


logger = get_logger(__name__)


def _require_bentoml():
    """Raise error if bentoml not available."""
    if not _BENTOML_AVAILABLE:
        raise RuntimeError(
            "bentoml not installed. Install with: pip install 'traffic-vision[bentoml]'"
        )


class DetectionResult(BaseModel):
    """Single detection result."""

    bbox: list[float] = Field(..., description="[x1, y1, x2, y2]")
    class_id: int
    class_name: str
    confidence: float


class DetectionResponse(BaseModel):
    """Detection response."""

    detections: list[DetectionResult]
    inference_time_ms: float
    frame_id: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str
    device: str


# BentoML service definition - only create if bentoml is available
if _BENTOML_AVAILABLE:
    @bentoml.service(
        name="traffic-yolo-detector",
        traffic={"timeout": 30},
        resources={"gpu": 1},
    )
    class YOLODetectorService:
        """BentoML service for YOLOv8 traffic detection."""

        def __init__(self):
            _require_bentoml()
            self._settings = get_settings().vision
            self._model: YOLO | None = None
            self._device = self._settings.device
            self._metrics = get_metrics_registry()
            self._class_names = {
                0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                4: "bus", 5: "truck", 7: "truck",
            }

        @bentoml.on_startup
        def load_model(self) -> None:
            """Load model on service startup."""
            model_path = self._settings.model_path or self._settings.model_name
            self._model = YOLO(model_path)

            if self._device != "auto":
                self._model.to(self._device)

            if self._settings.half_precision:
                self._model.model.half()

            # Warmup
            dummy = np.zeros((self._settings.img_size, self._settings.img_size, 3), dtype=np.uint8)
            _ = self._model(dummy, verbose=False)

            logger.info("YOLO model loaded in BentoML", model=model_path, device=self._device)

        @bentoml.api(
            route="/detect",
            input=Image(),
            output=JSON(pydantic_model=DetectionResponse),
        )
        def detect(self, image: np.ndarray) -> DetectionResponse:
            """Detect objects in uploaded image.

            Args:
                image: Input image as numpy array (RGB format from BentoML).

            Returns:
                DetectionResponse with list of detections.
            """
            start_time = time.perf_counter()

            # Convert RGB to BGR for OpenCV/YOLO
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    if cls_id not in self._settings.classes and cls_id != self._settings.pedestrian_class:
                        continue

                    if conf < self._settings.confidence_threshold:
                        continue

                    class_name = self._class_names.get(cls_id, f"class_{cls_id}")

                    detections.append(DetectionResult(
                        bbox=[float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                    ))

                    self._metrics.inc_counter(
                        "bentoml_detections_total",
                        labels={"class_name": class_name},
                    )

            inference_time = (time.perf_counter() - start_time) * 1000

            self._metrics.record_histogram(
                "bentoml_inference_latency_ms",
                inference_time,
            )
            self._metrics.inc_counter("bentoml_requests_total", labels={"status": "success"})

            return DetectionResponse(
                detections=detections,
                inference_time_ms=inference_time,
            )

        @bentoml.api(
            route="/detect_batch",
            input=Multipart(img=Image(), frame_id=JSON()),
            output=JSON(pydantic_model=DetectionResponse),
        )
        def detect_batch(self, img: np.ndarray, frame_id: str | None = None) -> DetectionResponse:
            """Detect objects with frame ID tracking."""
            result = self.detect(img)
            result.frame_id = frame_id
            return result

        @bentoml.api(
            route="/detect_base64",
            input=JSON(pydantic_model=BaseModel),
            output=JSON(pydantic_model=DetectionResponse),
        )
        def detect_base64(self, request: dict[str, Any]) -> DetectionResponse:
            """Detect objects from base64 encoded image.

            Expected request format:
            {
                "image": "base64_encoded_string",
                "frame_id": "optional_frame_id"
            }
            """
            import base64
            from PIL import Image as PILImage

            image_data = base64.b64decode(request["image"])
            image = PILImage.open(io.BytesIO(image_data))
            frame = np.array(image)

            # Convert RGB to BGR
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            result = self.detect(frame)
            result.frame_id = request.get("frame_id")
            return result

        @bentoml.api(route="/health", output=JSON(pydantic_model=HealthResponse))
        def health(self) -> HealthResponse:
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model=self._settings.model_name,
                device=str(self._model.device) if self._model else "unknown",
            )

        @bentoml.api(route="/metrics", output=JSON())
        def metrics(self) -> dict[str, Any]:
            """Prometheus metrics endpoint."""
            # BentoML automatically exposes /metrics for Prometheus
            return {"message": "Metrics available at /metrics (Prometheus format)"}

else:
    # Placeholder class when bentoml not installed
    class YOLODetectorService:
        """Placeholder - install bentoml to use this service."""

        def __init__(self):
            _require_bentoml()


# For backwards compatibility and direct usage
def create_bento_service() -> YOLODetectorService:
    """Create BentoML service instance."""
    return YOLODetectorService()


# Import cv2 at module level
import cv2