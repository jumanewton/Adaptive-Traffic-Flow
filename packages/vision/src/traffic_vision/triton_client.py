"""Triton Inference Server client for YOLO models."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    httpclient = None
    grpcclient = None
    np_to_triton_dtype = None

from traffic_config.settings import get_settings
from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry


logger = get_logger(__name__)


def _require_triton():
    """Raise error if tritonclient not available."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "tritonclient not installed. Install with: pip install 'traffic-vision[triton]'"
        )


class TritonClient:
    """Async Triton Inference Server client for YOLO detection."""

    def __init__(
        self,
        url: str = "localhost:8000",
        grpc_url: str = "localhost:8001",
        model_name: str = "yolo",
        model_version: str = "",
        use_grpc: bool = False,
    ):
        _require_triton()
        self._url = url
        self._grpc_url = grpc_url
        self._model_name = model_name
        self._model_version = model_version
        self._use_grpc = use_grpc
        self._client: httpclient.InferenceServerClient | grpcclient.InferenceServerClient | None = None
        self._metrics = get_metrics_registry()

    async def connect(self) -> None:
        """Initialize connection to Triton server."""
        _require_triton()
        if self._use_grpc:
            self._client = grpcclient.InferenceServerClient(url=self._grpc_url)
        else:
            self._client = httpclient.InferenceServerClient(url=self._url)

        # Wait for server ready
        for _ in range(30):
            try:
                if self._client.is_server_ready():
                    break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError(f"Triton server not ready at {self._url}")

        # Check model ready
        for _ in range(30):
            try:
                if self._client.is_model_ready(self._model_name, self._model_version):
                    break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError(f"Model {self._model_name} not ready")

        logger.info("Connected to Triton", url=self._url, model=self._model_name)

    def close(self) -> None:
        """Close connection."""
        if self._client:
            self._client.close()
            self._client = None

    async def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run inference on input tensor.

        Args:
            input_tensor: Preprocessed input (NCHW format, float32).

        Returns:
            List of output tensors.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        start_time = time.perf_counter()

        # Create input
        inputs = [
            httpclient.InferInput("images", input_tensor.shape, np_to_triton_dtype(input_tensor.dtype))
            if not self._use_grpc
            else grpcclient.InferInput("images", input_tensor.shape, np_to_triton_dtype(input_tensor.dtype))
        ]
        inputs[0].set_data_from_numpy(input_tensor)

        # Create output requests
        outputs = [
            httpclient.InferRequestedOutput("output0")
            if not self._use_grpc
            else grpcclient.InferRequestedOutput("output0")
        ]

        # Run inference
        loop = asyncio.get_event_loop()
        if self._use_grpc:
            result = await loop.run_in_executor(
                None, self._client.infer, self._model_name, inputs, outputs, self._model_version
            )
        else:
            result = await loop.run_in_executor(
                None, self._client.infer, self._model_name, inputs, outputs, self._model_version
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Get outputs
        output_data = []
        for output in outputs:
            output_data.append(result.as_numpy(output.name()))

        # Record metrics
        self._metrics.record_histogram(
            "triton_inference_latency_ms",
            latency_ms,
            labels={"model": self._model_name},
        )
        self._metrics.inc_counter(
            "triton_inferences_total",
            labels={"model": self._model_name, "status": "success"},
        )

        return output_data

    async def infer_batch(self, input_batch: np.ndarray) -> list[np.ndarray]:
        """Run batch inference."""
        return await self.infer(input_batch)

    async def get_model_metadata(self) -> dict[str, Any]:
        """Get model metadata from Triton."""
        if not self._client:
            raise RuntimeError("Not connected")

        loop = asyncio.get_event_loop()
        if self._use_grpc:
            metadata = await loop.run_in_executor(
                None, self._client.get_model_metadata, self._model_name, self._model_version
            )
        else:
            metadata = await loop.run_in_executor(
                None, self._client.get_model_metadata, self._model_name, self._model_version
            )
        return metadata

    async def get_model_config(self) -> dict[str, Any]:
        """Get model configuration from Triton."""
        if not self._client:
            raise RuntimeError("Not connected")

        loop = asyncio.get_event_loop()
        if self._use_grpc:
            config = await loop.run_in_executor(
                None, self._client.get_model_config, self._model_name, self._model_version
            )
        else:
            config = await loop.run_in_executor(
                None, self._client.get_model_config, self._model_name, self._model_version
            )
        return config


class TritonYOLODetector:
    """YOLO detector using Triton Inference Server."""

    def __init__(
        self,
        triton_url: str = "localhost:8000",
        model_name: str = "yolo",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        class_names: dict[int, str] | None = None,
    ):
        self._triton_url = triton_url
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._img_size = img_size
        self._class_names = class_names or {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            4: "bus", 5: "truck", 7: "truck",
        }
        self._client: TritonClient | None = None

    async def initialize(self) -> None:
        """Initialize Triton connection."""
        self._client = TritonClient(url=self._triton_url, model_name=self._model_name)
        await self._client.connect()

    def close(self) -> None:
        """Close connection."""
        if self._client:
            self._client.close()

    async def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run detection on frame via Triton.

        Args:
            frame: BGR image frame.

        Returns:
            List of detections with bbox, class, confidence.
        """
        if not self._client:
            raise RuntimeError("Not initialized. Call initialize() first.")

        # Preprocess
        input_tensor = self._preprocess(frame)

        # Inference
        outputs = await self._client.infer(input_tensor)

        # Postprocess
        detections = self._postprocess(outputs[0], frame.shape[:2])

        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for Triton (letterbox, normalize, NCHW)."""
        h, w = frame.shape[:2]
        scale = min(self._img_size / h, self._img_size / w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((self._img_size, self._img_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        normalized = padded.astype(np.float32) / 255.0
        nchw = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(nchw, axis=0)

        return batched

    def _postprocess(self, output: np.ndarray, orig_shape: tuple[int, int]) -> list[dict[str, Any]]:
        """Postprocess YOLO output to detections."""
        h, w = orig_shape
        detections = []

        # Output shape: [1, num_boxes, 85] or [1, 84, 8400] depending on model
        # Handle ultralytics YOLOv8 output format
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]  # Remove batch dim

        # YOLOv8: [84, 8400] -> transpose to [8400, 84]
        if output.shape[0] < output.shape[1]:
            output = output.T

        for detection in output:
            # Class scores start at index 4 (x, y, w, h, obj_conf, class_scores...)
            if output.shape[1] == 85:  # YOLOv5 format
                obj_conf = detection[4]
                class_scores = detection[5:]
            else:  # YOLOv8 format: [x, y, w, h, class_scores...]
                obj_conf = 1.0
                class_scores = detection[4:]

            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            confidence = obj_conf * class_conf

            if confidence < self._confidence_threshold:
                continue

            if class_id not in self._class_names:
                continue

            # Convert from center-x, center-y, width, height to x1, y1, x2, y2
            cx, cy, bw, bh = detection[0], detection[1], detection[2], detection[3]

            # Scale back to original image
            x1 = (cx - bw / 2) * w / self._img_size
            y1 = (cy - bh / 2) * h / self._img_size
            x2 = (cx + bw / 2) * w / self._img_size
            y2 = (cy + bh / 2) * h / self._img_size

            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class_id": class_id,
                "class_name": self._class_names.get(class_id, f"class_{class_id}"),
                "confidence": confidence,
            })

        # NMS
        detections = self._nms(detections)

        return detections

    def _nms(self, detections: list[dict], iou_threshold: float | None = None) -> list[dict]:
        """Non-maximum suppression."""
        if not detections:
            return []

        iou_threshold = iou_threshold or self._iou_threshold

        # Sort by confidence
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping
            remaining = []
            for det in detections:
                if self._iou(best["bbox"], det["bbox"]) < iou_threshold:
                    remaining.append(det)
            detections = remaining

        return keep

    def _iou(self, box1: list[float], box2: list[float]) -> float:
        """Calculate IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


def create_triton_model_repo(
    model_path: str | Path,
    repo_path: str | Path,
    model_name: str = "yolo",
    version_policy: str = "latest",
    max_batch_size: int = 8,
    input_name: str = "images",
    output_name: str = "output0",
    img_size: int = 640,
) -> None:
    """Create Triton model repository structure for YOLO.

    Args:
        model_path: Path to ONNX model.
        repo_path: Path to model repository root.
        model_name: Model name.
        version_policy: Version policy (latest, specific).
        max_batch_size: Maximum batch size.
        input_name: Input tensor name.
        output_name: Output tensor name.
        img_size: Input image size.
    """
    repo_path = Path(repo_path)
    model_dir = repo_path / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy ONNX model
    import shutil
    shutil.copy2(model_path, version_dir / "model.onnx")

    # Create config.pbtxt
    config = f'''
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}
input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [3, {img_size}, {img_size}]
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [84, 8400]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
dynamic_batching {{
  max_queue_delay_microseconds: 1000
  preferred_batch_size: [1, 2, 4, 8]
}}
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator: [{{ name: "tensorrt" parameters: {{ precision_mode: "FP16" }} }}]
  }}
}}
version_policy: {{
  latest: {{
    num_versions: 1
  }}
}}
'''

    (model_dir / "config.pbtxt").write_text(config.strip())
    logger.info("Triton model repository created", path=repo_path, model=model_name)


import cv2