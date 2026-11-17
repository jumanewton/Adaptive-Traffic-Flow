"""Model deployment utilities: ONNX export, TensorRT optimization, serving."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from traffic_config.settings import VisionSettings, get_settings
from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry


logger = get_logger(__name__)


class ONNXExporter:
    """Export YOLO model to ONNX format with optional TensorRT optimization."""

    def __init__(self, settings: VisionSettings | None = None):
        self._settings = settings or get_settings().vision
        self._metrics = get_metrics_registry()

    def export_to_onnx(
        self,
        model_path: str | None = None,
        output_path: str | None = None,
        img_size: int | None = None,
        opset_version: int = 17,
        simplify: bool = True,
        dynamic_batch: bool = True,
        half_precision: bool = False,
    ) -> Path:
        """Export YOLO model to ONNX format.

        Args:
            model_path: Path to PyTorch model (.pt). Defaults to settings.model_path or model_name.
            output_path: Output ONNX path. Defaults to same name with .onnx extension.
            img_size: Input image size. Defaults to settings.img_size.
            opset_version: ONNX opset version.
            simplify: Whether to run onnxsim simplification.
            dynamic_batch: Whether to enable dynamic batch dimension.
            half_precision: Whether to export in FP16.

        Returns:
            Path to exported ONNX model.
        """
        model_path = model_path or self._settings.model_path or self._settings.model_name
        img_size = img_size or self._settings.img_size
        output_path = output_path or str(Path(model_path).with_suffix(".onnx"))

        logger.info("Exporting YOLO to ONNX", model=model_path, output=output_path, img_size=img_size)

        # Load model
        model = YOLO(model_path)
        if self._settings.device != "auto":
            model.to(self._settings.device)

        if half_precision:
            model.model.half()

        # Export to ONNX
        start_time = time.time()
        exported_path = model.export(
            format="onnx",
            imgsz=img_size,
            opset=opset_version,
            simplify=simplify,
            dynamic=dynamic_batch,
            half=half_precision,
            device=self._settings.device if self._settings.device != "auto" else "cpu",
        )
        export_time = time.time() - start_time

        exported_path = Path(exported_path)
        if not exported_path.exists():
            raise RuntimeError(f"ONNX export failed: {exported_path} not found")

        # Move to desired output path if different
        if exported_path != Path(output_path):
            exported_path.rename(output_path)
            exported_path = Path(output_path)

        # Verify export
        self._verify_onnx(output_path, img_size)

        logger.info("ONNX export completed", output=output_path, duration_s=export_time)
        self._metrics.inc_counter("onnx_exports_total", labels={"status": "success"})

        return exported_path

    def _verify_onnx(self, onnx_path: str | Path, img_size: int) -> None:
        """Verify ONNX model runs correctly."""
        try:
            import onnx
            import onnxruntime as ort

            # Load and check ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            # Test inference with ONNX Runtime
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name

            # Create dummy input
            dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
            outputs = session.run(None, {input_name: dummy_input})

            logger.info("ONNX verification passed", output_shapes=[o.shape for o in outputs])

        except ImportError:
            logger.warning("ONNX verification skipped: onnx/onnxruntime not installed")
        except Exception as e:
            logger.error("ONNX verification failed", error=str(e))
            raise

    def optimize_with_tensorrt(
        self,
        onnx_path: str | Path,
        output_path: str | Path | None = None,
        precision: str = "fp16",
        max_batch_size: int = 1,
        workspace_size: int = 1 << 30,  # 1 GB
    ) -> Path:
        """Convert ONNX to TensorRT engine using trtexec.

        Args:
            onnx_path: Path to ONNX model.
            output_path: Output engine path. Defaults to same name with .engine extension.
            precision: Precision mode - fp32, fp16, int8.
            max_batch_size: Maximum batch size for dynamic shapes.
            workspace_size: Maximum workspace size in bytes.

        Returns:
            Path to TensorRT engine.
        """
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        output_path = output_path or onnx_path.with_suffix(".engine")

        logger.info("Building TensorRT engine", onnx=onnx_path, output=output_path, precision=precision)

        # Build trtexec command
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={output_path}",
            f"--workspace={workspace_size}",
            f"--maxBatch={max_batch_size}",
        ]

        if precision == "fp16":
            cmd.append("--fp16")
        elif precision == "int8":
            cmd.extend(["--int8", "--calib=calibration.cache"])
        # fp32 is default

        # Dynamic shapes for batch dimension
        cmd.append("--optShapes=input:1x3x640x640")  # Adjust based on img_size

        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            build_time = time.time() - start_time

            if result.returncode != 0:
                logger.error("TensorRT build failed", stderr=result.stderr)
                raise RuntimeError(f"TensorRT build failed: {result.stderr}")

            engine_path = Path(output_path)
            if not engine_path.exists():
                raise RuntimeError(f"TensorRT engine not created: {engine_path}")

            logger.info("TensorRT engine built", engine=engine_path, duration_s=build_time)
            self._metrics.inc_counter("tensorrt_builds_total", labels={"precision": precision, "status": "success"})

            return engine_path

        except FileNotFoundError:
            logger.error("trtexec not found. Install TensorRT or use ONNX Runtime with TensorRT EP.")
            raise
        except subprocess.TimeoutExpired:
            logger.error("TensorRT build timed out")
            raise


class TensorRTInference:
    """TensorRT inference engine for YOLO models."""

    def __init__(self, engine_path: str | Path):
        self._engine_path = Path(engine_path)
        self._context = None
        self._engine = None
        self._inputs = []
        self._outputs = []
        self._bindings = []
        self._stream = None

        self._load_engine()

    def _load_engine(self) -> None:
        """Load TensorRT engine and create execution context."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)

            with open(self._engine_path, "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())

            self._context = self._engine.create_execution_context()

            # Setup bindings
            self._stream = cuda.Stream()
            for i in range(self._engine.num_io_tensors):
                tensor_name = self._engine.get_tensor_name(i)
                dtype = trt.nptype(self._engine.get_tensor_dtype(tensor_name))
                shape = self._context.get_tensor_shape(tensor_name)

                if self._engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self._inputs.append({"name": tensor_name, "dtype": dtype, "shape": shape})
                else:
                    self._outputs.append({"name": tensor_name, "dtype": dtype, "shape": shape})

                # Allocate device memory
                size = trt.volume(shape) * np.dtype(dtype).itemsize
                d_mem = cuda.mem_alloc(size)
                self._bindings.append(int(d_mem))
                self._context.set_tensor_address(tensor_name, int(d_mem))

            logger.info("TensorRT engine loaded", engine=self._engine_path)

        except ImportError:
            logger.error("TensorRT or pycuda not installed. Cannot run TensorRT inference.")
            raise

    def infer(self, input_data: np.ndarray) -> list[np.ndarray]:
        """Run inference on input data.

        Args:
            input_data: Preprocessed input tensor (NCHW format).

        Returns:
            List of output tensors.
        """
        import pycuda.driver as cuda

        # Copy input to device
        cuda.memcpy_htod_async(self._bindings[0], input_data, self._stream)

        # Execute
        self._context.execute_async_v3(stream_handle=self._stream.handle)

        # Copy outputs back
        outputs = []
        for i, out_info in enumerate(self._outputs):
            host_mem = cuda.pagelocked_empty(out_info["shape"], out_info["dtype"])
            cuda.memcpy_dtoh_async(host_mem, self._bindings[i + len(self._inputs)], self._stream)
            outputs.append(host_mem)

        self._stream.synchronize()
        return outputs


class ONNXRuntimeInference:
    """ONNX Runtime inference with optional TensorRT execution provider."""

    def __init__(
        self,
        onnx_path: str | Path,
        providers: list[str] | None = None,
        session_options: Any | None = None,
    ):
        import onnxruntime as ort

        self._onnx_path = Path(onnx_path)
        self._providers = providers or [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self._session_options = session_options or ort.SessionOptions()
        self._session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self._onnx_path),
            sess_options=self._session_options,
            providers=self._providers,
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        logger.info("ONNX Runtime session created", model=onnx_path, providers=self._session.get_providers())

    def infer(self, input_data: np.ndarray) -> list[np.ndarray]:
        """Run inference."""
        return self._session.run(self._output_names, {self._input_name: input_data})

    def infer_batch(self, input_batch: np.ndarray) -> list[np.ndarray]:
        """Run batch inference."""
        return self.infer(input_batch)


def preprocess_frame(frame: np.ndarray, img_size: int, half: bool = False) -> np.ndarray:
    """Preprocess frame for YOLO inference (letterbox resize, normalize, NCHW)."""
    # Letterbox resize
    h, w = frame.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # Normalize and convert to NCHW
    normalized = padded.astype(np.float32) / 255.0
    nchw = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
    batched = np.expand_dims(nchw, axis=0)  # Add batch dim

    if half:
        batched = batched.astype(np.float16)

    return batched


def benchmark_inference(
    inference_fn,
    input_data: np.ndarray,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> dict[str, float]:
    """Benchmark inference latency.

    Args:
        inference_fn: Callable that runs inference.
        input_data: Preprocessed input tensor.
        warmup_runs: Number of warmup iterations.
        benchmark_runs: Number of benchmark iterations.

    Returns:
        Dictionary with latency statistics (mean, p50, p95, p99, min, max, fps).
    """
    latencies = []

    # Warmup
    for _ in range(warmup_runs):
        _ = inference_fn(input_data)

    # Benchmark
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        _ = inference_fn(input_data)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "fps": 1000.0 / float(np.mean(latencies)),
    }


def export_and_benchmark(
    model_path: str | None = None,
    img_size: int = 640,
    onnx_output: str | None = None,
    tensorrt_precision: str | None = None,
    benchmark_runs: int = 100,
) -> dict[str, Any]:
    """Complete export and benchmark pipeline.

    Args:
        model_path: PyTorch model path.
        img_size: Input image size.
        onnx_output: ONNX output path.
        tensorrt_precision: If set, also build TensorRT engine (fp16/int8).
        benchmark_runs: Number of benchmark runs.

    Returns:
        Dictionary with export paths and benchmark results.
    """
    settings = get_settings().vision
    model_path = model_path or settings.model_path or settings.model_name
    onnx_output = onnx_output or str(Path(model_path).with_suffix(".onnx"))

    results = {"model_path": model_path, "onnx_path": onnx_output}

    # Export ONNX
    exporter = ONNXExporter()
    onnx_path = exporter.export_to_onnx(model_path=model_path, output_path=onnx_output, img_size=img_size)
    results["onnx_path"] = str(onnx_path)

    # Prepare test input
    test_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # Benchmark PyTorch
    logger.info("Benchmarking PyTorch model...")
    model = YOLO(model_path)
    if settings.device != "auto":
        model.to(settings.device)

    def pytorch_infer(x):
        return model(x, verbose=False)

    results["pytorch"] = benchmark_inference(pytorch_infer, test_input, benchmark_runs=benchmark_runs)

    # Benchmark ONNX Runtime
    logger.info("Benchmarking ONNX Runtime...")
    ort_infer = ONNXRuntimeInference(onnx_path)
    results["onnx_runtime"] = benchmark_inference(ort_infer.infer, test_input, benchmark_runs=benchmark_runs)

    # Benchmark TensorRT if requested
    if tensorrt_precision:
        logger.info(f"Building TensorRT engine ({tensorrt_precision})...")
        engine_path = exporter.optimize_with_tensorrt(onnx_path, precision=tensorrt_precision)
        results["tensorrt_engine"] = str(engine_path)

        logger.info("Benchmarking TensorRT...")
        trt_infer = TensorRTInference(engine_path)
        # Convert input to correct format for TensorRT
        trt_input = test_input.copy()
        results["tensorrt"] = benchmark_inference(trt_infer.infer, trt_input, benchmark_runs=benchmark_runs)

    logger.info("Benchmark complete", results={k: v for k, v in results.items() if k not in ["pytorch", "onnx_runtime", "tensorrt"]})
    return results