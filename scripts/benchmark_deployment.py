#!/usr/bin/env python
"""Benchmark script for deployment options: PyTorch, ONNX, TensorRT, Triton, BentoML."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import cv2
import numpy as np

from traffic_vision.deployment import (
    ONNXExporter,
    ONNXRuntimeInference,
    TensorRTInference,
    preprocess_frame,
    benchmark_inference,
    export_and_benchmark,
)
from traffic_vision.triton_client import TritonClient, create_triton_model_repo
from traffic_config.settings import get_settings


def benchmark_pytorch(model_path: str, img_size: int, runs: int) -> dict:
    """Benchmark PyTorch model directly."""
    from ultralytics import YOLO

    settings = get_settings().vision
    model = YOLO(model_path)
    if settings.device != "auto":
        model.to(settings.device)

    test_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    def infer(x):
        return model(x, verbose=False)

    return benchmark_inference(infer, test_input, benchmark_runs=runs)


def benchmark_onnx(onnx_path: str, img_size: int, runs: int) -> dict:
    """Benchmark ONNX Runtime."""
    ort_infer = ONNXRuntimeInference(onnx_path)
    test_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    return benchmark_inference(ort_infer.infer, test_input, benchmark_runs=runs)


def benchmark_tensorrt(engine_path: str, img_size: int, runs: int) -> dict:
    """Benchmark TensorRT engine."""
    trt_infer = TensorRTInference(engine_path)
    test_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    return benchmark_inference(trt_infer.infer, test_input, benchmark_runs=runs)


async def benchmark_triton(url: str, model_name: str, img_size: int, runs: int) -> dict:
    """Benchmark Triton Inference Server."""
    client = TritonClient(url=url, model_name=model_name)
    await client.connect()

    test_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    latencies = []

    # Warmup
    for _ in range(10):
        await client.infer(test_input)

    # Benchmark
    for _ in range(runs):
        start = time.perf_counter()
        await client.infer(test_input)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    client.close()

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


def benchmark_bentoml(url: str, img_size: int, runs: int) -> dict:
    """Benchmark BentoML service via HTTP."""
    import requests

    test_frame = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", test_frame)
    import base64
    b64_image = base64.b64encode(encoded).decode()

    latencies = []

    # Warmup
    for _ in range(5):
        requests.post(f"{url}/detect_base64", json={"image": b64_image})

    # Benchmark
    for _ in range(runs):
        start = time.perf_counter()
        resp = requests.post(f"{url}/detect_base64", json={"image": b64_image})
        resp.raise_for_status()
        elapsed = (time.perf_counter() - start) * 1000
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


def export_onnx(model_path: str, output_path: str, img_size: int) -> str:
    """Export model to ONNX."""
    exporter = ONNXExporter()
    return str(exporter.export_to_onnx(model_path=model_path, output_path=output_path, img_size=img_size))


def build_tensorrt(onnx_path: str, precision: str, output_path: str | None = None) -> str:
    """Build TensorRT engine."""
    exporter = ONNXExporter()
    return str(exporter.optimize_with_tensorrt(onnx_path, output_path=output_path, precision=precision))


def create_triton_repo(onnx_path: str, repo_path: str, model_name: str, img_size: int) -> None:
    """Create Triton model repository."""
    create_triton_model_repo(onnx_path, repo_path, model_name=model_name, img_size=img_size)


def run_full_benchmark(
    model_path: str,
    img_size: int = 640,
    onnx_output: str | None = None,
    tensorrt_precision: str | None = None,
    triton_url: str | None = None,
    triton_model: str = "yolo",
    bentoml_url: str | None = None,
    runs: int = 100,
    output_json: str | None = None,
) -> dict:
    """Run complete benchmark suite."""
    results = {
        "model_path": model_path,
        "img_size": img_size,
        "runs": runs,
    }

    onnx_path = onnx_output or str(Path(model_path).with_suffix(".onnx"))

    print(f"\n{'='*60}")
    print("EXPORTING TO ONNX")
    print(f"{'='*60}")
    onnx_path = export_onnx(model_path, onnx_path, img_size)
    results["onnx_path"] = onnx_path

    print(f"\n{'='*60}")
    print("BENCHMARKING PYTORCH")
    print(f"{'='*60}")
    results["pytorch"] = benchmark_pytorch(model_path, img_size, runs)
    print(f"  Mean: {results['pytorch']['mean_ms']:.2f} ms, FPS: {results['pytorch']['fps']:.1f}")

    print(f"\n{'='*60}")
    print("BENCHMARKING ONNX RUNTIME")
    print(f"{'='*60}")
    results["onnx_runtime"] = benchmark_onnx(onnx_path, img_size, runs)
    print(f"  Mean: {results['onnx_runtime']['mean_ms']:.2f} ms, FPS: {results['onnx_runtime']['fps']:.1f}")

    if tensorrt_precision:
        print(f"\n{'='*60}")
        print(f"BUILDING TENSORRT ENGINE ({tensorrt_precision.upper()})")
        print(f"{'='*60}")
        engine_path = build_tensorrt(onnx_path, tensorrt_precision)
        results["tensorrt_engine"] = engine_path

        print(f"\n{'='*60}")
        print("BENCHMARKING TENSORRT")
        print(f"{'='*60}")
        results["tensorrt"] = benchmark_tensorrt(engine_path, img_size, runs)
        print(f"  Mean: {results['tensorrt']['mean_ms']:.2f} ms, FPS: {results['tensorrt']['fps']:.1f}")

    if triton_url:
        print(f"\n{'='*60}")
        print("BENCHMARKING TRITON")
        print(f"{'='*60}")
        results["triton"] = asyncio.run(benchmark_triton(triton_url, triton_model, img_size, runs))
        print(f"  Mean: {results['triton']['mean_ms']:.2f} ms, FPS: {results['triton']['fps']:.1f}")

    if bentoml_url:
        print(f"\n{'='*60}")
        print("BENCHMARKING BENTOML")
        print(f"{'='*60}")
        results["bentoml"] = benchmark_bentoml(bentoml_url, img_size, runs)
        print(f"  Mean: {results['bentoml']['mean_ms']:.2f} ms, FPS: {results['bentoml']['fps']:.1f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Backend':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'FPS':<8}")
    print("-" * 60)
    for backend in ["pytorch", "onnx_runtime", "tensorrt", "triton", "bentoml"]:
        if backend in results:
            r = results[backend]
            print(f"{backend:<20} {r['mean_ms']:<12.2f} {r['p95_ms']:<12.2f} {r['p99_ms']:<12.2f} {r['fps']:<8.1f}")

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark deployment options for YOLO")
    parser.add_argument("model_path", help="Path to PyTorch model (.pt)")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--onnx-output", help="ONNX output path")
    parser.add_argument("--tensorrt", choices=["fp32", "fp16", "int8"], help="Build TensorRT engine")
    parser.add_argument("--triton-url", help="Triton server URL (e.g., localhost:8000)")
    parser.add_argument("--triton-model", default="yolo", help="Triton model name")
    parser.add_argument("--bentoml-url", help="BentoML service URL (e.g., http://localhost:3000)")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark runs")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--export-only", action="store_true", help="Only export ONNX, no benchmark")
    parser.add_argument("--triton-repo", help="Create Triton model repo at this path")

    args = parser.parse_args()

    if args.export_only:
        export_onnx(args.model_path, args.onnx_output or str(Path(args.model_path).with_suffix(".onnx")), args.img_size)
        if args.triton_repo:
            create_triton_repo(
                args.onnx_output or str(Path(args.model_path).with_suffix(".onnx")),
                args.triton_repo,
                "yolo",
                args.img_size,
            )
        return

    if args.triton_repo:
        onnx_path = args.onnx_output or str(Path(args.model_path).with_suffix(".onnx"))
        create_triton_repo(onnx_path, args.triton_repo, "yolo", args.img_size)
        return

    run_full_benchmark(
        model_path=args.model_path,
        img_size=args.img_size,
        onnx_output=args.onnx_output,
        tensorrt_precision=args.tensorrt,
        triton_url=args.triton_url,
        triton_model=args.triton_model,
        bentoml_url=args.bentoml_url,
        runs=args.runs,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()