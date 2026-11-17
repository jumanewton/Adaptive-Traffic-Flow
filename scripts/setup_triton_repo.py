#!/usr/bin/env python
"""Set up Triton model repository from exported ONNX model."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from traffic_vision.deployment import ONNXExporter
from traffic_vision.triton_client import create_triton_model_repo
from traffic_config.settings import get_settings


def main():
    parser = argparse.ArgumentParser(description="Set up Triton model repository")
    parser.add_argument("--model-path", help="Path to PyTorch model (.pt)")
    parser.add_argument("--onnx-path", help="Path to existing ONNX model (skip export)")
    parser.add_argument("--repo-path", default="infra/docker/triton-model-repo", help="Triton model repo path")
    parser.add_argument("--model-name", default="yolo", help="Model name")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--max-batch", type=int, default=8, help="Max batch size")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16", help="TensorRT precision")
    parser.add_argument("--export-only", action="store_true", help="Only export ONNX, don't create repo")

    args = parser.parse_args()

    settings = get_settings().vision
    model_path = args.model_path or settings.model_path or settings.model_name
    onnx_path = args.onnx_path or str(Path(model_path).with_suffix(".onnx"))
    repo_path = Path(args.repo_path)

    # Export ONNX if needed
    if not args.onnx_path:
        print(f"Exporting {model_path} to ONNX...")
        exporter = ONNXExporter()
        onnx_path = exporter.export_to_onnx(
            model_path=model_path,
            output_path=onnx_path,
            img_size=args.img_size,
            half_precision=(args.precision == "fp16"),
        )
        print(f"ONNX exported to: {onnx_path}")

    if args.export_only:
        return

    # Create Triton model repository
    print(f"Creating Triton model repository at {repo_path}...")
    create_triton_model_repo(
        model_path=onnx_path,
        repo_path=repo_path,
        model_name=args.model_name,
        max_batch_size=args.max_batch,
        img_size=args.img_size,
    )

    # Create TensorRT optimized model directory if precision is fp16
    if args.precision == "fp16":
        trt_dir = repo_path / args.model_name / "1-trt"
        trt_dir.mkdir(parents=True, exist_ok=True)

        # Create TensorRT model config
        config = f'''
name: "{args.model_name}_trt"
platform: "tensorrt_plan"
max_batch_size: {args.max_batch}
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [3, {args.img_size}, {args.img_size}]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [84, 8400]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }}
]
version_policy: {{
  latest: {{
    num_versions: 1
  }}
}}
'''
        (trt_dir / "config.pbtxt").write_text(config.strip())

        # Build TensorRT engine
        print("Building TensorRT engine...")
        exporter = ONNXExporter()
        engine_path = exporter.optimize_with_tensorrt(
            onnx_path,
            output_path=trt_dir / "model.plan",
            precision="fp16",
            max_batch_size=args.max_batch,
        )
        print(f"TensorRT engine built at: {engine_path}")

    print(f"\nTriton model repository ready at: {repo_path}")
    print(f"Start Triton with: docker compose --profile inference up triton")


if __name__ == "__main__":
    main()