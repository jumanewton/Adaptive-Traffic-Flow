"""Tests for deployment module."""

from __future__ import annotations

import numpy as np
import pytest

from traffic_vision.deployment import (
    preprocess_frame,
    benchmark_inference,
    ONNXRuntimeInference,
)
from traffic_observability.latency import LatencyTracker, PipelineLatencyMonitor, LatencySLO


class TestPreprocessing:
    """Test frame preprocessing."""

    def test_preprocess_frame_shape(self):
        """Test preprocessing produces correct shape."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = preprocess_frame(frame, img_size=640)

        assert processed.shape == (1, 3, 640, 640)
        assert processed.dtype == np.float32
        assert processed.min() >= 0
        assert processed.max() <= 1

    def test_preprocess_frame_half(self):
        """Test preprocessing with half precision."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = preprocess_frame(frame, img_size=640, half=True)

        assert processed.dtype == np.float16


class TestBenchmarking:
    """Test benchmarking utilities."""

    def test_benchmark_inference(self):
        """Test benchmark_inference returns expected keys."""
        def dummy_infer(x):
            return [x]

        input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
        results = benchmark_inference(dummy_infer, input_data, warmup_runs=2, benchmark_runs=5)

        assert "mean_ms" in results
        assert "median_ms" in results
        assert "p95_ms" in results
        assert "p99_ms" in results
        assert "min_ms" in results
        assert "max_ms" in results
        assert "fps" in results

        assert results["mean_ms"] > 0
        assert results["fps"] > 0


class TestLatencyTracker:
    """Test LatencyTracker."""

    def test_basic_tracking(self):
        """Test basic latency tracking."""
        tracker = LatencyTracker("test_basic")

        with tracker.measure("stage1"):
            pass

        stats = tracker.get_stats()
        assert "stage1" in stats
        assert stats["stage1"]["count"] == 1

    def test_slo_check(self):
        """Test SLO compliance checking."""
        tracker = LatencyTracker("test_slo_check", slos=[
            LatencySLO("fast_stage", target_ms=100, percentile=0.95),
        ])

        # Add fast latencies
        for _ in range(100):
            tracker._stage_timers["fast_stage"].append(10.0)

        results = tracker.check_slos()
        assert results["fast_stage"]["status"] == "healthy"
        assert results["fast_stage"]["compliance"] == 1.0

    def test_slo_violation(self):
        """Test SLO violation detection."""
        tracker = LatencyTracker("test_slo_violation", slos=[
            LatencySLO("slow_stage", target_ms=10, percentile=0.95),
        ])

        # Add slow latencies
        for _ in range(100):
            tracker._stage_timers["slow_stage"].append(50.0)

        results = tracker.check_slos()
        assert results["slow_stage"]["status"] == "violated"
        assert results["slow_stage"]["compliance"] < 1.0


class TestPipelineLatencyMonitor:
    """Test PipelineLatencyMonitor."""

    def test_pipeline_timing(self):
        """Test end-to-end pipeline timing."""
        monitor = PipelineLatencyMonitor("test_pipeline")

        monitor.start_pipeline("pipe-1")
        monitor.record_stage("pipe-1", "stage1", 10.0)
        monitor.record_stage("pipe-1", "stage2", 20.0)
        total = monitor.end_pipeline("pipe-1")

        assert total > 0

    def test_multiple_pipelines(self):
        """Test multiple concurrent pipelines."""
        monitor = PipelineLatencyMonitor("test_multi_pipeline")

        monitor.start_pipeline("pipe-1")
        monitor.start_pipeline("pipe-2")

        monitor.record_stage("pipe-1", "stage1", 10.0)
        monitor.record_stage("pipe-2", "stage1", 20.0)

        total1 = monitor.end_pipeline("pipe-1")
        total2 = monitor.end_pipeline("pipe-2")

        assert total1 > 0
        assert total2 > 0


class TestLatencySLO:
    """Test LatencySLO dataclass."""

    def test_default_values(self):
        """Test default SLO values."""
        slo = LatencySLO("test", target_ms=50)
        assert slo.target_ms == 50
        assert slo.percentile == 0.95
        assert slo.window_seconds == 300

    def test_custom_values(self):
        """Test custom SLO values."""
        slo = LatencySLO("test", target_ms=100, percentile=0.99, window_seconds=600)
        assert slo.target_ms == 100
        assert slo.percentile == 0.99
        assert slo.window_seconds == 600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])