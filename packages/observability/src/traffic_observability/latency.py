"""Latency monitoring and SLO tracking for traffic pipeline."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, Summary

from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry


logger = get_logger(__name__)


@dataclass
class LatencySLO:
    """Service Level Objective for latency."""

    name: str
    target_ms: float
    percentile: float = 0.95
    window_seconds: int = 300  # 5 minute rolling window


# Default SLOs for traffic pipeline
DEFAULT_SLOs = [
    LatencySLO("vision_inference", target_ms=50, percentile=0.95),
    LatencySLO("vision_tracking", target_ms=10, percentile=0.99),
    LatencySLO("state_estimation", target_ms=20, percentile=0.95),
    LatencySLO("control_computation", target_ms=30, percentile=0.95),
    LatencySLO("kafka_publish", target_ms=5, percentile=0.99),
    LatencySLO("end_to_end", target_ms=200, percentile=0.95),
]


class LatencyTracker:
    """Track and monitor latency across pipeline stages."""

    def __init__(self, service_name: str, slos: list[LatencySLO] | None = None):
        self._service_name = service_name
        self._slos = slos or DEFAULT_SLOs
        self._metrics = get_metrics_registry()
        self._stage_timers: dict[str, list[float]] = defaultdict(list)
        self._stage_start: dict[str, float] = {}

        # Prometheus metrics
        self._latency_histogram = Histogram(
            f"{service_name}_stage_latency_ms",
            "Latency per pipeline stage",
            ["stage"],
            buckets=[1, 2, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000],
        )
        self._slo_compliance = Gauge(
            f"{service_name}_slo_compliance",
            "SLO compliance ratio (0-1)",
            ["slo_name"],
        )
        self._slo_violations = Counter(
            f"{service_name}_slo_violations_total",
            "Total SLO violations",
            ["slo_name"],
        )
        self._pipeline_latency = Summary(
            f"{service_name}_pipeline_latency_ms",
            "End-to-end pipeline latency",
            ["pipeline"],
        )

    @contextmanager
    def measure(self, stage: str):
        """Context manager to measure stage latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self._stage_timers[stage].append(elapsed)
            self._latency_histogram.labels(stage=stage).observe(elapsed)

            # Keep only recent measurements (last 1000)
            if len(self._stage_timers[stage]) > 1000:
                self._stage_timers[stage] = self._stage_timers[stage][-1000:]

    def start_stage(self, stage: str) -> None:
        """Start timing a stage."""
        self._stage_start[stage] = time.perf_counter()

    def end_stage(self, stage: str) -> float:
        """End timing a stage and return latency in ms."""
        if stage not in self._stage_start:
            logger.warning("Stage not started", stage=stage)
            return 0.0

        elapsed = (time.perf_counter() - self._stage_start[stage]) * 1000
        self._stage_timers[stage].append(elapsed)
        self._latency_histogram.labels(stage=stage).observe(elapsed)
        del self._stage_start[stage]
        return elapsed

    def record_pipeline(self, pipeline: str, latency_ms: float) -> None:
        """Record end-to-end pipeline latency."""
        self._pipeline_latency.labels(pipeline=pipeline).observe(latency_ms)

    def check_slos(self) -> dict[str, dict[str, Any]]:
        """Check SLO compliance for all stages."""
        results = {}

        for slo in self._slos:
            latencies = self._stage_timers.get(slo.name, [])
            if not latencies:
                results[slo.name] = {"status": "no_data", "compliance": 1.0}
                self._slo_compliance.labels(slo_name=slo.name).set(1.0)
                continue

            # Calculate percentile
            sorted_latencies = sorted(latencies)
            idx = int(len(sorted_latencies) * slo.percentile)
            percentile_latency = sorted_latencies[min(idx, len(sorted_latencies) - 1)]

            # Compliance ratio
            compliant = sum(1 for l in latencies if l <= slo.target_ms)
            compliance = compliant / len(latencies)

            # Check violation
            violated = percentile_latency > slo.target_ms
            if violated:
                self._slo_violations.labels(slo_name=slo.name).inc()

            self._slo_compliance.labels(slo_name=slo.name).set(compliance)

            results[slo.name] = {
                "status": "violated" if violated else "healthy",
                "target_ms": slo.target_ms,
                "percentile": slo.percentile,
                "measured_percentile_ms": percentile_latency,
                "compliance": compliance,
                "sample_count": len(latencies),
                "mean_ms": sum(latencies) / len(latencies),
            }

        return results

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get latency statistics for all stages."""
        stats = {}
        for stage, latencies in self._stage_timers.items():
            if not latencies:
                continue
            sorted_l = sorted(latencies)
            stats[stage] = {
                "count": len(latencies),
                "mean_ms": sum(latencies) / len(latencies),
                "median_ms": sorted_l[len(sorted_l) // 2],
                "p95_ms": sorted_l[int(len(sorted_l) * 0.95)],
                "p99_ms": sorted_l[int(len(sorted_l) * 0.99)],
                "min_ms": min(latencies),
                "max_ms": max(latencies),
            }
        return stats

    def reset(self) -> None:
        """Reset all tracked latencies."""
        self._stage_timers.clear()
        self._stage_start.clear()


class PipelineLatencyMonitor:
    """Monitor end-to-end pipeline latency across services."""

    def __init__(self, service_name: str):
        self._service_name = service_name
        self._metrics = get_metrics_registry()
        self._pipeline_timers: dict[str, float] = {}

        self._e2e_histogram = Histogram(
            f"{service_name}_e2e_latency_ms",
            "End-to-end latency from camera to control action",
            ["pipeline"],
            buckets=[50, 100, 200, 300, 500, 1000, 2000, 5000],
        )
        self._stage_breakdown = Histogram(
            f"{service_name}_stage_breakdown_ms",
            "Latency breakdown by stage",
            ["pipeline", "stage"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

    def start_pipeline(self, pipeline_id: str) -> None:
        """Start tracking a pipeline execution."""
        self._pipeline_timers[pipeline_id] = time.perf_counter()

    def record_stage(self, pipeline_id: str, stage: str, latency_ms: float) -> None:
        """Record a stage latency within a pipeline."""
        self._stage_breakdown.labels(pipeline=pipeline_id, stage=stage).observe(latency_ms)

    def end_pipeline(self, pipeline_id: str) -> float:
        """End pipeline tracking and return total latency."""
        if pipeline_id not in self._pipeline_timers:
            return 0.0

        total_ms = (time.perf_counter() - self._pipeline_timers[pipeline_id]) * 1000
        self._e2e_histogram.labels(pipeline=pipeline_id).observe(total_ms)
        del self._pipeline_timers[pipeline_id]
        return total_ms


def create_latency_middleware(service_name: str):
    """Create ASGI middleware for HTTP latency tracking."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    metrics = get_metrics_registry()

    http_latency = Histogram(
        f"{service_name}_http_latency_ms",
        "HTTP request latency",
        ["method", "endpoint", "status"],
        buckets=[5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    )

    class LatencyMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start = time.perf_counter()
            response: Response = await call_next(request)
            elapsed = (time.perf_counter() - start) * 1000

            http_latency.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
            ).observe(elapsed)

            response.headers["X-Process-Time-MS"] = str(round(elapsed, 2))
            return response

    return LatencyMiddleware


# Decorator for easy latency tracking
def track_latency(stage: str, tracker: LatencyTracker | None = None):
    """Decorator to track function latency."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            nonlocal tracker
            if tracker is None:
                # Try to get from global or create
                from traffic_observability.metrics import get_metrics_registry
                tracker = LatencyTracker(get_metrics_registry()._namespace or "traffic")

            with tracker.measure(stage):
                return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            nonlocal tracker
            if tracker is None:
                from traffic_observability.metrics import get_metrics_registry
                tracker = LatencyTracker(get_metrics_registry()._namespace or "traffic")

            with tracker.measure(stage):
                return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator