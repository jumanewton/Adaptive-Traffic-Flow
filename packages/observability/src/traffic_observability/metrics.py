"""Prometheus metrics registry and common metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info


@dataclass
class MetricsRegistry:
    """Thread-safe wrapper around Prometheus CollectorRegistry with common metrics."""

    registry: CollectorRegistry = field(default_factory=CollectorRegistry)
    _lock: Lock = field(default_factory=Lock, init=False)
    _counters: dict[str, Counter] = field(default_factory=dict, init=False)
    _gauges: dict[str, Gauge] = field(default_factory=dict, init=False)
    _histograms: dict[str, Histogram] = field(default_factory=dict, init=False)
    _infos: dict[str, Info] = field(default_factory=dict, init=False)

    # --- Counter ---

    def counter(
        self,
        name: str,
        documentation: str,
        labels: list[str] | None = None,
    ) -> Counter:
        """Get or create a counter metric."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(
                    name, documentation, labels or [], registry=self.registry
                )
            return self._counters[name]

    def inc_counter(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        """Increment a counter."""
        self.counter(name, "").labels(**labels or {}).inc(value)

    # --- Gauge ---

    def gauge(
        self,
        name: str,
        documentation: str,
        labels: list[str] | None = None,
    ) -> Gauge:
        """Get or create a gauge metric."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(
                    name, documentation, labels or [], registry=self.registry
                )
            return self._gauges[name]

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge value."""
        self.gauge(name, "").labels(**labels or {}).set(value)

    # --- Histogram ---

    def histogram(
        self,
        name: str,
        documentation: str,
        labels: list[str] | None = None,
        buckets: list[float] | None = None,
    ) -> Histogram:
        """Get or create a histogram metric."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name,
                    documentation,
                    labels or [],
                    buckets=buckets or Histogram.DEFAULT_BUCKETS,
                    registry=self.registry,
                )
            return self._histograms[name]

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Observe a value in a histogram."""
        self.histogram(name, "").labels(**labels or {}).observe(value)

    # --- Info ---

    def info(self, name: str, documentation: str, labels: list[str] | None = None) -> Info:
        """Get or create an info metric."""
        with self._lock:
            if name not in self._infos:
                self._infos[name] = Info(name, documentation, labels or [], registry=self.registry)
            return self._infos[name]

    def set_info(self, name: str, value: dict[str, str], labels: dict[str, str] | None = None) -> None:
        """Set info metric value."""
        self.info(name, "").labels(**labels or {}).info(value)


# Global registry instance
_global_registry: MetricsRegistry | None = None


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry (singleton)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MetricsRegistry()
    return _global_registry


# --- Common traffic metrics ---

def create_traffic_metrics(registry: MetricsRegistry, service: str) -> dict[str, Any]:
    """Create standard traffic-related metrics for a service."""
    prefix = f"{service}_traffic"

    return {
        # Perception metrics
        "detections_total": registry.counter(
            f"{prefix}_detections_total",
            "Total number of detections",
            ["class_name", "camera_id"],
        ),
        "tracks_active": registry.gauge(
            f"{prefix}_tracks_active",
            "Number of active tracks",
            ["camera_id"],
        ),
        "track_age_frames": registry.histogram(
            f"{prefix}_track_age_frames",
            "Track age in frames",
            ["camera_id"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
        ),
        "inference_latency_ms": registry.histogram(
            f"{prefix}_inference_latency_ms",
            "Model inference latency in milliseconds",
            ["model", "camera_id"],
            buckets=[5, 10, 20, 30, 50, 80, 120, 200, 500],
        ),
        # Traffic state metrics
        "vehicle_count": registry.gauge(
            f"{prefix}_vehicle_count",
            "Current vehicle count per approach",
            ["intersection_id", "approach_id"],
        ),
        "queue_length_m": registry.gauge(
            f"{prefix}_queue_length_m",
            "Queue length in meters per approach",
            ["intersection_id", "approach_id"],
        ),
        "avg_speed_kmh": registry.gauge(
            f"{prefix}_avg_speed_kmh",
            "Average speed in km/h per approach",
            ["intersection_id", "approach_id"],
        ),
        "pedestrian_count": registry.gauge(
            f"{prefix}_pedestrian_count",
            "Pedestrian count per approach",
            ["intersection_id", "approach_id"],
        ),
        # Control metrics
        "phase_duration_seconds": registry.histogram(
            f"{prefix}_phase_duration_seconds",
            "Signal phase duration in seconds",
            ["intersection_id", "phase"],
            buckets=[5, 10, 15, 20, 30, 45, 60, 90, 120],
        ),
        "controller_action_total": registry.counter(
            f"{prefix}_controller_action_total",
            "Total controller actions by type",
            ["intersection_id", "action"],
        ),
        "control_loop_latency_ms": registry.histogram(
            f"{prefix}_control_loop_latency_ms",
            "Control loop latency in milliseconds",
            ["intersection_id", "controller_type"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        ),
        # Streaming metrics
        "events_published_total": registry.counter(
            f"{prefix}_events_published_total",
            "Total events published to stream",
            ["intersection_id", "event_type"],
        ),
        "events_consumed_total": registry.counter(
            f"{prefix}_events_consumed_total",
            "Total events consumed from stream",
            ["intersection_id", "event_type", "consumer"],
        ),
        "stream_lag_ms": registry.gauge(
            f"{prefix}_stream_lag_ms",
            "Stream processing lag in milliseconds",
            ["intersection_id", "consumer"],
        ),
        # Experiment metrics
        "experiment_runs_total": registry.counter(
            f"{prefix}_experiment_runs_total",
            "Total experiment runs",
            ["experiment_name", "controller_type", "status"],
        ),
        "experiment_waiting_time": registry.histogram(
            f"{prefix}_experiment_waiting_time",
            "Average waiting time from experiments",
            ["experiment_name", "controller_type"],
            buckets=[5, 10, 20, 30, 45, 60, 90, 120, 180, 300],
        ),
    }


# --- Decorators for common patterns ---

def track_latency(metric_name: str, labels: dict[str, str] | None = None):
    """Decorator to track function execution latency."""
    import functools
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            registry = get_metrics_registry()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                registry.observe_histogram(metric_name, elapsed_ms, labels)

        return wrapper

    return decorator


def count_calls(metric_name: str, labels: dict[str, str] | None = None):
    """Decorator to count function calls."""
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            registry = get_metrics_registry()
            registry.inc_counter(metric_name, labels)
            return func(*args, **kwargs)

        return wrapper

    return decorator