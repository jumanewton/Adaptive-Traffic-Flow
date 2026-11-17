"""OpenTelemetry tracing configuration."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span, SpanKind, Status, StatusCode


_tracer_provider: TracerProvider | None = None


def init_tracing(
    service_name: str | None = None,
    otlp_endpoint: str | None = None,
    console_export: bool = False,
) -> TracerProvider:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name (defaults to SERVICE_NAME env var)
        otlp_endpoint: OTLP gRPC endpoint for trace export (e.g., "localhost:4317")
        console_export: If True, also export to console (for development)

    Returns:
        Configured TracerProvider
    """
    global _tracer_provider

    if service_name is None:
        service_name = os.getenv("SERVICE_NAME", "traffic-lab")

    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)

    # OTLP exporter
    if otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Console exporter for development
    if console_export or os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true":
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    return provider


def get_tracer(name: str | None = None) -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name or __name__)


@contextmanager
def start_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict | None = None,
) -> Generator[Span, None, None]:
    """Context manager for creating and ending a span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name, kind=kind, attributes=attributes or {}) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        else:
            span.set_status(Status(StatusCode.OK))


def add_span_attributes(**attributes: any) -> None:
    """Add attributes to the current span."""
    span = trace.get_current_span()
    if span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def set_span_status(status: StatusCode, message: str = "") -> None:
    """Set the status of the current span."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_status(Status(status, message))


def get_current_trace_id() -> str | None:
    """Get the current trace ID as hex string."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.trace_id:
        return format(ctx.trace_id, "032x")
    return None


def get_current_span_id() -> str | None:
    """Get the current span ID as hex string."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.span_id:
        return format(ctx.span_id, "016x")
    return None