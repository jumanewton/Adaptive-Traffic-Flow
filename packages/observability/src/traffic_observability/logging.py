"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor


def add_service_context(logger: structlog.BoundLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add service context to all log entries."""
    # Service name from environment or default
    import os

    event_dict.setdefault("service", os.getenv("SERVICE_NAME", "traffic-lab"))
    event_dict.setdefault("environment", os.getenv("ENVIRONMENT", "development"))
    return event_dict


def add_timestamp(logger: structlog.BoundLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO timestamp to all log entries."""
    from datetime import datetime, timezone

    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def drop_color_message_key(logger: structlog.BoundLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Remove color_message key added by structlog for console output."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    service_name: str | None = None,
) -> None:
    """Configure structlog for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output JSON lines; otherwise pretty console output
        service_name: Override service name (defaults to SERVICE_NAME env var)
    """
    import os

    if service_name:
        os.environ["SERVICE_NAME"] = service_name

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_service_context,
        add_timestamp,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        drop_color_message_key,
    ]

    if json_output:
        # JSON output for production / log aggregation
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Pretty console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to route through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Quiet noisy third-party loggers
    for noisy_logger in ["urllib3", "asyncio", "kafka", "numba"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str | None = None, **bind_kwargs: Any) -> structlog.BoundLogger:
    """Get a structured logger with optional bound context.

    Args:
        name: Logger name (typically __name__)
        **bind_kwargs: Additional context to bind to the logger

    Returns:
        Configured structlog logger
    """
    logger = structlog.get_logger(name)
    if bind_kwargs:
        logger = logger.bind(**bind_kwargs)
    return logger