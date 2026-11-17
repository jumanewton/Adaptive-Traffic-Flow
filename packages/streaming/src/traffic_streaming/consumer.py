"""Kafka event consumer for traffic events."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

from traffic_config.settings import StreamingSettings, get_settings
from traffic_domain.schemas import TrafficEvent, EventType
from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry


logger = get_logger(__name__)


class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    async def handle(self, event: TrafficEvent) -> None:
        """Handle a traffic event."""
        pass

    @property
    @abstractmethod
    def event_types(self) -> list[EventType]:
        """List of event types this handler can process."""
        pass


class EventConsumer:
    """Async Kafka consumer for traffic events."""

    def __init__(
        self,
        topics: list[str],
        group_id: str | None = None,
        handlers: list[EventHandler] | None = None,
        settings: StreamingSettings | None = None,
    ):
        """Initialize consumer.

        Args:
            topics: List of topics to subscribe to
            group_id: Consumer group ID (defaults to config)
            handlers: List of event handlers
            settings: Optional StreamingSettings override
        """
        self._settings = settings or get_settings().streaming
        self._topics = topics
        self._group_id = group_id or self._settings.group_id
        self._handlers = handlers or []
        self._consumer: AIOKafkaConsumer | None = None
        self._running = False
        self._metrics = get_metrics_registry()

        # Build handler map by event type
        self._handler_map: dict[EventType, list[EventHandler]] = {}
        for handler in self._handlers:
            for event_type in handler.event_types:
                self._handler_map.setdefault(event_type, []).append(handler)

    def add_handler(self, handler: EventHandler) -> None:
        """Add an event handler."""
        self._handlers.append(handler)
        for event_type in handler.event_types:
            self._handler_map.setdefault(event_type, []).append(handler)

    async def start(self) -> None:
        """Start the consumer."""
        if self._consumer is not None:
            return

        self._consumer = AIOKafkaConsumer(
            *self._topics,
            bootstrap_servers=self._settings.bootstrap_servers,
            group_id=self._group_id,
            auto_offset_reset=self._settings.auto_offset_reset,
            enable_auto_commit=self._settings.enable_auto_commit,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            key_deserializer=lambda m: m.decode("utf-8") if m else None,
        )

        await self._consumer.start()
        self._running = True
        logger.info("Event consumer started", topics=self._topics, group_id=self._group_id)

    async def stop(self) -> None:
        """Stop the consumer."""
        self._running = False
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None
            logger.info("Event consumer stopped")

    async def run(self) -> None:
        """Run the consumer loop."""
        if self._consumer is None:
            await self.start()

        try:
            async for msg in self._consumer:
                if not self._running:
                    break

                await self._process_message(msg)

        except KafkaError as e:
            logger.error("Consumer error", error=str(e))
            raise
        except asyncio.CancelledError:
            logger.info("Consumer cancelled")
        finally:
            await self.stop()

    async def _process_message(self, msg) -> None:
        """Process a single Kafka message."""
        try:
            # Parse event
            event_data = msg.value
            event = TrafficEvent.model_validate(event_data)

            # Record metrics
            self._metrics.inc_counter(
                "traffic_events_consumed_total",
                labels={
                    "intersection_id": event.intersection_id,
                    "event_type": event.event_type.value,
                    "consumer": self._group_id,
                },
            )

            # Find handlers for this event type
            handlers = self._handler_map.get(event.event_type, [])
            if not handlers:
                logger.debug("No handlers for event type", event_type=event.event_type.value)
                return

            # Dispatch to handlers
            for handler in handlers:
                try:
                    await handler.handle(event)
                except Exception as e:
                    logger.error(
                        "Handler error",
                        handler=handler.__class__.__name__,
                        event_type=event.event_type.value,
                        error=str(e),
                    )
                    # Continue with other handlers

        except Exception as e:
            logger.error("Failed to process message", error=str(e), topic=msg.topic, partition=msg.partition)


# Decorator-based handler registration
class EventHandlerRegistry:
    """Registry for event handlers using decorators."""

    def __init__(self):
        self._handlers: dict[EventType, list[Callable[[TrafficEvent], Any]]] = {}

    def register(self, *event_types: EventType):
        """Decorator to register a handler function."""
        def decorator(func: Callable[[TrafficEvent], Any]):
            for event_type in event_types:
                self._handlers.setdefault(event_type, []).append(func)
            return func
        return decorator

    def get_handlers(self, event_type: EventType) -> list[Callable[[TrafficEvent], Any]]:
        return self._handlers.get(event_type, [])


# Global registry instance
handler_registry = EventHandlerRegistry()


# Synchronous wrapper for scripts
class SyncEventConsumer:
    """Synchronous wrapper around EventConsumer for simple scripts."""

    def __init__(
        self,
        topics: list[str],
        group_id: str | None = None,
        handlers: list[EventHandler] | None = None,
        settings: StreamingSettings | None = None,
    ):
        self._consumer = EventConsumer(topics, group_id, handlers, settings)
        import asyncio

        asyncio.run(self._consumer.start())

    def add_handler(self, handler: EventHandler) -> None:
        self._consumer.add_handler(handler)

    def run_forever(self) -> None:
        import asyncio

        asyncio.run(self._consumer.run())

    def close(self) -> None:
        import asyncio

        asyncio.run(self._consumer.stop())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example handlers
class LoggingEventHandler(EventHandler):
    """Simple handler that logs events."""

    @property
    def event_types(self) -> list[EventType]:
        return list(EventType)

    async def handle(self, event: TrafficEvent) -> None:
        logger.info(
            "Received event",
            event_type=event.event_type.value,
            intersection_id=event.intersection_id,
            timestamp=event.timestamp.isoformat(),
        )


class MetricsEventHandler(EventHandler):
    """Handler that updates metrics from events."""

    @property
    def event_types(self) -> list[EventType]:
        return [EventType.TRAFFIC_STATE, EventType.CONTROLLER_RECOMMENDATION]

    async def handle(self, event: TrafficEvent) -> None:
        # Metrics are already recorded in consumer
        # This could do additional processing
        pass