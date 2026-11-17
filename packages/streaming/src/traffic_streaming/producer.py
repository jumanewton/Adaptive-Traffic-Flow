"""Kafka event producer for traffic events."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from traffic_config.settings import StreamingSettings, get_settings
from traffic_domain.schemas import TrafficEvent
from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry


logger = get_logger(__name__)


class EventProducer:
    """Async Kafka producer for traffic events."""

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        client_id: str | None = None,
        settings: StreamingSettings | None = None,
    ):
        """Initialize producer.

        Args:
            bootstrap_servers: Kafka bootstrap servers (defaults to config)
            client_id: Client ID (defaults to config)
            settings: Optional StreamingSettings override
        """
        self._settings = settings or get_settings().streaming
        self._bootstrap_servers = bootstrap_servers or self._settings.bootstrap_servers
        self._client_id = client_id or self._settings.client_id
        self._producer: AIOKafkaProducer | None = None
        self._metrics = get_metrics_registry()

    async def start(self) -> None:
        """Start the producer."""
        if self._producer is not None:
            return

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            client_id=self._client_id,
            acks=self._settings.acks,
            retries=self._settings.retries,
            linger_ms=self._settings.linger_ms,
            compression_type=self._settings.compression_type if self._settings.compression_type != "none" else None,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )

        await self._producer.start()
        logger.info("Event producer started", bootstrap_servers=self._bootstrap_servers)

    async def stop(self) -> None:
        """Stop the producer."""
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
            logger.info("Event producer stopped")

    async def publish(
        self,
        topic: str,
        event: TrafficEvent,
        key: str | None = None,
    ) -> None:
        """Publish a traffic event to a topic.

        Args:
            topic: Kafka topic name
            event: TrafficEvent to publish
            key: Optional partition key (defaults to intersection_id)
        """
        if self._producer is None:
            await self.start()

        if key is None:
            key = event.intersection_id

        try:
            # Serialize event to dict
            value = event.model_dump(mode="json")

            await self._producer.send_and_wait(
                topic,
                value=value,
                key=key,
            )

            # Record metrics
            self._metrics.inc_counter(
                "traffic_events_published_total",
                labels={"intersection_id": event.intersection_id, "event_type": event.event_type.value},
            )

            logger.debug(
                "Event published",
                topic=topic,
                event_type=event.event_type.value,
                intersection_id=event.intersection_id,
            )

        except KafkaError as e:
            logger.error("Failed to publish event", topic=topic, error=str(e))
            raise

    async def publish_batch(
        self,
        topic: str,
        events: list[TrafficEvent],
        key: str | None = None,
    ) -> None:
        """Publish multiple events in a batch."""
        if self._producer is None:
            await self.start()

        if key is None and events:
            key = events[0].intersection_id

        try:
            # Send all events
            futures = []
            for event in events:
                value = event.model_dump(mode="json")
                future = self._producer.send(topic, value=value, key=key)
                futures.append(future)

            # Wait for all
            for future in futures:
                await future

            logger.debug("Event batch published", topic=topic, count=len(events))

        except KafkaError as e:
            logger.error("Failed to publish event batch", topic=topic, error=str(e))
            raise

    def close(self) -> None:
        """Synchronous close (for non-async contexts)."""
        # In non-async context, we can't properly await stop()
        # This is a fallback for scripts
        logger.warning("Synchronous close called - producer may not be properly closed")


# Context manager for easy usage
@contextmanager
def event_producer(
    bootstrap_servers: str | None = None,
    client_id: str | None = None,
    settings: StreamingSettings | None = None,
):
    """Context manager for EventProducer."""
    producer = EventProducer(bootstrap_servers, client_id, settings)
    import asyncio

    asyncio.run(producer.start())
    try:
        yield producer
    finally:
        asyncio.run(producer.stop())


# Synchronous wrapper for scripts
class SyncEventProducer:
    """Synchronous wrapper around EventProducer for simple scripts."""

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        client_id: str | None = None,
        settings: StreamingSettings | None = None,
    ):
        self._producer = EventProducer(bootstrap_servers, client_id, settings)
        import asyncio

        asyncio.run(self._producer.start())

    def publish(self, topic: str, event: TrafficEvent, key: str | None = None) -> None:
        import asyncio

        asyncio.run(self._producer.publish(topic, event, key))

    def publish_batch(self, topic: str, events: list[TrafficEvent], key: str | None = None) -> None:
        import asyncio

        asyncio.run(self._producer.publish_batch(topic, events, key))

    def close(self) -> None:
        import asyncio

        asyncio.run(self._producer.stop())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()