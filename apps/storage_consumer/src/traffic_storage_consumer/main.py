"""Storage Consumer main entry point.

Consumes traffic events from Kafka and persists to TimescaleDB and ClickHouse.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from traffic_config.settings import get_settings
from traffic_domain.schemas import (
    TrafficEvent,
    EventType,
    TrafficState,
    ControllerRecommendation,
    ExperimentResult,
    ExperimentStatus,
)
from traffic_observability.logging import configure_logging, get_logger
from traffic_observability.metrics import get_metrics_registry, create_traffic_metrics
from traffic_streaming.consumer import EventConsumer, EventHandler, SyncEventConsumer
from traffic_storage.timeseries import TimescaleDBAdapter
from traffic_storage.analytical import ClickHouseAdapter

logger = get_logger(__name__)

_shutdown = False


def signal_handler(signum: int, frame) -> None:
    global _shutdown
    logger.info("Shutdown signal received", signal=signum)
    _shutdown = True


class TrafficStateHandler(EventHandler):
    """Handle traffic state events and persist to databases."""

    def __init__(
        self,
        timescale_adapter: TimescaleDBAdapter,
        clickhouse_adapter: ClickHouseAdapter | None = None,
    ):
        self._timescale = timescale_adapter
        self._clickhouse = clickhouse_adapter
        self._metrics = get_metrics_registry()
        self._traffic_metrics = create_traffic_metrics(self._metrics, "storage_consumer")

    @property
    def event_types(self) -> list[EventType]:
        return [EventType.TRAFFIC_STATE]

    async def handle(self, event: TrafficEvent) -> None:
        """Persist traffic state to databases."""
        try:
            state = TrafficState.model_validate(event.payload)

            # Write to TimescaleDB
            await self._timescale.insert_traffic_state(state)
            self._traffic_metrics.inc_counter(
                "traffic_state_persisted_total",
                labels={"db": "timescaledb", "intersection": state.intersection_id},
            )

            # Write to ClickHouse if available
            if self._clickhouse:
                try:
                    self._clickhouse.insert_traffic_state(state)
                    self._traffic_metrics.inc_counter(
                        "traffic_state_persisted_total",
                        labels={"db": "clickhouse", "intersection": state.intersection_id},
                    )
                except Exception as e:
                    logger.warning("ClickHouse write failed", error=str(e))
                    self._traffic_metrics.inc_counter(
                        "traffic_state_persisted_errors_total",
                        labels={"db": "clickhouse"},
                    )

        except Exception as e:
            logger.error("Failed to persist traffic state", error=str(e))
            self._traffic_metrics.inc_counter(
                "traffic_state_persisted_errors_total",
                labels={"db": "timescaledb"},
            )


class ControllerRecommendationHandler(EventHandler):
    """Handle controller recommendation events."""

    def __init__(
        self,
        timescale_adapter: TimescaleDBAdapter,
        clickhouse_adapter: ClickHouseAdapter | None = None,
    ):
        self._timescale = timescale_adapter
        self._clickhouse = clickhouse_adapter
        self._metrics = get_metrics_registry()
        self._traffic_metrics = create_traffic_metrics(self._metrics, "storage_consumer")

    @property
    def event_types(self) -> list[EventType]:
        return [EventType.CONTROLLER_RECOMMENDATION]

    async def handle(self, event: TrafficEvent) -> None:
        """Persist controller recommendation."""
        try:
            rec = ControllerRecommendation.model_validate(event.payload)

            await self._timescale.insert_controller_recommendation(rec)
            self._traffic_metrics.inc_counter(
                "controller_recommendation_persisted_total",
                labels={"db": "timescaledb", "intersection": rec.intersection_id},
            )

        except Exception as e:
            logger.error("Failed to persist controller recommendation", error=str(e))
            self._traffic_metrics.inc_counter(
                "controller_recommendation_persisted_errors_total",
                labels={"db": "timescaledb"},
            )


class ExperimentResultHandler(EventHandler):
    """Handle experiment result events."""

    def __init__(
        self,
        timescale_adapter: TimescaleDBAdapter,
        clickhouse_adapter: ClickHouseAdapter | None = None,
    ):
        self._timescale = timescale_adapter
        self._clickhouse = clickhouse_adapter
        self._metrics = get_metrics_registry()
        self._traffic_metrics = create_traffic_metrics(self._metrics, "storage_consumer")

    @property
    def event_types(self) -> list[EventType]:
        return [EventType.EXPERIMENT_RESULT]

    async def handle(self, event: TrafficEvent) -> None:
        """Persist experiment result."""
        try:
            result = ExperimentResult.model_validate(event.payload)

            # Insert into TimescaleDB
            if not self._timescale._pool:
                await self._timescale.connect()

            query = """
                INSERT INTO traffic.experiment_result (
                    experiment_id, experiment_name, controller_type, status,
                    started_at, completed_at, avg_waiting_time, max_queue_length,
                    throughput_veh_per_hour, avg_stops_per_veh, avg_travel_time,
                    total_emissions, approach_metrics, time_series, error
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """

            import json
            async with self._timescale._pool.acquire() as conn:
                await conn.execute(
                    query,
                    result.experiment_id,
                    result.experiment_name,
                    result.controller_type,
                    result.status.value,
                    result.started_at,
                    result.completed_at,
                    result.avg_waiting_time,
                    result.avg_queue_length,
                    result.throughput_veh_per_hour,
                    result.stops_per_vehicle,
                    result.avg_travel_time,
                    result.total_emissions_kg,
                    json.dumps(result.approach_metrics),
                    json.dumps(result.time_series_data),
                    result.error_message,
                )

            self._traffic_metrics.inc_counter(
                "experiment_result_persisted_total",
                labels={"db": "timescaledb", "controller": result.controller_type},
            )

        except Exception as e:
            logger.error("Failed to persist experiment result", error=str(e))


class PhaseChangeHandler(EventHandler):
    """Handle phase change events."""

    def __init__(self, timescale_adapter: TimescaleDBAdapter):
        self._timescale = timescale_adapter
        self._metrics = get_metrics_registry()

    @property
    def event_types(self) -> list[EventType]:
        return [EventType.PHASE_CHANGE]

    async def handle(self, event: TrafficEvent) -> None:
        """Persist phase change."""
        try:
            if not self._timescale._pool:
                await self._timescale.connect()

            query = """
                INSERT INTO traffic.phase_change (
                    time, intersection_id, from_phase, to_phase, duration_seconds, triggered_by
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """

            payload = event.payload
            async with self._timescale._pool.acquire() as conn:
                await conn.execute(
                    query,
                    event.timestamp,
                    event.intersection_id,
                    payload.get("from_phase"),
                    payload.get("to_phase"),
                    payload.get("duration_seconds"),
                    payload.get("triggered_by", "controller"),
                )

        except Exception as e:
            logger.error("Failed to persist phase change", error=str(e))


async def run_storage_consumer(
    enable_timescale: bool = True,
    enable_clickhouse: bool = False,
) -> int:
    """Run the storage consumer."""
    global _shutdown

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    settings = get_settings()
    streaming_cfg = settings.streaming
    storage_cfg = settings.storage
    obs_cfg = settings.observability

    configure_logging(
        level=obs_cfg.log_level,
        json_output=obs_cfg.log_json,
        service_name="storage_consumer",
    )

    metrics_registry = get_metrics_registry()
    traffic_metrics = create_traffic_metrics(metrics_registry, "storage_consumer")

    # Initialize TimescaleDB
    timescale_adapter = None
    if enable_timescale:
        timescale_adapter = TimescaleDBAdapter(storage_cfg)
        try:
            await timescale_adapter.connect()
            logger.info("TimescaleDB connected")
        except Exception as e:
            logger.warning("TimescaleDB connection failed", error=str(e))
            timescale_adapter = None

    # Initialize ClickHouse
    clickhouse_adapter = None
    if enable_clickhouse and storage_cfg.clickhouse_url:
        clickhouse_adapter = ClickHouseAdapter(storage_cfg)
        try:
            clickhouse_adapter.connect()
            logger.info("ClickHouse connected")
        except Exception as e:
            logger.warning("ClickHouse connection failed", error=str(e))
            clickhouse_adapter = None

    if not timescale_adapter and not clickhouse_adapter:
        logger.error("No database adapters available")
        return 1

    # Create event consumer with handlers
    consumer = EventConsumer(
        topics=[
            streaming_cfg.traffic_state_topic,
            streaming_cfg.controller_action_topic,
            streaming_cfg.phase_change_topic,
            "traffic.experiment.result",
        ],
        group_id=f"{streaming_cfg.group_id}-storage",
        settings=streaming_cfg,
    )

    # Add handlers
    if timescale_adapter:
        consumer.add_handler(TrafficStateHandler(timescale_adapter, clickhouse_adapter))
        consumer.add_handler(ControllerRecommendationHandler(timescale_adapter, clickhouse_adapter))
        consumer.add_handler(ExperimentResultHandler(timescale_adapter, clickhouse_adapter))
        consumer.add_handler(PhaseChangeHandler(timescale_adapter))

    try:
        logger.info("Starting storage consumer")
        await consumer.start()
        await consumer.run()
    except asyncio.CancelledError:
        logger.info("Consumer cancelled")
    except Exception as e:
        logger.exception("Consumer error", error=str(e))
        return 1
    finally:
        await consumer.stop()
        if timescale_adapter:
            await timescale_adapter.disconnect()
        if clickhouse_adapter:
            clickhouse_adapter.close()
        logger.info("Storage consumer stopped")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Traffic Storage Consumer")
    parser.add_argument(
        "--no-timescale",
        action="store_true",
        help="Disable TimescaleDB",
    )
    parser.add_argument(
        "--clickhouse",
        action="store_true",
        help="Enable ClickHouse",
    )

    args = parser.parse_args()

    return asyncio.run(run_storage_consumer(
        enable_timescale=not args.no_timescale,
        enable_clickhouse=args.clickhouse,
    ))


if __name__ == "__main__":
    sys.exit(main())