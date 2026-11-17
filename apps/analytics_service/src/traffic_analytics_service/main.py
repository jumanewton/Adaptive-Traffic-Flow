"""Analytics Service - REST API for traffic data queries and Grafana datasource."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from aiohttp import web
from aiohttp.web import Request, Response, json_response

from traffic_config.settings import get_settings
from traffic_domain.schemas import TrafficState
from traffic_observability.logging import configure_logging, get_logger
from traffic_observability.metrics import get_metrics_registry, create_traffic_metrics
from traffic_storage.timeseries import TimescaleDBAdapter
from traffic_storage.analytical import ClickHouseAdapter

logger = get_logger(__name__)


class AnalyticsService:
    """Analytics service providing REST API for traffic data."""

    def __init__(
        self,
        timescale_adapter: TimescaleDBAdapter | None = None,
        clickhouse_adapter: ClickHouseAdapter | None = None,
    ):
        self._timescale = timescale_adapter
        self._clickhouse = clickhouse_adapter
        self._settings = get_settings()
        self._metrics = get_metrics_registry()
        self._traffic_metrics = create_traffic_metrics(self._metrics, "analytics_service")

    async def get_current_state(self, intersection_id: str) -> dict[str, Any] | None:
        """Get latest traffic state for an intersection."""
        if self._timescale and self._timescale._pool:
            return await self._timescale.get_latest_state(intersection_id)
        return None

    async def get_traffic_history(
        self,
        intersection_id: str,
        hours: int = 1,
        approach_id: str | None = None,
    ) -> list[dict]:
        """Get traffic state history."""
        if not self._timescale or not self._timescale._pool:
            return []

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        return await self._timescale.query_traffic_state(
            intersection_id=intersection_id,
            start_time=start_time,
            end_time=end_time,
            approach_id=approach_id,
        )

    async def get_aggregated_metrics(
        self,
        intersection_id: str,
        hours: int = 24,
        granularity: str = "5min",
    ) -> list[dict]:
        """Get pre-aggregated metrics from ClickHouse."""
        if not self._clickhouse:
            return []

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        return self._clickhouse.query_aggregated(
            intersection_id=intersection_id,
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
        )

    async def get_experiment_comparison(
        self,
        controller_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get experiment results for comparison."""
        if not self._clickhouse:
            return []

        return self._clickhouse.query_experiment_results(
            controller_type=controller_type,
            limit=limit,
        )

    async def get_intersection_summary(
        self,
        intersection_id: str,
        hours: int = 1,
    ) -> dict[str, Any]:
        """Get summary statistics for an intersection."""
        history = await self.get_traffic_history(intersection_id, hours)

        if not history:
            return {"intersection_id": intersection_id, "data_available": False}

        # Compute summaries
        approaches = {}
        for row in history:
            approach = row.get("approach_id")
            if not approach:
                continue
            if approach not in approaches:
                approaches[approach] = {
                    "vehicle_counts": [],
                    "queue_lengths": [],
                    "speeds": [],
                    "pedestrians": 0,
                }
            approaches[approach]["vehicle_counts"].append(row.get("vehicle_count", 0))
            approaches[approach]["queue_lengths"].append(row.get("queue_length_m", 0))
            approaches[approach]["speeds"].append(row.get("avg_speed_kmh", 0))
            approaches[approach]["pedestrians"] += row.get("pedestrian_count", 0)

        summary = {"intersection_id": intersection_id, "data_available": True, "approaches": {}}
        for approach, data in approaches.items():
            summary["approaches"][approach] = {
                "avg_vehicles": sum(data["vehicle_counts"]) / len(data["vehicle_counts"]) if data["vehicle_counts"] else 0,
                "max_queue_m": max(data["queue_lengths"]) if data["queue_lengths"] else 0,
                "avg_queue_m": sum(data["queue_lengths"]) / len(data["queue_lengths"]) if data["queue_lengths"] else 0,
                "avg_speed_kmh": sum(data["speeds"]) / len(data["speeds"]) if data["speeds"] else 0,
                "total_pedestrians": data["pedestrians"],
                "samples": len(data["vehicle_counts"]),
            }

        return summary


async def create_app(
    timescale_adapter: TimescaleDBAdapter | None = None,
    clickhouse_adapter: ClickHouseAdapter | None = None,
) -> web.Application:
    """Create aiohttp application with routes."""
    service = AnalyticsService(timescale_adapter, clickhouse_adapter)

    async def handle_current_state(request: Request) -> Response:
        intersection_id = request.match_info.get("intersection_id", "intersection-01")
        state = await service.get_current_state(intersection_id)
        if state:
            return json_response(state)
        return json_response({"error": "No data available"}, status=404)

    async def handle_history(request: Request) -> Response:
        intersection_id = request.match_info.get("intersection_id", "intersection-01")
        hours = int(request.query.get("hours", "1"))
        approach_id = request.query.get("approach")
        history = await service.get_traffic_history(intersection_id, hours, approach_id)
        return json_response({"intersection_id": intersection_id, "data": history})

    async def handle_summary(request: Request) -> Response:
        intersection_id = request.match_info.get("intersection_id", "intersection-01")
        hours = int(request.query.get("hours", "1"))
        summary = await service.get_intersection_summary(intersection_id, hours)
        return json_response(summary)

    async def handle_aggregated(request: Request) -> Response:
        intersection_id = request.match_info.get("intersection_id", "intersection-01")
        hours = int(request.query.get("hours", "24"))
        granularity = request.query.get("granularity", "5min")
        data = await service.get_aggregated_metrics(intersection_id, hours, granularity)
        return json_response({"intersection_id": intersection_id, "granularity": granularity, "data": data})

    async def handle_experiments(request: Request) -> Response:
        controller_type = request.query.get("controller")
        limit = int(request.query.get("limit", "20"))
        data = await service.get_experiment_comparison(controller_type, limit)
        return json_response({"experiments": data})

    async def handle_health(request: Request) -> Response:
        return json_response({"status": "healthy", "service": "analytics"})

    async def handle_metrics(request: Request) -> Response:
        """Prometheus metrics endpoint."""
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(body=generate_latest(), content_type=CONTENT_TYPE_LATEST)

    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_get("/api/v1/intersections/{intersection_id}/state", handle_current_state)
    app.router.add_get("/api/v1/intersections/{intersection_id}/history", handle_history)
    app.router.add_get("/api/v1/intersections/{intersection_id}/summary", handle_summary)
    app.router.add_get("/api/v1/intersections/{intersection_id}/aggregated", handle_aggregated)
    app.router.add_get("/api/v1/experiments", handle_experiments)

    return app


async def run_analytics_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    enable_timescale: bool = True,
    enable_clickhouse: bool = False,
) -> None:
    """Run the analytics HTTP server."""
    settings = get_settings()
    storage_cfg = settings.storage
    obs_cfg = settings.observability

    configure_logging(
        level=obs_cfg.log_level,
        json_output=obs_cfg.log_json,
        service_name="analytics_service",
    )

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

    app = await create_app(timescale_adapter, clickhouse_adapter)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info("Analytics server started", host=host, port=port)

    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
        if timescale_adapter:
            await timescale_adapter.disconnect()
        if clickhouse_adapter:
            clickhouse_adapter.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Traffic Analytics Service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--no-timescale", action="store_true", help="Disable TimescaleDB")
    parser.add_argument("--clickhouse", action="store_true", help="Enable ClickHouse")

    args = parser.parse_args()

    try:
        asyncio.run(run_analytics_server(
            host=args.host,
            port=args.port,
            enable_timescale=not args.no_timescale,
            enable_clickhouse=args.clickhouse,
        ))
        return 0
    except Exception as e:
        logger.exception("Analytics server failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())