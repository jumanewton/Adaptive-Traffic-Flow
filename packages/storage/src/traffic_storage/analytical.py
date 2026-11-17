"""ClickHouse adapter for analytical traffic queries."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import clickhouse_connect

from traffic_config.settings import StorageSettings, get_settings
from traffic_domain.schemas import TrafficState, ControllerRecommendation
from traffic_observability.logging import get_logger


logger = get_logger(__name__)


class ClickHouseAdapter:
    """Adapter for ClickHouse analytical database."""

    def __init__(self, settings: StorageSettings | None = None):
        self._settings = settings or get_settings().storage
        self._client = None

        if not self._settings.clickhouse_url:
            logger.warning("ClickHouse URL not configured")

    def connect(self) -> None:
        """Initialize ClickHouse connection."""
        if not self._settings.clickhouse_url:
            raise RuntimeError("ClickHouse URL not configured")

        # Parse URL (format: http://user:pass@host:port/database)
        url = self._settings.clickhouse_url
        # Simple parsing - in production use proper URL parsing
        self._client = clickhouse_connect.get_client(
            host="localhost",
            port=8123,
            username="traffic",
            password="traffic",
            database="traffic_lab",
        )
        logger.info("ClickHouse connection established")

    def close(self) -> None:
        """Close ClickHouse connection."""
        if self._client:
            self._client.close()
            self._client = None
        logger.info("ClickHouse connection closed")

    def insert_traffic_state(self, state: TrafficState) -> None:
        """Insert traffic state (batch insert for performance)."""
        if not self._client:
            self.connect()

        data = []
        for approach in state.approaches:
            row = [
                state.observed_at,
                state.intersection_id,
                approach.approach_id,
                None,  # lane_id
                approach.vehicle_count,
                approach.queue_length_m,
                0.0,
                approach.average_speed_kmh,
                approach.pedestrian_count,
                "{}",
                state.perception_latency_ms,
                state.tracking_quality,
                state.source,
                state.camera_ids,
            ]
            data.append(row)

            for lane_id, lane_state in approach.lanes.items():
                row = [
                    state.observed_at,
                    state.intersection_id,
                    approach.approach_id,
                    lane_id,
                    lane_state.vehicle_count,
                    lane_state.queue_length_m,
                    lane_state.occupancy,
                    lane_state.average_speed_kmh,
                    0,
                    str(lane_state.vehicle_types).replace("'", '"'),
                    state.perception_latency_ms,
                    state.tracking_quality,
                    state.source,
                    state.camera_ids,
                ]
                data.append(row)

        self._client.insert(
            "traffic_lab.traffic_state",
            data,
            column_names=[
                "time", "intersection_id", "approach_id", "lane_id",
                "vehicle_count", "queue_length_m", "occupancy", "avg_speed_kmh",
                "pedestrian_count", "vehicle_types",
                "perception_latency_ms", "tracking_quality", "source", "camera_ids"
            ],
        )

    def query_aggregated(
        self,
        intersection_id: str,
        start_time: datetime,
        end_time: datetime,
        granularity: str = "1min",
    ) -> list[dict]:
        """Query pre-aggregated traffic data."""
        if not self._client:
            self.connect()

        table_map = {
            "1min": "traffic_lab.traffic_state_1min",
            "5min": "traffic_lab.traffic_state_5min",
            "1hour": "traffic_lab.traffic_state_1hour",
        }
        table = table_map.get(granularity, "traffic_lab.traffic_state_1min")

        query = f"""
            SELECT * FROM {table}
            WHERE intersection_id = %(intersection_id)s
            AND time >= %(start_time)s
            AND time <= %(end_time)s
            ORDER BY time
        """

        result = self._client.query(query, parameters={
            "intersection_id": intersection_id,
            "start_time": start_time,
            "end_time": end_time,
        })

        columns = result.column_names
        return [dict(zip(columns, row)) for row in result.result_rows]

    def query_experiment_results(
        self,
        controller_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query experiment results."""
        if not self._client:
            self.connect()

        query = """
            SELECT * FROM traffic_lab.experiment_result
        """
        params = {}

        if controller_type:
            query += " WHERE controller_type = %(controller_type)s"
            params["controller_type"] = controller_type

        query += " ORDER BY started_at DESC LIMIT %(limit)s"
        params["limit"] = limit

        result = self._client.query(query, parameters=params)
        columns = result.column_names
        return [dict(zip(columns, row)) for row in result.result_rows]