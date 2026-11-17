"""TimescaleDB adapter for time-series traffic data."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

from traffic_config.settings import StorageSettings, get_settings
from traffic_domain.schemas import TrafficState, ControllerRecommendation
from traffic_observability.logging import get_logger


logger = get_logger(__name__)


class TimescaleDBAdapter:
    """Async adapter for TimescaleDB (PostgreSQL)."""

    def __init__(self, settings: StorageSettings | None = None):
        self._settings = settings or get_settings().storage
        self._engine = None
        self._session_factory = None
        self._pool = None

    async def connect(self) -> None:
        """Initialize database connection."""
        # SQLAlchemy async engine
        self._engine = create_async_engine(
            self._settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
            pool_size=10,
            max_overflow=20,
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

        # Also create asyncpg pool for direct queries
        self._pool = await asyncpg.create_pool(
            self._settings.database_url,
            min_size=5,
            max_size=20,
        )

        logger.info("TimescaleDB connection established")

    async def disconnect(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
        if self._pool:
            await self._pool.close()
        logger.info("TimescaleDB connections closed")

    @asynccontextmanager
    async def session(self):
        """Get a database session."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def insert_traffic_state(self, state: TrafficState) -> None:
        """Insert traffic state measurement."""
        if not self._pool:
            await self.connect()

        query = """
            INSERT INTO traffic.traffic_state (
                time, intersection_id, approach_id, lane_id,
                vehicle_count, queue_length_m, occupancy, avg_speed_kmh,
                pedestrian_count, vehicle_types,
                perception_latency_ms, tracking_quality, source, camera_ids
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
            )
        """

        async with self._pool.acquire() as conn:
            for approach in state.approaches:
                await conn.execute(
                    query,
                    state.observed_at,
                    state.intersection_id,
                    approach.approach_id,
                    None,  # lane_id - will insert lane-level separately
                    approach.vehicle_count,
                    approach.queue_length_m,
                    0.0,  # occupancy - compute from lanes
                    approach.average_speed_kmh,
                    approach.pedestrian_count,
                    "{}",  # vehicle_types JSONB
                    state.perception_latency_ms,
                    state.tracking_quality,
                    state.source,
                    state.camera_ids,
                )

                # Insert lane-level data
                for lane_id, lane_state in approach.lanes.items():
                    await conn.execute(
                        query,
                        state.observed_at,
                        state.intersection_id,
                        approach.approach_id,
                        lane_id,
                        lane_state.vehicle_count,
                        lane_state.queue_length_m,
                        lane_state.occupancy,
                        lane_state.average_speed_kmh,
                        0,  # pedestrian_count at lane level
                        str(lane_state.vehicle_types).replace("'", '"'),
                        state.perception_latency_ms,
                        state.tracking_quality,
                        state.source,
                        state.camera_ids,
                    )

    async def insert_controller_recommendation(self, rec: ControllerRecommendation) -> None:
        """Insert controller recommendation."""
        if not self._pool:
            await self.connect()

        query = """
            INSERT INTO traffic.controller_recommendation (
                time, intersection_id, current_phase, action,
                duration_seconds, reason, controller_type,
                simulation_only, confidence
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """

        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                rec.timestamp,
                rec.intersection_id,
                rec.current_phase.value,
                rec.action.value,
                rec.duration_seconds,
                rec.reason,
                rec.controller_type,
                rec.simulation_only,
                rec.confidence,
            )

    async def query_traffic_state(
        self,
        intersection_id: str,
        start_time: datetime,
        end_time: datetime,
        approach_id: str | None = None,
    ) -> list[dict]:
        """Query traffic state measurements."""
        if not self._pool:
            await self.connect()

        query = """
            SELECT * FROM traffic.traffic_state
            WHERE intersection_id = $1
            AND time >= $2
            AND time <= $3
        """
        params = [intersection_id, start_time, end_time]

        if approach_id:
            query += " AND approach_id = $4"
            params.append(approach_id)

        query += " ORDER BY time DESC"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def get_latest_state(self, intersection_id: str) -> dict | None:
        """Get latest traffic state for an intersection."""
        if not self._pool:
            await self.connect()

        query = """
            SELECT * FROM traffic.traffic_state
            WHERE intersection_id = $1
            ORDER BY time DESC
            LIMIT 1
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, intersection_id)
            return dict(row) if row else None