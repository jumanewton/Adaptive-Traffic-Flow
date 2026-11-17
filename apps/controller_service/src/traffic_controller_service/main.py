"""Real-time Controller Service.

Consumes traffic state events, computes control actions, and publishes
controller recommendations and phase changes to Kafka.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from traffic_config.settings import get_settings
from traffic_domain.schemas import (
    TrafficEvent,
    EventType,
    TrafficState,
    ControllerRecommendation,
    ControllerAction,
    SignalPhase,
    IntersectionConfig,
)
from traffic_observability.logging import configure_logging, get_logger
from traffic_observability.metrics import get_metrics_registry, create_traffic_metrics
from traffic_streaming.consumer import EventConsumer, EventHandler
from traffic_streaming.producer import SyncEventProducer
from traffic_control.base import BaseController
from traffic_control.fixed_time import FixedTimeController
from traffic_control.max_pressure import MaxPressureController

logger = get_logger(__name__)

_shutdown = False


def signal_handler(signum: int, frame) -> None:
    global _shutdown
    logger.info("Shutdown signal received", signal=signum)
    _shutdown = True


class TrafficStateHandler(EventHandler):
    """Handle traffic state events and compute control actions."""

    def __init__(
        self,
        controller: BaseController,
        producer: SyncEventProducer,
        control_topic: str,
        phase_change_topic: str,
        intersection_id: str,
    ):
        self._controller = controller
        self._producer = producer
        self._control_topic = control_topic
        self._phase_change_topic = phase_change_topic
        self._intersection_id = intersection_id
        self._current_phase = None
        self._metrics = get_metrics_registry()
        self._traffic_metrics = create_traffic_metrics(self._metrics, "controller_service")

    @property
    def event_types(self) -> list[EventType]:
        return [EventType.TRAFFIC_STATE]

    async def handle(self, event: TrafficEvent) -> None:
        """Compute control action from traffic state and publish."""
        try:
            state = TrafficState.model_validate(event.payload)

            # Compute control action
            recommendation = self._controller.compute_action(state)

            # Check for phase change
            if recommendation.action == ControllerAction.SWITCH_PHASE:
                # Publish phase change event
                phase_event = TrafficEvent(
                    event_type=EventType.PHASE_CHANGE,
                    intersection_id=self._intersection_id,
                    timestamp=datetime.now(timezone.utc),
                    payload={
                        "from_phase": self._current_phase.value if self._current_phase else None,
                        "to_phase": recommendation.current_phase.value,
                        "duration_seconds": recommendation.duration_seconds,
                        "triggered_by": "controller",
                        "reason": recommendation.reason,
                    },
                    source_service="controller_service",
                )
                try:
                    self._producer.publish(self._phase_change_topic, phase_event)
                    self._traffic_metrics.inc_counter(
                        "phase_changes_published_total",
                        labels={"intersection": self._intersection_id},
                    )
                except Exception as e:
                    logger.warning("Failed to publish phase change", error=str(e))

                self._current_phase = recommendation.current_phase

            # Publish controller recommendation
            control_event = TrafficEvent(
                event_type=EventType.CONTROLLER_RECOMMENDATION,
                intersection_id=self._intersection_id,
                timestamp=datetime.now(timezone.utc),
                payload=recommendation.model_dump(mode="json"),
                source_service="controller_service",
            )

            try:
                self._producer.publish(self._control_topic, control_event)
                self._traffic_metrics.inc_counter(
                    "controller_recommendations_published_total",
                    labels={"intersection": self._intersection_id, "controller": recommendation.controller_type},
                )
            except Exception as e:
                logger.warning("Failed to publish controller recommendation", error=str(e))

        except Exception as e:
            logger.error("Failed to process traffic state", error=str(e))


class RealTimeController:
    """Real-time traffic signal controller with Kafka integration."""

    def __init__(
        self,
        intersection_config: IntersectionConfig,
        controller_type: str = "max_pressure",
        settings: Any | None = None,
    ):
        self._settings = settings or get_settings()
        self._intersection_config = intersection_config
        self._controller_type = controller_type

        # Create controller
        control_settings = self._settings.control
        if controller_type == "fixed_time":
            self._controller = FixedTimeController(intersection_config, control_settings)
        elif controller_type == "max_pressure":
            self._controller = MaxPressureController(intersection_config, control_settings)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        # Initialize producer
        self._producer = SyncEventProducer(
            bootstrap_servers=self._settings.streaming.bootstrap_servers,
            client_id=f"{self._settings.streaming.client_id}-controller",
        )

        # Initialize consumer
        self._consumer = EventConsumer(
            topics=[self._settings.streaming.traffic_state_topic],
            group_id=f"{self._settings.streaming.group_id}-controller",
            settings=self._settings.streaming,
        )

        # Add handler
        self._consumer.add_handler(TrafficStateHandler(
            controller=self._controller,
            producer=self._producer,
            control_topic=self._settings.streaming.controller_action_topic,
            phase_change_topic=self._settings.streaming.phase_change_topic,
            intersection_id=intersection_config.intersection_id,
        ))

    async def start(self) -> None:
        """Start the controller."""
        await self._consumer.start()
        logger.info("Real-time controller started", controller=self._controller_type)

    async def run(self) -> None:
        """Run the controller loop."""
        try:
            await self._consumer.run()
        except asyncio.CancelledError:
            logger.info("Controller cancelled")
        finally:
            await self._consumer.stop()
            self._producer.close()
            logger.info("Real-time controller stopped")


def create_default_intersection_config() -> IntersectionConfig:
    """Create default intersection configuration."""
    from traffic_domain.schemas import ApproachConfig, LaneConfig, Point, SignalPhase

    return IntersectionConfig(
        intersection_id="intersection-01",
        name="Main St & 1st Ave",
        approaches=[
            ApproachConfig(
                approach_id="northbound",
                name="Northbound Main St",
                lanes=[
                    LaneConfig(lane_id="nb_left", approach_id="northbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="nb_thru", approach_id="northbound"),
                    LaneConfig(lane_id="nb_right", approach_id="northbound", is_turning=True, turn_direction="right"),
                ],
                signal_group="NS",
                has_crosswalk=True,
            ),
            ApproachConfig(
                approach_id="southbound",
                name="Southbound Main St",
                lanes=[
                    LaneConfig(lane_id="sb_left", approach_id="southbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="sb_thru", approach_id="southbound"),
                    LaneConfig(lane_id="sb_right", approach_id="southbound", is_turning=True, turn_direction="right"),
                ],
                signal_group="NS",
                has_crosswalk=True,
            ),
            ApproachConfig(
                approach_id="eastbound",
                name="Eastbound 1st Ave",
                lanes=[
                    LaneConfig(lane_id="eb_left", approach_id="eastbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="eb_thru", approach_id="eastbound"),
                    LaneConfig(lane_id="eb_right", approach_id="eastbound", is_turning=True, turn_direction="right"),
                ],
                signal_group="EW",
                has_crosswalk=True,
            ),
            ApproachConfig(
                approach_id="westbound",
                name="Westbound 1st Ave",
                lanes=[
                    LaneConfig(lane_id="wb_left", approach_id="westbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="wb_thru", approach_id="westbound"),
                    LaneConfig(lane_id="wb_right", approach_id="westbound", is_turning=True, turn_direction="right"),
                ],
                signal_group="EW",
                has_crosswalk=True,
            ),
        ],
        phases=[
            SignalPhase.NORTH_SOUTH_GREEN,
            SignalPhase.NORTH_SOUTH_YELLOW,
            SignalPhase.EAST_WEST_GREEN,
            SignalPhase.EAST_WEST_YELLOW,
        ],
    )


async def run_controller(
    controller_type: str = "max_pressure",
    intersection_id: str = "intersection-01",
) -> int:
    """Run the real-time controller service."""
    global _shutdown

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    settings = get_settings()
    obs_cfg = settings.observability

    configure_logging(
        level=obs_cfg.log_level,
        json_output=obs_cfg.log_json,
        service_name="controller_service",
    )

    intersection_config = create_default_intersection_config()
    intersection_config.intersection_id = intersection_id

    controller_service = RealTimeController(
        intersection_config=intersection_config,
        controller_type=controller_type,
        settings=settings,
    )

    try:
        await controller_service.start()
        await controller_service.run()
        return 0
    except Exception as e:
        logger.exception("Controller service failed", error=str(e))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time Traffic Controller Service")
    parser.add_argument(
        "-c", "--controller",
        choices=["fixed_time", "max_pressure"],
        default="max_pressure",
        help="Controller type",
    )
    parser.add_argument(
        "-i", "--intersection",
        default="intersection-01",
        help="Intersection ID",
    )
    parser.add_argument(
        "--sumo-home",
        help="Path to SUMO installation",
    )

    args = parser.parse_args()

    if args.sumo_home:
        import os
        os.environ["SUMO_HOME"] = args.sumo_home

    try:
        return asyncio.run(run_controller(
            controller_type=args.controller,
            intersection_id=args.intersection,
        ))
    except Exception as e:
        logger.exception("Controller service failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())