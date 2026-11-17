"""Controller Service - Real-time traffic signal control with Kafka event publishing."""

from traffic_controller_service.main import run_controller, RealTimeController

__all__ = ["run_controller", "RealTimeController"]