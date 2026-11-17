"""Analytics Service - Provides aggregated traffic data for dashboards and APIs."""

from traffic_analytics_service.main import run_analytics_server, AnalyticsService

__all__ = ["run_analytics_server", "AnalyticsService"]