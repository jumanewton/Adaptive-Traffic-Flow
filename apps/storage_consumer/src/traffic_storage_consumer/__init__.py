"""Storage Consumer - Consumes Kafka events and writes to databases."""

from traffic_storage_consumer.main import run_consumer, StorageConsumer

__all__ = ["run_consumer", "StorageConsumer"]