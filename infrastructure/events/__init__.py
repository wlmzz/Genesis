"""Event-driven architecture components for Genesis"""
from .event_types import (
    FrameCapturedEvent,
    PersonDetectedEvent,
    FaceRecognizedEvent,
    ZoneEvent,
    AlertTriggeredEvent,
    MetricsAggregatedEvent,
    SessionEvent,
)
from .producer import RedisEventProducer
from .consumer import RedisEventConsumer
from .retry_policy import BackpressureError, RetryPolicy

__all__ = [
    "FrameCapturedEvent",
    "PersonDetectedEvent",
    "FaceRecognizedEvent",
    "ZoneEvent",
    "AlertTriggeredEvent",
    "MetricsAggregatedEvent",
    "SessionEvent",
    "RedisEventProducer",
    "RedisEventConsumer",
    "BackpressureError",
    "RetryPolicy",
]
