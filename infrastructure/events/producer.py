"""
Redis Streams event producer for publishing events
"""
import json
import logging
import redis
from typing import Optional, Dict, Any
from .event_types import BaseEvent
from .retry_policy import BackpressureManager, BackpressureError

logger = logging.getLogger(__name__)


class RedisEventProducer:
    """
    Publishes events to Redis Streams with backpressure handling
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        stream_prefix: str = "genesis",
        backpressure_enabled: bool = True,
        max_pending: int = 1000,
        block_threshold: int = 800,
        redis_client: Optional[redis.Redis] = None
    ):
        """
        Args:
            redis_url: Redis connection URL
            stream_prefix: Prefix for stream names
            backpressure_enabled: Enable backpressure management
            max_pending: Maximum pending messages
            block_threshold: Threshold to start blocking
            redis_client: Optional existing Redis client (for testing)
        """
        self.redis_url = redis_url
        self.stream_prefix = stream_prefix

        # Connect to Redis
        if redis_client:
            self.client = redis_client
        else:
            self.client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )

        # Backpressure manager
        self.backpressure = BackpressureManager(
            max_pending=max_pending,
            block_threshold=block_threshold,
            enabled=backpressure_enabled
        )

        # Test connection
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _get_stream_name(self, event_type: str) -> str:
        """
        Get Redis stream name for event type

        Args:
            event_type: Event class name

        Returns:
            Full stream name with prefix
        """
        # Map event types to stream names
        stream_map = {
            "FrameCapturedEvent": "frames",
            "PersonDetectedEvent": "persons",
            "FaceRecognizedEvent": "faces",
            "ZoneEvent": "zones",
            "AlertTriggeredEvent": "alerts",
            "MetricsAggregatedEvent": "metrics",
            "SessionEvent": "sessions",
        }

        stream_suffix = stream_map.get(event_type, "events")
        return f"{self.stream_prefix}:{stream_suffix}"

    def _get_pending_count(self, stream_name: str) -> int:
        """
        Get count of pending messages in stream

        Args:
            stream_name: Stream name

        Returns:
            Number of pending messages
        """
        try:
            # Get pending messages info for all consumer groups
            groups = self.client.xinfo_groups(stream_name)
            total_pending = sum(group.get("pending", 0) for group in groups)
            return total_pending
        except redis.ResponseError:
            # Stream or consumer groups don't exist yet
            return 0

    def publish(
        self,
        event: BaseEvent,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Publish event to Redis stream

        Args:
            event: Event instance to publish
            max_retries: Maximum number of retry attempts

        Returns:
            Message ID if successful, None if failed

        Raises:
            BackpressureError: If backpressure threshold exceeded
        """
        event_type = event.__class__.__name__
        stream_name = self._get_stream_name(event_type)

        # Check backpressure
        pending_count = self._get_pending_count(stream_name)
        self.backpressure.check_backpressure(pending_count)

        # Prepare message payload
        message = {
            "event_type": event_type,
            "data": event.to_json(),
            "timestamp": str(event.timestamp if hasattr(event, "timestamp") else 0)
        }

        # Publish to stream
        try:
            message_id = self.client.xadd(stream_name, message, maxlen=100000)
            logger.debug(f"Published {event_type} to {stream_name}: {message_id}")
            return message_id

        except redis.RedisError as e:
            logger.error(f"Failed to publish event to {stream_name}: {e}")
            raise

    def publish_batch(
        self,
        events: list[BaseEvent]
    ) -> list[Optional[str]]:
        """
        Publish multiple events in a pipeline for better performance

        Args:
            events: List of events to publish

        Returns:
            List of message IDs (None for failed publishes)
        """
        if not events:
            return []

        pipe = self.client.pipeline()
        stream_names = []

        for event in events:
            event_type = event.__class__.__name__
            stream_name = self._get_stream_name(event_type)
            stream_names.append(stream_name)

            message = {
                "event_type": event_type,
                "data": event.to_json(),
                "timestamp": str(event.timestamp if hasattr(event, "timestamp") else 0)
            }

            pipe.xadd(stream_name, message, maxlen=100000)

        try:
            results = pipe.execute()
            logger.debug(f"Published {len(events)} events in batch")
            return results
        except redis.RedisError as e:
            logger.error(f"Failed to publish batch: {e}")
            return [None] * len(events)

    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            logger.info("Redis producer connection closed")
