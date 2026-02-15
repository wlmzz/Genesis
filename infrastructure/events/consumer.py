"""
Redis Streams event consumer for processing events
"""
import json
import logging
import redis
import asyncio
from typing import Optional, Callable, List, Dict, Any
from abc import ABC, abstractmethod
from .event_types import BaseEvent, deserialize_event
from .retry_policy import RetryPolicy, RetryConfig

logger = logging.getLogger(__name__)


class RedisEventConsumer(ABC):
    """
    Base class for Redis Streams consumers
    Subclass this and implement process_event() for specific workers
    """

    def __init__(
        self,
        redis_url: str,
        stream: str,
        consumer_group: str,
        consumer_name: Optional[str] = None,
        batch_size: int = 10,
        block_ms: int = 5000,
        retry_config: Optional[RetryConfig] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        """
        Args:
            redis_url: Redis connection URL
            stream: Stream name to consume from
            consumer_group: Consumer group name
            consumer_name: Consumer instance name (defaults to hostname)
            batch_size: Number of messages to read per batch
            block_ms: Milliseconds to block waiting for messages
            retry_config: Retry configuration
            redis_client: Optional existing Redis client (for testing)
        """
        self.redis_url = redis_url
        self.stream = stream
        self.consumer_group = consumer_group

        # Generate consumer name if not provided
        if consumer_name is None:
            import socket
            consumer_name = f"{self.__class__.__name__}-{socket.gethostname()}"
        self.consumer_name = consumer_name

        self.batch_size = batch_size
        self.block_ms = block_ms

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

        # Retry policy
        self.retry_policy = RetryPolicy(retry_config or RetryConfig())

        # Running state
        self.running = False

        # Initialize consumer group
        self._init_consumer_group()

        logger.info(
            f"Consumer initialized: {self.consumer_name} "
            f"(group={self.consumer_group}, stream={self.stream})"
        )

    def _init_consumer_group(self):
        """Create consumer group if it doesn't exist"""
        try:
            self.client.xgroup_create(
                self.stream,
                self.consumer_group,
                id="0",
                mkstream=True
            )
            logger.info(f"Created consumer group: {self.consumer_group}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists
                logger.debug(f"Consumer group already exists: {self.consumer_group}")
            else:
                raise

    @abstractmethod
    async def process_event(self, event: BaseEvent) -> None:
        """
        Process a single event - override this in subclasses

        Args:
            event: Deserialized event to process

        Raises:
            Exception: Any exception will trigger retry logic
        """
        pass

    async def _handle_message(
        self,
        message_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Handle a single message with retry logic

        Args:
            message_id: Redis message ID
            message: Message payload

        Returns:
            True if successfully processed, False otherwise
        """
        try:
            # Deserialize event
            event_type = message.get("event_type")
            event_data = json.loads(message.get("data", "{}"))
            event = deserialize_event(event_type, event_data)

            # Process event with retry
            def process():
                # Run async process_event in sync context
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.process_event(event))

            success = self.retry_policy.execute_with_retry(
                process,
                message_id
            )

            if success:
                # ACK message
                await self.ack(message_id)
                return True
            else:
                # Send to DLQ
                await self._send_to_dlq(message_id, message, "max_retries_exceeded")
                return False

        except Exception as e:
            logger.error(f"Failed to process message {message_id}: {e}", exc_info=True)
            await self._send_to_dlq(message_id, message, str(e))
            return False

    async def ack(self, message_id: str) -> None:
        """
        Acknowledge message processing completion

        Args:
            message_id: Message ID to acknowledge
        """
        try:
            self.client.xack(self.stream, self.consumer_group, message_id)
            logger.debug(f"ACKed message: {message_id}")
        except redis.RedisError as e:
            logger.error(f"Failed to ACK message {message_id}: {e}")

    async def _send_to_dlq(
        self,
        message_id: str,
        message: Dict[str, Any],
        error: str
    ) -> None:
        """
        Send failed message to Dead Letter Queue

        Args:
            message_id: Original message ID
            message: Original message payload
            error: Error description
        """
        dlq_stream = f"{self.stream}:dlq"

        dlq_message = {
            **message,
            "original_message_id": message_id,
            "error": error,
            "consumer_name": self.consumer_name,
        }

        try:
            self.client.xadd(dlq_stream, dlq_message)
            logger.warning(f"Sent message to DLQ: {message_id} (error: {error})")

            # ACK original message to remove from pending
            await self.ack(message_id)

        except redis.RedisError as e:
            logger.error(f"Failed to send to DLQ: {e}")

    async def _claim_pending_messages(self) -> List[tuple]:
        """
        Claim pending messages that have been idle too long

        Returns:
            List of (message_id, message) tuples
        """
        try:
            # Check for pending messages
            pending = self.client.xpending_range(
                self.stream,
                self.consumer_group,
                min="-",
                max="+",
                count=self.batch_size
            )

            if not pending:
                return []

            # Claim messages idle for more than 60 seconds
            claimed_ids = [p["message_id"] for p in pending if p["time_since_delivered"] > 60000]

            if claimed_ids:
                claimed = self.client.xclaim(
                    self.stream,
                    self.consumer_group,
                    self.consumer_name,
                    min_idle_time=60000,
                    message_ids=claimed_ids
                )
                logger.info(f"Claimed {len(claimed)} pending messages")
                return claimed

        except redis.RedisError as e:
            logger.error(f"Failed to claim pending messages: {e}")

        return []

    async def run(self) -> None:
        """
        Main consumer loop - reads and processes messages
        """
        self.running = True
        logger.info(f"Consumer started: {self.consumer_name}")

        while self.running:
            try:
                # First, claim any pending messages
                claimed = await self._claim_pending_messages()
                for message_id, message in claimed:
                    await self._handle_message(message_id, message)

                # Read new messages
                messages = self.client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.stream: ">"},
                    count=self.batch_size,
                    block=self.block_ms
                )

                if not messages:
                    continue

                # Process messages
                for stream_name, stream_messages in messages:
                    for message_id, message in stream_messages:
                        await self._handle_message(message_id, message)

            except redis.ConnectionError as e:
                logger.error(f"Redis connection error: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting

            except KeyboardInterrupt:
                logger.info("Consumer interrupted by user")
                self.stop()
                break

            except Exception as e:
                logger.error(f"Unexpected error in consumer loop: {e}", exc_info=True)
                await asyncio.sleep(1)

        logger.info(f"Consumer stopped: {self.consumer_name}")

    def stop(self) -> None:
        """Stop the consumer gracefully"""
        self.running = False
        logger.info(f"Stopping consumer: {self.consumer_name}")

    def close(self) -> None:
        """Close Redis connection"""
        self.stop()
        if self.client:
            self.client.close()
            logger.info("Redis consumer connection closed")
