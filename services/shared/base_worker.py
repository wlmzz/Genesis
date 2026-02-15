"""
Base worker class that all Genesis workers inherit from
Provides common functionality: health checks, metrics, logging
"""
import asyncio
import logging
import signal
import sys
from abc import abstractmethod
from typing import Optional
from infrastructure.events import RedisEventConsumer, BaseEvent
from .health import HealthServer
from .metrics import WorkerMetrics

logger = logging.getLogger(__name__)


class BaseWorker(RedisEventConsumer):
    """
    Base class for all Genesis worker services
    Extends RedisEventConsumer with health checks and metrics
    """

    def __init__(
        self,
        redis_url: str,
        stream: str,
        consumer_group: str,
        consumer_name: Optional[str] = None,
        worker_name: str = "genesis-worker",
        health_port: int = 8080,
        **kwargs
    ):
        """
        Args:
            redis_url: Redis connection URL
            stream: Stream name to consume from
            consumer_group: Consumer group name
            consumer_name: Consumer instance name
            worker_name: Worker service name (for metrics)
            health_port: Port for health check server
            **kwargs: Additional args for RedisEventConsumer
        """
        super().__init__(
            redis_url=redis_url,
            stream=stream,
            consumer_group=consumer_group,
            consumer_name=consumer_name,
            **kwargs
        )

        self.worker_name = worker_name

        # Health check server
        self.health_server = HealthServer(
            port=health_port,
            worker_name=worker_name,
            get_status=self.get_status
        )

        # Metrics
        self.metrics = self.create_metrics()

        # Set worker info
        self.metrics.set_info(
            worker_name=worker_name,
            stream=stream,
            consumer_group=consumer_group,
            consumer_name=self.consumer_name
        )

        # Shutdown flag
        self._shutdown_requested = False

        # Setup signal handlers
        self._setup_signal_handlers()

    @abstractmethod
    def create_metrics(self) -> WorkerMetrics:
        """
        Create metrics instance for this worker
        Override in subclasses to use specialized metrics
        """
        return WorkerMetrics(self.worker_name)

    @abstractmethod
    async def process_event(self, event: BaseEvent) -> None:
        """
        Process a single event - must be implemented by subclasses

        Args:
            event: Event to process
        """
        pass

    def get_status(self) -> dict:
        """
        Get current worker status

        Returns:
            Status dictionary
        """
        try:
            # Get pending messages count
            pending = self.client.xpending(
                self.stream,
                self.consumer_group
            )
            pending_count = pending.get('pending', 0) if isinstance(pending, dict) else 0

        except Exception as e:
            logger.error(f"Error getting pending count: {e}")
            pending_count = 0

        return {
            "worker_name": self.worker_name,
            "consumer_name": self.consumer_name,
            "stream": self.stream,
            "consumer_group": self.consumer_group,
            "running": self.running,
            "pending_messages": pending_count,
        }

    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down gracefully...")
            self._shutdown_requested = True
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _handle_message(
        self,
        message_id: str,
        message: dict
    ) -> bool:
        """
        Override to add metrics tracking

        Args:
            message_id: Redis message ID
            message: Message payload

        Returns:
            True if successfully processed
        """
        import json
        from infrastructure.events.event_types import deserialize_event

        try:
            # Deserialize event
            event_type = message.get("event_type")
            event_data = json.loads(message.get("data", "{}"))
            event = deserialize_event(event_type, event_data)

            # Time the processing
            import time
            start_time = time.time()

            # Process event
            await self.process_event(event)

            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_processing_time(event_type, duration)
            self.metrics.record_event_processed(event_type, 'success')

            # ACK message
            await self.ack(message_id)

            return True

        except Exception as e:
            logger.error(
                f"Failed to process message {message_id}: {e}",
                exc_info=True
            )

            # Record error metrics
            event_type = message.get("event_type", "unknown")
            self.metrics.record_event_processed(event_type, 'error')
            self.metrics.record_error(type(e).__name__)

            # Send to DLQ
            await self._send_to_dlq(message_id, message, str(e))

            return False

    async def start_async(self):
        """
        Start the worker with health check server
        Use this instead of run() for async context
        """
        logger.info(f"Starting {self.worker_name}...")

        # Start health check server
        await self.health_server.start()
        self.health_server.set_healthy(True)

        # Initialize (subclass hook)
        await self.initialize()

        # Mark as ready
        self.health_server.set_ready(True)

        # Run consumer loop
        try:
            await self.run()
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            self.health_server.set_healthy(False)
            raise
        finally:
            await self.shutdown()

    async def initialize(self):
        """
        Initialize worker resources
        Override in subclasses for custom initialization
        """
        pass

    async def shutdown(self):
        """
        Cleanup worker resources
        Override in subclasses for custom cleanup
        """
        logger.info(f"Shutting down {self.worker_name}...")

        # Mark as unhealthy
        self.health_server.set_healthy(False)
        self.health_server.set_ready(False)

        # Stop consumer
        self.stop()

        # Stop health server
        await self.health_server.stop()

        # Close connections
        self.close()

        logger.info(f"{self.worker_name} shutdown complete")

    def run_worker(self):
        """
        Main entry point for running the worker
        Handles asyncio event loop setup
        """
        logger.info("="*60)
        logger.info(f"{self.worker_name} Starting")
        logger.info("="*60)

        try:
            # Run the worker
            asyncio.run(self.start_async())

        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")

        except Exception as e:
            logger.error(f"Worker failed: {e}", exc_info=True)
            sys.exit(1)

        finally:
            logger.info(f"{self.worker_name} exited")


class WorkerConfig:
    """Configuration helper for workers"""

    @staticmethod
    def load_from_yaml(config_path: str) -> dict:
        """Load worker configuration from YAML file"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_redis_url(config: dict) -> str:
        """Extract Redis URL from config"""
        return config.get('event_driven', {}).get(
            'redis_url',
            'redis://localhost:6379'
        )

    @staticmethod
    def get_stream_prefix(config: dict) -> str:
        """Extract stream prefix from config"""
        return config.get('event_driven', {}).get(
            'stream_prefix',
            'genesis'
        )
