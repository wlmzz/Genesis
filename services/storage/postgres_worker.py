#!/usr/bin/env python3
"""
Storage Worker (PostgreSQL version)
Consumes metrics, face, and session events and persists to PostgreSQL with pgvector
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.events import (
    MetricsAggregatedEvent,
    FaceRecognizedEvent,
    SessionEvent,
    AlertTriggeredEvent,
    BaseEvent,
)
from infrastructure.database import PostgresClient
from services.shared import BaseWorker, WorkerMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PostgresStorageWorker(BaseWorker):
    """
    Worker that persists events to PostgreSQL with pgvector
    """

    def __init__(
        self,
        redis_url: str,
        stream_prefix: str,
        pg_host: str = "localhost",
        pg_port: int = 5432,
        pg_database: str = "genesis",
        pg_user: str = "genesis",
        pg_password: str = "genesis_dev_password",
        health_port: int = 8083
    ):
        """
        Args:
            redis_url: Redis connection URL
            stream_prefix: Stream name prefix
            pg_host: PostgreSQL host
            pg_port: PostgreSQL port
            pg_database: PostgreSQL database
            pg_user: PostgreSQL username
            pg_password: PostgreSQL password
            health_port: Health check server port
        """
        # For storage, we consume from metrics stream
        super().__init__(
            redis_url=redis_url,
            stream=f"{stream_prefix}:metrics",
            consumer_group="storage-workers-pg",
            worker_name="postgres-storage-worker",
            health_port=health_port,
            batch_size=50,  # Larger batch for bulk inserts
            block_ms=5000,
        )

        # PostgreSQL client
        self.pg = PostgresClient(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password
        )

        # Stats
        self.metrics_stored = 0
        self.faces_stored = 0
        self.sessions_stored = 0
        self.alerts_stored = 0

    def create_metrics(self) -> WorkerMetrics:
        """Create storage metrics"""
        return WorkerMetrics("postgres_storage_worker")

    async def initialize(self):
        """Initialize PostgreSQL connection"""
        logger.info("Initializing PostgreSQL storage worker...")

        # Connect to PostgreSQL
        await self.pg.connect()

        # Verify connection
        if await self.pg.health_check():
            logger.info("✓ PostgreSQL connection healthy")

            # Get database stats
            stats = await self.pg.get_database_stats()
            logger.info(f"  Database size: {stats.get('database_size', 'unknown')}")
            logger.info(f"  Total identities: {stats.get('total_identities', 0)}")
            logger.info(f"  Total sessions: {stats.get('total_sessions', 0)}")
        else:
            raise RuntimeError("PostgreSQL health check failed")

        logger.info("✓ PostgreSQL storage worker initialized")

    async def process_event(self, event: BaseEvent) -> None:
        """
        Process storage events

        Args:
            event: Event to store
        """
        try:
            if isinstance(event, MetricsAggregatedEvent):
                await self._store_metrics(event)

            elif isinstance(event, FaceRecognizedEvent):
                await self._store_face_event(event)

            elif isinstance(event, SessionEvent):
                await self._store_session_event(event)

            elif isinstance(event, AlertTriggeredEvent):
                await self._store_alert_event(event)

            else:
                logger.debug(f"Ignoring event type: {type(event)}")
                return

        except Exception as e:
            logger.error(f"Error storing event: {e}", exc_info=True)
            self.metrics.record_error("storage_error")
            raise

    async def _store_metrics(self, event: MetricsAggregatedEvent):
        """Store metrics event in TimescaleDB hypertable"""
        try:
            metrics = event.metrics

            success = await self.pg.insert_metrics(
                timestamp=datetime.fromtimestamp(event.timestamp),
                camera_id=event.camera_id or 'unknown',
                people_total=metrics.get('people_total', 0),
                people_by_zone=metrics.get('people_by_zone', {}),
                queue_len=metrics.get('queue_len', 0),
                avg_wait_sec=metrics.get('avg_wait_sec', 0.0),
                new_faces=metrics.get('new_faces', 0),
                recognized_faces=metrics.get('recognized_faces', 0),
                alerts_triggered=metrics.get('alerts_triggered', 0),
                metadata=metrics
            )

            if success:
                self.metrics_stored += 1

                if self.metrics_stored % 100 == 0:
                    logger.info(f"Stored {self.metrics_stored} metrics records")

        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
            raise

    async def _store_face_event(self, event: FaceRecognizedEvent):
        """Store face recognition event with embedding"""
        try:
            camera_id = event.metadata.get("camera_id") if event.metadata else None

            # Store in face_embeddings table
            embedding_id = await self.pg.insert_face_embedding(
                embedding=event.embedding,
                timestamp=datetime.fromtimestamp(event.timestamp),
                person_id=event.person_id,
                confidence=event.confidence,
                camera_id=camera_id,
                zone_name=event.current_zone,
                is_new_face=event.is_new_face,
                metadata=event.metadata or {}
            )

            if embedding_id:
                self.faces_stored += 1

                # If this is a new face with high confidence, consider registering as identity
                if event.is_new_face and event.confidence > 0.8 and event.person_id:
                    await self.pg.insert_identity(
                        person_id=event.person_id,
                        embedding=event.embedding,
                        metadata={"auto_registered": True}
                    )

                # Insert identity event
                if event.person_id:
                    await self.pg.insert_identity_event(
                        person_id=event.person_id,
                        event_type="face_recognized",
                        timestamp=datetime.fromtimestamp(event.timestamp),
                        zone_name=event.current_zone,
                        camera_id=camera_id,
                        confidence=event.confidence
                    )

                if self.faces_stored % 50 == 0:
                    logger.info(f"Stored {self.faces_stored} face events")

        except Exception as e:
            logger.error(f"Error storing face event: {e}")
            raise

    async def _store_session_event(self, event: SessionEvent):
        """Store session event"""
        try:
            if event.event_type == "session_start":
                # Insert new session
                session_id = await self.pg.insert_session(
                    person_id=event.person_id,
                    camera_id=event.camera_id or 'unknown',
                    start_time=datetime.fromtimestamp(event.timestamp),
                    metadata=event.metadata or {}
                )

                if session_id:
                    self.sessions_stored += 1

            elif event.event_type == "session_end":
                # End existing session
                if event.session_id:
                    await self.pg.end_session(
                        session_id=event.session_id,
                        end_time=datetime.fromtimestamp(event.timestamp),
                        zones_visited=event.zones_visited
                    )

        except Exception as e:
            logger.error(f"Error storing session event: {e}")
            raise

    async def _store_alert_event(self, event: AlertTriggeredEvent):
        """Store alert event"""
        try:
            alert_id = await self.pg.insert_alert(
                alert_type=event.alert_type,
                severity=event.severity,
                message=event.message,
                timestamp=datetime.fromtimestamp(event.timestamp),
                camera_id=event.camera_id,
                person_id=event.person_id,
                context=event.context
            )

            if alert_id:
                self.alerts_stored += 1

                if self.alerts_stored % 20 == 0:
                    logger.info(f"Stored {self.alerts_stored} alerts")

        except Exception as e:
            logger.error(f"Error storing alert: {e}")
            raise

    async def shutdown(self):
        """Cleanup resources"""
        if self.pg:
            await self.pg.close()
            logger.info("PostgreSQL connection closed")

        await super().shutdown()

    def get_status(self) -> dict:
        """Get worker status"""
        status = super().get_status()

        # Get database stats
        status.update({
            "metrics_stored_session": self.metrics_stored,
            "faces_stored_session": self.faces_stored,
            "sessions_stored_session": self.sessions_stored,
            "alerts_stored_session": self.alerts_stored,
        })

        return status


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Genesis PostgreSQL Storage Worker")
    parser.add_argument(
        "--config",
        default="configs/settings.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--pg-host",
        default="localhost",
        help="PostgreSQL host"
    )
    parser.add_argument(
        "--pg-port",
        type=int,
        default=5432,
        help="PostgreSQL port"
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=8083,
        help="Health check server port"
    )

    args = parser.parse_args()

    # Load configuration
    import yaml
    config = yaml.safe_load(open(args.config, 'r'))

    # Get Redis config
    redis_url = config.get('event_driven', {}).get('redis_url', 'redis://localhost:6379')
    stream_prefix = config.get('event_driven', {}).get('stream_prefix', 'genesis')

    # Get PostgreSQL config (from settings.yaml or args)
    pg_config = config.get('database', {}).get('postgres', {})

    # Create worker
    worker = PostgresStorageWorker(
        redis_url=redis_url,
        stream_prefix=stream_prefix,
        pg_host=pg_config.get('host', args.pg_host),
        pg_port=pg_config.get('port', args.pg_port),
        pg_database=pg_config.get('database', 'genesis'),
        pg_user=pg_config.get('user', 'genesis'),
        pg_password=pg_config.get('password', 'genesis_dev_password'),
        health_port=args.health_port
    )

    # Run worker
    worker.run_worker()


if __name__ == "__main__":
    main()
