#!/usr/bin/env python3
"""
Storage Worker
Consumes metrics, face, and session events and persists to PostgreSQL
Note: For Phase 2, this uses SQLite. Will migrate to PostgreSQL in Phase 3.
"""
import argparse
import logging
import sys
import time
import sqlite3
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.events import (
    MetricsAggregatedEvent,
    FaceRecognizedEvent,
    SessionEvent,
    BaseEvent,
)
from services.shared import BaseWorker, WorkerMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StorageWorker(BaseWorker):
    """
    Worker that persists events to database
    Currently uses SQLite (will migrate to PostgreSQL in Phase 3)
    """

    def __init__(
        self,
        redis_url: str,
        stream_prefix: str,
        db_path: str = "data/outputs/events.db",
        health_port: int = 8083
    ):
        """
        Args:
            redis_url: Redis connection URL
            stream_prefix: Stream name prefix
            db_path: SQLite database path
            health_port: Health check server port
        """
        # For storage, we consume from metrics stream
        super().__init__(
            redis_url=redis_url,
            stream=f"{stream_prefix}:metrics",
            consumer_group="storage-workers",
            worker_name="storage-worker",
            health_port=health_port,
            batch_size=50,  # Larger batch for bulk inserts
            block_ms=5000,
        )

        self.db_path = db_path
        self.db_conn: Optional[sqlite3.Connection] = None

        # Stats
        self.metrics_stored = 0
        self.faces_stored = 0
        self.sessions_stored = 0

    def create_metrics(self) -> WorkerMetrics:
        """Create storage metrics"""
        return WorkerMetrics("storage_worker")

    async def initialize(self):
        """Initialize database connection"""
        logger.info("Initializing storage worker...")

        # Create database directory
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)

        # Create tables if they don't exist
        self._create_tables()

        logger.info(f"✓ Storage worker initialized (db: {self.db_path})")

    def _create_tables(self):
        """Create database tables"""
        cursor = self.db_conn.cursor()

        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                interval_start REAL,
                interval_end REAL,
                people_total INTEGER,
                active_tracks INTEGER,
                events_processed INTEGER,
                camera_id TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Create index on timestamp
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
            ON metrics(timestamp DESC)
        """)

        # Face events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                track_id INTEGER,
                person_id TEXT,
                confidence REAL,
                is_new_face INTEGER,
                current_zone TEXT,
                camera_id TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Session events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                session_id TEXT,
                person_id TEXT,
                event_type TEXT,
                duration_seconds REAL,
                camera_id TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        self.db_conn.commit()
        logger.info("✓ Database tables created/verified")

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

            else:
                logger.debug(f"Ignoring event type: {type(event)}")
                return

        except Exception as e:
            logger.error(f"Error storing event: {e}", exc_info=True)
            self.metrics.record_error("storage_error")
            raise

    async def _store_metrics(self, event: MetricsAggregatedEvent):
        """Store metrics event"""
        cursor = self.db_conn.cursor()

        metrics = event.metrics
        cursor.execute("""
            INSERT INTO metrics (
                timestamp, interval_start, interval_end,
                people_total, active_tracks, events_processed,
                camera_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp,
            event.interval_start,
            event.interval_end,
            metrics.get('people_total', 0),
            metrics.get('active_tracks', 0),
            metrics.get('events_processed', 0),
            event.camera_id,
        ))

        self.db_conn.commit()
        self.metrics_stored += 1

        if self.metrics_stored % 100 == 0:
            logger.info(f"Stored {self.metrics_stored} metrics records")

    async def _store_face_event(self, event: FaceRecognizedEvent):
        """Store face recognition event"""
        cursor = self.db_conn.cursor()

        camera_id = event.metadata.get("camera_id") if event.metadata else None

        cursor.execute("""
            INSERT INTO face_events (
                timestamp, track_id, person_id, confidence,
                is_new_face, current_zone, camera_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp,
            event.track_id,
            event.person_id,
            event.confidence,
            1 if event.is_new_face else 0,
            event.current_zone,
            camera_id,
        ))

        self.db_conn.commit()
        self.faces_stored += 1

        if self.faces_stored % 50 == 0:
            logger.info(f"Stored {self.faces_stored} face events")

    async def _store_session_event(self, event: SessionEvent):
        """Store session event"""
        cursor = self.db_conn.cursor()

        cursor.execute("""
            INSERT INTO session_events (
                timestamp, session_id, person_id, event_type,
                duration_seconds, camera_id
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp,
            event.session_id,
            event.person_id,
            event.event_type,
            event.duration_seconds,
            event.camera_id,
        ))

        self.db_conn.commit()
        self.sessions_stored += 1

    async def shutdown(self):
        """Cleanup resources"""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed")

        await super().shutdown()

    def get_status(self) -> dict:
        """Get worker status"""
        status = super().get_status()

        # Get database stats
        if self.db_conn:
            cursor = self.db_conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM metrics")
            total_metrics = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM face_events")
            total_faces = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM session_events")
            total_sessions = cursor.fetchone()[0]

            status.update({
                "db_path": self.db_path,
                "total_metrics_stored": total_metrics,
                "total_faces_stored": total_faces,
                "total_sessions_stored": total_sessions,
            })

        status.update({
            "metrics_stored_session": self.metrics_stored,
            "faces_stored_session": self.faces_stored,
            "sessions_stored_session": self.sessions_stored,
        })

        return status


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Genesis Storage Worker")
    parser.add_argument(
        "--config",
        default="configs/settings.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--db-path",
        default="data/outputs/events.db",
        help="SQLite database path"
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

    # Create worker
    worker = StorageWorker(
        redis_url=redis_url,
        stream_prefix=stream_prefix,
        db_path=args.db_path,
        health_port=args.health_port
    )

    # Run worker
    worker.run_worker()


if __name__ == "__main__":
    main()
