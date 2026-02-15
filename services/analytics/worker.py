#!/usr/bin/env python3
"""
Analytics Worker
Consumes detection/face events, generates metrics, heatmaps, anomalies
"""
import argparse
import logging
import sys
import time
import asyncio
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.events import (
    RedisEventProducer,
    PersonDetectedEvent,
    FaceRecognizedEvent,
    ZoneEvent,
    MetricsAggregatedEvent,
    AlertTriggeredEvent,
    BaseEvent,
)
from services.shared import BaseWorker
from services.shared.metrics import AnalyticsMetrics
from core.analytics import GenesisAnalytics
from core.advanced_analytics import AnomalyDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalyticsWorker(BaseWorker):
    """
    Worker that aggregates metrics and detects anomalies
    Consumes multiple event streams
    """

    def __init__(
        self,
        redis_url: str,
        stream_prefix: str,
        zone_names: list = None,
        aggregation_interval: int = 60,
        health_port: int = 8082
    ):
        """
        Args:
            redis_url: Redis connection URL
            stream_prefix: Stream name prefix
            zone_names: List of zone names
            aggregation_interval: Seconds between metric aggregation
            health_port: Health check server port
        """
        # For analytics, we consume from multiple streams
        # Start with persons stream, will add others
        super().__init__(
            redis_url=redis_url,
            stream=f"{stream_prefix}:persons",
            consumer_group="analytics-workers",
            worker_name="analytics-worker",
            health_port=health_port,
            batch_size=20,
            block_ms=1000,
        )

        # Event producer for metrics and alerts
        self.producer = RedisEventProducer(
            redis_url=redis_url,
            stream_prefix=stream_prefix
        )

        # Analytics engine
        self.analytics = GenesisAnalytics(
            zone_names=zone_names or [],
            queue_zone="queue_area"
        )

        # Anomaly detector
        self.anomaly_detector = AnomalyDetector(sensitivity=2.0)

        # Tracking state
        self.active_tracks = {}  # track_id -> last_seen_time
        self.track_positions = defaultdict(list)  # track_id -> [(x,y), ...]

        # Aggregation
        self.aggregation_interval = aggregation_interval
        self.last_aggregation = time.time()

        # Stats
        self.events_processed = 0

    def create_metrics(self) -> AnalyticsMetrics:
        """Create analytics metrics"""
        return AnalyticsMetrics()

    async def initialize(self):
        """Initialize analytics worker"""
        logger.info("Initializing analytics worker...")

        # Start periodic aggregation task
        asyncio.create_task(self.periodic_aggregation())

        logger.info("✓ Analytics worker initialized")

    async def process_event(self, event: BaseEvent) -> None:
        """
        Process analytics events

        Args:
            event: Event to process (PersonDetected, FaceRecognized, Zone)
        """
        try:
            if isinstance(event, PersonDetectedEvent):
                await self._process_person_detected(event)

            elif isinstance(event, FaceRecognizedEvent):
                await self._process_face_recognized(event)

            elif isinstance(event, ZoneEvent):
                await self._process_zone_event(event)

            else:
                logger.debug(f"Ignoring event type: {type(event)}")
                return

            self.events_processed += 1

        except Exception as e:
            logger.error(f"Error processing analytics event: {e}", exc_info=True)
            self.metrics.record_error("processing_error")
            raise

    async def _process_person_detected(self, event: PersonDetectedEvent):
        """Process person detection for analytics"""
        # Update active tracks
        self.active_tracks[event.track_id] = time.time()

        # Update position history
        x1, y1, x2, y2 = event.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        self.track_positions[event.track_id].append((cx, cy))

        # Keep last 100 positions
        if len(self.track_positions[event.track_id]) > 100:
            self.track_positions[event.track_id].pop(0)

        # Update people count metric
        camera_id = event.metadata.get("camera_id") if event.metadata else "unknown"
        self.metrics.people_count.labels(camera_id=camera_id).set(len(self.active_tracks))

    async def _process_face_recognized(self, event: FaceRecognizedEvent):
        """Process face recognition for analytics"""
        # Could update identity-based analytics here
        pass

    async def _process_zone_event(self, event: ZoneEvent):
        """Process zone entry/exit events"""
        # Update zone occupancy
        # This would require maintaining zone state
        pass

    async def periodic_aggregation(self):
        """Periodically aggregate and publish metrics"""
        while self.running:
            try:
                await asyncio.sleep(self.aggregation_interval)

                # Cleanup stale tracks (>30 seconds old)
                now = time.time()
                stale_threshold = 30.0
                stale_tracks = [
                    tid for tid, last_seen in self.active_tracks.items()
                    if (now - last_seen) > stale_threshold
                ]

                for tid in stale_tracks:
                    del self.active_tracks[tid]
                    if tid in self.track_positions:
                        del self.track_positions[tid]

                # Calculate current metrics
                current_people = len(self.active_tracks)

                # Check for anomalies
                is_anomaly, z_score = self.anomaly_detector.is_anomaly(current_people)

                if is_anomaly:
                    # Publish alert
                    alert = AlertTriggeredEvent(
                        alert_type="anomaly_crowd",
                        severity="warning",
                        message=f"Unusual crowd detected: {current_people} people (z-score: {z_score:.2f})",
                        context={
                            "people_count": current_people,
                            "z_score": z_score,
                        },
                        timestamp=now,
                    )
                    self.producer.publish(alert)

                    # Update metrics
                    self.metrics.anomalies_detected.labels(
                        anomaly_type="crowd"
                    ).inc()

                # Publish aggregated metrics
                metrics_event = MetricsAggregatedEvent(
                    interval_start=self.last_aggregation,
                    interval_end=now,
                    metrics={
                        "people_total": current_people,
                        "active_tracks": len(self.active_tracks),
                        "events_processed": self.events_processed,
                    },
                    timestamp=now,
                )

                self.producer.publish(metrics_event)

                self.last_aggregation = now

                if self.events_processed % 1000 == 0:
                    logger.info(
                        f"Analytics: {self.events_processed} events processed, "
                        f"{current_people} people tracked"
                    )

            except Exception as e:
                logger.error(f"Error in periodic aggregation: {e}", exc_info=True)

    async def shutdown(self):
        """Cleanup resources"""
        await super().shutdown()
        self.producer.close()

    def get_status(self) -> dict:
        """Get worker status"""
        status = super().get_status()
        status.update({
            "events_processed": self.events_processed,
            "active_tracks": len(self.active_tracks),
            "current_people": len(self.active_tracks),
        })
        return status


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Genesis Analytics Worker")
    parser.add_argument(
        "--config",
        default="configs/settings.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Aggregation interval in seconds"
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=8082,
        help="Health check server port"
    )

    args = parser.parse_args()

    # Load configuration
    import yaml
    config = yaml.safe_load(open(args.config, 'r'))

    # Get Redis config
    redis_url = config.get('event_driven', {}).get('redis_url', 'redis://localhost:6379')
    stream_prefix = config.get('event_driven', {}).get('stream_prefix', 'genesis')

    # Get zone names
    try:
        from core.zones import load_zones
        zones_path = "configs/zones.json"
        zones = load_zones(zones_path)
        zone_names = list(zones.keys())
    except:
        zone_names = []

    # Create worker
    worker = AnalyticsWorker(
        redis_url=redis_url,
        stream_prefix=stream_prefix,
        zone_names=zone_names,
        aggregation_interval=args.interval,
        health_port=args.health_port
    )

    # Run worker
    worker.run_worker()


if __name__ == "__main__":
    main()
