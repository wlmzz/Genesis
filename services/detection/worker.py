#!/usr/bin/env python3
"""
Detection Worker
Consumes FrameCapturedEvents, runs YOLO detection, publishes PersonDetectedEvents
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.events import (
    RedisEventProducer,
    FrameCapturedEvent,
    PersonDetectedEvent,
    BaseEvent,
)
from infrastructure.cache import FrameCache
from services.shared import BaseWorker
from services.shared.metrics import DetectionMetrics
from services.detection.detector import PersonDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectionWorker(BaseWorker):
    """
    Worker that performs YOLO person detection on captured frames
    """

    def __init__(
        self,
        redis_url: str,
        stream_prefix: str,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        device: str = "cpu",
        health_port: int = 8080
    ):
        """
        Args:
            redis_url: Redis connection URL
            stream_prefix: Stream name prefix
            model_path: YOLO model path
            conf_threshold: Detection confidence threshold
            iou_threshold: IOU threshold for NMS
            device: Device to run on (cpu, cuda, mps)
            health_port: Health check server port
        """
        super().__init__(
            redis_url=redis_url,
            stream=f"{stream_prefix}:frames",
            consumer_group="detection-workers",
            worker_name="detection-worker",
            health_port=health_port,
            batch_size=10,
            block_ms=1000,
        )

        # Event producer for PersonDetectedEvents
        self.producer = RedisEventProducer(
            redis_url=redis_url,
            stream_prefix=stream_prefix
        )

        # Frame cache
        self.frame_cache = FrameCache(redis_url=redis_url)

        # YOLO detector
        self.detector = PersonDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device
        )

        # Stats
        self.frames_processed = 0
        self.total_detections = 0

    def create_metrics(self) -> DetectionMetrics:
        """Create detection-specific metrics"""
        return DetectionMetrics()

    async def initialize(self):
        """Initialize detector resources"""
        logger.info("Initializing detection worker...")

        # Set detector info in metrics
        detector_info = self.detector.get_model_info()
        self.metrics.set_info(**detector_info)

        logger.info("✓ Detection worker initialized")

    async def process_event(self, event: BaseEvent) -> None:
        """
        Process FrameCapturedEvent

        Args:
            event: FrameCapturedEvent to process
        """
        if not isinstance(event, FrameCapturedEvent):
            logger.warning(f"Unexpected event type: {type(event)}")
            return

        try:
            # Load frame from cache
            frame = self.frame_cache.get(event.frame_id)

            if frame is None:
                logger.warning(f"Frame {event.frame_id} not found in cache")
                self.metrics.record_error("frame_not_found")
                return

            # Run detection
            start_time = time.time()
            detections = self.detector.detect(frame, enable_tracking=True)
            detection_time = time.time() - start_time

            # Update metrics
            self.metrics.detection_confidence.observe(
                sum(d.confidence for d in detections) / len(detections)
                if detections else 0
            )

            # Publish PersonDetectedEvents
            for detection in detections:
                person_event = PersonDetectedEvent(
                    frame_id=event.frame_id,
                    track_id=detection.track_id or 0,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    timestamp=time.time(),
                    metadata={
                        "camera_id": event.camera_id,
                        "detection_time_ms": detection_time * 1000,
                    }
                )

                self.producer.publish(person_event)

                # Update metrics
                self.metrics.detections.labels(camera_id=event.camera_id).inc()

            # Update active tracks gauge
            if detections:
                track_ids = {d.track_id for d in detections if d.track_id}
                self.metrics.active_tracks.labels(camera_id=event.camera_id).set(len(track_ids))

            # Update stats
            self.frames_processed += 1
            self.total_detections += len(detections)

            if self.frames_processed % 100 == 0:
                logger.info(
                    f"Processed {self.frames_processed} frames, "
                    f"{self.total_detections} detections"
                )

        except Exception as e:
            logger.error(f"Error processing frame {event.frame_id}: {e}", exc_info=True)
            self.metrics.record_error("processing_error")
            raise

    async def shutdown(self):
        """Cleanup resources"""
        await super().shutdown()
        self.producer.close()
        self.frame_cache.close()

    def get_status(self) -> dict:
        """Get worker status"""
        status = super().get_status()
        status.update({
            "frames_processed": self.frames_processed,
            "total_detections": self.total_detections,
            "detector": self.detector.get_model_info(),
        })
        return status


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Genesis Detection Worker")
    parser.add_argument(
        "--config",
        default="configs/settings.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on"
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=8080,
        help="Health check server port"
    )

    args = parser.parse_args()

    # Load configuration
    import yaml
    config = yaml.safe_load(open(args.config, 'r'))

    # Get Redis config
    redis_url = config.get('event_driven', {}).get('redis_url', 'redis://localhost:6379')
    stream_prefix = config.get('event_driven', {}).get('stream_prefix', 'genesis')

    # Get detector config
    detector_config = config.get('detector', {})
    conf_threshold = detector_config.get('conf', 0.45)
    iou_threshold = detector_config.get('iou', 0.5)

    # Create worker
    worker = DetectionWorker(
        redis_url=redis_url,
        stream_prefix=stream_prefix,
        model_path=args.model,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=args.device,
        health_port=args.health_port
    )

    # Run worker
    worker.run_worker()


if __name__ == "__main__":
    main()
