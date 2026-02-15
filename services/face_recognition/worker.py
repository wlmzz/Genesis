#!/usr/bin/env python3
"""
Face Recognition Worker
Consumes PersonDetectedEvents, extracts face embeddings, publishes FaceRecognizedEvents
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.events import (
    RedisEventProducer,
    PersonDetectedEvent,
    FaceRecognizedEvent,
    BaseEvent,
)
from infrastructure.cache import FrameCache
from services.shared import BaseWorker
from services.shared.metrics import FaceRecognitionMetrics
from core.face_recognition import FaceRecognizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceRecognitionWorker(BaseWorker):
    """
    Worker that performs face recognition on detected persons
    """

    def __init__(
        self,
        redis_url: str,
        stream_prefix: str,
        faces_dir: str = "data/faces",
        model: str = "Facenet512",
        distance_metric: str = "cosine",
        threshold: float = 0.6,
        min_face_size: int = 80,
        health_port: int = 8081
    ):
        """
        Args:
            redis_url: Redis connection URL
            stream_prefix: Stream name prefix
            faces_dir: Directory containing known faces
            model: Face recognition model
            distance_metric: Distance metric for matching
            threshold: Recognition threshold
            min_face_size: Minimum face size in pixels
            health_port: Health check server port
        """
        super().__init__(
            redis_url=redis_url,
            stream=f"{stream_prefix}:persons",
            consumer_group="face-workers",
            worker_name="face-recognition-worker",
            health_port=health_port,
            batch_size=5,  # Smaller batch for face processing
            block_ms=2000,
        )

        # Event producer for FaceRecognizedEvents
        self.producer = RedisEventProducer(
            redis_url=redis_url,
            stream_prefix=stream_prefix
        )

        # Frame cache
        self.frame_cache = FrameCache(redis_url=redis_url)

        # Face recognizer
        self.recognizer = FaceRecognizer(
            faces_dir=faces_dir,
            model=model,
            distance_metric=distance_metric,
            threshold=threshold
        )

        self.min_face_size = min_face_size

        # Stats
        self.faces_processed = 0
        self.faces_recognized = 0
        self.new_faces = 0

    def create_metrics(self) -> FaceRecognitionMetrics:
        """Create face recognition metrics"""
        return FaceRecognitionMetrics()

    async def initialize(self):
        """Initialize face recognition resources"""
        logger.info("Initializing face recognition worker...")

        # Set recognizer info in metrics
        self.metrics.set_info(
            model=self.recognizer.model_name,
            distance_metric=self.recognizer.distance_metric,
            threshold=self.recognizer.threshold,
            known_faces=len(self.recognizer.known_embeddings)
        )

        logger.info(
            f"✓ Face recognition worker initialized "
            f"({len(self.recognizer.known_embeddings)} known faces)"
        )

    async def process_event(self, event: BaseEvent) -> None:
        """
        Process PersonDetectedEvent

        Args:
            event: PersonDetectedEvent to process
        """
        if not isinstance(event, PersonDetectedEvent):
            logger.warning(f"Unexpected event type: {type(event)}")
            return

        try:
            # Load frame from cache
            frame = self.frame_cache.get(event.frame_id)

            if frame is None:
                logger.warning(f"Frame {event.frame_id} not found in cache")
                self.metrics.record_error("frame_not_found")
                return

            # Extract face ROI
            x1, y1, x2, y2 = event.bbox
            face_img = frame[int(y1):int(y2), int(x1):int(x2)]

            # Check minimum face size
            if face_img.shape[0] < self.min_face_size or face_img.shape[1] < self.min_face_size:
                logger.debug(f"Face too small: {face_img.shape}")
                self.metrics.record_error("face_too_small")
                return

            # Extract embedding and recognize
            start_embed = time.time()
            embedding = self.recognizer.extract_embedding(face_img)

            if embedding is None:
                logger.debug("Failed to extract embedding")
                self.metrics.record_error("embedding_failed")
                return

            embedding_time = time.time() - start_embed
            self.metrics.embedding_time.observe(embedding_time)

            # Search for match
            start_search = time.time()
            person_id, confidence = self.recognizer.recognize_face(face_img)
            search_time = time.time() - start_search
            self.metrics.database_search_time.observe(search_time)

            is_new_face = person_id is None

            # Publish FaceRecognizedEvent
            face_event = FaceRecognizedEvent(
                track_id=event.track_id,
                person_id=person_id,
                embedding=embedding.tolist() if embedding is not None else [0.0] * 512,
                confidence=confidence if person_id else 0.0,
                is_new_face=is_new_face,
                timestamp=time.time(),
                current_zone=event.metadata.get("zone") if event.metadata else None,
                metadata={
                    "camera_id": event.metadata.get("camera_id") if event.metadata else None,
                    "embedding_time_ms": embedding_time * 1000,
                    "search_time_ms": search_time * 1000,
                }
            )

            self.producer.publish(face_event)

            # Update metrics
            self.metrics.faces_recognized.labels(
                is_new_face=str(is_new_face)
            ).inc()

            if confidence > 0:
                self.metrics.recognition_confidence.observe(confidence)

            # Update stats
            self.faces_processed += 1
            if person_id:
                self.faces_recognized += 1
            else:
                self.new_faces += 1

            if self.faces_processed % 50 == 0:
                logger.info(
                    f"Processed {self.faces_processed} faces: "
                    f"{self.faces_recognized} recognized, {self.new_faces} new"
                )

        except Exception as e:
            logger.error(f"Error processing person detection: {e}", exc_info=True)
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
            "faces_processed": self.faces_processed,
            "faces_recognized": self.faces_recognized,
            "new_faces": self.new_faces,
            "known_faces": len(self.recognizer.known_embeddings),
        })
        return status


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Genesis Face Recognition Worker")
    parser.add_argument(
        "--config",
        default="configs/settings.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID (None for CPU)"
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=8081,
        help="Health check server port"
    )

    args = parser.parse_args()

    # Set GPU if specified
    if args.gpu is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load configuration
    import yaml
    config = yaml.safe_load(open(args.config, 'r'))

    # Get Redis config
    redis_url = config.get('event_driven', {}).get('redis_url', 'redis://localhost:6379')
    stream_prefix = config.get('event_driven', {}).get('stream_prefix', 'genesis')

    # Get face recognition config
    face_config = config.get('face_recognition', {})
    identity_config = config.get('identity', {})

    # Create worker
    worker = FaceRecognitionWorker(
        redis_url=redis_url,
        stream_prefix=stream_prefix,
        faces_dir=identity_config.get('faces_dir', 'data/faces'),
        model=face_config.get('model', 'Facenet512'),
        distance_metric=face_config.get('distance_metric', 'cosine'),
        threshold=face_config.get('recognition_threshold', 0.6),
        min_face_size=face_config.get('face_size_threshold', 80),
        health_port=args.health_port
    )

    # Run worker
    worker.run_worker()


if __name__ == "__main__":
    main()
