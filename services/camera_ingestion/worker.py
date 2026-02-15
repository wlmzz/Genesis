#!/usr/bin/env python3
"""
Camera Ingestion Worker
Captures frames from multiple cameras and publishes FrameCapturedEvents
"""
import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.events import RedisEventProducer, FrameCapturedEvent, BackpressureError
from infrastructure.cache import FrameCache
from services.shared import HealthServer, WorkerMetrics
from services.camera_ingestion.camera import Camera, CameraConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CameraIngestionWorker:
    """
    Worker service for multi-camera frame ingestion
    Captures frames and publishes FrameCapturedEvents to Redis Streams
    """

    def __init__(
        self,
        camera_configs: List[CameraConfig],
        redis_url: str = "redis://localhost:6379",
        stream_prefix: str = "genesis",
        health_port: int = 8080
    ):
        """
        Args:
            camera_configs: List of camera configurations
            redis_url: Redis connection URL
            stream_prefix: Stream name prefix
            health_port: Health check server port
        """
        self.camera_configs = camera_configs
        self.cameras: dict[str, Camera] = {}

        # Event producer
        self.producer = RedisEventProducer(
            redis_url=redis_url,
            stream_prefix=stream_prefix
        )

        # Frame cache
        self.frame_cache = FrameCache(redis_url=redis_url)

        # Health server
        self.health_server = HealthServer(
            port=health_port,
            worker_name="camera-ingestion-worker",
            get_status=self.get_status
        )

        # Metrics
        self.metrics = WorkerMetrics("camera_ingestion_worker")
        self.metrics.set_info(
            worker_type="camera-ingestion",
            num_cameras=len(camera_configs)
        )

        # Running flag
        self.running = False

        # Stats
        self.total_frames_captured = 0
        self.total_frames_published = 0

    async def initialize(self):
        """Initialize cameras and health server"""
        logger.info("Initializing camera ingestion worker...")

        # Start health server
        await self.health_server.start()
        self.health_server.set_healthy(True)

        # Initialize cameras
        for config in self.camera_configs:
            if not config.enabled:
                logger.info(f"Skipping disabled camera: {config.camera_id}")
                continue

            try:
                camera = Camera(config)
                self.cameras[config.camera_id] = camera
                logger.info(f"✓ Camera initialized: {config.camera_id}")

            except Exception as e:
                logger.error(f"✗ Failed to initialize camera {config.camera_id}: {e}")

        if not self.cameras:
            raise RuntimeError("No cameras initialized successfully")

        self.health_server.set_ready(True)
        logger.info(f"✓ {len(self.cameras)} cameras ready")

    async def process_camera(self, camera_id: str, camera: Camera):
        """
        Process frames from a single camera

        Args:
            camera_id: Camera identifier
            camera: Camera instance
        """
        logger.info(f"Starting frame capture for {camera_id}")

        while self.running:
            try:
                # Capture frame
                frame = camera.capture()

                if frame is None:
                    # No frame ready (FPS throttling or error)
                    await asyncio.sleep(0.01)
                    continue

                # Generate frame ID
                frame_id = f"{camera_id}_{int(time.time() * 1000)}_{camera.frame_count}"

                # Cache frame
                self.frame_cache.set(frame_id, frame)

                # Publish event
                try:
                    event = FrameCapturedEvent(
                        camera_id=camera_id,
                        frame_id=frame_id,
                        timestamp=time.time(),
                        frame_shape=tuple(frame.shape),
                    )

                    self.producer.publish(event)

                    # Update stats
                    self.total_frames_captured += 1
                    self.total_frames_published += 1

                    # Update metrics
                    self.metrics.record_event_processed("FrameCapturedEvent", "success")

                    if self.total_frames_published % 100 == 0:
                        logger.info(
                            f"{camera_id}: {self.total_frames_published} frames published"
                        )

                except BackpressureError as e:
                    logger.warning(f"Backpressure detected for {camera_id}: {e}")
                    self.metrics.record_error("backpressure")
                    await asyncio.sleep(0.5)  # Wait before retry

                except Exception as e:
                    logger.error(f"Failed to publish frame from {camera_id}: {e}")
                    self.metrics.record_error("publish_failed")

                # Small sleep to yield control
                await asyncio.sleep(0.001)

            except Exception as e:
                logger.error(f"Error processing camera {camera_id}: {e}")
                self.metrics.record_error("capture_error")
                await asyncio.sleep(1.0)

        logger.info(f"Stopped frame capture for {camera_id}")

    async def run(self):
        """Main worker loop - capture from all cameras concurrently"""
        self.running = True
        logger.info("Camera ingestion worker started")

        # Create tasks for all cameras
        tasks = [
            asyncio.create_task(self.process_camera(cam_id, camera))
            for cam_id, camera in self.cameras.items()
        ]

        try:
            # Run all camera tasks concurrently
            await asyncio.gather(*tasks)

        except asyncio.CancelledError:
            logger.info("Camera tasks cancelled")

        except Exception as e:
            logger.error(f"Error in camera ingestion: {e}", exc_info=True)
            raise

    def stop(self):
        """Stop the worker gracefully"""
        logger.info("Stopping camera ingestion worker...")
        self.running = False

    async def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down camera ingestion worker...")

        # Mark unhealthy
        self.health_server.set_healthy(False)
        self.health_server.set_ready(False)

        # Stop all cameras
        for camera in self.cameras.values():
            camera.release()

        # Close connections
        self.producer.close()
        self.frame_cache.close()

        # Stop health server
        await self.health_server.stop()

        logger.info("Camera ingestion worker shutdown complete")

    def get_status(self) -> dict:
        """Get worker status"""
        camera_status = {
            cam_id: camera.get_info()
            for cam_id, camera in self.cameras.items()
        }

        return {
            "running": self.running,
            "total_cameras": len(self.cameras),
            "active_cameras": sum(1 for c in self.cameras.values() if c.is_open),
            "total_frames_captured": self.total_frames_captured,
            "total_frames_published": self.total_frames_published,
            "cameras": camera_status,
        }


async def main_async(args):
    """Async main function"""
    # Load configuration
    import yaml
    config = yaml.safe_load(open(args.config, 'r'))

    # Parse camera configs
    camera_configs = []

    if args.cam is not None:
        # Single webcam mode
        camera_configs.append(CameraConfig(
            camera_id=f"cam_{args.cam}",
            source=args.cam,
            target_fps=config.get('video', {}).get('process_fps', 12),
            resize_width=config.get('video', {}).get('resize_width', 960),
        ))
    else:
        # Multi-camera mode from config
        cameras_config = config.get('cameras', [])
        for cam_cfg in cameras_config:
            camera_configs.append(CameraConfig(
                camera_id=cam_cfg['id'],
                source=cam_cfg['source'],
                target_fps=cam_cfg.get('fps', 12),
                resize_width=cam_cfg.get('resize_width', 960),
                enabled=cam_cfg.get('enabled', True),
            ))

    if not camera_configs:
        logger.error("No cameras configured!")
        sys.exit(1)

    # Get Redis config
    redis_url = config.get('event_driven', {}).get('redis_url', 'redis://localhost:6379')
    stream_prefix = config.get('event_driven', {}).get('stream_prefix', 'genesis')

    # Create worker
    worker = CameraIngestionWorker(
        camera_configs=camera_configs,
        redis_url=redis_url,
        stream_prefix=stream_prefix,
        health_port=args.health_port
    )

    # Initialize
    await worker.initialize()

    # Run
    try:
        await worker.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await worker.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Genesis Camera Ingestion Worker")
    parser.add_argument(
        "--config",
        default="configs/settings.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=None,
        help="Single webcam ID (overrides config cameras)"
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=8080,
        help="Health check server port"
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Genesis Camera Ingestion Worker")
    logger.info("="*60)

    try:
        asyncio.run(main_async(args))
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
