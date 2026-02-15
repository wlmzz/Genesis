#!/usr/bin/env python3
"""
Test script for Genesis worker services
Publishes test events and verifies workers process them correctly
"""
import sys
import time
import asyncio
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.events import RedisEventProducer, FrameCapturedEvent, PersonDetectedEvent
from infrastructure.cache import FrameCache
import numpy as np


async def test_workers():
    """Test all worker services"""
    print("="*60)
    print("Genesis Workers Test")
    print("="*60)

    # Initialize producer and cache
    print("\n1. Initializing event producer...")
    producer = RedisEventProducer(
        redis_url="redis://localhost:6379",
        stream_prefix="genesis-test"
    )
    cache = FrameCache(redis_url="redis://localhost:6379")
    print("   ✓ Producer and cache ready")

    # Check worker health
    print("\n2. Checking worker health endpoints...")
    workers = {
        "detection": "http://localhost:8080",
        "face-recognition": "http://localhost:8081",
        "analytics": "http://localhost:8082",
        "storage": "http://localhost:8083",
    }

    healthy_workers = []
    for name, url in workers.items():
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print(f"   ✓ {name} worker: healthy")
                healthy_workers.append(name)
            else:
                print(f"   ✗ {name} worker: unhealthy (status {response.status_code})")
        except requests.RequestException as e:
            print(f"   ⚠ {name} worker: not reachable ({e})")

    if not healthy_workers:
        print("\n   ⚠ No workers are healthy. Start workers first:")
        print("      python services/detection/worker.py &")
        print("      python services/face_recognition/worker.py &")
        print("      python services/analytics/worker.py &")
        print("      python services/storage/worker.py &")
        return False

    # Create test frame
    print("\n3. Publishing test events...")
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame_id = f"test_frame_{int(time.time() * 1000)}"

    # Cache frame
    cache.set(frame_id, test_frame)
    print(f"   ✓ Cached test frame: {frame_id}")

    # Publish FrameCapturedEvent
    frame_event = FrameCapturedEvent(
        camera_id="test_cam_0",
        frame_id=frame_id,
        timestamp=time.time(),
        frame_shape=test_frame.shape,
    )
    producer.publish(frame_event)
    print("   ✓ Published FrameCapturedEvent")

    # Publish PersonDetectedEvent
    person_event = PersonDetectedEvent(
        frame_id=frame_id,
        track_id=1,
        bbox=(100.0, 150.0, 300.0, 450.0),
        confidence=0.95,
        timestamp=time.time(),
        metadata={"camera_id": "test_cam_0"}
    )
    producer.publish(person_event)
    print("   ✓ Published PersonDetectedEvent")

    # Wait for processing
    print("\n4. Waiting for workers to process events (5 seconds)...")
    await asyncio.sleep(5)

    # Check worker status
    print("\n5. Checking worker status...")
    for name, url in workers.items():
        if name not in healthy_workers:
            continue

        try:
            response = requests.get(f"{url}/status", timeout=2)
            if response.status_code == 200:
                status = response.json()
                print(f"\n   {name.upper()} Worker:")
                for key, value in status.items():
                    if key not in ['service', 'worker_name']:
                        print(f"     - {key}: {value}")
        except requests.RequestException as e:
            print(f"   ⚠ {name}: Could not get status ({e})")

    # Cleanup
    producer.close()
    cache.close()

    print("\n" + "="*60)
    print("✓ Test completed")
    print(f"✓ {len(healthy_workers)}/{len(workers)} workers are operational")
    print("="*60)

    return True


def main():
    """Main test entry point"""
    try:
        result = asyncio.run(test_workers())
        return 0 if result else 1
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
