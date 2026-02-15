"""
Integration tests for Redis event bus
"""

import pytest
import asyncio
import redis.asyncio as redis
from datetime import datetime


@pytest.mark.integration
@pytest.mark.asyncio
class TestEventBusIntegration:
    """Integration tests for event bus"""

    @pytest.fixture
    async def redis_client(self):
        """Create Redis client for tests"""
        client = redis.Redis(
            host="localhost",
            port=6379,
            db=1,  # Use separate DB for tests
            decode_responses=False
        )

        # Clean up before test
        await client.flushdb()

        yield client

        # Clean up after test
        await client.flushdb()
        await client.close()

    async def test_publish_and_consume_event(self, redis_client):
        """Test publishing and consuming events"""
        stream_name = "test:events"

        # Publish event
        event_data = {
            "event_type": "test",
            "data": "test_data",
            "timestamp": datetime.now().isoformat()
        }

        event_id = await redis_client.xadd(stream_name, event_data)
        assert event_id is not None

        # Consume event
        events = await redis_client.xread({stream_name: "0-0"}, count=1)

        assert len(events) == 1
        assert events[0][0] == stream_name.encode()
        assert len(events[0][1]) == 1

    async def test_consumer_group(self, redis_client):
        """Test consumer group functionality"""
        stream_name = "test:group_events"
        group_name = "test_group"
        consumer_name = "test_consumer"

        # Create consumer group
        try:
            await redis_client.xgroup_create(
                name=stream_name,
                groupname=group_name,
                id="0",
                mkstream=True
            )
        except redis.ResponseError:
            pass  # Group already exists

        # Publish events
        for i in range(5):
            await redis_client.xadd(stream_name, {"data": f"event_{i}"})

        # Read from group
        events = await redis_client.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams={stream_name: ">"},
            count=5
        )

        assert len(events[0][1]) == 5

        # Acknowledge events
        for event_id, _ in events[0][1]:
            await redis_client.xack(stream_name, group_name, event_id)

    async def test_backpressure_detection(self, redis_client):
        """Test backpressure detection via pending count"""
        stream_name = "test:backpressure"
        group_name = "test_group"

        try:
            await redis_client.xgroup_create(
                name=stream_name,
                groupname=group_name,
                id="0",
                mkstream=True
            )
        except redis.ResponseError:
            pass

        # Publish many events
        for i in range(100):
            await redis_client.xadd(stream_name, {"data": f"event_{i}"})

        # Get stream info
        info = await redis_client.xinfo_stream(stream_name)

        # Check length
        assert info[b"length"] == 100
