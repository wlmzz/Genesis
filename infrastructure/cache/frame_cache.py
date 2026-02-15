"""
Frame cache for sharing frames between workers via Redis
Uses pickle for serialization and LZ4 compression for efficiency
"""
import pickle
import logging
import redis
import numpy as np
from typing import Optional
import lz4.frame

logger = logging.getLogger(__name__)


class FrameCache:
    """
    Redis-based cache for sharing camera frames between workers
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl_seconds: int = 60,
        compress: bool = True,
        redis_client: Optional[redis.Redis] = None
    ):
        """
        Args:
            redis_url: Redis connection URL
            ttl_seconds: Time-to-live for cached frames
            compress: Whether to compress frames with LZ4
            redis_client: Optional existing Redis client
        """
        self.ttl_seconds = ttl_seconds
        self.compress = compress

        # Connect to Redis (binary mode for frame data)
        if redis_client:
            self.client = redis_client
        else:
            self.client = redis.from_url(
                redis_url,
                decode_responses=False,  # Binary mode
                socket_connect_timeout=5,
                socket_keepalive=True
            )

        try:
            self.client.ping()
            logger.info(f"Frame cache connected to Redis at {redis_url}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _serialize_frame(self, frame: np.ndarray) -> bytes:
        """
        Serialize frame to bytes with optional compression

        Args:
            frame: NumPy array frame

        Returns:
            Serialized frame bytes
        """
        # Pickle the frame
        frame_bytes = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)

        # Compress if enabled
        if self.compress:
            frame_bytes = lz4.frame.compress(frame_bytes)

        return frame_bytes

    def _deserialize_frame(self, frame_bytes: bytes) -> np.ndarray:
        """
        Deserialize frame from bytes

        Args:
            frame_bytes: Serialized frame bytes

        Returns:
            NumPy array frame
        """
        # Decompress if enabled
        if self.compress:
            frame_bytes = lz4.frame.decompress(frame_bytes)

        # Unpickle the frame
        frame = pickle.loads(frame_bytes)

        return frame

    def set(
        self,
        frame_id: str,
        frame: np.ndarray,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store frame in cache

        Args:
            frame_id: Unique frame identifier
            frame: NumPy array frame
            ttl: Optional TTL override (seconds)

        Returns:
            True if successful, False otherwise
        """
        try:
            frame_bytes = self._serialize_frame(frame)
            cache_key = f"frame:{frame_id}"

            # Store with TTL
            ttl = ttl or self.ttl_seconds
            self.client.setex(cache_key, ttl, frame_bytes)

            logger.debug(
                f"Cached frame {frame_id} "
                f"(size: {len(frame_bytes) / 1024:.1f} KB, ttl: {ttl}s)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to cache frame {frame_id}: {e}")
            return False

    def get(self, frame_id: str) -> Optional[np.ndarray]:
        """
        Retrieve frame from cache

        Args:
            frame_id: Frame identifier

        Returns:
            NumPy array frame if found, None otherwise
        """
        try:
            cache_key = f"frame:{frame_id}"
            frame_bytes = self.client.get(cache_key)

            if frame_bytes is None:
                logger.debug(f"Frame {frame_id} not found in cache")
                return None

            frame = self._deserialize_frame(frame_bytes)
            logger.debug(f"Retrieved frame {frame_id} from cache")
            return frame

        except Exception as e:
            logger.error(f"Failed to retrieve frame {frame_id}: {e}")
            return None

    def delete(self, frame_id: str) -> bool:
        """
        Delete frame from cache

        Args:
            frame_id: Frame identifier

        Returns:
            True if deleted, False if not found or error
        """
        try:
            cache_key = f"frame:{frame_id}"
            deleted = self.client.delete(cache_key)
            return deleted > 0
        except Exception as e:
            logger.error(f"Failed to delete frame {frame_id}: {e}")
            return False

    def exists(self, frame_id: str) -> bool:
        """
        Check if frame exists in cache

        Args:
            frame_id: Frame identifier

        Returns:
            True if exists, False otherwise
        """
        try:
            cache_key = f"frame:{frame_id}"
            return self.client.exists(cache_key) > 0
        except Exception as e:
            logger.error(f"Failed to check frame existence {frame_id}: {e}")
            return False

    def get_cache_size(self) -> int:
        """
        Get approximate cache size in bytes

        Returns:
            Total size of all cached frames in bytes
        """
        try:
            keys = self.client.keys("frame:*")
            total_size = sum(
                self.client.memory_usage(key) or 0
                for key in keys
            )
            return total_size
        except Exception as e:
            logger.error(f"Failed to get cache size: {e}")
            return 0

    def clear_all(self) -> int:
        """
        Clear all cached frames

        Returns:
            Number of frames deleted
        """
        try:
            keys = self.client.keys("frame:*")
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Cleared {deleted} frames from cache")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0

    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            logger.info("Frame cache connection closed")
