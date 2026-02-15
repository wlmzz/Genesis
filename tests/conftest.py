"""
Pytest configuration and shared fixtures for Genesis tests
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator
import numpy as np
import cv2


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Temporary Resources
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_video_path(temp_dir: Path) -> Path:
    """Create temporary video file path"""
    return temp_dir / "test_video.mp4"


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def mock_frame() -> np.ndarray:
    """Create mock video frame (640x480 RGB)"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_face_image() -> np.ndarray:
    """Create mock face image (112x112 RGB)"""
    return np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)


@pytest.fixture
def mock_detection_result():
    """Mock YOLO detection result"""
    from dataclasses import dataclass

    @dataclass
    class MockBox:
        xyxy: np.ndarray  # [x1, y1, x2, y2]
        conf: float
        cls: int
        id: int = None

    @dataclass
    class MockResult:
        boxes: list

    return MockResult(boxes=[
        MockBox(
            xyxy=np.array([[100, 100, 200, 300]]),
            conf=0.95,
            cls=0,  # person class
            id=1
        )
    ])


@pytest.fixture
def mock_face_embedding() -> np.ndarray:
    """Mock face embedding (512-dimensional)"""
    embedding = np.random.randn(512).astype(np.float32)
    # Normalize (L2 norm)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def mock_zone_config():
    """Mock zone configuration"""
    return {
        "entrance": {
            "points": [[50, 50], [250, 50], [250, 200], [50, 200]],
            "type": "entrance"
        },
        "checkout": {
            "points": [[300, 100], [500, 100], [500, 300], [300, 300]],
            "type": "checkout"
        }
    }


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
async def mock_db_pool():
    """Mock database connection pool"""
    from unittest.mock import AsyncMock, MagicMock

    pool = AsyncMock()
    connection = AsyncMock()

    # Mock acquire context manager
    pool.acquire = MagicMock(return_value=connection)
    connection.__aenter__ = AsyncMock(return_value=connection)
    connection.__aexit__ = AsyncMock(return_value=None)

    # Mock query methods
    connection.execute = AsyncMock()
    connection.fetch = AsyncMock(return_value=[])
    connection.fetchrow = AsyncMock(return_value=None)
    connection.fetchval = AsyncMock(return_value=None)

    return pool


# ============================================================================
# Redis Fixtures
# ============================================================================

@pytest.fixture
async def mock_redis():
    """Mock Redis client"""
    from unittest.mock import AsyncMock

    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value=b"1234567890-0")
    redis.xread = AsyncMock(return_value=[])
    redis.xack = AsyncMock()
    redis.xpending = AsyncMock(return_value={})
    redis.ping = AsyncMock(return_value=True)

    return redis


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model"""
    from unittest.mock import MagicMock

    model = MagicMock()
    model.track = MagicMock(return_value=[mock_detection_result()])

    return model


@pytest.fixture
def mock_face_recognition_model():
    """Mock face recognition model"""
    from unittest.mock import MagicMock

    model = MagicMock()
    model.extract_embedding = MagicMock(return_value=mock_face_embedding())

    return model


# ============================================================================
# Service Fixtures
# ============================================================================

@pytest.fixture
async def mock_event_bus():
    """Mock event bus"""
    from unittest.mock import AsyncMock

    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.get_pending_count = AsyncMock(return_value=0)

    return bus


@pytest.fixture
def mock_feature_store():
    """Mock feature store"""
    from unittest.mock import AsyncMock

    store = AsyncMock()
    store.get_online_features = AsyncMock(return_value={"feature1": 1.0})
    store.write_online_feature = AsyncMock()
    store.write_offline_feature = AsyncMock()

    return store


@pytest.fixture
def mock_model_registry():
    """Mock MLflow model registry"""
    from unittest.mock import MagicMock

    registry = MagicMock()
    registry.register_yolo_model = MagicMock(return_value="1")
    registry.promote_to_production = MagicMock()
    registry.get_production_model = MagicMock(return_value=None)

    return registry


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "video": {
            "input_source": 0,
            "fps": 30,
            "resolution": [640, 480]
        },
        "detection": {
            "model": "yolov8n.pt",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45
        },
        "face_recognition": {
            "model": "Facenet512",
            "distance_threshold": 0.6
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "genesis_test",
            "user": "genesis",
            "password": "genesis"
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 1  # Use different DB for tests
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000"
        }
    }


# ============================================================================
# Environment Setup
# ============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables"""
    monkeypatch.setenv("GENESIS_ENV", "test")
    monkeypatch.setenv("GENESIS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Add cleanup logic here if needed
    await asyncio.sleep(0)  # Allow pending tasks to complete
