"""
Feature Store for Genesis Platform

Centralized feature storage for training and inference.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncpg
import json
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Feature:
    """Feature definition"""
    feature_name: str
    feature_group: str  # e.g., "person_features", "zone_features"
    value_type: str  # "float", "int", "string", "array"
    description: str
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class FeatureValue:
    """Feature value with metadata"""
    entity_id: str  # person_id, zone_id, etc.
    feature_name: str
    value: Any
    timestamp: datetime
    version: int = 1


class FeatureStore:
    """
    Feature store for ML training and inference.

    Features:
    - Online features (low-latency serving)
    - Offline features (batch training)
    - Feature versioning
    - Point-in-time correct joins
    - Feature monitoring
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "genesis",
        db_user: str = "genesis",
        db_password: str = "genesis"
    ):
        """
        Initialize feature store.

        Args:
            db_host: PostgreSQL host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password
        """
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user,
            "password": db_password
        }
        self.pool: Optional[asyncpg.Pool] = None
        self.features: Dict[str, Feature] = {}

        logger.info(f"Feature store initialized (DB: {db_name}@{db_host})")

    async def connect(self) -> None:
        """Establish database connection pool"""
        self.pool = await asyncpg.create_pool(**self.db_config, min_size=5, max_size=20)
        await self._create_tables()
        logger.info("Feature store connected to database")

    async def _create_tables(self) -> None:
        """Create feature store tables"""
        async with self.pool.acquire() as conn:
            # Features metadata table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_features (
                    feature_name VARCHAR(255) PRIMARY KEY,
                    feature_group VARCHAR(255) NOT NULL,
                    value_type VARCHAR(50) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_feature_group (feature_group)
                )
            """)

            # Online feature values table (latest values)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_feature_values_online (
                    entity_id VARCHAR(255) NOT NULL,
                    feature_name VARCHAR(255) NOT NULL,
                    value JSONB NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    version INT DEFAULT 1,
                    PRIMARY KEY (entity_id, feature_name),
                    FOREIGN KEY (feature_name) REFERENCES ml_features(feature_name),
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_entity (entity_id)
                )
            """)

            # Offline feature values table (historical values)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_feature_values_offline (
                    id SERIAL PRIMARY KEY,
                    entity_id VARCHAR(255) NOT NULL,
                    feature_name VARCHAR(255) NOT NULL,
                    value JSONB NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    version INT DEFAULT 1,
                    FOREIGN KEY (feature_name) REFERENCES ml_features(feature_name),
                    INDEX idx_entity_time (entity_id, timestamp),
                    INDEX idx_feature_time (feature_name, timestamp)
                )
            """)

        logger.info("Feature store tables created")

    async def register_feature(self, feature: Feature) -> None:
        """
        Register a new feature.

        Args:
            feature: Feature object
        """
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO ml_features (feature_name, feature_group, value_type, description, created_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (feature_name) DO UPDATE
                SET feature_group = $2, value_type = $3, description = $4
            """, feature.feature_name, feature.feature_group, feature.value_type,
                feature.description, feature.created_at)

        self.features[feature.feature_name] = feature
        logger.info(f"Registered feature: {feature.feature_name} ({feature.feature_group})")

    async def write_online_feature(
        self,
        entity_id: str,
        feature_name: str,
        value: Any,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Write feature value to online store (low-latency serving).

        Args:
            entity_id: Entity ID (person_id, zone_id, etc.)
            feature_name: Feature name
            value: Feature value
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Convert value to JSON
        if isinstance(value, (np.ndarray, list)):
            value_json = json.dumps(value.tolist() if isinstance(value, np.ndarray) else value)
        else:
            value_json = json.dumps(value)

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO ml_feature_values_online (entity_id, feature_name, value, timestamp, version)
                VALUES ($1, $2, $3::jsonb, $4, 1)
                ON CONFLICT (entity_id, feature_name) DO UPDATE
                SET value = $3::jsonb, timestamp = $4, version = ml_feature_values_online.version + 1
            """, entity_id, feature_name, value_json, timestamp)

    async def write_offline_feature(
        self,
        entity_id: str,
        feature_name: str,
        value: Any,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Write feature value to offline store (historical, for training).

        Args:
            entity_id: Entity ID
            feature_name: Feature name
            value: Feature value
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Convert value to JSON
        if isinstance(value, (np.ndarray, list)):
            value_json = json.dumps(value.tolist() if isinstance(value, np.ndarray) else value)
        else:
            value_json = json.dumps(value)

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO ml_feature_values_offline (entity_id, feature_name, value, timestamp, version)
                VALUES ($1, $2, $3::jsonb, $4, 1)
            """, entity_id, feature_name, value_json, timestamp)

    async def get_online_features(
        self,
        entity_id: str,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Get latest feature values for inference (online serving).

        Args:
            entity_id: Entity ID
            feature_names: List of feature names

        Returns:
            Dictionary of feature values
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT feature_name, value, timestamp
                FROM ml_feature_values_online
                WHERE entity_id = $1 AND feature_name = ANY($2)
            """, entity_id, feature_names)

        features = {}
        for row in rows:
            features[row["feature_name"]] = json.loads(row["value"])

        return features

    async def get_offline_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical feature values for training (offline batch).

        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            List of feature records
        """
        query = """
            SELECT entity_id, feature_name, value, timestamp
            FROM ml_feature_values_offline
            WHERE entity_id = ANY($1) AND feature_name = ANY($2)
        """
        params = [entity_ids, feature_names]

        if start_time:
            query += " AND timestamp >= $3"
            params.append(start_time)

        if end_time:
            if start_time:
                query += " AND timestamp <= $4"
            else:
                query += " AND timestamp <= $3"
            params.append(end_time)

        query += " ORDER BY timestamp DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        features = []
        for row in rows:
            features.append({
                "entity_id": row["entity_id"],
                "feature_name": row["feature_name"],
                "value": json.loads(row["value"]),
                "timestamp": row["timestamp"]
            })

        return features

    async def get_point_in_time_features(
        self,
        entity_id: str,
        feature_names: List[str],
        timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Get feature values as of a specific point in time (for training).

        Args:
            entity_id: Entity ID
            feature_names: List of feature names
            timestamp: Point in time

        Returns:
            Dictionary of feature values
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT ON (feature_name) feature_name, value, timestamp
                FROM ml_feature_values_offline
                WHERE entity_id = $1 AND feature_name = ANY($2) AND timestamp <= $3
                ORDER BY feature_name, timestamp DESC
            """, entity_id, feature_names, timestamp)

        features = {}
        for row in rows:
            features[row["feature_name"]] = json.loads(row["value"])

        return features

    async def close(self) -> None:
        """Close database connections"""
        if self.pool:
            await self.pool.close()
        logger.info("Feature store disconnected")
