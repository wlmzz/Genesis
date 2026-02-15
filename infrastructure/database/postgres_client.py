"""
Async PostgreSQL client for Genesis
Uses asyncpg for high-performance async database operations
"""
import asyncpg
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class PostgresClient:
    """
    Async PostgreSQL client with connection pooling
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "genesis",
        user: str = "genesis",
        password: str = "genesis_dev_password",
        min_pool_size: int = 10,
        max_pool_size: int = 20
    ):
        """
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Username
            password: Password
            min_pool_size: Minimum connection pool size
            max_pool_size: Maximum connection pool size
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size

        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Create connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                command_timeout=60
            )
            logger.info(
                f"Connected to PostgreSQL at {self.host}:{self.port}/{self.database}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

    async def execute(self, query: str, *args) -> str:
        """
        Execute a query without returning results

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Query result status
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *args)
            return result

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """
        Execute a query and return all results

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            List of records
        """
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchone(self, query: str, *args) -> Optional[asyncpg.Record]:
        """
        Execute a query and return first result

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            First record or None
        """
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """
        Execute a query and return single value

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single value
        """
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    # =============================================================================
    # IDENTITY OPERATIONS
    # =============================================================================

    async def insert_identity(
        self,
        person_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Insert new identity with face embedding

        Args:
            person_id: Unique person identifier
            embedding: Face embedding vector (512-dim)
            metadata: Optional metadata (name, attributes, etc.)

        Returns:
            True if inserted, False if already exists
        """
        try:
            # Convert embedding to pgvector format
            embedding_str = f"[{','.join(map(str, embedding))}]"

            await self.execute(
                """
                INSERT INTO identities (person_id, embedding, metadata)
                VALUES ($1, $2::vector, $3)
                ON CONFLICT (person_id) DO NOTHING
                """,
                person_id,
                embedding_str,
                json.dumps(metadata or {})
            )
            return True

        except Exception as e:
            logger.error(f"Error inserting identity {person_id}: {e}")
            return False

    async def update_identity_embedding(
        self,
        person_id: str,
        embedding: List[float]
    ) -> bool:
        """
        Update identity embedding (e.g., for retraining)

        Args:
            person_id: Person identifier
            embedding: New face embedding

        Returns:
            True if updated
        """
        try:
            embedding_str = f"[{','.join(map(str, embedding))}]"

            result = await self.execute(
                """
                UPDATE identities
                SET embedding = $1::vector,
                    updated_at = NOW()
                WHERE person_id = $2
                """,
                embedding_str,
                person_id
            )

            return "UPDATE" in result

        except Exception as e:
            logger.error(f"Error updating identity {person_id}: {e}")
            return False

    async def get_identity(self, person_id: str) -> Optional[Dict[str, Any]]:
        """
        Get identity by ID

        Args:
            person_id: Person identifier

        Returns:
            Identity dict or None
        """
        row = await self.fetchone(
            """
            SELECT person_id, first_seen, last_seen, total_appearances, metadata
            FROM identities
            WHERE person_id = $1
            """,
            person_id
        )

        if row:
            return dict(row)
        return None

    async def search_similar_faces(
        self,
        embedding: List[float],
        threshold: float = 0.6,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar faces using cosine similarity

        Args:
            embedding: Query face embedding
            threshold: Similarity threshold (0-1, higher = more similar)
            limit: Maximum results

        Returns:
            List of matching identities with similarity scores
        """
        try:
            embedding_str = f"[{','.join(map(str, embedding))}]"

            rows = await self.fetch(
                """
                SELECT * FROM search_similar_faces($1::vector, $2, $3)
                """,
                embedding_str,
                threshold,
                limit
            )

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            return []

    # =============================================================================
    # SESSION OPERATIONS
    # =============================================================================

    async def insert_session(
        self,
        person_id: Optional[str],
        camera_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        duration_seconds: Optional[int] = None,
        zones_visited: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Insert new session

        Args:
            person_id: Person identifier (None for unknown)
            camera_id: Camera identifier
            start_time: Session start time
            end_time: Session end time (None if ongoing)
            duration_seconds: Session duration
            zones_visited: List of zones visited
            metadata: Additional metadata

        Returns:
            Session ID (UUID) or None
        """
        try:
            session_id = await self.fetchval(
                """
                INSERT INTO sessions (
                    person_id, camera_id, start_time, end_time,
                    duration_seconds, zones_visited, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING session_id
                """,
                person_id,
                camera_id,
                start_time,
                end_time,
                duration_seconds,
                zones_visited or [],
                json.dumps(metadata or {})
            )

            return str(session_id)

        except Exception as e:
            logger.error(f"Error inserting session: {e}")
            return None

    async def end_session(
        self,
        session_id: str,
        end_time: datetime,
        zones_visited: Optional[List[str]] = None
    ) -> bool:
        """
        End an ongoing session

        Args:
            session_id: Session ID
            end_time: Session end time
            zones_visited: Updated list of zones visited

        Returns:
            True if updated
        """
        try:
            result = await self.execute(
                """
                UPDATE sessions
                SET end_time = $1,
                    duration_seconds = EXTRACT(EPOCH FROM ($1 - start_time))::INTEGER,
                    zones_visited = COALESCE($2, zones_visited),
                    updated_at = NOW()
                WHERE session_id = $3
                """,
                end_time,
                zones_visited,
                session_id
            )

            return "UPDATE" in result

        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
            return False

    # =============================================================================
    # EVENT OPERATIONS
    # =============================================================================

    async def insert_identity_event(
        self,
        person_id: str,
        event_type: str,
        timestamp: datetime,
        session_id: Optional[str] = None,
        zone_name: Optional[str] = None,
        camera_id: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Insert identity event

        Args:
            person_id: Person identifier
            event_type: Event type (zone_entered, zone_exited, etc.)
            timestamp: Event timestamp
            session_id: Optional session ID
            zone_name: Optional zone name
            camera_id: Optional camera ID
            confidence: Optional confidence score
            metadata: Additional metadata

        Returns:
            Event ID or None
        """
        try:
            event_id = await self.fetchval(
                """
                INSERT INTO identity_events (
                    person_id, session_id, event_type, zone_name,
                    camera_id, confidence, timestamp, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING event_id
                """,
                person_id,
                session_id,
                event_type,
                zone_name,
                camera_id,
                confidence,
                timestamp,
                json.dumps(metadata or {})
            )

            return event_id

        except Exception as e:
            logger.error(f"Error inserting identity event: {e}")
            return None

    # =============================================================================
    # METRICS OPERATIONS (TimescaleDB)
    # =============================================================================

    async def insert_metrics(
        self,
        timestamp: datetime,
        camera_id: str,
        people_total: int = 0,
        people_by_zone: Optional[Dict[str, int]] = None,
        queue_len: int = 0,
        avg_wait_sec: float = 0,
        new_faces: int = 0,
        recognized_faces: int = 0,
        alerts_triggered: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Insert metrics record (TimescaleDB hypertable)

        Args:
            timestamp: Metric timestamp
            camera_id: Camera identifier
            people_total: Total people count
            people_by_zone: People count per zone
            queue_len: Queue length
            avg_wait_sec: Average wait time
            new_faces: New faces detected
            recognized_faces: Recognized faces
            alerts_triggered: Alerts triggered
            metadata: Additional metadata

        Returns:
            True if inserted
        """
        try:
            await self.execute(
                """
                INSERT INTO metrics (
                    time, camera_id, people_total, people_by_zone,
                    queue_len, avg_wait_sec, new_faces, recognized_faces,
                    alerts_triggered, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                timestamp,
                camera_id,
                people_total,
                json.dumps(people_by_zone or {}),
                queue_len,
                avg_wait_sec,
                new_faces,
                recognized_faces,
                alerts_triggered,
                json.dumps(metadata or {})
            )

            return True

        except Exception as e:
            logger.error(f"Error inserting metrics: {e}")
            return False

    async def get_metrics_range(
        self,
        camera_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get metrics for time range

        Args:
            camera_id: Camera identifier
            start_time: Start of range
            end_time: End of range

        Returns:
            List of metrics records
        """
        rows = await self.fetch(
            """
            SELECT * FROM metrics
            WHERE camera_id = $1
              AND time >= $2
              AND time <= $3
            ORDER BY time ASC
            """,
            camera_id,
            start_time,
            end_time
        )

        return [dict(row) for row in rows]

    # =============================================================================
    # FACE EMBEDDINGS OPERATIONS
    # =============================================================================

    async def insert_face_embedding(
        self,
        embedding: List[float],
        timestamp: datetime,
        person_id: Optional[str] = None,
        confidence: Optional[float] = None,
        camera_id: Optional[str] = None,
        zone_name: Optional[str] = None,
        is_new_face: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Insert face embedding record

        Args:
            embedding: Face embedding vector
            timestamp: Detection timestamp
            person_id: Person ID (if recognized)
            confidence: Recognition confidence
            camera_id: Camera ID
            zone_name: Zone name
            is_new_face: Whether this is a new face
            metadata: Additional metadata

        Returns:
            Embedding ID or None
        """
        try:
            embedding_str = f"[{','.join(map(str, embedding))}]"

            embedding_id = await self.fetchval(
                """
                INSERT INTO face_embeddings (
                    person_id, embedding, confidence, camera_id,
                    zone_name, timestamp, is_new_face, metadata
                )
                VALUES ($1, $2::vector, $3, $4, $5, $6, $7, $8)
                RETURNING embedding_id
                """,
                person_id,
                embedding_str,
                confidence,
                camera_id,
                zone_name,
                timestamp,
                is_new_face,
                json.dumps(metadata or {})
            )

            return embedding_id

        except Exception as e:
            logger.error(f"Error inserting face embedding: {e}")
            return None

    # =============================================================================
    # ALERT OPERATIONS
    # =============================================================================

    async def insert_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        timestamp: datetime,
        camera_id: Optional[str] = None,
        person_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Insert alert record

        Args:
            alert_type: Type of alert
            severity: Severity level (info, warning, critical)
            message: Alert message
            timestamp: Alert timestamp
            camera_id: Camera ID
            person_id: Person ID (if relevant)
            context: Additional context

        Returns:
            Alert ID or None
        """
        try:
            alert_id = await self.fetchval(
                """
                INSERT INTO alerts (
                    alert_type, severity, message, timestamp,
                    camera_id, person_id, context
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING alert_id
                """,
                alert_type,
                severity,
                message,
                timestamp,
                camera_id,
                person_id,
                json.dumps(context or {})
            )

            return alert_id

        except Exception as e:
            logger.error(f"Error inserting alert: {e}")
            return None

    # =============================================================================
    # STATISTICS & UTILITIES
    # =============================================================================

    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Statistics dictionary
        """
        row = await self.fetchone("SELECT * FROM database_stats")
        return dict(row) if row else {}

    async def health_check(self) -> bool:
        """
        Check database connection health

        Returns:
            True if healthy
        """
        try:
            await self.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
