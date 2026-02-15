"""
Vector similarity search client for face recognition
Wraps pgvector operations for face embedding search
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from .postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class VectorSearchClient:
    """
    Client for face embedding similarity search using pgvector
    """

    def __init__(self, postgres_client: PostgresClient):
        """
        Args:
            postgres_client: PostgreSQL client instance
        """
        self.db = postgres_client

    async def find_matching_face(
        self,
        embedding: List[float],
        threshold: float = 0.6
    ) -> Optional[Tuple[str, float]]:
        """
        Find best matching face for given embedding

        Args:
            embedding: Query face embedding (512-dim)
            threshold: Minimum similarity threshold (0-1)

        Returns:
            Tuple of (person_id, similarity_score) or None if no match
        """
        try:
            matches = await self.db.search_similar_faces(
                embedding=embedding,
                threshold=threshold,
                limit=1
            )

            if matches:
                match = matches[0]
                return (match['person_id'], match['distance'])

            return None

        except Exception as e:
            logger.error(f"Error finding matching face: {e}")
            return None

    async def find_all_matches(
        self,
        embedding: List[float],
        threshold: float = 0.6,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find all matching faces above threshold

        Args:
            embedding: Query face embedding
            threshold: Minimum similarity threshold
            limit: Maximum number of results

        Returns:
            List of matches with person_id and similarity scores
        """
        try:
            matches = await self.db.search_similar_faces(
                embedding=embedding,
                threshold=threshold,
                limit=limit
            )

            return matches

        except Exception as e:
            logger.error(f"Error finding all matches: {e}")
            return []

    async def register_new_face(
        self,
        person_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register new face in database

        Args:
            person_id: Unique person identifier
            embedding: Face embedding vector
            metadata: Optional metadata (name, attributes, etc.)

        Returns:
            True if registered successfully
        """
        return await self.db.insert_identity(
            person_id=person_id,
            embedding=embedding,
            metadata=metadata
        )

    async def update_face_embedding(
        self,
        person_id: str,
        embedding: List[float]
    ) -> bool:
        """
        Update existing face embedding (e.g., after retraining)

        Args:
            person_id: Person identifier
            embedding: New face embedding

        Returns:
            True if updated
        """
        return await self.db.update_identity_embedding(
            person_id=person_id,
            embedding=embedding
        )

    async def get_total_faces(self) -> int:
        """
        Get total number of registered faces

        Returns:
            Total face count
        """
        try:
            count = await self.db.fetchval(
                "SELECT COUNT(*) FROM identities"
            )
            return count or 0

        except Exception as e:
            logger.error(f"Error getting total faces: {e}")
            return 0

    async def get_face_info(self, person_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered face

        Args:
            person_id: Person identifier

        Returns:
            Face information dictionary or None
        """
        return await self.db.get_identity(person_id)

    async def get_recent_faces(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recently seen faces

        Args:
            limit: Maximum number of results

        Returns:
            List of face records sorted by last_seen
        """
        try:
            rows = await self.db.fetch(
                """
                SELECT person_id, first_seen, last_seen,
                       total_appearances, metadata
                FROM identities
                ORDER BY last_seen DESC
                LIMIT $1
                """,
                limit
            )

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error getting recent faces: {e}")
            return []

    async def search_by_metadata(
        self,
        metadata_query: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search faces by metadata

        Args:
            metadata_query: Metadata key-value pairs to match
            limit: Maximum results

        Returns:
            List of matching face records
        """
        try:
            # Build JSONB query
            # This searches for records where metadata contains all specified keys/values
            import json
            metadata_json = json.dumps(metadata_query)

            rows = await self.db.fetch(
                """
                SELECT person_id, first_seen, last_seen,
                       total_appearances, metadata
                FROM identities
                WHERE metadata @> $1::jsonb
                ORDER BY last_seen DESC
                LIMIT $2
                """,
                metadata_json,
                limit
            )

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
