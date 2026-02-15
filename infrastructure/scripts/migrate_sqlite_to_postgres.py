#!/usr/bin/env python3
"""
Migration script: SQLite → PostgreSQL
Migrates identities, sessions, and historical data from SQLite to PostgreSQL with pgvector
"""
import asyncio
import sqlite3
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.database import PostgresClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SQLiteToPostgresMigration:
    """
    Migrates data from SQLite to PostgreSQL
    """

    def __init__(
        self,
        sqlite_path: str,
        postgres_client: PostgresClient
    ):
        """
        Args:
            sqlite_path: Path to SQLite database
            postgres_client: PostgreSQL client instance
        """
        self.sqlite_path = sqlite_path
        self.pg = postgres_client
        self.sqlite_conn: Optional[sqlite3.Connection] = None

        # Migration statistics
        self.stats = {
            'identities_migrated': 0,
            'identities_skipped': 0,
            'sessions_migrated': 0,
            'events_migrated': 0,
            'errors': 0
        }

    def connect_sqlite(self):
        """Connect to SQLite database"""
        try:
            self.sqlite_conn = sqlite3.connect(self.sqlite_path)
            self.sqlite_conn.row_factory = sqlite3.Row
            logger.info(f"Connected to SQLite database: {self.sqlite_path}")

        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise

    def close_sqlite(self):
        """Close SQLite connection"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
            logger.info("SQLite connection closed")

    async def migrate_identities(self) -> int:
        """
        Migrate identities from SQLite to PostgreSQL

        Returns:
            Number of identities migrated
        """
        logger.info("Migrating identities...")

        try:
            # Get identities from SQLite
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT person_id, embedding, first_seen, last_seen, metadata
                FROM identities
            """)

            rows = cursor.fetchall()
            logger.info(f"Found {len(rows)} identities in SQLite")

            migrated = 0

            for row in rows:
                try:
                    person_id = row['person_id']
                    embedding_blob = row['embedding']

                    # Deserialize embedding (stored as pickled numpy array in SQLite)
                    import pickle
                    import numpy as np

                    try:
                        embedding_array = pickle.loads(embedding_blob)
                        if isinstance(embedding_array, np.ndarray):
                            embedding = embedding_array.tolist()
                        else:
                            embedding = embedding_array
                    except:
                        # If pickle fails, assume it's already a list or array
                        logger.warning(f"Could not unpickle embedding for {person_id}, skipping")
                        self.stats['identities_skipped'] += 1
                        continue

                    # Parse metadata (JSON string in SQLite)
                    import json
                    try:
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    except:
                        metadata = {}

                    # Insert into PostgreSQL
                    success = await self.pg.insert_identity(
                        person_id=person_id,
                        embedding=embedding,
                        metadata=metadata
                    )

                    if success:
                        # Update timestamps
                        await self.pg.execute(
                            """
                            UPDATE identities
                            SET first_seen = $1,
                                last_seen = $2
                            WHERE person_id = $3
                            """,
                            datetime.fromtimestamp(row['first_seen']) if row['first_seen'] else datetime.now(),
                            datetime.fromtimestamp(row['last_seen']) if row['last_seen'] else datetime.now(),
                            person_id
                        )

                        migrated += 1
                        self.stats['identities_migrated'] += 1

                        if migrated % 10 == 0:
                            logger.info(f"  Migrated {migrated}/{len(rows)} identities...")

                    else:
                        logger.warning(f"  Failed to migrate identity: {person_id}")
                        self.stats['identities_skipped'] += 1

                except Exception as e:
                    logger.error(f"  Error migrating identity {row['person_id']}: {e}")
                    self.stats['errors'] += 1
                    continue

            logger.info(f"✓ Migrated {migrated} identities")
            return migrated

        except Exception as e:
            logger.error(f"Error migrating identities: {e}")
            raise

    async def migrate_sessions(self) -> int:
        """
        Migrate sessions from SQLite to PostgreSQL

        Returns:
            Number of sessions migrated
        """
        logger.info("Migrating sessions...")

        try:
            # Check if sessions table exists in SQLite
            cursor = self.sqlite_conn.cursor()

            try:
                cursor.execute("""
                    SELECT session_id, person_id, camera_id, start_time,
                           end_time, duration_seconds, zones_visited
                    FROM sessions
                """)
                rows = cursor.fetchall()

            except sqlite3.OperationalError:
                logger.warning("Sessions table not found in SQLite, skipping")
                return 0

            logger.info(f"Found {len(rows)} sessions in SQLite")

            migrated = 0

            for row in rows:
                try:
                    # Parse zones_visited (may be JSON or comma-separated)
                    zones = []
                    if row['zones_visited']:
                        import json
                        try:
                            zones = json.loads(row['zones_visited'])
                        except:
                            zones = row['zones_visited'].split(',')

                    # Insert session
                    session_id = await self.pg.insert_session(
                        person_id=row['person_id'],
                        camera_id=row['camera_id'] or 'unknown',
                        start_time=datetime.fromtimestamp(row['start_time']),
                        end_time=datetime.fromtimestamp(row['end_time']) if row['end_time'] else None,
                        duration_seconds=row['duration_seconds'],
                        zones_visited=zones
                    )

                    if session_id:
                        migrated += 1
                        self.stats['sessions_migrated'] += 1

                        if migrated % 50 == 0:
                            logger.info(f"  Migrated {migrated}/{len(rows)} sessions...")

                except Exception as e:
                    logger.error(f"  Error migrating session: {e}")
                    self.stats['errors'] += 1
                    continue

            logger.info(f"✓ Migrated {migrated} sessions")
            return migrated

        except Exception as e:
            logger.error(f"Error migrating sessions: {e}")
            return 0

    async def migrate_events(self) -> int:
        """
        Migrate identity events from SQLite to PostgreSQL

        Returns:
            Number of events migrated
        """
        logger.info("Migrating identity events...")

        try:
            cursor = self.sqlite_conn.cursor()

            try:
                cursor.execute("""
                    SELECT event_id, person_id, event_type, zone_name,
                           timestamp, metadata
                    FROM identity_events
                """)
                rows = cursor.fetchall()

            except sqlite3.OperationalError:
                logger.warning("Identity events table not found in SQLite, skipping")
                return 0

            logger.info(f"Found {len(rows)} events in SQLite")

            migrated = 0

            for row in rows:
                try:
                    # Parse metadata
                    import json
                    try:
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    except:
                        metadata = {}

                    # Insert event
                    event_id = await self.pg.insert_identity_event(
                        person_id=row['person_id'],
                        event_type=row['event_type'],
                        timestamp=datetime.fromtimestamp(row['timestamp']),
                        zone_name=row['zone_name'],
                        metadata=metadata
                    )

                    if event_id:
                        migrated += 1
                        self.stats['events_migrated'] += 1

                        if migrated % 100 == 0:
                            logger.info(f"  Migrated {migrated}/{len(rows)} events...")

                except Exception as e:
                    logger.error(f"  Error migrating event: {e}")
                    self.stats['errors'] += 1
                    continue

            logger.info(f"✓ Migrated {migrated} events")
            return migrated

        except Exception as e:
            logger.error(f"Error migrating events: {e}")
            return 0

    async def verify_migration(self) -> bool:
        """
        Verify migration was successful

        Returns:
            True if verification passed
        """
        logger.info("Verifying migration...")

        try:
            # Get counts from both databases
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM identities")
            sqlite_identities = cursor.fetchone()[0]

            pg_identities = await self.pg.fetchval("SELECT COUNT(*) FROM identities")

            logger.info(f"  SQLite identities: {sqlite_identities}")
            logger.info(f"  PostgreSQL identities: {pg_identities}")

            if sqlite_identities != pg_identities:
                logger.warning(
                    f"  ⚠ Identity count mismatch: "
                    f"SQLite={sqlite_identities}, PostgreSQL={pg_identities}"
                )
                return False

            # Test vector search
            logger.info("  Testing vector similarity search...")
            test_embedding = [0.1] * 512  # Dummy embedding
            matches = await self.pg.search_similar_faces(test_embedding, threshold=0.0, limit=1)

            if matches:
                logger.info(f"  ✓ Vector search working (found {len(matches)} matches)")
            else:
                logger.info("  ✓ Vector search working (no matches, which is expected)")

            logger.info("✓ Verification passed")
            return True

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    async def run_migration(self) -> bool:
        """
        Run complete migration process

        Returns:
            True if successful
        """
        logger.info("="*60)
        logger.info("SQLite → PostgreSQL Migration")
        logger.info("="*60)

        try:
            # Connect to SQLite
            self.connect_sqlite()

            # Connect to PostgreSQL
            await self.pg.connect()

            # Check PostgreSQL health
            if not await self.pg.health_check():
                logger.error("PostgreSQL is not healthy, aborting migration")
                return False

            # Migrate identities
            await self.migrate_identities()

            # Migrate sessions
            await self.migrate_sessions()

            # Migrate events
            await self.migrate_events()

            # Verify migration
            verified = await self.verify_migration()

            # Print statistics
            logger.info("")
            logger.info("="*60)
            logger.info("Migration Statistics:")
            logger.info("="*60)
            for key, value in self.stats.items():
                logger.info(f"  {key.replace('_', ' ').title()}: {value}")
            logger.info("="*60)

            if verified and self.stats['errors'] == 0:
                logger.info("✓ Migration completed successfully!")
                return True
            elif verified and self.stats['errors'] > 0:
                logger.warning(f"⚠ Migration completed with {self.stats['errors']} errors")
                return True
            else:
                logger.error("✗ Migration failed verification")
                return False

        except Exception as e:
            logger.error(f"Migration failed: {e}", exc_info=True)
            return False

        finally:
            # Cleanup
            self.close_sqlite()
            await self.pg.close()


async def main():
    """Main migration entry point"""
    parser = argparse.ArgumentParser(description="Migrate Genesis data from SQLite to PostgreSQL")
    parser.add_argument(
        "--sqlite-db",
        default="data/outputs/identities.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--pg-host",
        default="localhost",
        help="PostgreSQL host"
    )
    parser.add_argument(
        "--pg-port",
        type=int,
        default=5432,
        help="PostgreSQL port"
    )
    parser.add_argument(
        "--pg-database",
        default="genesis",
        help="PostgreSQL database name"
    )
    parser.add_argument(
        "--pg-user",
        default="genesis",
        help="PostgreSQL username"
    )
    parser.add_argument(
        "--pg-password",
        default="genesis_dev_password",
        help="PostgreSQL password"
    )

    args = parser.parse_args()

    # Check if SQLite database exists
    sqlite_path = Path(args.sqlite_db)
    if not sqlite_path.exists():
        logger.error(f"SQLite database not found: {args.sqlite_db}")
        logger.info("Nothing to migrate. This is normal for new installations.")
        return 0

    # Create PostgreSQL client
    pg_client = PostgresClient(
        host=args.pg_host,
        port=args.pg_port,
        database=args.pg_database,
        user=args.pg_user,
        password=args.pg_password
    )

    # Run migration
    migration = SQLiteToPostgresMigration(
        sqlite_path=str(sqlite_path),
        postgres_client=pg_client
    )

    success = await migration.run_migration()

    if success:
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Update configs/settings.yaml to use PostgreSQL")
        logger.info("  2. Restart workers to use new database")
        logger.info("  3. Keep SQLite backup for 2 weeks, then archive")
        return 0
    else:
        logger.error("Migration failed. SQLite database remains unchanged.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
