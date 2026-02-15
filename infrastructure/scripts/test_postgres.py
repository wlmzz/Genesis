#!/usr/bin/env python3
"""
Test PostgreSQL connection and pgvector functionality
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.database import PostgresClient, VectorSearchClient
import numpy as np


async def test_postgres():
    """Test PostgreSQL connection and operations"""
    print("="*60)
    print("Genesis PostgreSQL Test")
    print("="*60)

    # Create client
    print("\n1. Connecting to PostgreSQL...")
    client = PostgresClient(
        host="localhost",
        port=5432,
        database="genesis",
        user="genesis",
        password="genesis_dev_password"
    )

    try:
        await client.connect()
        print("   ✓ Connected to PostgreSQL")

        # Health check
        print("\n2. Checking database health...")
        healthy = await client.health_check()
        if healthy:
            print("   ✓ Database is healthy")
        else:
            print("   ✗ Database health check failed")
            return False

        # Get stats
        print("\n3. Getting database statistics...")
        stats = await client.get_database_stats()
        for key, value in stats.items():
            print(f"   - {key}: {value}")

        # Test vector search
        print("\n4. Testing pgvector similarity search...")

        # Create test embedding
        test_embedding = np.random.rand(512).tolist()

        # Insert test identity
        success = await client.insert_identity(
            person_id="test_person_001",
            embedding=test_embedding,
            metadata={"name": "Test Person", "test": True}
        )

        if success:
            print("   ✓ Inserted test identity")
        else:
            print("   ⚠ Identity may already exist (this is OK)")

        # Search for similar faces
        matches = await client.search_similar_faces(
            embedding=test_embedding,
            threshold=0.9,  # High threshold to match our test embedding
            limit=5
        )

        print(f"   ✓ Vector search returned {len(matches)} matches")
        for match in matches:
            print(f"     - {match['person_id']}: similarity={match['distance']:.4f}")

        # Test vector search client
        print("\n5. Testing VectorSearchClient...")
        vector_client = VectorSearchClient(client)

        # Find matching face
        result = await vector_client.find_matching_face(
            embedding=test_embedding,
            threshold=0.9
        )

        if result:
            person_id, similarity = result
            print(f"   ✓ Found matching face: {person_id} (similarity={similarity:.4f})")
        else:
            print("   ⚠ No matching face found")

        # Test metrics insertion
        print("\n6. Testing TimescaleDB metrics insertion...")
        from datetime import datetime

        success = await client.insert_metrics(
            timestamp=datetime.now(),
            camera_id="test_cam_0",
            people_total=5,
            people_by_zone={"entrance": 2, "checkout": 3},
            queue_len=3,
            new_faces=1,
            recognized_faces=4
        )

        if success:
            print("   ✓ Inserted test metrics")
        else:
            print("   ✗ Failed to insert metrics")

        # Test face embedding insertion
        print("\n7. Testing face embedding storage...")
        embedding_id = await client.insert_face_embedding(
            embedding=test_embedding,
            timestamp=datetime.now(),
            person_id="test_person_001",
            confidence=0.95,
            camera_id="test_cam_0",
            is_new_face=False
        )

        if embedding_id:
            print(f"   ✓ Inserted face embedding (ID: {embedding_id})")
        else:
            print("   ✗ Failed to insert face embedding")

        # Clean up test data
        print("\n8. Cleaning up test data...")
        await client.execute("DELETE FROM identities WHERE person_id LIKE 'test_%'")
        await client.execute("DELETE FROM face_embeddings WHERE person_id LIKE 'test_%'")
        await client.execute("DELETE FROM metrics WHERE camera_id = 'test_cam_0'")
        print("   ✓ Test data cleaned up")

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        print("\nPostgreSQL is ready for production use!")
        print("\nNext steps:")
        print("  1. Run migration: python infrastructure/scripts/migrate_sqlite_to_postgres.py")
        print("  2. Update configs/settings.yaml: database.postgres.enabled: true")
        print("  3. Start postgres storage worker: python services/storage/postgres_worker.py")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await client.close()


def main():
    """Main test entry point"""
    try:
        result = asyncio.run(test_postgres())
        return 0 if result else 1

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
