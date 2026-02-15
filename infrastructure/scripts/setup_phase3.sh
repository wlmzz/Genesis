#!/bin/bash
# Setup script for Genesis Phase 3: PostgreSQL Migration

set -e

echo "========================================"
echo "Genesis Phase 3 Setup"
echo "PostgreSQL + pgvector + TimescaleDB"
echo "========================================"
echo

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running from correct directory
if [ ! -f "configs/settings.yaml" ]; then
    echo -e "${RED}✗ Error: Must run from project root directory${NC}"
    exit 1
fi

echo "1. Installing Python dependencies..."
pip install -q asyncpg pgvector psycopg2-binary
echo -e "${GREEN}   ✓ Dependencies installed${NC}"

echo
echo "2. Starting PostgreSQL with TimescaleDB..."
cd infrastructure
docker-compose -f docker-compose.dev.yml up -d postgres
cd ..
sleep 5

echo
echo "3. Waiting for PostgreSQL to be healthy..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker exec genesis-postgres pg_isready -U genesis > /dev/null 2>&1; then
        echo -e "${GREEN}   ✓ PostgreSQL is healthy${NC}"
        break
    fi

    echo "   Waiting for PostgreSQL... (attempt $((attempt + 1))/$max_attempts)"
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}   ✗ PostgreSQL failed to start${NC}"
    exit 1
fi

echo
echo "4. Installing pgvector extension..."
echo "   This may take a few minutes..."

docker exec genesis-postgres bash -c '
  apt-get update -qq && \
  apt-get install -y -qq postgresql-server-dev-15 build-essential git wget && \
  cd /tmp && \
  wget -q https://github.com/pgvector/pgvector/archive/refs/tags/v0.5.1.tar.gz && \
  tar -xzf v0.5.1.tar.gz && \
  cd pgvector-0.5.1 && \
  make -s && \
  make -s install && \
  cd / && \
  rm -rf /tmp/pgvector-0.5.1 /tmp/v0.5.1.tar.gz
' > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}   ✓ pgvector extension installed${NC}"
else
    echo -e "${RED}   ✗ Failed to install pgvector${NC}"
    exit 1
fi

echo
echo "5. Initializing database schema..."
docker exec -i genesis-postgres psql -U genesis -d genesis < infrastructure/postgres/init.sql > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}   ✓ Database schema initialized${NC}"
else
    echo -e "${RED}   ✗ Failed to initialize schema${NC}"
    exit 1
fi

echo
echo "6. Testing PostgreSQL connection..."
python infrastructure/scripts/test_postgres.py

if [ $? -ne 0 ]; then
    echo -e "${RED}   ✗ PostgreSQL test failed${NC}"
    exit 1
fi

echo
echo "7. Checking for existing SQLite data..."
if [ -f "data/outputs/identities.db" ]; then
    echo -e "${YELLOW}   ⚠ Found existing SQLite database${NC}"
    echo "   Would you like to migrate data to PostgreSQL? (y/n)"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "   Running migration..."
        python infrastructure/scripts/migrate_sqlite_to_postgres.py

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}   ✓ Migration completed${NC}"
        else
            echo -e "${RED}   ✗ Migration failed${NC}"
            echo "   You can run migration manually later:"
            echo "   python infrastructure/scripts/migrate_sqlite_to_postgres.py"
        fi
    else
        echo "   Skipping migration. You can run it later:"
        echo "   python infrastructure/scripts/migrate_sqlite_to_postgres.py"
    fi
else
    echo "   No existing SQLite database found (fresh installation)"
fi

echo
echo "========================================"
echo -e "${GREEN}✓ Phase 3 Setup Complete!${NC}"
echo "========================================"
echo
echo "PostgreSQL is running with:"
echo "  • Host:     localhost:5432"
echo "  • Database: genesis"
echo "  • User:     genesis"
echo "  • Password: genesis_dev_password"
echo
echo "Extensions enabled:"
echo "  ✓ pgvector (face embedding similarity search)"
echo "  ✓ TimescaleDB (time-series metrics)"
echo
echo "To enable PostgreSQL in your application:"
echo "  1. Edit configs/settings.yaml"
echo "  2. Set: database.postgres.enabled: true"
echo "  3. Set: database.sqlite.enabled: false"
echo
echo "To start PostgreSQL storage worker:"
echo "  python services/storage/postgres_worker.py"
echo
echo "To view database:"
echo "  • psql: docker exec -it genesis-postgres psql -U genesis -d genesis"
echo "  • pgAdmin: http://localhost:8082 (from docker-compose)"
echo
echo "To stop PostgreSQL:"
echo "  cd infrastructure && docker-compose -f docker-compose.dev.yml stop postgres"
echo
