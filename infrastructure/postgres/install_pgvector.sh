#!/bin/bash
# Install pgvector extension in TimescaleDB container
# Run this after starting the postgres container

echo "Installing pgvector extension in TimescaleDB..."

docker exec -it genesis-postgres bash -c '
  apt-get update && \
  apt-get install -y postgresql-server-dev-15 build-essential git && \
  cd /tmp && \
  git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git && \
  cd pgvector && \
  make && \
  make install && \
  cd / && \
  rm -rf /tmp/pgvector
'

echo "✓ pgvector installed"
echo
echo "Now connect to database and run:"
echo "  docker exec -it genesis-postgres psql -U genesis -d genesis"
echo "  CREATE EXTENSION vector;"
