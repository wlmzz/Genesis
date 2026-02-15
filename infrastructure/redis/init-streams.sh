#!/bin/bash
# Initialize Redis Streams and consumer groups for Genesis

set -e

echo "Initializing Genesis Redis Streams..."

# Wait for Redis to be ready
until redis-cli ping; do
  echo "Waiting for Redis..."
  sleep 1
done

# Create streams and consumer groups
redis-cli << EOF
# Create streams with consumer groups
XGROUP CREATE genesis:frames frame-workers 0 MKSTREAM
XGROUP CREATE genesis:frames detection-workers 0 MKSTREAM

XGROUP CREATE genesis:persons person-workers 0 MKSTREAM
XGROUP CREATE genesis:persons face-workers 0 MKSTREAM

XGROUP CREATE genesis:faces face-storage-workers 0 MKSTREAM

XGROUP CREATE genesis:zones zone-workers 0 MKSTREAM
XGROUP CREATE genesis:zones analytics-workers 0 MKSTREAM

XGROUP CREATE genesis:alerts alert-workers 0 MKSTREAM

XGROUP CREATE genesis:metrics metrics-workers 0 MKSTREAM
XGROUP CREATE genesis:metrics storage-workers 0 MKSTREAM

XGROUP CREATE genesis:sessions session-workers 0 MKSTREAM
XGROUP CREATE genesis:sessions storage-workers 0 MKSTREAM

# Create DLQ streams
XGROUP CREATE genesis:frames:dlq dlq-monitor 0 MKSTREAM
XGROUP CREATE genesis:persons:dlq dlq-monitor 0 MKSTREAM
XGROUP CREATE genesis:faces:dlq dlq-monitor 0 MKSTREAM
XGROUP CREATE genesis:zones:dlq dlq-monitor 0 MKSTREAM
XGROUP CREATE genesis:alerts:dlq dlq-monitor 0 MKSTREAM
XGROUP CREATE genesis:metrics:dlq dlq-monitor 0 MKSTREAM
XGROUP CREATE genesis:sessions:dlq dlq-monitor 0 MKSTREAM

EOF

echo "✓ Redis Streams initialized successfully"
echo "✓ Consumer groups created"
echo "✓ DLQ streams ready"
