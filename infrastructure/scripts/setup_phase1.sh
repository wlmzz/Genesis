#!/bin/bash
# Setup script for Genesis Phase 1: Event-Driven Foundation

set -e

echo "========================================"
echo "Genesis Phase 1 Setup"
echo "Event-Driven Foundation"
echo "========================================"
echo

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running from correct directory
if [ ! -f "configs/settings.yaml" ]; then
    echo -e "${RED}✗ Error: Must run from project root directory${NC}"
    echo "  cd to StoreOps Ai directory and run:"
    echo "  bash infrastructure/scripts/setup_phase1.sh"
    exit 1
fi

echo "1. Installing Python dependencies..."
echo "   Installing infrastructure requirements..."
pip install -q -r infrastructure/requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}   ✓ Dependencies installed${NC}"
else
    echo -e "${RED}   ✗ Failed to install dependencies${NC}"
    exit 1
fi

echo
echo "2. Starting infrastructure services..."
echo "   Launching Docker Compose..."

cd infrastructure
docker-compose -f docker-compose.dev.yml up -d

if [ $? -eq 0 ]; then
    echo -e "${GREEN}   ✓ Services started${NC}"
else
    echo -e "${RED}   ✗ Failed to start services${NC}"
    cd ..
    exit 1
fi
cd ..

echo
echo "3. Waiting for services to be healthy..."
sleep 5

# Check Redis
echo -n "   Checking Redis... "
if docker exec genesis-redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}   Redis is not responding${NC}"
    exit 1
fi

# Check PostgreSQL
echo -n "   Checking PostgreSQL... "
if docker exec genesis-postgres pg_isready -U genesis > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠ PostgreSQL not ready yet (this is OK for Phase 1)${NC}"
fi

echo
echo "4. Initializing Redis Streams..."
docker exec genesis-redis sh -c '
redis-cli XGROUP CREATE genesis:frames frame-workers 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE genesis:frames detection-workers 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE genesis:persons person-workers 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE genesis:persons face-workers 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE genesis:faces face-storage-workers 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE genesis:zones zone-workers 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE genesis:zones analytics-workers 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE genesis:alerts alert-workers 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE genesis:metrics metrics-workers 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE genesis:metrics storage-workers 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE genesis:sessions session-workers 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE genesis:sessions storage-workers 0 MKSTREAM 2>/dev/null || true
echo "✓ Streams initialized"
'

echo -e "${GREEN}   ✓ Redis Streams ready${NC}"

echo
echo "5. Running event bus test..."
python infrastructure/scripts/test_event_bus.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}   ✓ Event bus test passed${NC}"
else
    echo -e "${RED}   ✗ Event bus test failed${NC}"
    exit 1
fi

echo
echo "========================================"
echo -e "${GREEN}✓ Phase 1 Setup Complete!${NC}"
echo "========================================"
echo
echo "Services running:"
echo "  • Redis (events):        http://localhost:6379"
echo "  • Redis Commander:       http://localhost:8081"
echo "  • PostgreSQL:            localhost:5432"
echo "  • pgAdmin:               http://localhost:8082"
echo "  • Prometheus:            http://localhost:9090"
echo "  • Grafana:               http://localhost:3000 (admin/admin)"
echo
echo "To enable event-driven mode:"
echo "  1. Edit configs/settings.yaml"
echo "  2. Set: event_driven.enabled: true"
echo "  3. Run: python app/run_camera.py"
echo
echo "To stop services:"
echo "  cd infrastructure && docker-compose -f docker-compose.dev.yml down"
echo
echo "Next: Phase 2 - Worker Services Implementation"
echo
