#!/bin/bash
# Setup script for Genesis Phase 4: API Gateway & Service Mesh

set -e

echo "========================================"
echo "Genesis Phase 4 Setup"
echo "API Gateway & Service Mesh"
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
pip install -q fastapi uvicorn[standard] pydantic websockets python-multipart aiofiles
echo -e "${GREEN}   ✓ Dependencies installed${NC}"

echo
echo "2. Creating Kong network..."
docker network create genesis-network 2>/dev/null || true
echo -e "${GREEN}   ✓ Network created${NC}"

echo
echo "3. Starting Kong Gateway..."
cd infrastructure
docker-compose -f kong/docker-compose.kong.yml up -d
cd ..
sleep 5

echo
echo "4. Waiting for Kong to be healthy..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8001 > /dev/null 2>&1; then
        echo -e "${GREEN}   ✓ Kong is healthy${NC}"
        break
    fi

    echo "   Waiting for Kong... (attempt $((attempt + 1))/$max_attempts)"
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}   ✗ Kong failed to start${NC}"
    exit 1
fi

echo
echo "5. Starting API Gateway..."
python services/api_gateway/main.py &
API_PID=$!
echo $API_PID > logs/api_gateway.pid
sleep 3

echo
echo "6. Testing API Gateway..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}   ✓ API Gateway is healthy${NC}"
else
    echo -e "${RED}   ✗ API Gateway failed to start${NC}"
    kill $API_PID 2>/dev/null || true
    exit 1
fi

echo
echo "7. Testing Kong routes..."
if curl -s http://localhost:8000/api/v1/system/status > /dev/null 2>&1; then
    echo -e "${GREEN}   ✓ Kong routes working${NC}"
else
    echo -e "${YELLOW}   ⚠ Kong routes not accessible yet${NC}"
fi

echo
echo "========================================"
echo -e "${GREEN}✓ Phase 4 Setup Complete!${NC}"
echo "========================================"
echo
echo "Services running:"
echo "  • API Gateway (FastAPI): http://localhost:8000"
echo "  • Kong Gateway:          http://localhost:8000 (proxy)"
echo "  • Kong Admin API:        http://localhost:8001"
echo "  • Konga (Admin UI):      http://localhost:1337"
echo
echo "API Documentation:"
echo "  • Swagger UI:  http://localhost:8000/docs"
echo "  • ReDoc:       http://localhost:8000/redoc"
echo
echo "Test endpoints:"
echo "  curl http://localhost:8000/health"
echo "  curl http://localhost:8000/api/v1/system/status"
echo "  curl http://localhost:8000/api/v1/faces"
echo
echo "To stop services:"
echo "  kill \$(cat logs/api_gateway.pid)"
echo "  cd infrastructure && docker-compose -f kong/docker-compose.kong.yml down"
echo
