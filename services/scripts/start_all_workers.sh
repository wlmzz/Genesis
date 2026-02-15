#!/bin/bash
# Start all Genesis worker services locally

set -e

echo "========================================"
echo "Genesis Workers - Start All"
echo "========================================"
echo

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running from correct directory
if [ ! -f "configs/settings.yaml" ]; then
    echo -e "${RED}✗ Error: Must run from project root directory${NC}"
    exit 1
fi

# Check if Redis is running
echo "1. Checking Redis..."
if docker ps | grep -q genesis-redis; then
    echo -e "${GREEN}   ✓ Redis is running${NC}"
else
    echo -e "${YELLOW}   ⚠ Redis not running, starting...${NC}"
    cd infrastructure
    docker-compose -f docker-compose.dev.yml up -d redis
    cd ..
    sleep 3
fi

# Create log directory
mkdir -p logs/workers

# Start Detection Worker
echo
echo "2. Starting Detection Worker (port 8080)..."
python services/detection/worker.py > logs/workers/detection.log 2>&1 &
DETECTION_PID=$!
echo -e "${GREEN}   ✓ Started (PID: $DETECTION_PID)${NC}"

# Start Face Recognition Worker
echo
echo "3. Starting Face Recognition Worker (port 8081)..."
python services/face_recognition/worker.py > logs/workers/face-recognition.log 2>&1 &
FACE_PID=$!
echo -e "${GREEN}   ✓ Started (PID: $FACE_PID)${NC}"

# Start Analytics Worker
echo
echo "4. Starting Analytics Worker (port 8082)..."
python services/analytics/worker.py > logs/workers/analytics.log 2>&1 &
ANALYTICS_PID=$!
echo -e "${GREEN}   ✓ Started (PID: $ANALYTICS_PID)${NC}"

# Start Storage Worker
echo
echo "5. Starting Storage Worker (port 8083)..."
python services/storage/worker.py > logs/workers/storage.log 2>&1 &
STORAGE_PID=$!
echo -e "${GREEN}   ✓ Started (PID: $STORAGE_PID)${NC}"

# Save PIDs to file
echo "$DETECTION_PID" > logs/workers/pids.txt
echo "$FACE_PID" >> logs/workers/pids.txt
echo "$ANALYTICS_PID" >> logs/workers/pids.txt
echo "$STORAGE_PID" >> logs/workers/pids.txt

# Wait for workers to start
echo
echo "6. Waiting for workers to become healthy (15 seconds)..."
sleep 15

# Check health
echo
echo "7. Checking worker health..."
all_healthy=true

for port in 8080 8081 8082 8083; do
    if curl -s -f http://localhost:$port/health > /dev/null 2>&1; then
        echo -e "${GREEN}   ✓ Worker on port $port: healthy${NC}"
    else
        echo -e "${YELLOW}   ⚠ Worker on port $port: not ready yet${NC}"
        all_healthy=false
    fi
done

echo
echo "========================================"
echo -e "${GREEN}✓ All workers started!${NC}"
echo "========================================"
echo
echo "Worker logs:"
echo "  - Detection:        logs/workers/detection.log"
echo "  - Face Recognition: logs/workers/face-recognition.log"
echo "  - Analytics:        logs/workers/analytics.log"
echo "  - Storage:          logs/workers/storage.log"
echo
echo "Health endpoints:"
echo "  - Detection:        http://localhost:8080/health"
echo "  - Face Recognition: http://localhost:8081/health"
echo "  - Analytics:        http://localhost:8082/health"
echo "  - Storage:          http://localhost:8083/health"
echo
echo "To stop all workers:"
echo "  bash services/scripts/stop_all_workers.sh"
echo
echo "To test workers:"
echo "  python services/scripts/test_workers.py"
echo
