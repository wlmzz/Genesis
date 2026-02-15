#!/bin/bash
# Setup script for Genesis Phase 5: Observability Stack

set -e

echo "========================================"
echo "Genesis Phase 5 Setup"
echo "Observability Stack"
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
pip install -q prometheus-client opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-instrumentation-redis \
    opentelemetry-instrumentation-asyncpg \
    opentelemetry-instrumentation-requests \
    opentelemetry-instrumentation-logging
echo -e "${GREEN}   ✓ Dependencies installed${NC}"

echo
echo "2. Ensuring genesis-network exists..."
docker network create genesis-network 2>/dev/null || true
echo -e "${GREEN}   ✓ Network ready${NC}"

echo
echo "3. Starting observability stack..."
cd infrastructure/observability
docker-compose -f docker-compose.observability.yml up -d
cd ../..

echo
echo "4. Waiting for services to be healthy..."
max_attempts=60
attempt=0

# Wait for Prometheus
echo "   Checking Prometheus..."
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
        echo -e "${GREEN}   ✓ Prometheus is healthy${NC}"
        break
    fi
    
    if [ $attempt -eq $((max_attempts - 1)) ]; then
        echo -e "${RED}   ✗ Prometheus failed to start${NC}"
        exit 1
    fi
    
    sleep 2
    attempt=$((attempt + 1))
done

# Wait for Grafana
attempt=0
echo "   Checking Grafana..."
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}   ✓ Grafana is healthy${NC}"
        break
    fi
    
    if [ $attempt -eq $((max_attempts - 1)) ]; then
        echo -e "${RED}   ✗ Grafana failed to start${NC}"
        exit 1
    fi
    
    sleep 2
    attempt=$((attempt + 1))
done

# Wait for Loki
attempt=0
echo "   Checking Loki..."
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:3100/ready > /dev/null 2>&1; then
        echo -e "${GREEN}   ✓ Loki is healthy${NC}"
        break
    fi
    
    if [ $attempt -eq $((max_attempts - 1)) ]; then
        echo -e "${YELLOW}   ⚠ Loki may not be ready${NC}"
        break
    fi
    
    sleep 2
    attempt=$((attempt + 1))
done

# Wait for Tempo
attempt=0
echo "   Checking Tempo..."
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:3200/ready > /dev/null 2>&1; then
        echo -e "${GREEN}   ✓ Tempo is healthy${NC}"
        break
    fi
    
    if [ $attempt -eq $((max_attempts - 1)) ]; then
        echo -e "${YELLOW}   ⚠ Tempo may not be ready${NC}"
        break
    fi
    
    sleep 2
    attempt=$((attempt + 1))
done

echo
echo "5. Verifying Prometheus targets..."
sleep 5
targets=$(curl -s http://localhost:9090/api/v1/targets | grep -o '"health":"up"' | wc -l)
echo -e "${GREEN}   ✓ $targets Prometheus targets UP${NC}"

echo
echo "6. Verifying Grafana datasources..."
datasources=$(curl -s -u admin:genesis-admin-change-me http://localhost:3000/api/datasources | grep -o '"name"' | wc -l)
echo -e "${GREEN}   ✓ $datasources datasources configured${NC}"

echo
echo "========================================"
echo -e "${GREEN}✓ Phase 5 Setup Complete!${NC}"
echo "========================================"
echo
echo "Observability services running:"
echo "  • Prometheus:     http://localhost:9090"
echo "  • Grafana:        http://localhost:3000"
echo "  • Loki:           http://localhost:3100"
echo "  • Tempo:          http://localhost:3200"
echo "  • Alertmanager:   http://localhost:9093"
echo "  • Node Exporter:  http://localhost:9100"
echo
echo "Grafana credentials:"
echo "  Username: admin"
echo "  Password: genesis-admin-change-me"
echo
echo "Available dashboards:"
echo "  • Genesis - System Overview"
echo "  • Genesis - Face Recognition"
echo
echo "Next steps:"
echo "  1. Open Grafana: http://localhost:3000"
echo "  2. Navigate to Dashboards > Genesis"
echo "  3. Start workers to see metrics flowing"
echo "  4. Configure alerts in Alertmanager"
echo
echo "To stop services:"
echo "  cd infrastructure/observability"
echo "  docker-compose -f docker-compose.observability.yml down"
echo
echo "To view logs:"
echo "  docker logs -f genesis-prometheus"
echo "  docker logs -f genesis-grafana"
echo "  docker logs -f genesis-loki"
echo
