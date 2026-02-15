#!/bin/bash

# Phase 6: MLOps Pipeline Setup Script
# This script sets up the complete MLOps infrastructure for Genesis

set -e

PROJECT_ROOT="/Users/wlmzz/StudioProjects/StoreOps Ai"
cd "$PROJECT_ROOT"

echo "🚀 Setting up Phase 6: MLOps Pipeline"
echo "======================================"

# Step 1: Install Python dependencies
echo ""
echo "📦 Installing MLOps dependencies..."
pip install mlflow==2.9.2 \
    dvc==3.50.0 \
    onnx==1.16.0 \
    onnxruntime==1.17.0 \
    scipy==1.13.0 \
    scikit-learn==1.4.0 \
    boto3==1.34.0

# Step 2: Create genesis-network if not exists
echo ""
echo "🔧 Ensuring Docker network exists..."
if ! docker network inspect genesis-network &>/dev/null; then
    docker network create genesis-network
    echo "✅ Created genesis-network"
else
    echo "✅ genesis-network already exists"
fi

# Step 3: Start MLOps stack
echo ""
echo "🐳 Starting MLOps stack (MLflow, PostgreSQL, MinIO, DVC)..."
cd infrastructure/mlops
docker-compose -f docker-compose.mlops.yml up -d

# Step 4: Wait for services
echo ""
echo "⏳ Waiting for services to be healthy..."

echo "  - Waiting for PostgreSQL..."
timeout 60 bash -c 'until docker exec genesis-postgres-mlflow pg_isready -U genesis -d mlflow &>/dev/null; do sleep 2; done'
echo "    ✅ PostgreSQL ready"

echo "  - Waiting for MLflow..."
timeout 90 bash -c 'until curl -sf http://localhost:5000/health &>/dev/null; do sleep 3; done'
echo "    ✅ MLflow ready"

echo "  - Waiting for MinIO..."
timeout 60 bash -c 'until curl -sf http://localhost:9000/minio/health/live &>/dev/null; do sleep 2; done'
echo "    ✅ MinIO ready"

# Step 5: Create MLflow database tables
echo ""
echo "🗄️  Initializing MLflow database..."
docker exec genesis-mlflow mlflow db upgrade postgresql://genesis:genesis@postgres:5432/mlflow || true

# Step 6: Create MinIO bucket for DVC
echo ""
echo "🗃️  Creating MinIO bucket for DVC..."
docker exec genesis-minio sh -c "mc alias set myminio http://localhost:9000 minioadmin minioadmin; mc mb myminio/genesis-datasets || true"

# Step 7: Initialize DVC
echo ""
echo "📊 Initializing DVC..."
if [ ! -d ".dvc" ]; then
    dvc init
    echo "  ✅ DVC initialized"
else
    echo "  ✅ DVC already initialized"
fi

# Step 8: Verify services
echo ""
echo "✅ Verifying services..."

# Check MLflow
if curl -sf http://localhost:5000/health &>/dev/null; then
    echo "  ✅ MLflow is running at http://localhost:5000"
else
    echo "  ❌ MLflow is not responding"
    exit 1
fi

# Check MinIO
if curl -sf http://localhost:9000/minio/health/live &>/dev/null; then
    echo "  ✅ MinIO is running at http://localhost:9000 (Console: http://localhost:9001)"
else
    echo "  ❌ MinIO is not responding"
    exit 1
fi

# Check PostgreSQL
if docker exec genesis-postgres-mlflow pg_isready -U genesis -d mlflow &>/dev/null; then
    echo "  ✅ PostgreSQL is running (port 5433)"
else
    echo "  ❌ PostgreSQL is not responding"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ Phase 6 Setup Complete!"
echo "======================================"
echo ""
echo "🎉 MLOps Infrastructure is ready!"
echo ""
echo "Access Points:"
echo "  - MLflow UI:    http://localhost:5000"
echo "  - MinIO Console: http://localhost:9001 (admin/minioadmin)"
echo "  - PostgreSQL:   localhost:5433 (user: genesis, db: mlflow)"
echo ""
echo "Next Steps:"
echo "  1. Register your first model:"
echo "     python -m mlops.mlflow.model_registry"
echo ""
echo "  2. Track a training run:"
echo "     import mlflow"
echo "     mlflow.set_tracking_uri('http://localhost:5000')"
echo "     mlflow.log_metric('accuracy', 0.95)"
echo ""
echo "  3. Version your datasets with DVC:"
echo "     dvc add data/training_set.csv"
echo "     dvc push"
echo ""
echo "📚 Documentation: See PHASE6_COMPLETE.md"
echo ""
