# Genesis Worker Services

Microservices architecture for Genesis facial recognition platform.

## Services Overview

| Service | Purpose | Input Events | Output Events | Scaling |
|---------|---------|--------------|---------------|---------|
| **Camera Ingestion** | Frame capture from multiple cameras | - | FrameCaptured | 1 per camera |
| **Detection** | YOLO person detection & tracking | FrameCaptured | PersonDetected | Horizontal |
| **Face Recognition** | DeepFace embedding & matching | PersonDetected | FaceRecognized | GPU-based |
| **Analytics** | Metrics, heatmaps, anomalies | PersonDetected, Zone, Face | MetricsAggregated, Alert | 2-3 instances |
| **Storage** | Database persistence | Metrics, Face, Session | - | 1-2 instances |

## Architecture

```
Camera Ingestion → Detection → Face Recognition → Analytics → Storage
      │               │              │                │           │
      └───────────────┴──────────────┴────────────────┴───────────┘
                            Redis Streams
```

## Running Workers

### Development (Local)

```bash
# Camera Ingestion
python services/camera_ingestion/worker.py --config configs/settings.yaml

# Detection Worker
python services/detection/worker.py --workers 2

# Face Recognition Worker
python services/face_recognition/worker.py --gpu 0

# Analytics Worker
python services/analytics/worker.py

# Storage Worker
python services/storage/worker.py
```

### Production (Docker)

```bash
# Build all services
docker-compose -f services/docker-compose.yml build

# Start all workers
docker-compose -f services/docker-compose.yml up -d

# Scale detection workers
docker-compose -f services/docker-compose.yml up -d --scale detection=3
```

## Monitoring

Each worker exposes:
- **Health endpoint**: `GET /health`
- **Metrics endpoint**: `GET /metrics` (Prometheus format)
- **Status endpoint**: `GET /status`

View worker status:
```bash
curl http://localhost:8080/health  # Detection worker
curl http://localhost:8081/health  # Face recognition worker
```

## Directory Structure

```
services/
├── shared/              # Shared utilities
│   ├── base_worker.py   # Base worker class
│   ├── health.py        # Health check server
│   └── metrics.py       # Prometheus metrics
├── camera_ingestion/
│   ├── worker.py
│   ├── camera.py
│   └── Dockerfile
├── detection/
│   ├── worker.py
│   ├── detector.py
│   └── Dockerfile
├── face_recognition/
│   ├── worker.py
│   ├── recognizer.py
│   └── Dockerfile
├── analytics/
│   ├── worker.py
│   ├── metrics_engine.py
│   └── Dockerfile
├── storage/
│   ├── worker.py
│   ├── database.py
│   └── Dockerfile
└── docker-compose.yml
```

## Next Steps

Phase 2 implementation creates all 5 worker services with:
- ✅ Independent deployment
- ✅ Horizontal scaling
- ✅ Health checks
- ✅ Retry logic + DLQ
- ✅ Prometheus metrics
- ✅ Async event processing
