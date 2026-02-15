# Genesis Infrastructure

Infrastructure components for Genesis enterprise architecture including event bus, caching, and deployment configurations.

## Phase 1: Event-Driven Foundation ✅

### Components

- **Events System** (`events/`): Redis Streams-based event bus
  - Event types: 7 domain events (FrameCaptured, PersonDetected, FaceRecognized, Zone, Alert, Metrics, Session)
  - Producer: Publishes events with backpressure management
  - Consumer: Base class for workers with retry logic and DLQ
  - Retry policy: Configurable backoff and dead letter queue

- **Frame Cache** (`cache/`): Redis-based frame sharing
  - LZ4 compression for efficient storage
  - TTL-based expiration
  - Binary serialization with pickle

- **Docker Compose** (`docker-compose.dev.yml`): Development infrastructure
  - Redis (event streaming)
  - PostgreSQL with pgvector (identity storage)
  - Prometheus + Grafana (monitoring)
  - Redis Commander + pgAdmin (debugging tools)

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r infrastructure/requirements.txt
   ```

2. **Start infrastructure:**
   ```bash
   cd infrastructure
   docker-compose -f docker-compose.dev.yml up -d
   ```

3. **Initialize Redis Streams:**
   ```bash
   chmod +x redis/init-streams.sh
   docker exec -it genesis-redis /bin/sh -c "$(cat redis/init-streams.sh)"
   ```

4. **Enable event-driven mode:**
   Edit `configs/settings.yaml`:
   ```yaml
   event_driven:
     enabled: true  # Set to true
   ```

5. **Run camera with events:**
   ```bash
   python app/run_camera.py
   ```

### Verification

1. **Check Redis is healthy:**
   ```bash
   docker exec -it genesis-redis redis-cli ping
   # Should return: PONG
   ```

2. **View events in Redis:**
   ```bash
   docker exec -it genesis-redis redis-cli
   > XREAD COUNT 10 STREAMS genesis:frames 0
   > XREAD COUNT 10 STREAMS genesis:persons 0
   ```

3. **Monitor with Redis Commander:**
   Open http://localhost:8081

4. **Check backpressure:**
   ```bash
   docker exec -it genesis-redis redis-cli
   > XPENDING genesis:frames detection-workers
   ```

### Feature Flags

Event-driven mode is controlled by `event_driven.enabled` in `settings.yaml`:

- `false` (default): Sync mode - traditional frame-by-frame processing
- `true`: Event-driven mode - publishes events to Redis Streams

Both modes work simultaneously (dual-mode) for gradual migration.

### Event Streams

| Stream | Purpose | Consumer Groups |
|--------|---------|-----------------|
| `genesis:frames` | Frame capture events | `frame-workers`, `detection-workers` |
| `genesis:persons` | Person detection events | `person-workers`, `face-workers` |
| `genesis:faces` | Face recognition events | `face-storage-workers` |
| `genesis:zones` | Zone entry/exit events | `zone-workers`, `analytics-workers` |
| `genesis:alerts` | Alert events | `alert-workers` |
| `genesis:metrics` | Aggregated metrics | `metrics-workers`, `storage-workers` |
| `genesis:sessions` | Session events | `session-workers`, `storage-workers` |

### Backpressure Management

Automatic backpressure prevents overwhelming downstream services:

- **Block threshold**: 800 pending messages (warning)
- **Max pending**: 1000 messages (hard block)
- **Behavior**: When exceeded, `BackpressureError` is raised

Configure in `settings.yaml`:
```yaml
event_driven:
  backpressure:
    enabled: true
    max_pending: 1000
    block_threshold: 800
```

### Retry Policy

Failed events are retried with exponential backoff:

- **Max attempts**: 3
- **Backoff**: [1s, 5s, 15s]
- **DLQ**: Messages exceeding max retries go to `{stream}:dlq`

Monitor DLQ:
```bash
docker exec -it genesis-redis redis-cli XLEN genesis:frames:dlq
```

### Frame Cache

Frames are cached in Redis for sharing between workers:

- **TTL**: 60 seconds (configurable)
- **Compression**: LZ4 (70-80% size reduction)
- **Max size**: 500 MB (configurable)

Cache statistics:
```python
from infrastructure.cache import FrameCache

cache = FrameCache()
size_bytes = cache.get_cache_size()
print(f"Cache size: {size_bytes / 1024 / 1024:.1f} MB")
```

### Next Phase

Phase 2 will implement worker services that consume these events:
- Camera Ingestion Worker
- Detection Worker (YOLO)
- Face Recognition Worker
- Analytics Worker
- Storage Worker

See `../docs/architecture/phase2-workers.md` for details.

## Troubleshooting

### Redis connection refused
```bash
# Check Redis is running
docker ps | grep genesis-redis

# Check logs
docker logs genesis-redis
```

### Events not appearing
```bash
# Verify event producer is publishing
docker exec -it genesis-redis redis-cli
> XINFO STREAM genesis:frames

# Check consumer groups exist
> XINFO GROUPS genesis:frames
```

### Backpressure activating too often
- Increase `block_threshold` in settings.yaml
- Scale up worker instances (Phase 2)
- Reduce camera FPS

### Frame cache full
- Reduce TTL: `frame_cache.ttl_seconds`
- Increase max size: `frame_cache.max_size_mb`
- Clear cache: `docker exec -it genesis-redis redis-cli FLUSHDB`

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│         app/run_camera.py (Dual Mode)           │
│  ┌──────────────┐        ┌──────────────────┐  │
│  │  Sync Mode   │   +    │ Event Publisher  │  │
│  │  (existing)  │        │  (new)           │  │
│  └──────────────┘        └────────┬─────────┘  │
└──────────────────────────────────┼──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Redis Streams           │
                    │  (7 event types)            │
                    │  + Frame Cache (LZ4)        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    Worker Services          │
                    │  (Phase 2 - Coming Soon)    │
                    └─────────────────────────────┘
```

## License

Part of Genesis Facial Recognition Platform
