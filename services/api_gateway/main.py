#!/usr/bin/env python3
"""
Genesis API Gateway
Unified REST API built with FastAPI
"""
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.database import PostgresClient, VectorSearchClient
from infrastructure.events import RedisEventProducer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Pydantic Models (Request/Response Schemas)
# =============================================================================

class FaceRecognitionRequest(BaseModel):
    """Face recognition request"""
    image_base64: str = Field(..., description="Base64-encoded image")
    camera_id: str = Field(default="unknown", description="Camera identifier")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class FaceRecognitionResponse(BaseModel):
    """Face recognition response"""
    person_id: Optional[str] = Field(None, description="Identified person ID")
    confidence: float = Field(..., description="Recognition confidence")
    is_new_face: bool = Field(..., description="Whether this is a new face")
    message: str = Field(..., description="Status message")


class RegisterFaceRequest(BaseModel):
    """Register new face request"""
    image_base64: str
    person_id: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class RegisterFaceResponse(BaseModel):
    """Register face response"""
    success: bool
    person_id: str
    message: str


class FaceInfo(BaseModel):
    """Face information"""
    person_id: str
    first_seen: datetime
    last_seen: datetime
    total_appearances: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricsSnapshot(BaseModel):
    """Current metrics snapshot"""
    timestamp: datetime
    camera_id: str
    people_total: int = 0
    queue_len: int = 0
    avg_wait_sec: float = 0.0
    people_by_zone: Dict[str, int] = Field(default_factory=dict)
    new_faces: int = 0
    recognized_faces: int = 0


class Session(BaseModel):
    """Session information"""
    session_id: str
    person_id: Optional[str]
    camera_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[int]
    zones_visited: List[str] = Field(default_factory=list)


class Alert(BaseModel):
    """Alert information"""
    alert_id: int
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    camera_id: Optional[str]
    person_id: Optional[str]
    acknowledged: bool = False


class SystemStatus(BaseModel):
    """System status"""
    total_cameras: int
    active_cameras: int
    total_identities: int
    active_sessions: int
    total_metrics: int
    database_size: str
    uptime_seconds: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    healthy: bool
    version: str
    timestamp: datetime


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Genesis API",
    description="Unified REST API for Genesis Facial Recognition Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global database client
db_client: Optional[PostgresClient] = None
vector_client: Optional[VectorSearchClient] = None
start_time = datetime.now()


# =============================================================================
# Dependency Injection
# =============================================================================

async def get_db() -> PostgresClient:
    """Get database client"""
    if db_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized"
        )
    return db_client


async def get_vector_client() -> VectorSearchClient:
    """Get vector search client"""
    if vector_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector search not initialized"
        )
    return vector_client


# Optional API key authentication
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key (if configured)"""
    # TODO: Implement proper API key validation
    # For now, this is a placeholder
    return True


# =============================================================================
# Startup/Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global db_client, vector_client

    logger.info("Starting Genesis API Gateway...")

    # Connect to PostgreSQL
    db_client = PostgresClient(
        host="localhost",
        port=5432,
        database="genesis",
        user="genesis",
        password="genesis_dev_password"
    )

    await db_client.connect()
    logger.info("✓ Connected to PostgreSQL")

    # Initialize vector search
    vector_client = VectorSearchClient(db_client)
    logger.info("✓ Vector search initialized")

    logger.info("✓ API Gateway ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API Gateway...")

    if db_client:
        await db_client.close()

    logger.info("✓ API Gateway shutdown complete")


# =============================================================================
# Health & System Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    healthy = True

    # Check database
    if db_client:
        healthy = await db_client.health_check()

    return HealthResponse(
        status="healthy" if healthy else "unhealthy",
        healthy=healthy,
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.get("/api/v1/system/status", response_model=SystemStatus, tags=["System"])
async def get_system_status(db: PostgresClient = Depends(get_db)):
    """Get system status"""
    stats = await db.get_database_stats()

    # Get additional stats
    total_cameras = 1  # TODO: Get from camera registry
    active_cameras = 1  # TODO: Get from active camera list
    active_sessions = await db.fetchval(
        "SELECT COUNT(*) FROM sessions WHERE end_time IS NULL"
    )

    uptime = (datetime.now() - start_time).total_seconds()

    return SystemStatus(
        total_cameras=total_cameras,
        active_cameras=active_cameras,
        total_identities=stats.get('total_identities', 0),
        active_sessions=active_sessions or 0,
        total_metrics=stats.get('total_metrics', 0),
        database_size=stats.get('database_size', 'unknown'),
        uptime_seconds=uptime
    )


# =============================================================================
# Face Recognition Endpoints
# =============================================================================

@app.post("/api/v1/faces/recognize", response_model=FaceRecognitionResponse, tags=["Faces"])
async def recognize_face(
    request: FaceRecognitionRequest,
    vector: VectorSearchClient = Depends(get_vector_client)
):
    """
    Recognize a face from base64-encoded image

    Note: This endpoint requires face embedding extraction.
    For full implementation, integrate with face_recognition worker.
    """
    # TODO: Implement face embedding extraction from image
    # This would require deploying face_recognition as a service
    # For now, return a placeholder response

    return FaceRecognitionResponse(
        person_id=None,
        confidence=0.0,
        is_new_face=True,
        message="Face recognition requires face_recognition worker to be running as a service"
    )


@app.post("/api/v1/faces/register", response_model=RegisterFaceResponse, tags=["Faces"])
async def register_face(
    request: RegisterFaceRequest,
    vector: VectorSearchClient = Depends(get_vector_client)
):
    """Register a new face"""
    # TODO: Extract embedding from image
    # For now, return placeholder

    return RegisterFaceResponse(
        success=False,
        person_id=request.person_id,
        message="Face registration requires face_recognition worker"
    )


@app.get("/api/v1/faces", response_model=List[FaceInfo], tags=["Faces"])
async def list_faces(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: PostgresClient = Depends(get_db)
):
    """List all registered faces"""
    rows = await db.fetch(
        """
        SELECT person_id, first_seen, last_seen, total_appearances, metadata
        FROM identities
        ORDER BY last_seen DESC
        LIMIT $1 OFFSET $2
        """,
        limit,
        offset
    )

    faces = []
    for row in rows:
        import json
        faces.append(FaceInfo(
            person_id=row['person_id'],
            first_seen=row['first_seen'],
            last_seen=row['last_seen'],
            total_appearances=row['total_appearances'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        ))

    return faces


@app.get("/api/v1/faces/{person_id}", response_model=FaceInfo, tags=["Faces"])
async def get_face_info(
    person_id: str,
    db: PostgresClient = Depends(get_db)
):
    """Get information about a specific face"""
    face = await db.get_identity(person_id)

    if not face:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face {person_id} not found"
        )

    import json
    return FaceInfo(
        person_id=face['person_id'],
        first_seen=face['first_seen'],
        last_seen=face['last_seen'],
        total_appearances=face['total_appearances'],
        metadata=json.loads(face['metadata']) if face.get('metadata') else {}
    )


# =============================================================================
# Metrics Endpoints
# =============================================================================

@app.get("/api/v1/metrics/current", response_model=MetricsSnapshot, tags=["Metrics"])
async def get_current_metrics(
    camera_id: str = Query("unknown"),
    db: PostgresClient = Depends(get_db)
):
    """Get current metrics snapshot"""
    # Get most recent metrics
    row = await db.fetchone(
        """
        SELECT * FROM metrics
        WHERE camera_id = $1
        ORDER BY time DESC
        LIMIT 1
        """,
        camera_id
    )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No metrics found for camera {camera_id}"
        )

    import json
    return MetricsSnapshot(
        timestamp=row['time'],
        camera_id=row['camera_id'],
        people_total=row['people_total'],
        queue_len=row['queue_len'],
        avg_wait_sec=row['avg_wait_sec'],
        people_by_zone=json.loads(row['people_by_zone']) if row['people_by_zone'] else {},
        new_faces=row['new_faces'],
        recognized_faces=row['recognized_faces']
    )


@app.get("/api/v1/metrics/range", response_model=List[MetricsSnapshot], tags=["Metrics"])
async def get_metrics_range(
    camera_id: str = Query("unknown"),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    db: PostgresClient = Depends(get_db)
):
    """Get metrics for time range"""
    if not start_time:
        start_time = datetime.now() - timedelta(hours=1)
    if not end_time:
        end_time = datetime.now()

    metrics = await db.get_metrics_range(camera_id, start_time, end_time)

    import json
    return [
        MetricsSnapshot(
            timestamp=m['time'],
            camera_id=m['camera_id'],
            people_total=m['people_total'],
            queue_len=m['queue_len'],
            avg_wait_sec=m['avg_wait_sec'],
            people_by_zone=json.loads(m['people_by_zone']) if m['people_by_zone'] else {},
            new_faces=m['new_faces'],
            recognized_faces=m['recognized_faces']
        )
        for m in metrics
    ]


# =============================================================================
# Session Endpoints
# =============================================================================

@app.get("/api/v1/sessions/active", response_model=List[Session], tags=["Sessions"])
async def get_active_sessions(
    camera_id: Optional[str] = Query(None),
    db: PostgresClient = Depends(get_db)
):
    """Get active sessions"""
    query = """
        SELECT session_id, person_id, camera_id, start_time, end_time,
               duration_seconds, zones_visited
        FROM sessions
        WHERE end_time IS NULL
    """
    params = []

    if camera_id:
        query += " AND camera_id = $1"
        params.append(camera_id)

    query += " ORDER BY start_time DESC LIMIT 50"

    rows = await db.fetch(query, *params)

    return [
        Session(
            session_id=str(row['session_id']),
            person_id=row['person_id'],
            camera_id=row['camera_id'],
            start_time=row['start_time'],
            end_time=row['end_time'],
            duration_seconds=row['duration_seconds'],
            zones_visited=row['zones_visited'] or []
        )
        for row in rows
    ]


@app.get("/api/v1/sessions/{session_id}", response_model=Session, tags=["Sessions"])
async def get_session(
    session_id: str,
    db: PostgresClient = Depends(get_db)
):
    """Get session by ID"""
    row = await db.fetchone(
        """
        SELECT session_id, person_id, camera_id, start_time, end_time,
               duration_seconds, zones_visited
        FROM sessions
        WHERE session_id = $1
        """,
        session_id
    )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    return Session(
        session_id=str(row['session_id']),
        person_id=row['person_id'],
        camera_id=row['camera_id'],
        start_time=row['start_time'],
        end_time=row['end_time'],
        duration_seconds=row['duration_seconds'],
        zones_visited=row['zones_visited'] or []
    )


# =============================================================================
# Alert Endpoints
# =============================================================================

@app.get("/api/v1/alerts", response_model=List[Alert], tags=["Alerts"])
async def get_recent_alerts(
    camera_id: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    unacknowledged_only: bool = Query(False),
    db: PostgresClient = Depends(get_db)
):
    """Get recent alerts"""
    query = "SELECT * FROM alerts WHERE 1=1"
    params = []
    param_count = 0

    if camera_id:
        param_count += 1
        query += f" AND camera_id = ${param_count}"
        params.append(camera_id)

    if severity:
        param_count += 1
        query += f" AND severity = ${param_count}"
        params.append(severity)

    if unacknowledged_only:
        query += " AND NOT acknowledged"

    param_count += 1
    query += f" ORDER BY timestamp DESC LIMIT ${param_count}"
    params.append(limit)

    rows = await db.fetch(query, *params)

    return [
        Alert(
            alert_id=row['alert_id'],
            alert_type=row['alert_type'],
            severity=row['severity'],
            message=row['message'],
            timestamp=row['timestamp'],
            camera_id=row['camera_id'],
            person_id=row['person_id'],
            acknowledged=row['acknowledged']
        )
        for row in rows
    ]


@app.post("/api/v1/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge_alert(
    alert_id: int,
    acknowledged_by: str = Query(...),
    db: PostgresClient = Depends(get_db)
):
    """Acknowledge an alert"""
    result = await db.execute(
        """
        UPDATE alerts
        SET acknowledged = TRUE,
            acknowledged_at = NOW(),
            acknowledged_by = $1
        WHERE alert_id = $2
        """,
        acknowledged_by,
        alert_id
    )

    if "UPDATE 0" in result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )

    return {"success": True, "message": "Alert acknowledged"}


# =============================================================================
# WebSocket Endpoints (for real-time updates)
# =============================================================================

@app.websocket("/ws/metrics/{camera_id}")
async def websocket_metrics(websocket, camera_id: str):
    """WebSocket endpoint for real-time metrics streaming"""
    await websocket.accept()

    try:
        while True:
            # Get latest metrics
            # TODO: Implement real-time metrics streaming
            await asyncio.sleep(1)
            # await websocket.send_json({...})

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run API gateway"""
    uvicorn.run(
        "services.api_gateway.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
