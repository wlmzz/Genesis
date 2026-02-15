#!/usr/bin/env python3
"""
Genesis - REST API Server
API per accesso dati e controllo sistema
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
import uvicorn

app = FastAPI(
    title="Genesis API",
    description="REST API per sistema Genesis facial recognition & analytics",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = "data/outputs"
CSV_PATH = os.path.join(DATA_DIR, "metrics.csv")
DB_PATH = os.path.join(DATA_DIR, "identities.db")

# Models
class MetricsResponse(BaseModel):
    timestamp: int
    people_total: int
    queue_len: int
    avg_queue_wait_sec: float
    zones: Dict[str, int]

class IdentityEvent(BaseModel):
    person_id: str
    timestamp: float
    zone: Optional[str]
    confidence: float
    event_type: str

class SessionInfo(BaseModel):
    person_id: str
    start_time: float
    end_time: Optional[float]
    total_duration: Optional[float]
    zones_visited: str

class StatsResponse(BaseModel):
    total_people: int
    avg_queue_length: float
    avg_wait_time: float
    unique_identities: int
    total_sessions: int

# Endpoints

@app.get("/")
def root():
    """Health check"""
    return {
        "service": "Genesis API",
        "status": "online",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics/latest", response_model=MetricsResponse)
def get_latest_metrics():
    """Ottieni metriche più recenti"""
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="Metrics file not found")

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise HTTPException(status_code=404, detail="No metrics available")

    latest = df.iloc[-1]

    # Estrai zone
    zones = {}
    for col in df.columns:
        if col.startswith("zone_"):
            zone_name = col.replace("zone_", "")
            zones[zone_name] = int(latest[col])

    return MetricsResponse(
        timestamp=int(latest["ts"]),
        people_total=int(latest["people_total"]),
        queue_len=int(latest["queue_len"]),
        avg_queue_wait_sec=float(latest["avg_queue_wait_sec"]),
        zones=zones
    )

@app.get("/metrics/history")
def get_metrics_history(
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0)
):
    """Ottieni storico metriche"""
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="Metrics file not found")

    df = pd.read_csv(CSV_PATH)
    total = len(df)

    # Paginazione
    df_page = df.iloc[offset:offset+limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": df_page.to_dict(orient="records")
    }

@app.get("/metrics/stats", response_model=StatsResponse)
def get_overall_stats():
    """Statistiche aggregate"""
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="Metrics file not found")

    df = pd.read_csv(CSV_PATH)

    conn = sqlite3.connect(DB_PATH) if os.path.exists(DB_PATH) else None

    unique_identities = 0
    total_sessions = 0

    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT person_id) FROM identities")
        unique_identities = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]
        conn.close()

    return StatsResponse(
        total_people=int(df["people_total"].sum()),
        avg_queue_length=float(df["queue_len"].mean()),
        avg_wait_time=float(df["avg_queue_wait_sec"].mean()),
        unique_identities=unique_identities,
        total_sessions=total_sessions
    )

@app.get("/identities/events", response_model=List[IdentityEvent])
def get_identity_events(
    person_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """Eventi identità"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Identity database not found")

    conn = sqlite3.connect(DB_PATH)

    if person_id:
        query = f"SELECT * FROM identities WHERE person_id = ? ORDER BY timestamp DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(person_id, limit))
    else:
        query = f"SELECT * FROM identities ORDER BY timestamp DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(limit,))

    conn.close()

    events = []
    for _, row in df.iterrows():
        events.append(IdentityEvent(
            person_id=row["person_id"],
            timestamp=row["timestamp"],
            zone=row["zone"],
            confidence=row["confidence"],
            event_type=row["event_type"]
        ))

    return events

@app.get("/identities/sessions", response_model=List[SessionInfo])
def get_sessions(
    person_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500)
):
    """Sessioni identità"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Identity database not found")

    conn = sqlite3.connect(DB_PATH)

    if person_id:
        query = "SELECT * FROM sessions WHERE person_id = ? ORDER BY start_time DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(person_id, limit))
    else:
        query = "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(limit,))

    conn.close()

    sessions = []
    for _, row in df.iterrows():
        sessions.append(SessionInfo(
            person_id=row["person_id"],
            start_time=row["start_time"],
            end_time=row.get("end_time"),
            total_duration=row.get("total_duration"),
            zones_visited=row["zones_visited"]
        ))

    return sessions

@app.get("/identities/list")
def list_identities():
    """Lista tutte le identità conosciute"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Identity database not found")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT person_id, COUNT(*) as visit_count, MAX(timestamp) as last_seen
        FROM identities
        GROUP BY person_id
        ORDER BY visit_count DESC
    """)

    results = cursor.fetchall()
    conn.close()

    identities = []
    for row in results:
        identities.append({
            "person_id": row[0],
            "visit_count": row[1],
            "last_seen": row[2]
        })

    return {"identities": identities}

@app.get("/analytics/heatmap")
def get_heatmap_data():
    """Dati per heatmap (placeholder - implementare in run_camera)"""
    # TODO: implementare salvataggio heatmap data
    return {"message": "Heatmap data not yet implemented"}

@app.get("/analytics/predictions")
def get_predictions():
    """Predizioni future (placeholder)"""
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="Metrics file not found")

    df = pd.read_csv(CSV_PATH)

    # Simple prediction: media ultimi 10 valori
    recent_queue = df["queue_len"].tail(10).mean()

    return {
        "predicted_queue_next_hour": round(recent_queue, 1),
        "confidence": "low",
        "note": "Simple moving average prediction"
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Genesis API Server")
    print("="*60)
    print("\nStarting server on http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
