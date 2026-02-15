from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import time
import sqlite3
from pathlib import Path

@dataclass
class IdentityState:
    """Stato di un'identità tracciata"""
    person_id: str  # "person_001", "John_Doe", etc.
    track_id: int   # ID tracking temporaneo YOLO
    first_seen: float
    last_seen: float
    confidence: float
    frame_count: int = 0
    zone_history: List[Tuple[str, float]] = field(default_factory=list)

class IdentityTracker:
    """Tracker identità con persistenza su database SQLite"""

    def __init__(self, db_path: str):
        self.identities: Dict[int, IdentityState] = {}  # track_id -> IdentityState
        self.db_path = Path(db_path)
        self.init_database()

    def init_database(self):
        """Inizializza database SQLite per timeline identità"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                zone TEXT,
                confidence REAL,
                event_type TEXT
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                total_duration REAL,
                zones_visited TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def update_identity(self, track_id: int, person_id: Optional[str],
                       confidence: float, zone: Optional[str] = None):
        """Aggiorna o crea identity per un track_id"""
        now = time.time()

        if track_id not in self.identities:
            if person_id is None:
                person_id = f"unknown_{track_id}"

            self.identities[track_id] = IdentityState(
                person_id=person_id,
                track_id=track_id,
                first_seen=now,
                last_seen=now,
                confidence=confidence
            )
            self.log_event(person_id, now, zone, confidence, "first_seen")
        else:
            state = self.identities[track_id]
            state.last_seen = now
            state.frame_count += 1

            # Aggiorna person_id se riconosciuto (da unknown a known)
            if person_id and state.person_id.startswith("unknown_"):
                old_id = state.person_id
                state.person_id = person_id
                self.log_event(person_id, now, zone, confidence, "identified")

            # Log cambio zona
            if zone and (not state.zone_history or state.zone_history[-1][0] != zone):
                state.zone_history.append((zone, now))
                self.log_event(state.person_id, now, zone, confidence, "zone_enter")

    def log_event(self, person_id: str, timestamp: float, zone: Optional[str],
                  confidence: float, event_type: str):
        """Salva evento nel database"""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute('''
            INSERT INTO identities (person_id, timestamp, zone, confidence, event_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (person_id, timestamp, zone, confidence, event_type))
        conn.commit()
        conn.close()

    def get_identity(self, track_id: int) -> Optional[IdentityState]:
        """Ottiene identity state per track_id"""
        return self.identities.get(track_id)

    def cleanup_stale(self, timeout: float = 3.0):
        """Rimuove identità non viste da timeout secondi"""
        now = time.time()
        stale = [tid for tid, state in self.identities.items()
                 if (now - state.last_seen) > timeout]

        for tid in stale:
            state = self.identities[tid]
            duration = state.last_seen - state.first_seen
            zones = ",".join([z[0] for z in state.zone_history])

            # Salva sessione completata
            conn = sqlite3.connect(str(self.db_path))
            c = conn.cursor()
            c.execute('''
                INSERT INTO sessions (person_id, start_time, end_time, total_duration, zones_visited)
                VALUES (?, ?, ?, ?, ?)
            ''', (state.person_id, state.first_seen, state.last_seen, duration, zones))
            conn.commit()
            conn.close()

            del self.identities[tid]
