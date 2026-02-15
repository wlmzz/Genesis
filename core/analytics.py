from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List
import time

from core.zones import point_in_polygon

@dataclass
class TrackState:
    """Stato di un track (persona tracciata)"""
    first_seen_ts: float
    last_seen_ts: float
    zone_enter_ts: Dict[str, float] = field(default_factory=dict)
    zone_dwell_sec: Dict[str, float] = field(default_factory=dict)

@dataclass
class MetricsSnapshot:
    """Snapshot metriche in un istante"""
    ts: float
    people_total: int
    people_by_zone: Dict[str, int]
    avg_queue_wait_sec: float
    queue_len: int

class GenesisAnalytics:
    """Analytics engine per Genesis - gestisce KPI, code, zone"""

    def __init__(self, zone_names: List[str], queue_zone: str = "queue_area"):
        self.tracks: Dict[int, TrackState] = {}
        self.zone_names = zone_names
        self.queue_zone = queue_zone

    def update(self, track_centers: Dict[int, Tuple[int, int]], zones: Dict[str, List[Tuple[int, int]]], now: float | None = None):
        """Aggiorna stato analytics con nuove posizioni"""
        now = now or time.time()

        # update per-track state
        active_ids: Set[int] = set(track_centers.keys())
        for tid, center in track_centers.items():
            if tid not in self.tracks:
                self.tracks[tid] = TrackState(first_seen_ts=now, last_seen_ts=now)
            st = self.tracks[tid]
            st.last_seen_ts = now

            # zone membership + dwell
            for zname, poly in zones.items():
                inside = point_in_polygon(center, poly)
                in_zone = zname in st.zone_enter_ts
                if inside and not in_zone:
                    st.zone_enter_ts[zname] = now
                elif (not inside) and in_zone:
                    enter = st.zone_enter_ts.pop(zname)
                    st.zone_dwell_sec[zname] = st.zone_dwell_sec.get(zname, 0.0) + (now - enter)

        # cleanup stale tracks (se non visti da 2s)
        stale = [tid for tid, st in self.tracks.items() if (now - st.last_seen_ts) > 2.0]
        for tid in stale:
            st = self.tracks[tid]
            # close any open zone intervals
            for zname, enter in list(st.zone_enter_ts.items()):
                st.zone_dwell_sec[zname] = st.zone_dwell_sec.get(zname, 0.0) + (now - enter)
                st.zone_enter_ts.pop(zname, None)
            del self.tracks[tid]

    def snapshot(self, track_centers: Dict[int, Tuple[int, int]], zones: Dict[str, List[Tuple[int, int]]], now: float | None = None) -> MetricsSnapshot:
        """Genera snapshot corrente delle metriche"""
        now = now or time.time()
        people_by_zone: Dict[str, int] = {z: 0 for z in zones.keys()}

        # who is currently inside each zone (based on center)
        for tid, c in track_centers.items():
            for zname, poly in zones.items():
                if point_in_polygon(c, poly):
                    people_by_zone[zname] += 1

        queue_len = people_by_zone.get(self.queue_zone, 0)

        # estimated avg wait = average time since entering queue_area among currently-in-queue tracks
        waits: List[float] = []
        for tid, st in self.tracks.items():
            if self.queue_zone in st.zone_enter_ts:
                waits.append(now - st.zone_enter_ts[self.queue_zone])

        avg_wait = sum(waits) / len(waits) if waits else 0.0

        return MetricsSnapshot(
            ts=now,
            people_total=len(track_centers),
            people_by_zone=people_by_zone,
            avg_queue_wait_sec=avg_wait,
            queue_len=queue_len,
        )
