from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import cv2
import numpy as np

Point = Tuple[int, int]
Polygon = List[Point]

def load_zones(path: str) -> Dict[str, Polygon]:
    """Carica zone da file JSON"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    zones: Dict[str, Polygon] = {}
    for k, pts in raw.items():
        zones[k] = [(int(x), int(y)) for x, y in pts]
    return zones

def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Verifica se un punto è dentro un poligono"""
    # cv2.pointPolygonTest expects contour as Nx1x2
    contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0

def draw_zones(frame, zones: Dict[str, Polygon]):
    """Disegna le zone sul frame"""
    for name, poly in zones.items():
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
        x, y = poly[0]
        cv2.putText(frame, name, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
