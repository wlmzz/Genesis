from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import cv2

class HeatMapGenerator:
    """Genera heat maps di traffico"""

    def __init__(self, frame_width: int, frame_height: int):
        self.width = frame_width
        self.height = frame_height
        self.heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

    def add_position(self, x: int, y: int, weight: float = 1.0):
        """Aggiungi posizione alla heatmap"""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Gaussian blur per smooth heatmap
            radius = 30
            y1 = max(0, y - radius)
            y2 = min(self.height, y + radius)
            x1 = max(0, x - radius)
            x2 = min(self.width, x + radius)

            self.heatmap[y1:y2, x1:x2] += weight

    def get_heatmap_image(self, colormap=cv2.COLORMAP_JET) -> np.ndarray:
        """Genera immagine heatmap colorata"""
        # Normalizza
        normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)

        # Applica colormap
        colored = cv2.applyColorMap(normalized, colormap)
        return colored

    def get_hotspots(self, threshold: float = 0.7) -> List[Tuple[int, int]]:
        """Identifica hotspots (zone più trafficate)"""
        max_val = self.heatmap.max()
        threshold_val = max_val * threshold

        hotspots = []
        for y in range(self.height):
            for x in range(self.width):
                if self.heatmap[y, x] >= threshold_val:
                    hotspots.append((x, y))

        return hotspots

    def save(self, path: str):
        """Salva heatmap come immagine"""
        img = self.get_heatmap_image()
        cv2.imwrite(path, img)


class PathTracker:
    """Traccia percorsi individuali"""

    def __init__(self):
        self.paths: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)

    def add_position(self, track_id: int, x: int, y: int, timestamp: float):
        """Aggiungi posizione al percorso"""
        self.paths[track_id].append((x, y, timestamp))

    def get_path(self, track_id: int) -> List[Tuple[int, int, float]]:
        """Ottieni percorso completo per track_id"""
        return self.paths.get(track_id, [])

    def get_path_length(self, track_id: int) -> float:
        """Calcola lunghezza percorso in pixel"""
        path = self.paths.get(track_id, [])
        if len(path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(path)):
            x1, y1, _ = path[i-1]
            x2, y2, _ = path[i]
            total_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)

        return total_length

    def draw_path(self, frame: np.ndarray, track_id: int, color=(0, 255, 0), thickness=2):
        """Disegna percorso su frame"""
        path = self.paths.get(track_id, [])
        if len(path) < 2:
            return frame

        for i in range(1, len(path)):
            x1, y1, _ = path[i-1]
            x2, y2, _ = path[i]
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

        return frame

    def get_all_paths(self) -> Dict[int, List[Tuple[int, int, float]]]:
        """Ottieni tutti i percorsi"""
        return dict(self.paths)

    def analyze_path_patterns(self) -> Dict[str, any]:
        """Analizza pattern nei percorsi"""
        if not self.paths:
            return {}

        avg_length = np.mean([self.get_path_length(tid) for tid in self.paths.keys()])
        max_length = max([self.get_path_length(tid) for tid in self.paths.keys()])

        return {
            "total_tracks": len(self.paths),
            "avg_path_length_px": avg_length,
            "max_path_length_px": max_length
        }


class AnomalyDetector:
    """Rileva anomalie nei pattern di comportamento"""

    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.historical_data: List[float] = []

    def add_datapoint(self, value: float):
        """Aggiungi datapoint allo storico"""
        self.historical_data.append(value)

        # Mantieni solo ultimi 1000 punti
        if len(self.historical_data) > 1000:
            self.historical_data.pop(0)

    def is_anomaly(self, value: float) -> Tuple[bool, float]:
        """
        Verifica se valore è anomalo usando z-score
        Returns: (is_anomaly, z_score)
        """
        if len(self.historical_data) < 10:
            return False, 0.0

        mean = np.mean(self.historical_data)
        std = np.std(self.historical_data)

        if std == 0:
            return False, 0.0

        z_score = abs((value - mean) / std)
        is_anomaly = z_score > self.sensitivity

        return is_anomaly, z_score

    def get_anomaly_description(self, value: float, metric_name: str) -> str:
        """Genera descrizione anomalia"""
        is_anom, z_score = self.is_anomaly(value)

        if not is_anom:
            return ""

        mean = np.mean(self.historical_data)
        deviation = ((value - mean) / mean) * 100

        direction = "superiore" if value > mean else "inferiore"

        return f"ANOMALIA: {metric_name} = {value:.1f} ({direction} alla media di {deviation:+.1f}%, z-score: {z_score:.2f})"


class BehaviorClustering:
    """Clustering comportamenti visitatori"""

    def __init__(self, eps: float = 0.5, min_samples: int = 3):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()

    def cluster_behaviors(self, behavior_data: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster visitatori in base a comportamento
        Input: DataFrame con colonne [duration, zones_count, revisit_count, ...]
        """
        if len(behavior_data) < self.min_samples:
            behavior_data["cluster"] = 0
            return behavior_data

        # Normalizza features
        features = self.scaler.fit_transform(behavior_data)

        # DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(features)

        behavior_data["cluster"] = labels

        return behavior_data

    def get_cluster_profiles(self, clustered_data: pd.DataFrame) -> Dict[int, Dict]:
        """Genera profili per ogni cluster"""
        profiles = {}

        for cluster_id in clustered_data["cluster"].unique():
            if cluster_id == -1:  # Noise in DBSCAN
                continue

            cluster_data = clustered_data[clustered_data["cluster"] == cluster_id]

            profiles[cluster_id] = {
                "count": len(cluster_data),
                "avg_duration": cluster_data.get("duration", pd.Series([0])).mean(),
                "avg_zones": cluster_data.get("zones_count", pd.Series([0])).mean(),
                "description": self._describe_cluster(cluster_id, cluster_data)
            }

        return profiles

    def _describe_cluster(self, cluster_id: int, data: pd.DataFrame) -> str:
        """Descrivi cluster in linguaggio naturale"""
        avg_duration = data.get("duration", pd.Series([0])).mean()
        avg_zones = data.get("zones_count", pd.Series([0])).mean()

        if avg_duration < 60 and avg_zones < 2:
            return "Quick Browsers - Visitatori rapidi che esplorano poco"
        elif avg_duration > 300 and avg_zones >= 3:
            return "Deep Shoppers - Visitatori interessati che esplorano molto"
        elif avg_duration > 180:
            return "Time Spenders - Visitatori che passano molto tempo"
        else:
            return "Regular Visitors - Visitatori con comportamento standard"


class PredictiveAnalytics:
    """Analytics predittive semplici"""

    @staticmethod
    def predict_queue_length(historical_queue: List[int], horizon: int = 5) -> List[float]:
        """Predice lunghezza coda futura (media mobile)"""
        if len(historical_queue) < 3:
            return [0.0] * horizon

        # Simple moving average prediction
        window = min(10, len(historical_queue))
        recent_avg = np.mean(historical_queue[-window:])
        trend = (historical_queue[-1] - historical_queue[-window]) / window

        predictions = []
        for i in range(1, horizon + 1):
            pred = recent_avg + (trend * i)
            predictions.append(max(0, pred))

        return predictions

    @staticmethod
    def estimate_wait_time(queue_length: int, avg_service_time: float = 120.0) -> float:
        """Stima tempo di attesa basato su lunghezza coda"""
        return queue_length * avg_service_time

    @staticmethod
    def suggest_optimal_staff(current_queue: int, target_wait: float = 300.0,
                            service_time: float = 120.0) -> int:
        """Suggerisci numero ottimale operatori"""
        if current_queue == 0:
            return 1

        # Calcola operatori necessari per target wait time
        total_service_time = current_queue * service_time
        staff_needed = int(np.ceil(total_service_time / target_wait))

        return max(1, staff_needed)
