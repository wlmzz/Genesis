from __future__ import annotations
import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from deepface import DeepFace
import cv2

class FaceRecognizer:
    """Facial recognition engine usando DeepFace"""

    def __init__(self, faces_dir: str, model: str = "Facenet512",
                 distance_metric: str = "cosine", threshold: float = 0.6):
        self.faces_dir = Path(faces_dir)
        self.model = model
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.known_embeddings: Dict[str, np.ndarray] = {}
        self.load_known_faces()

    def load_known_faces(self):
        """Carica embeddings dei volti noti dal database"""
        if not self.faces_dir.exists():
            os.makedirs(self.faces_dir, exist_ok=True)
            return

        for person_dir in self.faces_dir.iterdir():
            if person_dir.is_dir():
                embedding_path = person_dir / "embedding.npy"
                if embedding_path.exists():
                    self.known_embeddings[person_dir.name] = np.load(embedding_path)
                else:
                    # Genera embedding da foto di riferimento
                    photo_path = person_dir / "photo.jpg"
                    if photo_path.exists():
                        embedding = self.get_embedding(str(photo_path))
                        if embedding is not None:
                            np.save(embedding_path, embedding)
                            self.known_embeddings[person_dir.name] = embedding

    def get_embedding(self, image_path_or_array) -> Optional[np.ndarray]:
        """Estrae embedding da un'immagine"""
        try:
            result = DeepFace.represent(
                img_path=image_path_or_array,
                model_name=self.model,
                enforce_detection=False
            )
            return np.array(result[0]["embedding"])
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def recognize_face(self, face_img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Riconosce un volto confrontandolo con il database
        Returns: (identity_name, confidence_distance)
        """
        if len(self.known_embeddings) == 0:
            return None, 1.0

        embedding = self.get_embedding(face_img)
        if embedding is None:
            return None, 1.0

        best_match = None
        best_distance = float('inf')

        for person_id, known_emb in self.known_embeddings.items():
            distance = self._compute_distance(embedding, known_emb)
            if distance < best_distance:
                best_distance = distance
                best_match = person_id

        if best_distance < self.threshold:
            return best_match, best_distance
        else:
            return None, best_distance

    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcola distanza tra due embeddings"""
        if self.distance_metric == "cosine":
            return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        elif self.distance_metric == "euclidean":
            return np.linalg.norm(emb1 - emb2)
        else:
            return np.linalg.norm(emb1 - emb2)

    def register_new_face(self, face_img: np.ndarray, person_id: str):
        """Registra un nuovo volto nel database"""
        person_dir = self.faces_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        # Salva foto
        photo_path = person_dir / "photo.jpg"
        cv2.imwrite(str(photo_path), face_img)

        # Genera e salva embedding
        embedding = self.get_embedding(face_img)
        if embedding is not None:
            embedding_path = person_dir / "embedding.npy"
            np.save(embedding_path, embedding)
            self.known_embeddings[person_id] = embedding
            return True
        return False
