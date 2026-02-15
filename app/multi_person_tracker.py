#!/usr/bin/env python3
"""
Multi-Person Tracker
Traccia multiple persone attraverso i frame assegnando ID univoci
"""
import numpy as np
import cv2
from collections import defaultdict
import time

class Person:
    """Rappresenta una persona tracciata"""

    def __init__(self, person_id, bbox):
        self.id = person_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.color = self._generate_color(person_id)
        self.last_seen = time.time()
        self.frames_tracked = 0
        self.emotion_history = []
        self.pose_landmarks = None
        self.face_landmarks = None
        self.hand_landmarks = []

    def _generate_color(self, person_id):
        """Genera colore unico per ogni ID"""
        colors = [
            (255, 0, 0),    # Rosso
            (0, 255, 0),    # Verde
            (0, 0, 255),    # Blu
            (255, 255, 0),  # Ciano
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Giallo
            (128, 0, 128),  # Viola
            (255, 128, 0),  # Arancione
            (0, 128, 255),  # Azzurro
            (128, 255, 0),  # Verde lime
        ]
        return colors[person_id % len(colors)]

    def update(self, bbox, pose_landmarks=None, face_landmarks=None, hand_landmarks=None, emotion=None):
        """Aggiorna dati persona"""
        self.bbox = bbox
        self.last_seen = time.time()
        self.frames_tracked += 1

        if pose_landmarks is not None:
            self.pose_landmarks = pose_landmarks
        if face_landmarks is not None:
            self.face_landmarks = face_landmarks
        if hand_landmarks:
            self.hand_landmarks = hand_landmarks
        if emotion:
            self.emotion_history.append(emotion)
            # Mantieni solo ultime 10 emozioni
            if len(self.emotion_history) > 10:
                self.emotion_history.pop(0)

    def get_dominant_emotion(self):
        """Ottieni emozione dominante dalla history"""
        if not self.emotion_history:
            return None

        # Conta emozioni
        emotion_counts = defaultdict(int)
        for emo in self.emotion_history:
            emotion_counts[emo['emotion']] += 1

        # Emozione più frequente
        dominant = max(emotion_counts, key=emotion_counts.get)
        return dominant

    def is_lost(self, timeout=2.0):
        """Check se persona è persa (non vista da troppo tempo)"""
        return (time.time() - self.last_seen) > timeout


class MultiPersonTracker:
    """Traccia multiple persone attraverso i frame"""

    def __init__(self, max_people=5, iou_threshold=0.3, lost_timeout=2.0):
        self.max_people = max_people
        self.iou_threshold = iou_threshold
        self.lost_timeout = lost_timeout

        self.people = {}  # person_id -> Person
        self.next_id = 0

    def calculate_iou(self, bbox1, bbox2):
        """
        Calcola Intersection over Union tra due bounding boxes
        bbox format: (x1, y1, x2, y2)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersezione
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Unione
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-6)
        return iou

    def bbox_from_landmarks(self, landmarks, frame_width, frame_height):
        """
        Calcola bounding box da landmarks
        """
        if not landmarks:
            return None

        # Estrai coordinate x,y
        xs = [lm.x * frame_width for lm in landmarks]
        ys = [lm.y * frame_height for lm in landmarks]

        if not xs or not ys:
            return None

        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))

        # Espandi box del 20% per margine
        width = x2 - x1
        height = y2 - y1
        margin_w = int(width * 0.2)
        margin_h = int(height * 0.2)

        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(frame_width, x2 + margin_w)
        y2 = min(frame_height, y2 + margin_h)

        return (x1, y1, x2, y2)

    def update(self, pose_results, face_results, hand_results, emotion_results, frame_shape):
        """
        Aggiorna tracker con nuove detection

        Args:
            pose_results: Lista di pose landmarks
            face_results: Lista di face landmarks
            hand_results: Lista di hand landmarks
            emotion_results: Lista di emotion results
            frame_shape: (height, width, channels)
        """
        h, w = frame_shape[:2]

        # Pulisci persone perse
        lost_ids = [pid for pid, person in self.people.items() if person.is_lost(self.lost_timeout)]
        for pid in lost_ids:
            del self.people[pid]

        # Crea detection da pose landmarks (principale)
        detections = []
        if pose_results:
            for i, pose_landmarks in enumerate(pose_results):
                bbox = self.bbox_from_landmarks(pose_landmarks, w, h)
                if bbox:
                    detections.append({
                        'bbox': bbox,
                        'pose': pose_landmarks,
                        'face': face_results[i] if i < len(face_results) else None,
                        'hands': hand_results[i] if i < len(hand_results) else [],
                        'emotion': emotion_results[i] if i < len(emotion_results) else None
                    })

        # Se non ci sono pose, usa face landmarks
        elif face_results:
            for i, face_landmarks in enumerate(face_results):
                bbox = self.bbox_from_landmarks(face_landmarks, w, h)
                if bbox:
                    detections.append({
                        'bbox': bbox,
                        'pose': None,
                        'face': face_landmarks,
                        'hands': hand_results[i] if i < len(hand_results) else [],
                        'emotion': emotion_results[i] if i < len(emotion_results) else None
                    })

        # Match detection con persone esistenti usando IoU
        matched_people = set()
        matched_detections = set()

        for det_idx, detection in enumerate(detections):
            best_iou = 0
            best_person_id = None

            for person_id, person in self.people.items():
                if person_id in matched_people:
                    continue

                iou = self.calculate_iou(detection['bbox'], person.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_person_id = person_id

            if best_person_id is not None:
                # Match trovato - aggiorna persona esistente
                self.people[best_person_id].update(
                    bbox=detection['bbox'],
                    pose_landmarks=detection['pose'],
                    face_landmarks=detection['face'],
                    hand_landmarks=detection['hands'],
                    emotion=detection['emotion']
                )
                matched_people.add(best_person_id)
                matched_detections.add(det_idx)

        # Crea nuove persone per detection non matchate
        for det_idx, detection in enumerate(detections):
            if det_idx in matched_detections:
                continue

            # Non superare max persone
            if len(self.people) >= self.max_people:
                continue

            # Crea nuova persona
            person_id = self.next_id
            self.next_id += 1

            person = Person(person_id, detection['bbox'])
            person.update(
                bbox=detection['bbox'],
                pose_landmarks=detection['pose'],
                face_landmarks=detection['face'],
                hand_landmarks=detection['hands'],
                emotion=detection['emotion']
            )

            self.people[person_id] = person

        return self.people

    def draw_people(self, frame):
        """Disegna info di tutte le persone sul frame"""
        for person_id, person in self.people.items():
            x1, y1, x2, y2 = person.bbox
            color = person.color

            # Bounding box principale
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Label con ID
            label = f"Person #{person_id}"

            # Aggiungi emozione se disponibile
            dominant_emotion = person.get_dominant_emotion()
            if dominant_emotion:
                label += f" - {dominant_emotion}"

            # Background label
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1-label_h-15), (x1+label_w+10, y1), color, -1)

            # Testo label
            cv2.putText(frame, label, (x1+5, y1-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Indicatore frames tracked
            frames_text = f"Frames: {person.frames_tracked}"
            cv2.putText(frame, frames_text, (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Stats globali
        stats_text = f"People tracked: {len(self.people)}"
        cv2.putText(frame, stats_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def get_person_color(self, person_id):
        """Ottieni colore di una persona specifica"""
        if person_id in self.people:
            return self.people[person_id].color
        return (255, 255, 255)
