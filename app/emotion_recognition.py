#!/usr/bin/env python3
"""
Emotion Recognition Module
Riconosce emozioni dai 478 face landmarks di MediaPipe
"""
import numpy as np
import cv2

class EmotionRecognizer:
    """
    Riconosce 7 emozioni base dai face landmarks:
    - Happy (Felice)
    - Sad (Triste)
    - Angry (Arrabbiato)
    - Surprised (Sorpreso)
    - Fearful (Spaventato)
    - Disgusted (Disgustato)
    - Neutral (Neutro)
    """

    # MediaPipe Face Mesh landmark indices (key points)
    LEFT_EYE = [33, 133, 160, 159, 158, 157, 173]
    RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398]
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336]
    MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    NOSE_TIP = 1
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    UPPER_LIP = 13
    LOWER_LIP = 14

    # Emozioni
    EMOTIONS = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Fearful', 'Disgusted']

    # Colori per ogni emozione (BGR)
    EMOTION_COLORS = {
        'Neutral': (200, 200, 200),
        'Happy': (0, 255, 0),
        'Sad': (255, 100, 100),
        'Angry': (0, 0, 255),
        'Surprised': (255, 255, 0),
        'Fearful': (128, 0, 128),
        'Disgusted': (0, 128, 128)
    }

    # Emoji per ogni emozione
    EMOTION_EMOJI = {
        'Neutral': '😐',
        'Happy': '😊',
        'Sad': '😢',
        'Angry': '😠',
        'Surprised': '😲',
        'Fearful': '😨',
        'Disgusted': '🤢'
    }

    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold

    def get_landmark_coords(self, face_landmarks, indices):
        """Estrae coordinate di specifici landmark"""
        if isinstance(indices, int):
            indices = [indices]

        coords = []
        for idx in indices:
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                coords.append([lm.x, lm.y, lm.z])

        return np.array(coords) if coords else None

    def calculate_distance(self, p1, p2):
        """Calcola distanza euclidea tra due punti"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def calculate_mouth_aspect_ratio(self, face_landmarks):
        """
        Calcola apertura bocca (MAR - Mouth Aspect Ratio)
        Alto valore = bocca aperta
        """
        # Punti verticali bocca
        upper = self.get_landmark_coords(face_landmarks, self.UPPER_LIP)
        lower = self.get_landmark_coords(face_landmarks, self.LOWER_LIP)

        # Punti orizzontali bocca
        left = self.get_landmark_coords(face_landmarks, self.LEFT_MOUTH_CORNER)
        right = self.get_landmark_coords(face_landmarks, self.RIGHT_MOUTH_CORNER)

        if upper is None or lower is None or left is None or right is None:
            return 0.0

        # Distanza verticale
        vertical = self.calculate_distance(upper[0][:2], lower[0][:2])

        # Distanza orizzontale
        horizontal = self.calculate_distance(left[0][:2], right[0][:2])

        # Ratio
        mar = vertical / (horizontal + 1e-6)
        return mar

    def calculate_mouth_curvature(self, face_landmarks):
        """
        Calcola curvatura bocca (positivo = sorriso, negativo = triste)
        """
        left = self.get_landmark_coords(face_landmarks, self.LEFT_MOUTH_CORNER)
        right = self.get_landmark_coords(face_landmarks, self.RIGHT_MOUTH_CORNER)
        upper = self.get_landmark_coords(face_landmarks, self.UPPER_LIP)

        if left is None or right is None or upper is None:
            return 0.0

        # Centro bocca
        center_y = (left[0][1] + right[0][1]) / 2

        # Se angoli bocca sono sopra il centro -> sorriso
        # Se sotto -> triste
        curvature = center_y - upper[0][1]

        return curvature

    def calculate_eye_aspect_ratio(self, face_landmarks, eye_indices):
        """
        Calcola apertura occhio (EAR - Eye Aspect Ratio)
        """
        eye_coords = self.get_landmark_coords(face_landmarks, eye_indices)

        if eye_coords is None or len(eye_coords) < 6:
            return 0.0

        # Distanze verticali
        v1 = self.calculate_distance(eye_coords[1][:2], eye_coords[5][:2])
        v2 = self.calculate_distance(eye_coords[2][:2], eye_coords[4][:2])

        # Distanza orizzontale
        h = self.calculate_distance(eye_coords[0][:2], eye_coords[3][:2])

        # Ratio
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear

    def calculate_eyebrow_position(self, face_landmarks, eyebrow_indices, eye_indices):
        """
        Calcola posizione sopracciglio rispetto all'occhio
        Valore alto = sopracciglio alzato
        """
        eyebrow = self.get_landmark_coords(face_landmarks, eyebrow_indices)
        eye = self.get_landmark_coords(face_landmarks, eye_indices)

        if eyebrow is None or eye is None:
            return 0.0

        # Media Y sopracciglio e occhio
        eyebrow_y = np.mean([p[1] for p in eyebrow])
        eye_y = np.mean([p[1] for p in eye])

        # Distanza (Y invertito in coordinate immagine)
        distance = eye_y - eyebrow_y

        return distance

    def recognize_emotion(self, face_landmarks):
        """
        Riconosce emozione dai face landmarks

        Returns:
            dict: {
                'emotion': str,
                'confidence': float,
                'scores': dict  # score per ogni emozione
            }
        """
        if not face_landmarks:
            return {
                'emotion': 'Neutral',
                'confidence': 0.0,
                'scores': {}
            }

        # Calcola features
        mouth_openness = self.calculate_mouth_aspect_ratio(face_landmarks)
        mouth_curve = self.calculate_mouth_curvature(face_landmarks)

        left_eye_open = self.calculate_eye_aspect_ratio(face_landmarks, self.LEFT_EYE)
        right_eye_open = self.calculate_eye_aspect_ratio(face_landmarks, self.RIGHT_EYE)
        avg_eye_open = (left_eye_open + right_eye_open) / 2

        left_eyebrow_height = self.calculate_eyebrow_position(face_landmarks, self.LEFT_EYEBROW, self.LEFT_EYE)
        right_eyebrow_height = self.calculate_eyebrow_position(face_landmarks, self.RIGHT_EYEBROW, self.RIGHT_EYE)
        avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2

        # Score per ogni emozione (0-1)
        scores = {}

        # HAPPY: bocca aperta + curvatura positiva + occhi normali
        scores['Happy'] = 0.0
        if mouth_curve > 0.01 and mouth_openness > 0.1:
            scores['Happy'] = min(1.0, (mouth_curve * 100 + mouth_openness * 2))

        # SAD: bocca chiusa + curvatura negativa + sopracciglia abbassate
        scores['Sad'] = 0.0
        if mouth_curve < -0.005 and avg_eyebrow_height < 0.05:
            scores['Sad'] = min(1.0, abs(mouth_curve) * 80 + (0.05 - avg_eyebrow_height) * 5)

        # ANGRY: bocca tesa + sopracciglia abbassate + occhi stretti
        scores['Angry'] = 0.0
        if avg_eyebrow_height < 0.04 and avg_eye_open < 0.25:
            scores['Angry'] = min(1.0, (0.04 - avg_eyebrow_height) * 10 + (0.25 - avg_eye_open) * 2)

        # SURPRISED: bocca molto aperta + occhi molto aperti + sopracciglia alzate
        scores['Surprised'] = 0.0
        if mouth_openness > 0.3 and avg_eye_open > 0.3 and avg_eyebrow_height > 0.06:
            scores['Surprised'] = min(1.0, mouth_openness * 2 + avg_eye_open + (avg_eyebrow_height - 0.06) * 5)

        # FEARFUL: occhi molto aperti + sopracciglia alzate + bocca leggermente aperta
        scores['Fearful'] = 0.0
        if avg_eye_open > 0.28 and avg_eyebrow_height > 0.055:
            scores['Fearful'] = min(1.0, avg_eye_open * 2 + (avg_eyebrow_height - 0.055) * 8)

        # DISGUSTED: naso arricciato + labbro superiore alzato
        scores['Disgusted'] = 0.0
        if mouth_curve < 0 and mouth_openness < 0.15:
            scores['Disgusted'] = min(1.0, abs(mouth_curve) * 50)

        # NEUTRAL: baseline se nessuna emozione forte
        scores['Neutral'] = 0.3  # baseline

        # Trova emozione con score più alto
        if scores:
            max_emotion = max(scores, key=scores.get)
            max_confidence = scores[max_emotion]

            # Se confidence troppo bassa, default a Neutral
            if max_confidence < self.confidence_threshold:
                max_emotion = 'Neutral'
                max_confidence = scores.get('Neutral', 0.3)
        else:
            max_emotion = 'Neutral'
            max_confidence = 0.3
            scores = {'Neutral': 0.3}

        return {
            'emotion': max_emotion,
            'confidence': float(max_confidence),
            'scores': {k: float(v) for k, v in scores.items()}
        }

    def draw_emotion_info(self, frame, emotion_result, position=(10, 30)):
        """Disegna info emozione sul frame"""
        x, y = position

        emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']

        # Colore ed emoji
        color = self.EMOTION_COLORS.get(emotion, (255, 255, 255))
        emoji = self.EMOTION_EMOJI.get(emotion, '')

        # Background semi-trasparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (x-5, y-25), (x+300, y+80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Emozione
        text = f"{emoji} {emotion}"
        cv2.putText(frame, text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        y += 30

        # Confidenza
        conf_text = f"Confidence: {confidence:.1%}"
        cv2.putText(frame, conf_text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y += 30

        # Top 3 emozioni
        if emotion_result.get('scores'):
            sorted_emotions = sorted(
                emotion_result['scores'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            for emo, score in sorted_emotions:
                if score > 0.1:
                    bar_width = int(score * 200)
                    cv2.rectangle(frame, (x, y-10), (x+bar_width, y),
                                self.EMOTION_COLORS.get(emo, (255, 255, 255)), -1)
                    y += 15

        return frame
