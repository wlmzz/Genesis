from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from deepface import DeepFace

class EmotionDetector:
    """
    Analisi emozioni da volti usando DeepFace
    ATTENZIONE: Questa è una funzionalità sensibile - usare solo con consenso
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def detect_emotion(self, face_img: np.ndarray) -> Optional[Dict[str, any]]:
        """
        Rileva emozione dominante da immagine volto
        Returns: {emotion: str, confidence: float, all_emotions: dict}
        """
        if not self.enabled:
            return None

        try:
            # DeepFace emotion analysis
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            emotions = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'neutral')

            return {
                "emotion": dominant_emotion,
                "confidence": emotions.get(dominant_emotion, 0.0),
                "all_emotions": emotions
            }

        except Exception as e:
            print(f"Emotion detection error: {e}")
            return None

    def get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """Colore BGR per visualizzazione emozione"""
        colors = {
            "happy": (0, 255, 0),      # Verde
            "neutral": (255, 255, 255), # Bianco
            "sad": (255, 0, 0),         # Blu
            "angry": (0, 0, 255),       # Rosso
            "surprise": (0, 255, 255),  # Giallo
            "fear": (128, 0, 128),      # Viola
            "disgust": (0, 128, 128)    # Marrone
        }
        return colors.get(emotion, (255, 255, 255))

    def get_emotion_emoji(self, emotion: str) -> str:
        """Emoji per emozione"""
        emojis = {
            "happy": "😊",
            "neutral": "😐",
            "sad": "😢",
            "angry": "😠",
            "surprise": "😲",
            "fear": "😨",
            "disgust": "🤢"
        }
        return emojis.get(emotion, "😐")

    def analyze_batch_emotions(self, emotions_list: list[str]) -> Dict[str, any]:
        """Analizza distribuzione emozioni in un gruppo"""
        if not emotions_list:
            return {}

        from collections import Counter
        emotion_counts = Counter(emotions_list)

        total = len(emotions_list)
        distribution = {
            emotion: (count / total) * 100
            for emotion, count in emotion_counts.items()
        }

        # Emozione dominante
        dominant = emotion_counts.most_common(1)[0][0]

        # Sentiment score (-1 negativo, +1 positivo)
        sentiment_scores = {
            "happy": 1.0,
            "surprise": 0.5,
            "neutral": 0.0,
            "fear": -0.3,
            "sad": -0.7,
            "angry": -0.9,
            "disgust": -0.8
        }

        avg_sentiment = np.mean([sentiment_scores.get(e, 0) for e in emotions_list])

        return {
            "total_people": total,
            "dominant_emotion": dominant,
            "distribution": distribution,
            "sentiment_score": avg_sentiment,
            "sentiment_label": self._sentiment_label(avg_sentiment)
        }

    def _sentiment_label(self, score: float) -> str:
        """Converti sentiment score in label"""
        if score > 0.5:
            return "Very Positive"
        elif score > 0.1:
            return "Positive"
        elif score > -0.1:
            return "Neutral"
        elif score > -0.5:
            return "Negative"
        else:
            return "Very Negative"


class AgeGenderDetector:
    """Analisi età e genere (opzionale, privacy-sensitive)"""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def detect(self, face_img: np.ndarray) -> Optional[Dict[str, any]]:
        """
        Rileva età e genere
        ATTENZIONE: Funzionalità molto sensibile per privacy
        """
        if not self.enabled:
            return None

        try:
            result = DeepFace.analyze(
                face_img,
                actions=['age', 'gender'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            return {
                "age": result.get('age'),
                "gender": result.get('dominant_gender'),
                "gender_confidence": result.get('gender', {})
            }

        except Exception as e:
            print(f"Age/Gender detection error: {e}")
            return None
