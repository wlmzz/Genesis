#!/usr/bin/env python3
"""
AI-Powered Lip Reading Module
Usa Visual Speech Recognition + Llama Vision per lip reading VERO
"""
import numpy as np
import cv2
from collections import deque
import base64
import io
from PIL import Image
import requests
import json

class AILipReader:
    """
    Lip Reading con AI usando:
    1. Estrazione ROI bocca da face landmarks
    2. Analisi sequenza con Llama Vision
    3. Post-processing per text cleanup
    """

    # Landmark indices per bocca (MediaPipe Face Mesh)
    MOUTH_LANDMARKS = [
        # Labbra esterne
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
        # Labbra interne
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        # Angoli e punti chiave
        324, 318, 402, 317, 14, 87, 178, 88, 95, 62, 76
    ]

    def __init__(self,
                 ollama_url="http://localhost:11434",
                 model_name="llama3.2-vision:11b",
                 buffer_size=15,  # frames da analizzare insieme
                 analysis_interval=10):  # analizza ogni N frame
        """
        Args:
            ollama_url: URL del server Ollama
            model_name: Nome modello Llama Vision
            buffer_size: Numero frame da accumulare prima di analizzare
            analysis_interval: Analizza ogni N frame (per performance)
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.buffer_size = buffer_size
        self.analysis_interval = analysis_interval

        # Buffer per accumulo frame
        self.mouth_frames = deque(maxlen=buffer_size)
        self.frame_counter = 0

        # Risultati
        self.current_text = ""
        self.confidence = 0.0
        self.last_analysis_time = 0

        # History
        self.recognized_words = []

    def extract_mouth_roi(self, frame, face_landmarks):
        """
        Estrae Region of Interest (ROI) della bocca dal frame

        Args:
            frame: Frame originale
            face_landmarks: Landmarks del volto da MediaPipe

        Returns:
            mouth_roi: Immagine ritagliata della bocca (None se fallisce)
        """
        if not face_landmarks or len(face_landmarks) < max(self.MOUTH_LANDMARKS):
            return None

        h, w, _ = frame.shape

        # Estrai coordinate bocca
        mouth_points = []
        for idx in self.MOUTH_LANDMARKS:
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                mouth_points.append((x, y))

        if not mouth_points:
            return None

        # Calcola bounding box
        xs = [p[0] for p in mouth_points]
        ys = [p[1] for p in mouth_points]

        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        # Espandi box del 40% per contesto
        width = x2 - x1
        height = y2 - y1

        margin_w = int(width * 0.4)
        margin_h = int(height * 0.4)

        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)

        # Ritaglia ROI
        mouth_roi = frame[y1:y2, x1:x2]

        # Resize a dimensione standard per consistenza
        if mouth_roi.size > 0:
            mouth_roi = cv2.resize(mouth_roi, (128, 64))
            return mouth_roi

        return None

    def encode_image_base64(self, image):
        """Converte immagine numpy in base64 per Llama Vision"""
        # Converti BGR (OpenCV) a RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Converti in PIL Image
        pil_image = Image.fromarray(image)

        # Encode in base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return img_base64

    def analyze_with_llama(self, mouth_images):
        """
        Analizza sequenza di immagini bocca con Llama Vision

        Args:
            mouth_images: Lista di immagini ROI bocca

        Returns:
            dict: {'text': str, 'confidence': float}
        """
        if not mouth_images:
            return {'text': '', 'confidence': 0.0}

        try:
            # Usa solo frame chiave (primo, medio, ultimo) per performance
            key_frames = [
                mouth_images[0],
                mouth_images[len(mouth_images)//2] if len(mouth_images) > 2 else mouth_images[0],
                mouth_images[-1]
            ]

            # Encode immagini
            encoded_images = [self.encode_image_base64(img) for img in key_frames]

            # Prompt per Llama Vision
            prompt = """You are a lip reading AI. Analyze these images of a person's mouth movement and determine what they are saying.

Instructions:
- Look at the sequence of mouth positions
- Consider lip shape, opening, and movement
- Output ONLY the word or short phrase being spoken (max 3-4 words)
- If uncertain, output your best guess
- Common Italian words: ciao, buongiorno, sì, no, grazie, prego, aiuto

What is being said?"""

            # Chiamata a Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": encoded_images,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Bassa per output più deterministico
                        "top_p": 0.9,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '').strip()

                # Cleanup: prendi solo prime 3 parole
                words = text.split()[:3]
                clean_text = ' '.join(words)

                # Stima confidence (semplificato - basato su lunghezza risposta)
                confidence = min(0.9, len(clean_text) / 20.0) if clean_text else 0.0

                return {
                    'text': clean_text,
                    'confidence': confidence
                }
            else:
                return {'text': '', 'confidence': 0.0}

        except Exception as e:
            print(f"Llama Vision error: {e}")
            return {'text': '', 'confidence': 0.0}

    def process_frame(self, frame, face_landmarks):
        """
        Processa un frame e aggiorna lo stato del lip reading

        Args:
            frame: Frame video corrente
            face_landmarks: Landmarks del volto

        Returns:
            dict: {
                'text': str,
                'confidence': float,
                'is_analyzing': bool,
                'buffer_size': int
            }
        """
        self.frame_counter += 1

        # Estrai ROI bocca
        mouth_roi = self.extract_mouth_roi(frame, face_landmarks)

        if mouth_roi is not None:
            self.mouth_frames.append(mouth_roi)

        result = {
            'text': self.current_text,
            'confidence': self.confidence,
            'is_analyzing': False,
            'buffer_size': len(self.mouth_frames)
        }

        # Analizza ogni N frame quando buffer è pieno
        if (self.frame_counter % self.analysis_interval == 0 and
            len(self.mouth_frames) >= self.buffer_size // 2):

            result['is_analyzing'] = True

            # Analizza con Llama Vision
            analysis = self.analyze_with_llama(list(self.mouth_frames))

            if analysis['text']:
                self.current_text = analysis['text']
                self.confidence = analysis['confidence']

                # Aggiungi a history
                self.recognized_words.append({
                    'text': analysis['text'],
                    'confidence': analysis['confidence'],
                    'frame': self.frame_counter
                })

            result['text'] = self.current_text
            result['confidence'] = self.confidence

        return result

    def draw_lip_reading_info(self, frame, result, position=(10, 100)):
        """Disegna informazioni lip reading sul frame"""
        x, y = position

        # Background semi-trasparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (x-5, y-25), (x+400, y+120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Titolo
        cv2.putText(frame, "AI LIP READING (Llama Vision)", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        y += 30

        # Testo riconosciuto
        text = result.get('text', '')
        if text:
            text_color = (0, 255, 0) if result['confidence'] > 0.5 else (0, 200, 200)
            cv2.putText(frame, f"Text: {text}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            y += 30
            cv2.putText(frame, f"Confidence: {result['confidence']:.0%}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Text: --- (analyzing...)", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

        y += 30

        # Buffer status
        buffer_size = result.get('buffer_size', 0)
        is_analyzing = result.get('is_analyzing', False)

        status = "ANALYZING..." if is_analyzing else f"Buffer: {buffer_size}/15"
        status_color = (0, 255, 255) if is_analyzing else (200, 200, 200)
        cv2.putText(frame, status, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        return frame

    def get_recent_words(self, n=5):
        """Ottieni ultime N parole/frasi riconosciute"""
        return self.recognized_words[-n:] if self.recognized_words else []
