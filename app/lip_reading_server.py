#!/usr/bin/env python3
"""
Lip Reading Data Server
Espone dati lip reading per la dashboard via Redis
"""
import json
import time
from datetime import datetime

class LipReadingDataServer:
    """Server per condividere dati lip reading con dashboard"""

    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.data_key = "genesis:lip_reading:current"
        self.history_key = "genesis:lip_reading:history"
        self.max_history = 50

    def update_data(self, lip_result):
        """
        Aggiorna dati lip reading in Redis

        Args:
            lip_result: dict con risultati da LipReader.process_frame()
        """
        if not self.redis_client:
            return

        try:
            # Prepara dati correnti - converti esplicitamente a tipi Python nativi
            current_data = {
                'is_speaking': bool(lip_result.get('is_speaking', False)),
                'word': str(lip_result.get('word')) if lip_result.get('word') is not None else None,
                'confidence': float(lip_result.get('confidence', 0.0)),
                'mouth_state': str(lip_result.get('mouth_state', 'closed')),
                'timestamp': datetime.now().isoformat()
            }

            # Salva stato corrente
            self.redis_client.set(
                self.data_key,
                json.dumps(current_data),
                ex=60  # Expire dopo 60 secondi
            )

            # Se c'è una parola riconosciuta, aggiungila alla history
            if lip_result.get('word') and float(lip_result.get('confidence', 0)) > 0.5:
                history_entry = {
                    'word': str(lip_result['word']),
                    'confidence': float(lip_result['confidence']),
                    'timestamp': datetime.now().isoformat()
                }

                # Aggiungi a lista Redis (LPUSH per nuovi in testa)
                self.redis_client.lpush(
                    self.history_key,
                    json.dumps(history_entry)
                )

                # Mantieni solo ultimi N elementi
                self.redis_client.ltrim(self.history_key, 0, self.max_history - 1)

        except Exception as e:
            print(f"Error updating lip reading data: {e}")

    def get_current_data(self):
        """Recupera dati correnti"""
        if not self.redis_client:
            return None

        try:
            data = self.redis_client.get(self.data_key)
            return json.loads(data) if data else None
        except Exception as e:
            print(f"Error getting lip reading data: {e}")
            return None

    def get_history(self, limit=20):
        """Recupera cronologia parole riconosciute"""
        if not self.redis_client:
            return []

        try:
            history_raw = self.redis_client.lrange(self.history_key, 0, limit - 1)
            return [json.loads(item) for item in history_raw]
        except Exception as e:
            print(f"Error getting lip reading history: {e}")
            return []
