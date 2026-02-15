from __future__ import annotations
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

class LlamaAnalyzer:
    """
    Integrazione con Llama locale per analisi intelligenti
    Richiede Ollama in esecuzione (ollama serve)
    """

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"

    def _call_llama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Chiama Llama locale via Ollama API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except Exception as e:
            print(f"Error calling Llama: {e}")
            return ""

    def analyze_behavior_patterns(self, metrics_df: pd.DataFrame) -> str:
        """Analizza pattern comportamentali dai dati metrici"""

        # Prepara summary statistico
        summary = {
            "total_records": len(metrics_df),
            "avg_people": metrics_df["people_total"].mean(),
            "max_people": metrics_df["people_total"].max(),
            "avg_queue_len": metrics_df["queue_len"].mean(),
            "max_queue_len": metrics_df["queue_len"].max(),
            "avg_wait_time": metrics_df["avg_queue_wait_sec"].mean(),
            "peak_hours": self._find_peak_hours(metrics_df)
        }

        prompt = f"""Analizza questi dati di traffico e comportamento in un negozio/spazio commerciale.

Dati statistici:
- Totale osservazioni: {summary['total_records']}
- Media persone presenti: {summary['avg_people']:.1f}
- Picco massimo persone: {summary['max_people']}
- Media lunghezza coda: {summary['avg_queue_len']:.1f}
- Coda massima: {summary['max_queue_len']}
- Tempo attesa medio: {summary['avg_wait_time']:.1f} secondi
- Ore di punta: {summary['peak_hours']}

Fornisci:
1. Analisi dei pattern identificati
2. Anomalie o problemi critici
3. Suggerimenti operativi concreti (staffing, gestione code, layout)
4. Previsioni e raccomandazioni

Rispondi in italiano, in modo conciso e pratico (max 300 parole)."""

        system = "Sei un esperto di retail analytics e ottimizzazione operativa. Fornisci analisi pratiche e actionable."

        return self._call_llama(prompt, system)

    def _find_peak_hours(self, df: pd.DataFrame) -> str:
        """Identifica ore di punta"""
        if "ts" not in df.columns:
            return "N/A"

        df["hour"] = pd.to_datetime(df["ts"], unit="s").dt.hour
        peak_hours = df.groupby("hour")["people_total"].mean().nlargest(3)
        return ", ".join([f"{int(h)}:00" for h in peak_hours.index])

    def generate_alert_description(self, alert_type: str, metrics: Dict[str, Any]) -> str:
        """Genera descrizione naturale per alert usando LLM"""

        prompt = f"""Sei un sistema di alert per un negozio. Genera un messaggio di allerta BREVE (max 2 frasi) per questo evento:

Tipo alert: {alert_type}
Metriche attuali:
{json.dumps(metrics, indent=2)}

Il messaggio deve essere:
- Chiaro e urgente
- Include raccomandazione immediata
- In italiano
- Max 2 frasi"""

        return self._call_llama(prompt, "Sei un assistente di gestione retail. Comunica in modo chiaro e conciso.")

    def analyze_identity_behavior(self, person_id: str, session_data: Dict) -> str:
        """Analizza comportamento di una specifica identità"""

        prompt = f"""Analizza il comportamento di questa persona in un negozio:

ID: {person_id}
Durata visita: {session_data.get('duration', 0):.0f} secondi
Zone visitate: {session_data.get('zones', 'N/A')}
Numero visite precedenti: {session_data.get('previous_visits', 0)}

Fornisci breve profilo comportamentale (max 100 parole):
- Tipo di visitatore (browser, buyer, frequente, occasionale)
- Pattern interessanti
- Suggerimenti per engagement"""

        return self._call_llama(prompt)

    def generate_daily_report(self, date: str, daily_stats: Dict) -> str:
        """Genera report giornaliero completo"""

        prompt = f"""Genera un report giornaliero professionale per il giorno {date}.

Statistiche:
{json.dumps(daily_stats, indent=2)}

Il report deve includere:
1. Executive Summary (2-3 frasi)
2. KPI principali e confronto con obiettivi
3. Highlights positivi
4. Criticità emerse
5. Action items per domani

Formato: professionale ma conciso. In italiano. Max 400 parole."""

        system = "Sei un retail manager esperto. Genera report chiari e actionable."

        return self._call_llama(prompt, system)

    def answer_question(self, question: str, context_data: Dict) -> str:
        """Risponde a domande sui dati usando RAG pattern"""

        prompt = f"""Rispondi a questa domanda basandoti sui dati forniti:

DOMANDA: {question}

DATI DISPONIBILI:
{json.dumps(context_data, indent=2)}

Fornisci risposta precisa e basata sui dati. Se i dati non sono sufficienti, dillo chiaramente.
Rispondi in italiano."""

        system = "Sei un data analyst. Rispondi solo con informazioni supportate dai dati forniti."

        return self._call_llama(prompt, system)

    def suggest_optimizations(self, current_metrics: Dict, historical_data: pd.DataFrame) -> str:
        """Suggerisce ottimizzazioni basate su dati storici"""

        # Calcola trend
        trends = {
            "people_trend": "crescente" if historical_data["people_total"].is_monotonic_increasing else "decrescente",
            "queue_trend": "crescente" if historical_data["queue_len"].mean() > historical_data["queue_len"].head(10).mean() else "decrescente"
        }

        prompt = f"""Analizza questa situazione e suggerisci ottimizzazioni:

SITUAZIONE ATTUALE:
{json.dumps(current_metrics, indent=2)}

TREND STORICI:
- Traffico: {trends['people_trend']}
- Code: {trends['queue_trend']}

Fornisci 3-5 suggerimenti concreti per:
1. Ridurre tempi di attesa
2. Ottimizzare layout/zone
3. Migliorare staffing
4. Incrementare conversione

Ogni suggerimento deve essere specifico e implementabile. In italiano."""

        return self._call_llama(prompt)

    def detect_anomalies_description(self, anomaly_data: Dict) -> str:
        """Descrive anomalie rilevate in linguaggio naturale"""

        prompt = f"""Descrivi questa anomalia rilevata nel sistema:

{json.dumps(anomaly_data, indent=2)}

Spiega:
1. Cosa è successo (in modo semplice)
2. Perché è anomalo
3. Possibili cause
4. Cosa fare

Max 150 parole. Italiano."""

        return self._call_llama(prompt)

    def check_health(self) -> bool:
        """Verifica se Ollama è raggiungibile"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
