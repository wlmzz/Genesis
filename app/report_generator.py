#!/usr/bin/env python3
"""
Genesis - Automatic Report Generator
Genera report automatici usando LLM
"""
from __future__ import annotations
import argparse
import os
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_integration import LlamaAnalyzer

def generate_daily_report(date: str, output_dir: str = "data/outputs/reports"):
    """Genera report giornaliero"""

    csv_path = "data/outputs/metrics.csv"
    db_path = "data/outputs/identities.db"

    if not os.path.exists(csv_path):
        print("❌ Nessun dato metriche disponibile")
        return

    # Carica dati
    df = pd.read_csv(csv_path)
    df["dt"] = pd.to_datetime(df["ts"], unit="s")

    # Filtra per giorno
    target_date = pd.to_datetime(date)
    daily_df = df[df["dt"].dt.date == target_date.date()]

    if daily_df.empty:
        print(f"❌ Nessun dato per {date}")
        return

    # Calcola statistiche giornaliere
    daily_stats = {
        "date": date,
        "total_observations": len(daily_df),
        "avg_people": daily_df["people_total"].mean(),
        "max_people": daily_df["people_total"].max(),
        "min_people": daily_df["people_total"].min(),
        "avg_queue_len": daily_df["queue_len"].mean(),
        "max_queue_len": daily_df["queue_len"].max(),
        "avg_wait_time_sec": daily_df["avg_queue_wait_sec"].mean(),
        "max_wait_time_sec": daily_df["avg_queue_wait_sec"].max(),
        "peak_hour": daily_df.groupby(daily_df["dt"].dt.hour)["people_total"].mean().idxmax()
    }

    # Identità
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)

        # Sessioni giornaliere
        start_ts = target_date.timestamp()
        end_ts = (target_date + timedelta(days=1)).timestamp()

        sessions_df = pd.read_sql_query(
            f"SELECT * FROM sessions WHERE start_time >= {start_ts} AND start_time < {end_ts}",
            conn
        )

        daily_stats["total_sessions"] = len(sessions_df)
        daily_stats["unique_visitors"] = sessions_df["person_id"].nunique()
        daily_stats["avg_session_duration"] = sessions_df["total_duration"].mean() if len(sessions_df) > 0 else 0

        conn.close()

    # Usa LLM per generare report
    llm = LlamaAnalyzer()

    if not llm.check_health():
        print("⚠️  Ollama non disponibile. Generando report semplice...")
        report = generate_simple_report(daily_stats)
    else:
        print("✓ Generazione report con Llama...")
        report = llm.generate_daily_report(date, daily_stats)

    # Salva report
    os.makedirs(output_dir, exist_ok=True)
    report_filename = f"report_{date}.txt"
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f"GENESIS - REPORT GIORNALIERO\n")
        f.write(f"Data: {date}\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n\n" + "="*60 + "\n")
        f.write(f"Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n✓ Report salvato: {report_path}\n")
    print(report)

    return report_path

def generate_simple_report(stats: dict) -> str:
    """Report semplice senza LLM"""
    report = f"""
STATISTICHE PRINCIPALI
----------------------
- Osservazioni totali: {stats['total_observations']}
- Persone medie: {stats['avg_people']:.1f}
- Picco massimo: {stats['max_people']}
- Coda media: {stats['avg_queue_len']:.1f}
- Coda massima: {stats['max_queue_len']}
- Attesa media: {stats['avg_wait_time_sec']:.0f} secondi
- Attesa massima: {stats['max_wait_time_sec']:.0f} secondi
- Ora di punta: {stats.get('peak_hour', 'N/A')}:00

IDENTITÀ
--------
- Sessioni totali: {stats.get('total_sessions', 0)}
- Visitatori unici: {stats.get('unique_visitors', 0)}
- Durata media sessione: {stats.get('avg_session_duration', 0):.0f} secondi

NOTE
----
Report generato senza LLM. Per analisi avanzate, avvia Ollama.
"""
    return report

def main():
    ap = argparse.ArgumentParser(description="Genesis - Report Generator")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                   help="Data report (YYYY-MM-DD)")
    ap.add_argument("--outdir", default="data/outputs/reports",
                   help="Directory output")
    args = ap.parse_args()

    print("\n" + "="*60)
    print("Genesis - Report Generator")
    print("="*60 + "\n")

    generate_daily_report(args.date, args.outdir)

if __name__ == "__main__":
    main()
