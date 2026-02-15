#!/usr/bin/env python3
"""
Genesis - Dashboard Streamlit
Dashboard interattiva per visualizzazione KPI e profili identità
"""
from __future__ import annotations
import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Genesis Dashboard", layout="wide", page_icon="🔍")

# Custom CSS
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #E74C3C;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Genesis Dashboard")
st.caption("Sistema avanzato di facial recognition e tracking - INTERNAL TESTING ONLY ⚠️")

# Sidebar
st.sidebar.header("⚙️ Configurazione")
outdir = st.sidebar.text_input("Output directory", "data/outputs")
csv_path = os.path.join(outdir, "metrics.csv")
db_path = os.path.join(outdir, "identities.db")

refresh = st.sidebar.checkbox("Auto-refresh (5s)", False)
if refresh:
    import time
    time.sleep(5)
    st.rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "👥 Identità", "📈 Timeline", "🗄️ Database"])

# TAB 1: OVERVIEW
with tab1:
    st.header("KPI Operativi")

    if not os.path.exists(csv_path):
        st.warning(f"⚠️ Nessun file metrics.csv trovato in: {csv_path}")
        st.info("Avvia run_camera.py o run_video.py per generare dati")
        st.stop()

    df = pd.read_csv(csv_path)
    if df.empty:
        st.warning("metrics.csv è vuoto")
        st.stop()

    df["dt"] = pd.to_datetime(df["ts"], unit="s")
    latest = df.iloc[-1].to_dict()

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 People Total", int(latest.get("people_total", 0)))
    col2.metric("🚶 Queue Length", int(latest.get("queue_len", 0)))
    col3.metric("⏱️ Avg Wait (s)", int(float(latest.get("avg_queue_wait_sec", 0))))

    zone_cols = [c for c in df.columns if c.startswith("zone_")]
    col4.metric("📍 Zones Tracked", len(zone_cols))

    st.divider()

    # Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("People & Queue Timeline")
        fig = px.line(df, x="dt", y=["people_total", "queue_len"],
                     labels={"value": "Count", "dt": "Time"},
                     title="People Total vs Queue Length")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Average Queue Wait Time")
        fig = px.area(df, x="dt", y="avg_queue_wait_sec",
                     labels={"avg_queue_wait_sec": "Seconds", "dt": "Time"},
                     title="Queue Wait Time Evolution")
        st.plotly_chart(fig, use_container_width=True)

    # Zone distribution
    if zone_cols:
        st.subheader("Distribuzione per Zone")
        zone_data = df[zone_cols].iloc[-1]
        fig = px.bar(x=zone_data.index, y=zone_data.values,
                    labels={"x": "Zone", "y": "People Count"},
                    title="Current People per Zone")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📋 Recent Metrics")
    st.dataframe(df.tail(50), use_container_width=True, height=300)

# TAB 2: IDENTITÀ
with tab2:
    st.header("👥 Identità Riconosciute")

    if not os.path.exists(db_path):
        st.warning(f"⚠️ Database identità non trovato: {db_path}")
        st.info("Assicurati che face_recognition sia abilitato in settings.yaml")
    else:
        conn = sqlite3.connect(db_path)

        # Sessions overview
        sessions_df = pd.read_sql_query("SELECT * FROM sessions ORDER BY start_time DESC LIMIT 100", conn)

        if not sessions_df.empty:
            sessions_df["start_dt"] = pd.to_datetime(sessions_df["start_time"], unit="s")
            sessions_df["end_dt"] = pd.to_datetime(sessions_df["end_time"], unit="s")

            st.subheader("📊 Sessioni Recenti")

            col1, col2, col3 = st.columns(3)
            unique_persons = sessions_df["person_id"].nunique()
            total_sessions = len(sessions_df)
            avg_duration = sessions_df["total_duration"].mean() if "total_duration" in sessions_df.columns else 0

            col1.metric("Persone Uniche", unique_persons)
            col2.metric("Sessioni Totali", total_sessions)
            col3.metric("Durata Media (s)", f"{avg_duration:.1f}")

            # Sessions table
            st.dataframe(sessions_df[["person_id", "start_dt", "total_duration", "zones_visited"]],
                        use_container_width=True, height=400)

            # Top visitors
            st.subheader("🏆 Top Visitors")
            top_visitors = sessions_df["person_id"].value_counts().head(10)
            fig = px.bar(x=top_visitors.index, y=top_visitors.values,
                        labels={"x": "Person ID", "y": "Visit Count"},
                        title="Most Frequent Visitors")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Nessuna sessione registrata ancora")

        conn.close()

# TAB 3: TIMELINE
with tab3:
    st.header("📈 Timeline Eventi")

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        events_df = pd.read_sql_query("SELECT * FROM identities ORDER BY timestamp DESC LIMIT 500", conn)

        if not events_df.empty:
            events_df["dt"] = pd.to_datetime(events_df["timestamp"], unit="s")

            # Filter by person
            persons = ["All"] + sorted(events_df["person_id"].unique().tolist())
            selected_person = st.selectbox("Filtra per persona", persons)

            if selected_person != "All":
                events_df = events_df[events_df["person_id"] == selected_person]

            st.dataframe(events_df[["person_id", "dt", "zone", "event_type", "confidence"]],
                        use_container_width=True, height=500)

            # Events distribution
            event_counts = events_df["event_type"].value_counts()
            fig = px.pie(values=event_counts.values, names=event_counts.index,
                        title="Event Type Distribution")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Nessun evento registrato")

        conn.close()
    else:
        st.warning("Database non disponibile")

# TAB 4: DATABASE
with tab4:
    st.header("🗄️ Database Info")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Files")
        files_info = []
        if os.path.exists(csv_path):
            size = os.path.getsize(csv_path)
            files_info.append({"File": "metrics.csv", "Size (KB)": f"{size/1024:.1f}", "Status": "✅"})
        else:
            files_info.append({"File": "metrics.csv", "Size (KB)": "-", "Status": "❌"})

        if os.path.exists(db_path):
            size = os.path.getsize(db_path)
            files_info.append({"File": "identities.db", "Size (KB)": f"{size/1024:.1f}", "Status": "✅"})
        else:
            files_info.append({"File": "identities.db", "Size (KB)": "-", "Status": "❌"})

        st.dataframe(pd.DataFrame(files_info), use_container_width=True)

    with col2:
        st.subheader("📊 Database Stats")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM identities")
            total_events = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT person_id) FROM identities")
            unique_persons = cursor.fetchone()[0]

            stats = pd.DataFrame({
                "Metric": ["Total Events", "Total Sessions", "Unique Persons"],
                "Value": [total_events, total_sessions, unique_persons]
            })
            st.dataframe(stats, use_container_width=True)

            conn.close()
        else:
            st.info("Database non disponibile")

st.divider()
st.caption("Genesis © 2026 - Internal Testing Only ⚠️ Not GDPR Compliant")
