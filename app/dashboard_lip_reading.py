#!/usr/bin/env python3
"""
Dashboard Lip Reading Component
Visualizza dati lettura labiale in Streamlit
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from lip_reading_server import LipReadingDataServer

def render_lip_reading_section(redis_client):
    """
    Render lip reading section in dashboard

    Args:
        redis_client: Redis connection
    """
    st.header("Lip Reading - Lettura Labiale")

    if not redis_client:
        st.warning("Redis not connected - Lip reading data unavailable")
        return

    server = LipReadingDataServer(redis_client)

    # Current status
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Current Status")

        current_data = server.get_current_data()

        if current_data:
            # Speaking status
            if current_data.get('is_speaking'):
                st.success("SPEAKING")
            else:
                st.info("SILENCE")

            # Current word
            if current_data.get('word'):
                st.metric(
                    label="Current Word",
                    value=current_data['word'],
                    delta=f"{current_data.get('confidence', 0):.0%} confidence"
                )
            else:
                st.metric(label="Current Word", value="---")

            # Mouth state
            mouth_state = current_data.get('mouth_state', 'unknown')
            st.text(f"Mouth: {mouth_state}")

            # Timestamp
            timestamp = current_data.get('timestamp', '')
            if timestamp:
                st.caption(f"Updated: {timestamp}")

        else:
            st.warning("No lip reading data available")
            st.caption("Make sure the camera tracking is running")

    with col2:
        st.subheader("Word History")

        # Get history
        history = server.get_history(limit=20)

        if history:
            # Create dataframe
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")

            # Display table
            st.dataframe(
                df[['word', 'confidence', 'timestamp']],
                column_config={
                    "word": "Word",
                    "confidence": "Confidence",
                    "timestamp": st.column_config.DatetimeColumn(
                        "Time",
                        format="HH:mm:ss"
                    )
                },
                hide_index=True,
                use_container_width=True
            )

            # Word frequency chart
            st.subheader("Word Frequency")
            word_counts = df['word'].value_counts()

            fig = go.Figure(data=[
                go.Bar(
                    x=word_counts.index,
                    y=word_counts.values,
                    marker_color='lightblue'
                )
            ])

            fig.update_layout(
                xaxis_title="Word",
                yaxis_title="Count",
                showlegend=False,
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No words recognized yet")
            st.caption("Words will appear here as they are recognized")

    # Refresh info
    st.caption("Dashboard auto-refreshes every 2 seconds")


def render_lip_reading_stats(redis_client):
    """Render quick stats for overview page"""
    if not redis_client:
        return

    server = LipReadingDataServer(redis_client)
    current_data = server.get_current_data()

    if current_data:
        col1, col2 = st.columns(2)

        with col1:
            status = "SPEAKING" if current_data.get('is_speaking') else "SILENCE"
            st.metric("Speech Status", status)

        with col2:
            word = current_data.get('word', '---')
            st.metric("Last Word", word)
