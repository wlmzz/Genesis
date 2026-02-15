"""
Prometheus metrics definitions for Genesis workers.

This module provides standardized metrics for all Genesis microservices.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, Summary
from typing import Dict, List

# ==============================================================================
# System Metrics
# ==============================================================================

# Worker info metric
worker_info = Info(
    'genesis_worker',
    'Genesis worker information'
)

# Worker uptime
worker_uptime_seconds = Gauge(
    'genesis_worker_uptime_seconds',
    'Worker uptime in seconds',
    ['service', 'worker_id']
)

# ==============================================================================
# Frame Processing Metrics
# ==============================================================================

frames_processed_total = Counter(
    'genesis_frames_processed_total',
    'Total frames processed',
    ['camera_id', 'status']  # status: success, error, skipped
)

frames_dropped_total = Counter(
    'genesis_frames_dropped_total',
    'Total frames dropped due to backpressure or errors',
    ['camera_id', 'reason']  # reason: backpressure, error, timeout
)

frame_processing_latency_seconds = Histogram(
    'genesis_frame_processing_latency_seconds',
    'Frame processing latency in seconds',
    ['camera_id', 'worker_type'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

current_fps = Gauge(
    'genesis_current_fps',
    'Current frames per second',
    ['camera_id']
)

# ==============================================================================
# Detection Metrics
# ==============================================================================

detections_total = Counter(
    'genesis_detections_total',
    'Total person detections',
    ['camera_id', 'confidence_range']  # confidence_range: high, medium, low
)

detection_latency_seconds = Histogram(
    'genesis_detection_latency_seconds',
    'YOLO detection processing time',
    ['model_version'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

active_tracks = Gauge(
    'genesis_active_tracks',
    'Current number of active person tracks',
    ['camera_id']
)

detections_per_frame = Histogram(
    'genesis_detections_per_frame',
    'Number of detections per frame',
    ['camera_id'],
    buckets=[0, 1, 2, 5, 10, 20, 50, 100]
)

# ==============================================================================
# Face Recognition Metrics
# ==============================================================================

faces_recognized_total = Counter(
    'genesis_faces_recognized_total',
    'Total faces recognized',
    ['person_id', 'is_new']  # is_new: true, false
)

face_recognition_latency_seconds = Histogram(
    'genesis_face_recognition_latency_seconds',
    'Face recognition processing time',
    ['model_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
)

face_embedding_extraction_seconds = Histogram(
    'genesis_face_embedding_extraction_seconds',
    'Time to extract face embedding',
    ['model_name'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

face_similarity_search_seconds = Histogram(
    'genesis_face_similarity_search_seconds',
    'pgvector similarity search time',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

face_recognition_confidence = Histogram(
    'genesis_face_recognition_confidence',
    'Face recognition confidence scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

unique_faces_total = Gauge(
    'genesis_unique_faces_total',
    'Total number of unique faces in database'
)

# ==============================================================================
# Event Bus Metrics (Redis Streams)
# ==============================================================================

redis_events_published_total = Counter(
    'genesis_redis_events_published_total',
    'Total events published to Redis Streams',
    ['stream_name', 'event_type', 'status']  # status: success, error
)

redis_events_consumed_total = Counter(
    'genesis_redis_events_consumed_total',
    'Total events consumed from Redis Streams',
    ['stream_name', 'event_type', 'consumer_group']
)

redis_pending_messages = Gauge(
    'genesis_redis_pending_messages',
    'Number of pending messages in Redis stream',
    ['stream_name', 'consumer_group']
)

redis_consumer_lag_seconds = Gauge(
    'genesis_redis_consumer_lag_seconds',
    'Consumer lag in seconds',
    ['stream_name', 'consumer_id']
)

redis_connection_errors_total = Counter(
    'genesis_redis_connection_errors_total',
    'Total Redis connection errors',
    ['worker_id', 'error_type']
)

redis_retry_attempts_total = Counter(
    'genesis_redis_retry_attempts_total',
    'Total retry attempts for failed messages',
    ['stream_name', 'attempt_number']  # attempt_number: 1, 2, 3
)

dlq_messages_total = Counter(
    'genesis_dlq_messages_total',
    'Total messages sent to Dead Letter Queue',
    ['stream_name', 'error_reason']
)

backpressure_triggered_total = Counter(
    'genesis_backpressure_triggered_total',
    'Total times backpressure was triggered',
    ['stream_name']
)

# ==============================================================================
# Database Metrics
# ==============================================================================

db_query_duration_seconds = Histogram(
    'genesis_db_query_duration_seconds',
    'Database query execution time',
    ['query_type', 'table'],  # query_type: select, insert, update, delete
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

db_connection_pool_size = Gauge(
    'genesis_db_connection_pool_size',
    'Current database connection pool size',
    ['pool_name']
)

db_connection_pool_available = Gauge(
    'genesis_db_connection_pool_available',
    'Available connections in pool',
    ['pool_name']
)

db_errors_total = Counter(
    'genesis_db_errors_total',
    'Total database errors',
    ['error_type', 'table']
)

# ==============================================================================
# Analytics Metrics
# ==============================================================================

people_count_current = Gauge(
    'genesis_people_count_current',
    'Current number of people detected',
    ['camera_id', 'zone_name']
)

queue_length_current = Gauge(
    'genesis_queue_length_current',
    'Current queue length',
    ['zone_name']
)

queue_avg_wait_seconds = Gauge(
    'genesis_queue_avg_wait_seconds',
    'Average queue wait time in seconds',
    ['zone_name']
)

zone_occupancy_ratio = Gauge(
    'genesis_zone_occupancy_ratio',
    'Zone occupancy ratio (0-1)',
    ['zone_name']
)

session_duration_seconds = Histogram(
    'genesis_session_duration_seconds',
    'Customer session duration',
    ['person_id'],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600]
)

anomaly_detected = Counter(
    'genesis_anomaly_detected',
    'Anomalies detected by analytics',
    ['type', 'zone_name']  # type: crowd, loitering, unusual_movement
)

# ==============================================================================
# ML Model Metrics
# ==============================================================================

model_inference_latency_seconds = Histogram(
    'genesis_model_inference_latency_seconds',
    'ML model inference latency',
    ['model_name', 'model_version'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

model_predictions_total = Counter(
    'genesis_model_predictions_total',
    'Total model predictions',
    ['model_name', 'model_version', 'ab_variant']  # ab_variant: A, B, control
)

prediction_confidence = Histogram(
    'genesis_prediction_confidence',
    'Model prediction confidence scores',
    ['model_name'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

model_drift_score = Gauge(
    'genesis_model_drift_score',
    'Model drift score (KS statistic)',
    ['model_name']
)

model_accuracy = Gauge(
    'genesis_model_accuracy',
    'Model accuracy (last evaluation)',
    ['model_name', 'model_version']
)

# ==============================================================================
# API Gateway Metrics
# ==============================================================================

http_requests_total = Counter(
    'genesis_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'genesis_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

websocket_connections_active = Gauge(
    'genesis_websocket_connections_active',
    'Active WebSocket connections',
    ['endpoint']
)

api_rate_limit_exceeded_total = Counter(
    'genesis_api_rate_limit_exceeded_total',
    'API rate limit exceeded count',
    ['consumer', 'endpoint']
)

# ==============================================================================
# LLM Service Metrics
# ==============================================================================

llm_requests_total = Counter(
    'genesis_llm_requests_total',
    'Total LLM requests',
    ['model', 'request_type']  # request_type: narrative, analysis, summary
)

llm_latency_seconds = Histogram(
    'genesis_llm_latency_seconds',
    'LLM request latency',
    ['model'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

llm_tokens_generated = Counter(
    'genesis_llm_tokens_generated',
    'Total tokens generated by LLM',
    ['model']
)

llm_cache_hits_total = Counter(
    'genesis_llm_cache_hits_total',
    'LLM cache hits',
    ['cache_type']
)

# ==============================================================================
# Helper Functions
# ==============================================================================

def record_frame_processed(camera_id: str, success: bool, latency: float):
    """Record frame processing metrics."""
    status = "success" if success else "error"
    frames_processed_total.labels(camera_id=camera_id, status=status).inc()
    frame_processing_latency_seconds.labels(
        camera_id=camera_id,
        worker_type="ingestion"
    ).observe(latency)


def record_detection(camera_id: str, detections: int, latency: float, confidence: float):
    """Record detection metrics."""
    confidence_range = (
        "high" if confidence > 0.8 else
        "medium" if confidence > 0.5 else
        "low"
    )
    detections_total.labels(
        camera_id=camera_id,
        confidence_range=confidence_range
    ).inc()
    detection_latency_seconds.labels(model_version="yolov8n").observe(latency)
    detections_per_frame.labels(camera_id=camera_id).observe(detections)


def record_face_recognition(person_id: str, is_new: bool, confidence: float, latency: float):
    """Record face recognition metrics."""
    faces_recognized_total.labels(
        person_id=person_id,
        is_new=str(is_new).lower()
    ).inc()
    face_recognition_latency_seconds.labels(model_name="facenet512").observe(latency)
    face_recognition_confidence.observe(confidence)


def record_redis_event(stream_name: str, event_type: str, success: bool):
    """Record Redis event publication."""
    status = "success" if success else "error"
    redis_events_published_total.labels(
        stream_name=stream_name,
        event_type=event_type,
        status=status
    ).inc()


def record_db_query(query_type: str, table: str, duration: float):
    """Record database query metrics."""
    db_query_duration_seconds.labels(
        query_type=query_type,
        table=table
    ).observe(duration)


def update_people_count(camera_id: str, zone_name: str, count: int):
    """Update current people count."""
    people_count_current.labels(
        camera_id=camera_id,
        zone_name=zone_name
    ).set(count)


def update_queue_metrics(zone_name: str, length: int, avg_wait: float):
    """Update queue metrics."""
    queue_length_current.labels(zone_name=zone_name).set(length)
    queue_avg_wait_seconds.labels(zone_name=zone_name).set(avg_wait)
