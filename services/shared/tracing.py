"""
OpenTelemetry distributed tracing for Genesis workers.

This module provides standardized tracing instrumentation for all microservices.
"""

import logging
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

logger = logging.getLogger(__name__)


def init_tracing(
    service_name: str,
    service_version: str = "1.0.0",
    otlp_endpoint: str = "http://tempo:4317",
    environment: str = "development"
) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing for a Genesis service.
    
    Args:
        service_name: Name of the service (e.g., "detection-worker")
        service_version: Version of the service
        otlp_endpoint: OTLP exporter endpoint (Tempo)
        environment: Deployment environment
        
    Returns:
        Configured tracer instance
    """
    # Create resource with service metadata
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "deployment.environment": environment,
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.language": "python",
        "service.namespace": "genesis"
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Configure OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=True  # Use TLS in production
    )
    
    # Add batch span processor
    provider.add_span_processor(
        BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000
        )
    )
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    # Auto-instrument libraries
    try:
        RedisInstrumentor().instrument()
        AsyncPGInstrumentor().instrument()
        RequestsInstrumentor().instrument()
        LoggingInstrumentor().instrument()
        logger.info(f"Tracing initialized for {service_name}")
    except Exception as e:
        logger.warning(f"Failed to auto-instrument libraries: {e}")
    
    # Return tracer
    return trace.get_tracer(service_name, service_version)


class TracingContext:
    """
    Context manager for creating traced operations.
    
    Example:
        with TracingContext(tracer, "process_frame") as span:
            span.set_attribute("frame_id", frame_id)
            span.set_attribute("camera_id", camera_id)
            # ... processing logic
    """
    
    def __init__(
        self,
        tracer: trace.Tracer,
        operation_name: str,
        attributes: Optional[dict] = None
    ):
        self.tracer = tracer
        self.operation_name = operation_name
        self.attributes = attributes or {}
        self.span = None
        
    def __enter__(self):
        self.span = self.tracer.start_as_current_span(self.operation_name)
        self.span.__enter__()
        
        # Set attributes
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
            
        return self.span
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.span.record_exception(exc_val)
            self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
        
        self.span.__exit__(exc_type, exc_val, exc_tb)


def trace_async_function(tracer: trace.Tracer, span_name: Optional[str] = None):
    """
    Decorator for tracing async functions.
    
    Example:
        @trace_async_function(tracer, "process_event")
        async def process_event(self, event):
            # ... processing logic
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            operation_name = span_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(operation_name) as span:
                # Add function arguments as attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
                    
        return wrapper
    return decorator


def trace_sync_function(tracer: trace.Tracer, span_name: Optional[str] = None):
    """
    Decorator for tracing synchronous functions.
    
    Example:
        @trace_sync_function(tracer, "extract_embedding")
        def extract_embedding(self, face_img):
            # ... processing logic
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            operation_name = span_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(operation_name) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
                    
        return wrapper
    return decorator


def add_event_attributes(span: trace.Span, event: dict):
    """
    Add event-specific attributes to a span.
    
    Args:
        span: Active span
        event: Event data dictionary
    """
    span.set_attribute("event.type", event.get("type", "unknown"))
    span.set_attribute("event.timestamp", event.get("timestamp", 0))
    
    if "camera_id" in event:
        span.set_attribute("camera.id", event["camera_id"])
    
    if "frame_id" in event:
        span.set_attribute("frame.id", event["frame_id"])
        
    if "track_id" in event:
        span.set_attribute("track.id", event["track_id"])
        
    if "person_id" in event:
        span.set_attribute("person.id", event["person_id"])


def create_child_span(
    tracer: trace.Tracer,
    parent_span: trace.Span,
    operation_name: str,
    attributes: Optional[dict] = None
):
    """
    Create a child span linked to a parent.
    
    Args:
        tracer: Tracer instance
        parent_span: Parent span
        operation_name: Name of the child operation
        attributes: Optional attributes
        
    Returns:
        Child span context manager
    """
    with trace.use_span(parent_span):
        child_span = tracer.start_span(operation_name)
        
        if attributes:
            for key, value in attributes.items():
                child_span.set_attribute(key, value)
                
        return child_span
