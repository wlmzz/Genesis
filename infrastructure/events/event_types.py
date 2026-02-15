"""
Event type definitions for Genesis event-driven architecture
All events are immutable dataclasses with JSON serialization support
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Literal, Dict, Any, Tuple
import json


@dataclass
class BaseEvent:
    """Base class for all events"""

    def to_dict(self) -> dict:
        """Convert event to dictionary for serialization"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict):
        """Create event from dictionary"""
        return cls(**data)


@dataclass
class FrameCapturedEvent(BaseEvent):
    """
    Event published when a new frame is captured from camera
    Frame data is stored in Redis cache or S3, referenced by frame_id
    """
    camera_id: str
    frame_id: str
    timestamp: float
    frame_shape: Tuple[int, int, int]  # (height, width, channels)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PersonDetectedEvent(BaseEvent):
    """
    Event published when YOLO detects a person in a frame
    """
    frame_id: str
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FaceRecognizedEvent(BaseEvent):
    """
    Event published when a face is recognized or registered as new
    """
    track_id: int
    person_id: Optional[str]  # None if new face
    embedding: List[float]  # Face embedding vector (512-dim for Facenet512)
    confidence: float
    is_new_face: bool
    timestamp: float
    current_zone: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ZoneEvent(BaseEvent):
    """
    Event published when a person enters or exits a zone
    """
    track_id: int
    person_id: Optional[str]
    zone_name: str
    event_type: Literal["entered", "exited"]
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AlertTriggeredEvent(BaseEvent):
    """
    Event published when an alert is triggered (anomaly, threshold, etc.)
    """
    alert_type: str  # e.g., "anomaly_crowd", "queue_threshold", "vip_detected"
    severity: Literal["info", "warning", "critical"]
    message: str
    context: Dict[str, Any]  # Additional alert context
    timestamp: float
    camera_id: Optional[str] = None
    person_id: Optional[str] = None


@dataclass
class MetricsAggregatedEvent(BaseEvent):
    """
    Event published periodically with aggregated metrics snapshot
    """
    interval_start: float
    interval_end: float
    metrics: Dict[str, Any]  # Flexible metrics payload
    timestamp: float
    camera_id: Optional[str] = None


@dataclass
class SessionEvent(BaseEvent):
    """
    Event published when a person session starts or ends
    """
    person_id: str
    session_id: str
    event_type: Literal["session_start", "session_end"]
    timestamp: float
    camera_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    # Session end details
    duration_seconds: Optional[float] = None
    zones_visited: Optional[List[str]] = None


# Event type registry for deserialization
EVENT_TYPE_REGISTRY = {
    "FrameCapturedEvent": FrameCapturedEvent,
    "PersonDetectedEvent": PersonDetectedEvent,
    "FaceRecognizedEvent": FaceRecognizedEvent,
    "ZoneEvent": ZoneEvent,
    "AlertTriggeredEvent": AlertTriggeredEvent,
    "MetricsAggregatedEvent": MetricsAggregatedEvent,
    "SessionEvent": SessionEvent,
}


def deserialize_event(event_type: str, data: dict) -> BaseEvent:
    """
    Deserialize event from type name and data dictionary

    Args:
        event_type: Name of event class
        data: Event data dictionary

    Returns:
        Deserialized event instance

    Raises:
        ValueError: If event type is unknown
    """
    event_class = EVENT_TYPE_REGISTRY.get(event_type)
    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class.from_dict(data)
