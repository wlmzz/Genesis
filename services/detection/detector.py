"""
YOLO-based person detector with tracking
Wraps Ultralytics YOLO for person detection and tracking
"""
import logging
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single person detection result"""
    track_id: Optional[int]
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # 0 = person in COCO


class PersonDetector:
    """
    YOLO-based person detector with tracking
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Args:
            model_path: Path to YOLO model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run on (cpu, cuda, mps)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Load YOLO model
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        # Set device
        if device != "cpu":
            self.model.to(device)

        logger.info(f"✓ YOLO model loaded (device: {device})")

        # Tracking state
        self.track_history = {}  # track_id -> history of positions

    def detect(
        self,
        frame: np.ndarray,
        enable_tracking: bool = True
    ) -> List[Detection]:
        """
        Detect persons in frame

        Args:
            frame: Input frame (BGR numpy array)
            enable_tracking: Enable multi-object tracking

        Returns:
            List of Detection objects
        """
        try:
            # Run detection/tracking
            if enable_tracking:
                results = self.model.track(
                    source=frame,
                    persist=True,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    classes=[0],  # person class only
                )
            else:
                results = self.model(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    classes=[0],
                )

            # Parse results
            detections = []

            if len(results) == 0:
                return detections

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return detections

            # Extract boxes
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            # Extract track IDs if tracking enabled
            if enable_tracking and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [None] * len(boxes)

            # Build detections
            for bbox, conf, track_id in zip(boxes, confidences, track_ids):
                detection = Detection(
                    track_id=int(track_id) if track_id is not None else None,
                    bbox=tuple(bbox.tolist()),
                    confidence=float(conf),
                    class_id=0
                )
                detections.append(detection)

                # Update track history
                if track_id is not None:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2

                    if track_id not in self.track_history:
                        self.track_history[track_id] = []

                    self.track_history[track_id].append((cx, cy))

                    # Keep last 30 positions
                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id].pop(0)

            return detections

        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return []

    def get_track_path(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Get historical path for a track

        Args:
            track_id: Track identifier

        Returns:
            List of (x, y) positions
        """
        return self.track_history.get(track_id, [])

    def reset_tracking(self):
        """Reset tracking state (useful for new video/camera)"""
        self.track_history.clear()
        logger.info("Tracking state reset")

    def get_model_info(self) -> dict:
        """Get detector model information"""
        return {
            "model_path": self.model_path,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "active_tracks": len(self.track_history),
        }
