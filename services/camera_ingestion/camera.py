"""
Camera interface for capturing frames from various sources
Supports webcams, RTSP streams, video files
"""
import cv2
import logging
import time
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class CameraConfig:
    """Configuration for a camera source"""

    def __init__(
        self,
        camera_id: str,
        source: str | int,
        target_fps: int = 12,
        resize_width: int = 960,
        enabled: bool = True,
        rtsp_transport: str = "tcp"
    ):
        """
        Args:
            camera_id: Unique camera identifier
            source: Camera source (int for webcam, string for RTSP/file)
            target_fps: Target frames per second
            resize_width: Resize frame to this width (maintaining aspect ratio)
            enabled: Whether camera is enabled
            rtsp_transport: RTSP transport protocol (tcp/udp)
        """
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.resize_width = resize_width
        self.enabled = enabled
        self.rtsp_transport = rtsp_transport


class Camera:
    """
    Camera capture interface with automatic reconnection
    """

    def __init__(self, config: CameraConfig):
        """
        Args:
            config: Camera configuration
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.last_frame_time = 0.0
        self.is_open = False

        # Frame timing
        self.frame_interval = 1.0 / config.target_fps

        # Reconnection settings
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5.0

        # Open camera
        self._open_camera()

    def _open_camera(self) -> bool:
        """
        Open camera capture

        Returns:
            True if successful, False otherwise
        """
        try:
            # Close existing capture
            if self.cap is not None:
                self.cap.release()

            # Open camera
            if isinstance(self.config.source, int):
                # Webcam
                self.cap = cv2.VideoCapture(self.config.source)
            else:
                # RTSP or file
                if self.config.source.startswith("rtsp://"):
                    # RTSP stream
                    self.cap = cv2.VideoCapture(
                        self.config.source,
                        cv2.CAP_FFMPEG
                    )
                    # Set RTSP transport
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    # Video file
                    self.cap = cv2.VideoCapture(self.config.source)

            # Verify opened
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.config.camera_id}")
                return False

            self.is_open = True
            self.reconnect_attempts = 0

            logger.info(
                f"Camera {self.config.camera_id} opened successfully "
                f"(source: {self.config.source})"
            )

            return True

        except Exception as e:
            logger.error(f"Error opening camera {self.config.camera_id}: {e}")
            self.is_open = False
            return False

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame maintaining aspect ratio

        Args:
            frame: Input frame

        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        if w == self.config.resize_width:
            return frame

        scale = self.config.resize_width / w
        new_h = int(h * scale)

        return cv2.resize(frame, (self.config.resize_width, new_h))

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a frame from camera with FPS throttling

        Returns:
            Frame array if successful, None otherwise
        """
        # Check if enough time has passed for next frame
        now = time.time()
        time_since_last = now - self.last_frame_time

        if time_since_last < self.frame_interval:
            # Too soon, skip frame
            return None

        # Check if camera is open
        if not self.is_open:
            # Attempt reconnection
            if self.reconnect_attempts < self.max_reconnect_attempts:
                logger.warning(
                    f"Camera {self.config.camera_id} not open, "
                    f"attempting reconnection ({self.reconnect_attempts + 1}/"
                    f"{self.max_reconnect_attempts})"
                )
                time.sleep(self.reconnect_delay)
                self.reconnect_attempts += 1
                self._open_camera()

            return None

        try:
            # Read frame
            ret, frame = self.cap.read()

            if not ret or frame is None:
                logger.warning(f"Failed to read frame from {self.config.camera_id}")
                self.is_open = False
                return None

            # Resize frame
            frame = self._resize_frame(frame)

            # Update counters
            self.frame_count += 1
            self.last_frame_time = now

            return frame

        except Exception as e:
            logger.error(f"Error capturing frame from {self.config.camera_id}: {e}")
            self.is_open = False
            return None

    def get_info(self) -> dict:
        """
        Get camera information

        Returns:
            Camera info dictionary
        """
        info = {
            "camera_id": self.config.camera_id,
            "source": str(self.config.source),
            "is_open": self.is_open,
            "frame_count": self.frame_count,
            "target_fps": self.config.target_fps,
            "enabled": self.config.enabled,
        }

        if self.cap is not None and self.is_open:
            try:
                info.update({
                    "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": self.cap.get(cv2.CAP_PROP_FPS),
                })
            except:
                pass

        return info

    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_open = False
            logger.info(f"Camera {self.config.camera_id} released")

    def __del__(self):
        """Cleanup on destruction"""
        self.release()
