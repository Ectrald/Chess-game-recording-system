"""MJPEG Video Streamer for Chess Game Recording System Web UI.

Encodes and streams live video frames with composite HUD overlays over HTTP
using the multipart/x-mixed-replace standard.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncGenerator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MJPEGStreamer:
    """Thread-safe frame buffer and asynchronous MJPEG generator."""

    def __init__(self, target_fps: int = 25, quality: int = 80) -> None:
        """Initializes streamer with target FPS and JPEG compression quality.

        Args:
            target_fps: Stream frame rate in frames per second.
            quality: JPEG compression quality factor (1-100).
        """
        self.target_fps = max(1, min(60, target_fps))
        self.quality = max(10, min(100, quality))
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._placeholder_jpeg: Optional[bytes] = None

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Encodes BGR image to JPEG bytes and updates the buffer atomically.

        Args:
            frame_bgr: Composite HUD image [H, W, 3] in BGR color format.
        """
        success, buffer = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
        )
        if success:
            with self._lock:
                self._latest_jpeg = buffer.tobytes()

    def update_jpeg_bytes(self, jpeg_bytes: bytes) -> None:
        """Sets pre-encoded JPEG bytes into the buffer atomically."""
        with self._lock:
            self._latest_jpeg = jpeg_bytes

    def get_latest_jpeg(self) -> bytes:
        """Returns the latest JPEG frame, falling back to a placeholder if none available."""
        with self._lock:
            if self._latest_jpeg is not None:
                return self._latest_jpeg

        return self._get_placeholder()

    def _get_placeholder(self) -> bytes:
        """Generates or returns a cached dark placeholder image."""
        if self._placeholder_jpeg is None:
            canvas = np.zeros((360, 640, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            text = "Awaiting Video Stream..."
            cv2.putText(
                canvas,
                text,
                (140, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
            success, buffer = cv2.imencode(
                ".jpg",
                canvas,
                [int(cv2.IMWRITE_JPEG_QUALITY), 75],
            )
            self._placeholder_jpeg = buffer.tobytes() if success else b""

        return self._placeholder_jpeg

    async def frame_generator(self, fps: Optional[int] = None) -> AsyncGenerator[bytes, None]:
        """Asynchronously yields multipart MJPEG boundary frames for HTTP streaming.

        Args:
            fps: Optional streaming rate override.

        Yields:
            HTTP multipart chunk bytes formatted as `multipart/x-mixed-replace`.
        """
        target_fps = fps or self.target_fps
        delay = 1.0 / max(1.0, float(target_fps))

        while True:
            jpeg_data = self.get_latest_jpeg()
            chunk = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg_data)).encode("ascii") + b"\r\n\r\n"
                + jpeg_data
                + b"\r\n"
            )
            yield chunk
            await asyncio.sleep(delay)
