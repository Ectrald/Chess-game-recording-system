"""Video stream abstraction layer for physical cameras and synthetic generators.

Isolates computer vision inference from underlying capture sources
(RTSP, USB webcam, file, synthetic board generator).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Union
import cv2

from src.core.config import CameraConfig
from src.core.datatypes import Frame
from src.core.exceptions import VideoSourceError
from src.vision.synthetic_board import SyntheticBoardGenerator


class BaseVideoAdapter(ABC):
    """Abstract interface for video frame acquisition."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Reads and returns the next video frame, or None if stream ended (EOF)."""
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Returns stream frame rate in frames per second."""
        pass

    @abstractmethod
    def get_frame_count(self) -> int:
        """Returns total frame count, or -1 for indefinite live streams."""
        pass

    @abstractmethod
    def seek(self, frame_id: int) -> bool:
        """Seeks to the given frame index. Returns True if seek was successful."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Returns True if the underlying capture device/stream is currently active."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases underlying video resources and capture handles."""
        pass

    def __enter__(self) -> BaseVideoAdapter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __iter__(self) -> Iterator[Frame]:
        while self.is_opened():
            frame = self.read()
            if frame is None:
                break
            yield frame


class OpenCVVideoAdapter(BaseVideoAdapter):
    """Adapter for OpenCV-supported video streams (webcam, file, RTSP)."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._current_frame_id = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = config.fps
        self._total_frames: int = -1

        self._open()

    def _open(self) -> None:
        source: Union[int, str] = self.config.source_path
        # Try converting numeric string to camera index (e.g. "0" -> 0)
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise VideoSourceError(f"Failed to open video source: '{self.config.source_path}'")

        # Configure camera resolution if device
        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            if self.config.fps > 0:
                self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        detected_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if detected_fps and detected_fps > 0:
            self._fps = detected_fps
        else:
            self._fps = self.config.fps or 30.0

        cnt = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._total_frames = int(cnt) if cnt > 0 else -1
        self._current_frame_id = 0

    def read(self) -> Optional[Frame]:
        if self._cap is None or not self._cap.isOpened():
            return None

        success, img = self._cap.read()
        if not success or img is None:
            return None

        # Apply rotation if configured
        if self.config.rotate == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif self.config.rotate == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif self.config.rotate == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = img.shape[:2]
        timestamp = float(self._current_frame_id) / self._fps

        frame = Frame(
            frame_id=self._current_frame_id,
            timestamp=timestamp,
            image=img,
            width=w,
            height=h,
        )
        self._current_frame_id += 1
        return frame

    def get_fps(self) -> float:
        return self._fps

    def get_frame_count(self) -> int:
        return self._total_frames

    def seek(self, frame_id: int) -> bool:
        if self._cap is None or not self._cap.isOpened():
            return False
        if frame_id < 0:
            return False
        if self._total_frames > 0 and frame_id >= self._total_frames:
            return False

        ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_id))
        if ok:
            self._current_frame_id = frame_id
        return ok

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticVideoAdapter(BaseVideoAdapter):
    """Adapter wrapping SyntheticBoardGenerator for repeatable testing."""

    def __init__(
        self,
        generator: Optional[SyntheticBoardGenerator] = None,
        config: Optional[CameraConfig] = None,
        pgn_file: Optional[Union[str, Path]] = None,
    ) -> None:
        if generator is not None:
            self.generator = generator
        else:
            self.generator = SyntheticBoardGenerator(
                canvas_width=config.width if config else 1920,
                canvas_height=config.height if config else 1080,
                fps=config.fps if config else 30.0,
            )
            # If pgn_file provided or configured
            pgn_source = pgn_file or (config.source_path if config and config.source_path else None)
            if pgn_source and Path(str(pgn_source)).is_file():
                self.generator.set_game_from_pgn(str(pgn_source))

        self._current_frame_id = 0
        self._is_active = True

    def read(self) -> Optional[Frame]:
        if not self._is_active or self._current_frame_id >= self.generator.total_frames:
            return None

        frame = self.generator.render_frame(self._current_frame_id)
        self._current_frame_id += 1
        return frame

    def get_fps(self) -> float:
        return self.generator.fps

    def get_frame_count(self) -> int:
        return self.generator.total_frames

    def seek(self, frame_id: int) -> bool:
        if 0 <= frame_id < self.generator.total_frames:
            self._current_frame_id = frame_id
            return True
        return False

    def is_opened(self) -> bool:
        return self._is_active

    def release(self) -> None:
        self._is_active = False


def create_video_adapter(config: CameraConfig) -> BaseVideoAdapter:
    """Factory function creating the appropriate video adapter for the given configuration.
    
    Args:
        config: CameraConfig instance.
        
    Returns:
        BaseVideoAdapter instance.
        
    Raises:
        VideoSourceError: If source type is unrecognized or opening fails.
    """
    source_type = config.source_type.lower()
    if source_type == "synthetic":
        return SyntheticVideoAdapter(config=config)
    elif source_type in ("file", "device", "rtsp"):
        return OpenCVVideoAdapter(config=config)
    else:
        raise VideoSourceError(f"Unsupported video source type: '{config.source_type}'")
