"""Video device discovery, raw stream recording, and batch video transcription utilities."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

import cv2
import numpy as np

from src.core.config import AppConfig
from src.core.datatypes import Frame
from src.orchestrator import SystemOrchestrator

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _read_v4l2_name(index: int) -> str:
    sys_name = Path(f"/sys/class/video4linux/video{index}/name")
    try:
        if sys_name.is_file():
            return sys_name.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return f"Camera {index}"


def discover_video_devices(
    active_source: Optional[str] = None,
    active_opened: bool = False,
    max_probe: int = 6,
) -> List[Dict[str, Any]]:
    """Discovers connected camera devices without interrupting the active capture stream."""
    devices: List[Dict[str, Any]] = []
    seen_sources: set[str] = set()

    active_norm = str(active_source).strip() if active_source is not None else ""
    if active_norm.startswith("/dev/video") and active_norm.replace("/dev/video", "").isdigit():
        active_idx: Optional[int] = int(active_norm.replace("/dev/video", ""))
    elif active_norm.isdigit():
        active_idx = int(active_norm)
    else:
        active_idx = None

    candidate_indices: List[int] = []
    dev_dir = Path("/dev")
    if dev_dir.exists():
        for p in sorted(dev_dir.glob("video*")):
            suffix = p.name.replace("video", "")
            if suffix.isdigit():
                candidate_indices.append(int(suffix))
    for i in range(max_probe):
        if i not in candidate_indices:
            candidate_indices.append(i)

    for idx in sorted(set(candidate_indices)):
        src_str = str(idx)
        hw_name = _read_v4l2_name(idx)
        is_active = active_idx is not None and idx == active_idx

        if is_active:
            devices.append(
                {
                    "source": src_str,
                    "device_path": f"/dev/video{idx}",
                    "label": f"{hw_name} (/dev/video{idx})",
                    "available": bool(active_opened),
                    "active": True,
                }
            )
            seen_sources.add(src_str)
            continue

        cap = None
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if w > 0 and h > 0:
                    devices.append(
                        {
                            "source": src_str,
                            "device_path": f"/dev/video{idx}",
                            "label": f"{hw_name} (/dev/video{idx}) [{int(w)}x{int(h)}]",
                            "available": True,
                            "active": False,
                        }
                    )
                    seen_sources.add(src_str)
        except Exception:
            pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    if active_norm and active_norm not in seen_sources and active_idx is None:
        devices.insert(
            0,
            {
                "source": active_norm,
                "device_path": active_norm,
                "label": f"Custom Source ({Path(active_norm).name})",
                "available": bool(active_opened),
                "active": True,
            },
        )

    return devices


def list_available_videos(sessions_base_dir: Path | str, recordings_dir: Path | str) -> List[Dict[str, Any]]:
    """Lists saved recordings, uploaded videos, and workspace benchmark videos."""
    base = Path(sessions_base_dir)
    rec_root = Path(recordings_dir)
    search_dirs: List[tuple[str, Path]] = [
        ("recorded", rec_root),
        ("uploaded", rec_root / "uploads"),
        ("benchmark", Path("data/benchmark_videos")),
    ]
    if base.exists():
        for s_dir in base.iterdir():
            if s_dir.is_dir():
                r_sub = s_dir / "recordings"
                if r_sub.is_dir():
                    search_dirs.append(("session", r_sub))

    items: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for category, directory in search_dirs:
        if not directory.is_dir():
            continue
        for f in sorted(directory.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not f.is_file() or f.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            resolved = str(f.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            st = f.stat()
            items.append(
                {
                    "name": f.name,
                    "path": str(f),
                    "category": category,
                    "size_mb": round(st.st_size / (1024 * 1024), 2),
                    "modified_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
    return items


class LiveVideoRecorder:
    """Thread-safe raw camera video writer with MP4 (mp4v) / AVI (MJPG) support."""

    def __init__(self, recordings_dir: Path | str) -> None:
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._writer: Optional[cv2.VideoWriter] = None
        self._file_path: Optional[Path] = None
        self._size: Optional[tuple[int, int]] = None
        self._fps: float = 20.0
        self._frame_count: int = 0
        self._started_at: Optional[float] = None
        self._session_id: Optional[str] = None

    def is_recording(self) -> bool:
        with self._lock:
            return self._writer is not None and self._writer.isOpened()

    def start(
        self,
        width: int,
        height: int,
        fps: float = 20.0,
        session_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Starts writing incoming raw camera frames to disk."""
        with self._lock:
            if self._writer is not None:
                raise RuntimeError("Video recording is already active")

            target_dir = Path(output_dir) if output_dir else self.recordings_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_sid = (session_id or "game").replace("/", "_")
            fps_val = float(fps) if 1.0 <= fps <= 60.0 else 20.0
            w = max(16, int(width))
            h = max(16, int(height))

            mp4_path = target_dir / f"{safe_sid}_{ts}.mp4"
            writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_val, (w, h))
            chosen_path = mp4_path
            if not writer.isOpened():
                writer.release()
                avi_path = target_dir / f"{safe_sid}_{ts}.avi"
                writer = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*"MJPG"), fps_val, (w, h))
                chosen_path = avi_path

            if not writer.isOpened():
                writer.release()
                raise RuntimeError("Failed to initialize OpenCV VideoWriter")

            self._writer = writer
            self._file_path = chosen_path
            self._size = (w, h)
            self._fps = fps_val
            self._frame_count = 0
            self._started_at = time.monotonic()
            self._session_id = session_id

            logger.info("Started raw video recording -> %s (%dx%d @ %.1f FPS)", chosen_path, w, h, fps_val)
            return self.status()

    def write_frame(self, image_bgr: np.ndarray) -> None:
        """Appends a raw BGR frame to the active video file."""
        with self._lock:
            if self._writer is None or not self._writer.isOpened() or self._size is None:
                return
            h, w = image_bgr.shape[:2]
            tw, th = self._size
            frame_to_write = (
                image_bgr if (w == tw and h == th) else cv2.resize(image_bgr, (tw, th))
            )
            self._writer.write(frame_to_write)
            self._frame_count += 1

    def stop(self) -> Optional[Dict[str, Any]]:
        """Stops recording, flushes file, and returns metadata."""
        with self._lock:
            if self._writer is None:
                return None
            writer = self._writer
            file_path = self._file_path
            frames = self._frame_count
            started = self._started_at
            self._writer = None
            self._file_path = None
            self._size = None
            self._frame_count = 0
            self._started_at = None

        writer.release()
        duration = (time.monotonic() - started) if started is not None else 0.0
        size_bytes = file_path.stat().st_size if (file_path and file_path.exists()) else 0
        logger.info("Stopped raw video recording: %s (frames=%d, bytes=%d)", file_path, frames, size_bytes)
        return {
            "filename": file_path.name if file_path else None,
            "path": str(file_path) if file_path else None,
            "frame_count": frames,
            "duration_sec": round(duration, 1),
            "size_bytes": size_bytes,
            "session_id": self._session_id,
        }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            active = self._writer is not None and self._writer.isOpened()
            duration = (time.monotonic() - self._started_at) if (active and self._started_at) else 0.0
            return {
                "is_recording": active,
                "filename": self._file_path.name if self._file_path else None,
                "path": str(self._file_path) if self._file_path else None,
                "frame_count": self._frame_count,
                "duration_sec": round(duration, 1),
            }


class VideoTranscriptionManager:
    """Runs asynchronous offline chess move decoding on an uploaded or saved video file."""

    def __init__(self, base_config: AppConfig) -> None:
        self.base_config = base_config
        self._lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status: Dict[str, Any] = {
            "status": "idle",
            "job_id": None,
            "video_name": None,
            "session_id": None,
            "progress_pct": 0.0,
            "frames_processed": 0,
            "total_frames": 0,
            "plies_found": 0,
            "moves": [],
            "error": None,
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status, moves=list(self._status["moves"]))

    def start(
        self,
        video_path: Path,
        white_player: str = "White Player",
        black_player: str = "Black Player",
        session_id: Optional[str] = None,
        target_fps: float = 6.0,
        on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_finished: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Launches background decoding task for the given video file."""
        with self._lock:
            if self._status["status"] == "running":
                raise RuntimeError("A video transcription job is already running")
            if not video_path.is_file():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            job_id = uuid.uuid4().hex[:8]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_sid = session_id or f"video_{ts}_{job_id[:6]}"
            self._cancel_event.clear()
            self._status = {
                "status": "running",
                "job_id": job_id,
                "video_name": video_path.name,
                "session_id": target_sid,
                "progress_pct": 0.0,
                "frames_processed": 0,
                "total_frames": 0,
                "plies_found": 0,
                "moves": [],
                "error": None,
            }

        self._thread = threading.Thread(
            target=self._run_job,
            args=(video_path, target_sid, white_player, black_player, target_fps, on_update, on_finished),
            name=f"VideoTranscribe-{job_id}",
            daemon=True,
        )
        self._thread.start()
        return self.get_status()

    def cancel(self) -> None:
        self._cancel_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_job(
        self,
        video_path: Path,
        session_id: str,
        white_player: str,
        black_player: str,
        target_fps: float,
        on_update: Optional[Callable[[Dict[str, Any]], None]],
        on_finished: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        cfg = self.base_config.model_copy(deep=True)
        cfg.analysis.stockfish_enabled = False
        orch: Optional[SystemOrchestrator] = None
        cap: Optional[cv2.VideoCapture] = None

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video file: {video_path}")

            video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if video_fps <= 0:
                video_fps = 30.0
            total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            stride = max(1, int(round(video_fps / max(1.0, target_fps))))

            orch = SystemOrchestrator(config=cfg, session_id=session_id)
            orch.start_session(white_player=white_player, black_player=black_player)

            with self._lock:
                self._status["total_frames"] = max(1, total_raw // stride) if total_raw > 0 else 0

            raw_idx = 0
            processed = 0
            last_notify = 0.0

            while not self._cancel_event.is_set():
                ok = cap.grab()
                if not ok:
                    break
                if raw_idx % stride == 0:
                    ret, img = cap.retrieve()
                    if not ret or img is None:
                        break
                    h, w = img.shape[:2]
                    frame = Frame(
                        frame_id=raw_idx,
                        timestamp=float(raw_idx) / video_fps,
                        image=img,
                        width=w,
                        height=h,
                    )
                    orch.process_frame(frame)
                    processed += 1

                    now = time.monotonic()
                    if now - last_notify >= 0.4:
                        last_notify = now
                        hist = orch.state_manager.history
                        pct = (
                            min(99.0, 100.0 * raw_idx / total_raw)
                            if total_raw > 0
                            else 0.0
                        )
                        with self._lock:
                            self._status["progress_pct"] = round(pct, 1)
                            self._status["frames_processed"] = processed
                            self._status["plies_found"] = len(hist)
                            self._status["moves"] = [m.move_san for m in hist]
                        if on_update:
                            on_update(self.get_status())
                raw_idx += 1

            manifest = orch.stop()
            moves_list = [m.move_san for m in orch.state_manager.history]
            with self._lock:
                self._status["status"] = "cancelled" if self._cancel_event.is_set() else "completed"
                self._status["progress_pct"] = 100.0 if not self._cancel_event.is_set() else self._status["progress_pct"]
                self._status["frames_processed"] = processed
                self._status["plies_found"] = manifest.total_plies
                self._status["moves"] = moves_list

            snap = self.get_status()
            if on_update:
                on_update(snap)
            if on_finished and snap["status"] == "completed":
                on_finished(snap)

        except Exception as exc:
            logger.error("Video transcription failed (%s): %s", video_path, exc, exc_info=True)
            with self._lock:
                self._status["status"] = "failed"
                self._status["error"] = str(exc)
            if orch is not None:
                try:
                    orch.stop()
                except Exception:
                    pass
            if on_update:
                on_update(self.get_status())
        finally:
            if cap is not None:
                cap.release()
