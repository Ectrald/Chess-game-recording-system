"""FastAPI Application Factory for Chess Game Recording System Web UI.

Provides RESTful endpoints, MJPEG video streaming, WebSocket event delivery,
session lifecycle management, video device switching, raw video recording,
and offline video game transcription.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import urllib.parse

import chess
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.core.config import CameraConfig
from src.core.datatypes import FSMState
from src.orchestrator import SystemOrchestrator
from src.ui.visualizer import VisualHUDComposer
from src.ui.web.streamer import MJPEGStreamer
from src.ui.web.video_services import (
    VIDEO_EXTENSIONS,
    LiveVideoRecorder,
    VideoTranscriptionManager,
    discover_video_devices,
    list_available_videos,
)
from src.ui.web.ws_manager import WebSocketManager
from src.vision.video_adapter import BaseVideoAdapter, create_video_adapter

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


class ResolvePayload(BaseModel):
    """Payload schema for manual or ranked UNCERTAIN move resolution."""
    move_uci: Optional[str] = Field(default=None, description="Direct UCI move string, e.g. 'e2e4'")
    auto_accept_top: bool = Field(default=False, description="Automatically accept top-ranked candidate")
    candidate_index: Optional[int] = Field(default=None, description="0-based index of candidate to accept")


class SetCornersPayload(BaseModel):
    """Payload schema for manual board corner calibration."""
    corners: List[List[float]] = Field(
        ...,
        description="4 [x, y] corner coordinates in video frame space: [TL, TR, BR, BL]",
    )


class NewSessionPayload(BaseModel):
    """Payload schema for starting a new game session."""
    white_player: Optional[str] = None
    black_player: Optional[str] = None
    session_id: Optional[str] = None
    pause: bool = True


class SwitchVideoSourcePayload(BaseModel):
    """Payload schema for switching active video input source."""
    source: str
    source_type: Optional[str] = None


class TranscribeVideoPayload(BaseModel):
    """Payload schema for decoding chess moves from an uploaded or recorded video."""
    video_path: str
    white_player: str = "White Player"
    black_player: str = "Black Player"
    target_fps: float = 6.0


SetCornersPayload.model_rebuild()


def get_state_payload(
    orchestrator: SystemOrchestrator,
    video_source: Optional[str] = None,
    recording_status: Optional[Dict[str, Any]] = None,
    transcribe_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serializes authoritative orchestrator and board state into a JSON dictionary."""
    board = orchestrator.state_manager.current_board
    turn = "white" if board.turn == chess.WHITE else "black"
    is_check = board.is_check()
    is_game_over = board.is_game_over()
    result = board.result() if is_game_over else "*"
    fsm_state = orchestrator.fsm.current_state.value
    history = orchestrator.state_manager.history

    move_history: List[Dict[str, Any]] = [
        {
            "ply": m.ply,
            "san": m.move_san,
            "uci": m.move_uci,
            "fen_after": m.fen_after,
        }
        for m in history
    ]
    last_move = (
        {"ply": history[-1].ply, "san": history[-1].move_san, "uci": history[-1].move_uci}
        if history else None
    )

    eval_dict: Optional[Dict[str, Any]] = None
    if orchestrator._latest_eval is not None:
        ev = orchestrator._latest_eval.eval_after
        eval_dict = {
            "score_cp": ev.eval_cp if ev is not None else None,
            "mate_in": ev.mate_in if ev is not None else None,
            "depth": ev.depth if ev is not None else 0,
            "best_move": ev.best_move_uci if ev is not None else None,
            "quality": orchestrator._latest_eval.quality.value if orchestrator._latest_eval.quality else None,
        }

    candidates: List[Dict[str, Any]] = []
    last_decision = orchestrator.fsm.last_decision or orchestrator._last_decision
    if orchestrator._active_uncertainty and last_decision is not None:
        hypos = last_decision.top_hypotheses
        if not hypos and last_decision.accepted_hypothesis:
            hypos = [last_decision.accepted_hypothesis]
        candidates = [
            {
                "rank": idx + 1,
                "uci": h.move_uci,
                "san": h.move_san or h.move_uci,
                "score": round(float(h.total_score), 3),
            }
            for idx, h in enumerate(hypos)
        ]

    board_corners: Optional[List[List[float]]] = None
    if orchestrator._board_geometry is not None and orchestrator._board_geometry.corners:
        board_corners = [
            [
                round(float(c[0] if isinstance(c, (tuple, list)) else c.x), 1),
                round(float(c[1] if isinstance(c, (tuple, list)) else c.y), 1),
            ]
            for c in orchestrator._board_geometry.corners
        ]

    payload: Dict[str, Any] = {
        "type": "gameState",
        "session_id": orchestrator.session_id,
        "initial_fen": orchestrator.state_manager._initial_fen,
        "fen": orchestrator.state_manager.current_fen,
        "turn": turn,
        "ply": orchestrator.state_manager.ply,
        "is_check": is_check,
        "is_game_over": is_game_over,
        "result": result,
        "fsm_state": fsm_state,
        "is_paused": orchestrator._is_paused,
        "active_uncertainty": orchestrator._active_uncertainty,
        "illegal_attempts": getattr(orchestrator, "_illegal_attempts_count", 0),
        "uncertainty_reason": (
            last_decision.rejection_reason
            if (orchestrator._active_uncertainty and last_decision is not None)
            else None
        ),
        "uncertain_squares": (
            list(last_decision.uncertain_squares)
            if (orchestrator._active_uncertainty and last_decision is not None)
            else []
        ),
        "uncertain_candidates": candidates,
        "fps": round(float(orchestrator._current_fps), 1),
        "board_detected": orchestrator._board_geometry is not None,
        "board_orientation": orchestrator._orientation.value,
        "board_corners": board_corners,
        "last_move": last_move,
        "move_history": move_history,
        "engine_eval": eval_dict,
        "white_player": orchestrator.white_player,
        "black_player": orchestrator.black_player,
        "video_source": video_source or str(orchestrator.config.camera.source_path),
    }
    if recording_status is not None:
        payload["video_recording"] = recording_status
    if transcribe_status is not None:
        payload["video_transcription"] = transcribe_status
    return payload


def _infer_source_type(src: str, explicit_type: Optional[str] = None) -> str:
    if explicit_type:
        return explicit_type
    if src.endswith(".pgn"):
        return "synthetic"
    if src.isdigit() or src.startswith("/dev/video"):
        return "device"
    if src.startswith("rtsp://") or src.startswith("http://"):
        return "rtsp"
    return "file"


def create_web_app(
    orchestrator: SystemOrchestrator,
    video_adapter: Optional[BaseVideoAdapter] = None,
    ws_manager: Optional[WebSocketManager] = None,
    streamer: Optional[MJPEGStreamer] = None,
    composer: Optional[VisualHUDComposer] = None,
    start_worker: bool = True,
    start_paused: bool = False,
) -> FastAPI:
    """Builds and configures the FastAPI application instance."""
    ws = ws_manager or WebSocketManager()
    stream = streamer or MJPEGStreamer(
        target_fps=orchestrator.config.web_ui.stream_fps,
        quality=orchestrator.config.web_ui.jpeg_quality,
    )
    hud_composer = composer or VisualHUDComposer(
        width=orchestrator.config.ui.display_width,
        height=orchestrator.config.ui.display_height,
        font_scale=orchestrator.config.ui.font_scale,
    )

    if start_paused:
        orchestrator.pause()

    base_sessions_dir = Path(orchestrator.config.persistence.base_dir)
    recordings_dir = base_sessions_dir.parent / "recordings"
    uploads_dir = recordings_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    recorder = LiveVideoRecorder(recordings_dir=recordings_dir)
    transcriber = VideoTranscriptionManager(base_config=orchestrator.config)

    source_lock = threading.RLock()
    adapter_state: Dict[str, Any] = {
        "adapter": video_adapter,
        "source": str(orchestrator.config.camera.source_path),
        "eof": False,
    }

    stop_event = threading.Event()
    worker_thread: Optional[threading.Thread] = None

    def build_full_state() -> Dict[str, Any]:
        with source_lock:
            cur_src = adapter_state["source"]
        return get_state_payload(
            orchestrator,
            video_source=cur_src,
            recording_status=recorder.status(),
            transcribe_status=transcriber.get_status(),
        )

    def broadcast_state_sync() -> None:
        ws.broadcast_threadsafe(build_full_state())

    def worker_loop() -> None:
        """Background thread acquiring video frames, feeding recorder, and driving orchestrator."""
        orchestrator.start_session(
            white_player=orchestrator.white_player,
            black_player=orchestrator.black_player,
        )
        orchestrator.save_session_snapshot(is_completed=False)

        logger.info("Web background video worker started (start_paused=%s)", orchestrator._is_paused)
        last_ply = orchestrator.state_manager.ply
        last_fsm = orchestrator.fsm.current_state
        last_uncertain = orchestrator._active_uncertainty
        last_session_id = orchestrator.session_id
        last_illegal_count = getattr(orchestrator, "_illegal_attempts_count", 0)
        frame_counter = 0

        try:
            while not stop_event.is_set():
                t0 = time.perf_counter()
                with source_lock:
                    cur_adapter: Optional[BaseVideoAdapter] = adapter_state["adapter"]
                    is_eof = adapter_state["eof"]

                if cur_adapter is None or not cur_adapter.is_opened() or is_eof:
                    time.sleep(0.1)
                    continue

                nominal_fps = cur_adapter.get_fps()
                delay = 1.0 / max(1.0, nominal_fps) if nominal_fps > 0 else 0.033

                frame = cur_adapter.read()
                if frame is None:
                    with source_lock:
                        if adapter_state["adapter"] is cur_adapter:
                            adapter_state["eof"] = True
                    logger.info("Video stream reached end of file or closed.")
                    continue

                # Write raw BGR camera frame if operator started video recording
                if recorder.is_recording():
                    recorder.write_frame(frame.image)

                state = orchestrator.process_frame(frame)
                frame_counter += 1

                corners = (
                    orchestrator._board_geometry.corners
                    if orchestrator._board_geometry is not None
                    else None
                )
                vis = hud_composer.compose_video_overlay(
                    frame.image,
                    state,
                    corners=corners,
                )
                stream.update_frame(vis)

                curr_ply = orchestrator.state_manager.ply
                curr_fsm = orchestrator.fsm.current_state
                curr_uncertain = orchestrator._active_uncertainty
                curr_sid = orchestrator.session_id
                curr_illegal = getattr(orchestrator, "_illegal_attempts_count", 0)

                state_changed = (
                    curr_ply != last_ply
                    or curr_fsm != last_fsm
                    or curr_uncertain != last_uncertain
                    or curr_sid != last_session_id
                    or curr_illegal != last_illegal_count
                )

                if state_changed or (frame_counter % 15 == 0):
                    last_ply = curr_ply
                    last_fsm = curr_fsm
                    last_uncertain = curr_uncertain
                    last_session_id = curr_sid
                    last_illegal_count = curr_illegal
                    ws.broadcast_threadsafe(build_full_state())

                elapsed = time.perf_counter() - t0
                rem_delay = delay - elapsed
                if rem_delay > 0.001:
                    time.sleep(rem_delay)

        except Exception as exc:
            logger.error("Error in web video worker loop: %s", exc, exc_info=True)
        finally:
            if recorder.is_recording():
                recorder.stop()
            logger.info("Web background video worker finished.")

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        nonlocal worker_thread
        loop = asyncio.get_running_loop()
        ws.set_loop(loop)

        if start_worker and video_adapter is not None:
            worker_thread = threading.Thread(target=worker_loop, name="WebVideoWorker", daemon=True)
            worker_thread.start()

        yield

        stop_event.set()
        transcriber.cancel()
        if recorder.is_recording():
            recorder.stop()
        if worker_thread is not None and worker_thread.is_alive():
            worker_thread.join(timeout=2.0)
        with source_lock:
            cur_adapter = adapter_state["adapter"]
            if cur_adapter is not None:
                cur_adapter.release()

    app = FastAPI(
        title="Chess Game Recording System - Web Dashboard",
        version="1.1.0",
        lifespan=lifespan,
    )

    cors_origins = orchestrator.config.web_ui.cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # -------------------------------------------------------------------------
    # Core Page, Video Stream & WebSocket
    # -------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def get_index(request: Request):
        """Renders single page application dashboard."""
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "title": "Chess Game Recording System",
                "session_id": orchestrator.session_id,
                "white_player": orchestrator.white_player,
                "black_player": orchestrator.black_player,
            },
        )

    @app.get("/video_feed")
    async def get_video_feed():
        """Streams real-time MJPEG video frames with composite HUD overlays."""
        return StreamingResponse(
            stream.frame_generator(fps=orchestrator.config.web_ui.stream_fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.websocket("/ws/live")
    async def websocket_live_endpoint(websocket: WebSocket):
        """Maintains live bi-directional WebSocket session with browser client."""
        await ws.connect(websocket)
        await websocket.send_json(build_full_state())
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            ws.disconnect(websocket)
        except Exception:
            ws.disconnect(websocket)

    @app.get("/api/state")
    async def get_state():
        """Returns current JSON snapshot of game state and perception telemetry."""
        return build_full_state()

    @app.get("/api/pgn")
    async def get_pgn(download: bool = False):
        """Returns recorded game history in authoritative FIDE PGN format."""
        pgn_str = orchestrator.export_pgn()
        headers = {}
        if download:
            headers["Content-Disposition"] = 'attachment; filename="game.pgn"'
        return PlainTextResponse(pgn_str, headers=headers)

    # -------------------------------------------------------------------------
    # Operator Controls & Pause Calibration
    # -------------------------------------------------------------------------
    @app.post("/api/action/pause")
    async def action_pause():
        """Toggles video processing pause state."""
        is_paused = orchestrator.toggle_pause()
        await ws.broadcast(build_full_state())
        return {"success": True, "is_paused": is_paused}

    @app.post("/api/action/rollback")
    async def action_rollback():
        """Rolls back the most recently committed move (operator undo)."""
        popped = orchestrator.rollback_move()
        await ws.broadcast(build_full_state())
        return {
            "success": True,
            "rolled_back_move": popped.move_san if popped else None,
            "current_ply": orchestrator.state_manager.ply,
        }

    @app.post("/api/action/recalibrate")
    async def action_recalibrate():
        """Clears locked board geometry and triggers re-detection on resume (or manual calibrate)."""
        orchestrator.recalibrate()
        await ws.broadcast(build_full_state())
        return {"success": True}

    @app.post("/api/action/calibrate")
    async def action_manual_calibrate():
        """Calibrates board geometry & orientation on the current frame immediately (works while paused)."""
        ok = orchestrator.manual_calibrate()
        await ws.broadcast(build_full_state())
        if not ok:
            raise HTTPException(
                status_code=400,
                detail="Failed to detect chess board on the current frame. Ensure camera feed is live or adjust corners manually.",
            )
        return {
            "success": True,
            "board_detected": orchestrator._board_geometry is not None,
            "board_orientation": orchestrator._orientation.value,
        }

    @app.post("/api/action/reset_session")
    async def action_reset_session():
        """Resets current game moves and journal back to starting position."""
        manifest = orchestrator.reset_session()
        await ws.broadcast(build_full_state())
        return {
            "success": True,
            "session_id": manifest.session_id,
            "ply": manifest.total_plies,
        }

    @app.post("/api/action/set_corners")
    async def action_set_corners(req: SetCornersPayload):
        """Sets board corners manually by operator."""
        if len(req.corners) != 4:
            raise HTTPException(status_code=400, detail="Exactly 4 corners required")
        tuples_corners = [(float(p[0]), float(p[1])) for p in req.corners]
        success = orchestrator.set_manual_corners(tuples_corners)
        if not success:
            raise HTTPException(status_code=400, detail="Invalid quadrilateral corners")
        await ws.broadcast(build_full_state())
        return {"success": True}

    @app.post("/api/action/resolve")
    async def action_resolve(req: ResolvePayload):
        """Resolves active UNCERTAIN state by operator intervention."""
        resolved = orchestrator.resolve_uncertainty(
            move_uci=req.move_uci,
            auto_accept_top=req.auto_accept_top,
            candidate_index=req.candidate_index,
        )
        await ws.broadcast(build_full_state())
        if not resolved:
            raise HTTPException(status_code=400, detail="Failed to resolve uncertainty with provided parameters")
        return {"success": True}

    # -------------------------------------------------------------------------
    # Sessions Lifecycle (New / Resume / List)
    # -------------------------------------------------------------------------
    @app.get("/api/sessions")
    async def api_list_sessions():
        """Lists persisted recording sessions on disk."""
        return {
            "current_session_id": orchestrator.session_id,
            "sessions": orchestrator.list_sessions(),
        }

    @app.post("/api/sessions/new")
    async def api_new_session(req: Optional[NewSessionPayload] = None):
        """Saves current session and creates a fresh game session."""
        body = req or NewSessionPayload()
        manifest = orchestrator.new_session(
            white_player=body.white_player,
            black_player=body.black_player,
            session_id=body.session_id,
            pause=body.pause,
        )
        await ws.broadcast(build_full_state())
        return {
            "success": True,
            "session_id": manifest.session_id,
            "is_paused": orchestrator._is_paused,
        }

    @app.post("/api/sessions/{session_id}/resume")
    async def api_resume_session(session_id: str):
        """Resumes a previously persisted session from disk."""
        try:
            manifest = orchestrator.resume_session(session_id=session_id, pause=True)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await ws.broadcast(build_full_state())
        return {
            "success": True,
            "session_id": manifest.session_id,
            "ply": manifest.total_plies,
            "is_paused": orchestrator._is_paused,
        }

    # -------------------------------------------------------------------------
    # Video Device Selection, Recording & Offline Video Decoding
    # -------------------------------------------------------------------------
    @app.get("/api/video/devices")
    async def api_video_devices():
        """Enumerates available camera devices without disrupting the active stream."""
        with source_lock:
            cur_adapter = adapter_state["adapter"]
            cur_source = adapter_state["source"]
            opened = bool(cur_adapter is not None and cur_adapter.is_opened())
        devices = discover_video_devices(active_source=cur_source, active_opened=opened)
        return {"devices": devices, "active_source": cur_source}

    @app.post("/api/video/source")
    async def api_switch_video_source(req: SwitchVideoSourcePayload):
        """Hot-swaps the active video source (camera index/device/file)."""
        src = req.source.strip()
        if not src:
            raise HTTPException(status_code=400, detail="Video source cannot be empty")
        src_type = _infer_source_type(src, req.source_type)

        if recorder.is_recording():
            recorder.stop()

        new_cfg = CameraConfig(
            source_type=src_type,
            source_path=src,
            width=orchestrator.config.camera.width,
            height=orchestrator.config.camera.height,
            fps=orchestrator.config.camera.fps,
            rotate=orchestrator.config.camera.rotate,
        )

        # Release current adapter before re-opening device if same or overlapping V4L2 node
        with source_lock:
            old_adapter: Optional[BaseVideoAdapter] = adapter_state["adapter"]
            prev_source = adapter_state["source"]
            adapter_state["adapter"] = None
            if old_adapter is not None:
                try:
                    old_adapter.release()
                except Exception:
                    pass

            try:
                new_adapter = create_video_adapter(new_cfg)
            except Exception as exc:
                # Best-effort recovery of previous source so UI stream does not stay dead
                try:
                    fallback_cfg = CameraConfig(
                        source_type=_infer_source_type(prev_source),
                        source_path=prev_source,
                        width=orchestrator.config.camera.width,
                        height=orchestrator.config.camera.height,
                        fps=orchestrator.config.camera.fps,
                        rotate=orchestrator.config.camera.rotate,
                    )
                    adapter_state["adapter"] = create_video_adapter(fallback_cfg)
                    adapter_state["eof"] = False
                except Exception:
                    pass
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not open video source '{src}': {exc}",
                ) from exc

            adapter_state["adapter"] = new_adapter
            adapter_state["source"] = src
            adapter_state["eof"] = False

        orchestrator.config.camera.source_path = src
        orchestrator.config.camera.source_type = src_type
        orchestrator.pause()
        orchestrator.recalibrate()
        await ws.broadcast(build_full_state())
        return {"success": True, "source": src, "source_type": src_type}

    @app.post("/api/recording/start")
    async def api_recording_start():
        """Starts recording raw camera frames into an MP4/AVI file."""
        if recorder.is_recording():
            return {"success": True, "recording": recorder.status()}

        raw_frame = orchestrator._last_raw_frame
        if raw_frame is not None:
            w, h = raw_frame.width, raw_frame.height
        else:
            w, h = orchestrator.config.camera.width, orchestrator.config.camera.height

        with source_lock:
            cur_adapter = adapter_state["adapter"]
            fps = cur_adapter.get_fps() if cur_adapter is not None else orchestrator.config.camera.fps

        rec_out_dir = orchestrator.session_dir / "recordings"
        try:
            info = recorder.start(
                width=w,
                height=h,
                fps=fps,
                session_id=orchestrator.session_id,
                output_dir=rec_out_dir,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await ws.broadcast(build_full_state())
        return {"success": True, "recording": info}

    @app.post("/api/recording/stop")
    async def api_recording_stop():
        """Stops active raw video recording and returns download metadata."""
        result = recorder.stop()
        if result is None:
            raise HTTPException(status_code=400, detail="No active video recording to stop")
        result["download_url"] = f"/api/videos/download?path={urllib.parse.quote(result['path'] or '')}"
        await ws.broadcast(build_full_state())
        return {"success": True, "video": result}

    @app.get("/api/videos")
    async def api_list_videos():
        """Lists recorded and uploaded videos available for download or move decoding."""
        videos = list_available_videos(base_sessions_dir, recordings_dir)
        for v in videos:
            v["download_url"] = f"/api/videos/download?path={urllib.parse.quote(v['path'])}"
        return {"videos": videos}

    @app.get("/api/videos/download")
    async def api_download_video(path: str):
        """Downloads a recorded or uploaded video file safely."""
        target = Path(path).resolve()
        allowed_roots = [
            base_sessions_dir.resolve(),
            recordings_dir.resolve(),
            Path("data/benchmark_videos").resolve(),
        ]
        if not any(target.is_relative_to(root) for root in allowed_roots) or not target.is_file():
            raise HTTPException(status_code=404, detail="Video file not found")
        return FileResponse(path=str(target), filename=target.name)

    MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024

    @app.post("/api/video/upload")
    async def api_upload_video(request: Request, filename: str = "uploaded_game.mp4"):
        """Uploads a video file via raw binary request body (no python-multipart needed)."""
        import re as _re
        safe_name = _re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name).lstrip(".") or "uploaded_game.mp4"
        suffix = Path(safe_name).suffix.lower() or ".mp4"
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported video format: {suffix}")

        dest_path = uploads_dir / f"{int(time.time())}_{safe_name}"
        total_bytes = 0
        try:
            with dest_path.open("wb") as f:
                async for chunk in request.stream():
                    if chunk:
                        total_bytes += len(chunk)
                        if total_bytes > MAX_UPLOAD_BYTES:
                            raise HTTPException(status_code=413, detail="Uploaded file exceeds 4 GB limit")
                        f.write(chunk)
        except Exception:
            dest_path.unlink(missing_ok=True)
            raise

        if total_bytes == 0:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        return {
            "success": True,
            "filename": dest_path.name,
            "path": str(dest_path),
            "size_bytes": total_bytes,
        }

    @app.post("/api/video/transcribe")
    async def api_transcribe_video(req: TranscribeVideoPayload):
        """Starts background move decoding for the given video file."""
        video_path = Path(req.video_path).resolve()
        allowed_roots = [
            base_sessions_dir.resolve(),
            recordings_dir.resolve(),
            Path("data/benchmark_videos").resolve(),
        ]
        if not any(video_path.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(status_code=400, detail="Video path outside allowed directories")
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail=f"Video file not found: {req.video_path}")

        def on_update(status_snap: Dict[str, Any]) -> None:
            ws.broadcast_threadsafe({"type": "transcribeStatus", **status_snap})

        def on_finished(status_snap: Dict[str, Any]) -> None:
            try:
                orchestrator.resume_session(status_snap["session_id"], pause=True)
                broadcast_state_sync()
            except Exception as exc:
                logger.error("Failed to load transcribed session: %s", exc)

        try:
            job = transcriber.start(
                video_path=video_path,
                white_player=req.white_player,
                black_player=req.black_player,
                target_fps=req.target_fps,
                on_update=on_update,
                on_finished=on_finished,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {"success": True, **job}

    @app.get("/api/video/transcribe/status")
    async def api_transcribe_status():
        """Returns current background video decoding job status."""
        return transcriber.get_status()

    return app
