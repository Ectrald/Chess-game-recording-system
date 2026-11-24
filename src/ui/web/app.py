"""FastAPI Application Factory for Chess Game Recording System Web UI.

Provides RESTful endpoints, MJPEG video streaming, WebSocket event delivery,
and Jinja2 template rendering for browser-based operator control.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import chess
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.core.datatypes import FSMState, PipelineState
from src.orchestrator import SystemOrchestrator
from src.ui.visualizer import VisualHUDComposer
from src.ui.web.streamer import MJPEGStreamer
from src.ui.web.ws_manager import WebSocketManager
from src.vision.video_adapter import BaseVideoAdapter

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


SetCornersPayload.model_rebuild()


def get_state_payload(orchestrator: SystemOrchestrator) -> Dict[str, Any]:
    """Serializes authoritative orchestrator and board state into a JSON dictionary."""
    board = orchestrator.state_manager.current_board
    turn = "white" if board.turn == chess.WHITE else "black"
    is_check = board.is_check()
    is_game_over = board.is_game_over()
    result = board.result() if is_game_over else "*"
    fsm_state = orchestrator.fsm.current_state.value
    history = orchestrator.state_manager.history

    move_history: List[Dict[str, Any]] = [
        {"ply": m.ply, "san": m.move_san, "uci": m.move_uci}
        for m in history
    ]
    last_move = (
        {"ply": history[-1].ply, "san": history[-1].move_san, "uci": history[-1].move_uci}
        if history else None
    )

    eval_dict: Optional[Dict[str, Any]] = None
    if orchestrator._latest_eval is not None:
        eval_dict = {
            "score_cp": orchestrator._latest_eval.score_cp,
            "mate_in": orchestrator._latest_eval.mate_in,
            "depth": orchestrator._latest_eval.depth,
            "best_move": orchestrator._latest_eval.best_move,
            "quality": orchestrator._latest_eval.quality.value if orchestrator._latest_eval.quality else None,
        }

    candidates: List[Dict[str, Any]] = []
    if orchestrator._active_uncertainty and orchestrator._last_decision is not None:
        hypos = orchestrator._last_decision.top_hypotheses
        if not hypos and orchestrator._last_decision.accepted_hypothesis:
            hypos = [orchestrator._last_decision.accepted_hypothesis]
        candidates = [
            {
                "rank": idx + 1,
                "uci": h.move_uci,
                "san": h.move_san,
                "score": round(float(h.confidence_score), 3),
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

    return {
        "type": "gameState",
        "fen": orchestrator.state_manager.current_fen,
        "turn": turn,
        "ply": orchestrator.state_manager.ply,
        "is_check": is_check,
        "is_game_over": is_game_over,
        "result": result,
        "fsm_state": fsm_state,
        "is_paused": orchestrator._is_paused,
        "active_uncertainty": orchestrator._active_uncertainty,
        "uncertain_candidates": candidates,
        "fps": round(float(orchestrator._current_fps), 1),
        "board_detected": orchestrator._board_geometry is not None,
        "board_corners": board_corners,
        "last_move": last_move,
        "move_history": move_history,
        "engine_eval": eval_dict,
        "white_player": orchestrator.white_player,
        "black_player": orchestrator.black_player,
    }


def create_web_app(
    orchestrator: SystemOrchestrator,
    video_adapter: Optional[BaseVideoAdapter] = None,
    ws_manager: Optional[WebSocketManager] = None,
    streamer: Optional[MJPEGStreamer] = None,
    composer: Optional[VisualHUDComposer] = None,
    start_worker: bool = True,
) -> FastAPI:
    """Builds and configures the FastAPI application instance.

    Args:
        orchestrator: Initialized SystemOrchestrator facade.
        video_adapter: Optional video capture adapter for live worker loop.
        ws_manager: Optional WebSocketManager instance.
        streamer: Optional MJPEGStreamer instance.
        composer: Optional VisualHUDComposer instance.
        start_worker: Whether to automatically launch video worker thread on startup.

    Returns:
        Configured FastAPI application instance.
    """
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

    stop_event = threading.Event()
    worker_thread: Optional[threading.Thread] = None

    def worker_loop() -> None:
        """Background thread acquiring video frames and driving orchestrator pipeline."""
        if video_adapter is None:
            return

        orchestrator.start_session()
        nominal_fps = video_adapter.get_fps()
        delay = 1.0 / max(1.0, nominal_fps) if nominal_fps > 0 else 0.033

        logger.info("Web background video worker started (nominal_fps=%.1f)", nominal_fps)
        last_ply = orchestrator.state_manager.ply
        last_fsm = orchestrator.fsm.current_state
        last_uncertain = orchestrator._active_uncertainty
        frame_counter = 0

        try:
            while not stop_event.is_set() and video_adapter.is_opened():
                t0 = time.perf_counter()
                frame = video_adapter.read()
                if frame is None:
                    logger.info("Video stream reached end of file or closed.")
                    break

                state = orchestrator.process_frame(frame)
                frame_counter += 1

                # Render video frame overlay with board corners & telemetry
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

                # Broadcast on significant state events or periodic telemetry
                curr_ply = orchestrator.state_manager.ply
                curr_fsm = orchestrator.fsm.current_state
                curr_uncertain = orchestrator._active_uncertainty

                state_changed = (
                    curr_ply != last_ply
                    or curr_fsm != last_fsm
                    or curr_uncertain != last_uncertain
                )

                if state_changed or (frame_counter % 15 == 0):
                    last_ply = curr_ply
                    last_fsm = curr_fsm
                    last_uncertain = curr_uncertain
                    payload = get_state_payload(orchestrator)
                    ws.broadcast_threadsafe(payload)

                elapsed = time.perf_counter() - t0
                rem_delay = delay - elapsed
                if rem_delay > 0.001:
                    time.sleep(rem_delay)

        except Exception as exc:
            logger.error("Error in web video worker loop: %s", exc, exc_info=True)
        finally:
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
        if worker_thread is not None and worker_thread.is_alive():
            worker_thread.join(timeout=2.0)
        if video_adapter is not None:
            video_adapter.release()

    app = FastAPI(
        title="Chess Game Recording System - Web Dashboard",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Enable CORS
    cors_origins = orchestrator.config.web_ui.cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files and Templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # -------------------------------------------------------------------------
    # Route Handlers
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
        # Transmit immediate full state snapshot upon connection
        initial_state = get_state_payload(orchestrator)
        await websocket.send_json(initial_state)

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
        return get_state_payload(orchestrator)

    @app.get("/api/pgn")
    async def get_pgn(download: bool = False):
        """Returns recorded game history in authoritative FIDE PGN format."""
        pgn_str = orchestrator.export_pgn()
        headers = {}
        if download:
            headers["Content-Disposition"] = 'attachment; filename="game.pgn"'
        return PlainTextResponse(pgn_str, headers=headers)

    @app.post("/api/action/pause")
    async def action_pause():
        """Toggles video processing pause state."""
        is_paused = orchestrator.toggle_pause()
        payload = get_state_payload(orchestrator)
        await ws.broadcast(payload)
        return {"success": True, "is_paused": is_paused}

    @app.post("/api/action/rollback")
    async def action_rollback():
        """Rolls back the most recently committed move (operator undo)."""
        popped = orchestrator.rollback_move()
        payload = get_state_payload(orchestrator)
        await ws.broadcast(payload)
        return {
            "success": True,
            "rolled_back_move": popped.move_san if popped else None,
            "current_ply": orchestrator.state_manager.ply,
        }

    @app.post("/api/action/recalibrate")
    async def action_recalibrate():
        """Triggers board re-detection and orientation calibration."""
        orchestrator.recalibrate()
        payload = get_state_payload(orchestrator)
        await ws.broadcast(payload)
        return {"success": True}

    @app.post("/api/action/set_corners")
    async def action_set_corners(req: SetCornersPayload):
        """Sets board corners manually by operator."""
        if len(req.corners) != 4:
            raise HTTPException(status_code=400, detail="Exactly 4 corners required")
        tuples_corners = [(float(p[0]), float(p[1])) for p in req.corners]
        success = orchestrator.set_manual_corners(tuples_corners)
        if not success:
            raise HTTPException(status_code=400, detail="Invalid quadrilateral corners")
        payload = get_state_payload(orchestrator)
        await ws.broadcast(payload)
        return {"success": True}

    @app.post("/api/action/resolve")
    async def action_resolve(req: ResolvePayload):
        """Resolves active UNCERTAIN state by operator intervention."""
        resolved = orchestrator.resolve_uncertainty(
            move_uci=req.move_uci,
            auto_accept_top=req.auto_accept_top,
            candidate_index=req.candidate_index,
        )
        payload = get_state_payload(orchestrator)
        await ws.broadcast(payload)
        if not resolved:
            raise HTTPException(status_code=400, detail="Failed to resolve uncertainty with provided parameters")
        return {"success": True}

    return app
