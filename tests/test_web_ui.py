"""Automated test suite for Web UI, REST endpoints, MJPEG streamer, and WebSocket manager.

Validates:
- Application factory initialization.
- HTML single-page application rendering.
- State snapshot serialization (/api/state).
- PGN export endpoint (/api/pgn).
- Operator action routes: pause, rollback, recalibrate, resolve.
- Video feed MJPEG streaming headers and boundary chunks.
- Bi-directional WebSocket communication.
- CLI argument parsing for `web` subcommand.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading
import time

import chess
from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.core.config import AppConfig, load_config
from src.core.datatypes import MoveHypothesis
from src.orchestrator import SystemOrchestrator
from src.ui.cli import build_parser
from src.ui.web.app import create_web_app, get_state_payload
from src.ui.web.streamer import MJPEGStreamer
from src.ui.web.ws_manager import WebSocketManager


@pytest.fixture
def orchestrator(tmp_path: Path) -> SystemOrchestrator:
    """Fixture providing isolated SystemOrchestrator with temp base directory."""
    config = load_config()
    config.persistence.base_dir = str(tmp_path / "sessions")
    config.analysis.stockfish_enabled = False  # Disable engine worker in unit tests
    orch = SystemOrchestrator(config=config, session_id="test_web_session")
    orch.start_session()
    return orch


@pytest.fixture
def client(orchestrator: SystemOrchestrator) -> TestClient:
    """Fixture providing FastAPI test client with non-running worker loop."""
    app = create_web_app(orchestrator=orchestrator, start_worker=False)
    return TestClient(app)


def test_index_page(client: TestClient) -> None:
    """Verifies that GET / renders HTML single page operator dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "Chess Game Recording System" in html
    assert "chessboard" in html
    assert "videoStream" in html
    assert "uncertainModal" in html


def test_api_state_endpoint(client: TestClient, orchestrator: SystemOrchestrator) -> None:
    """Verifies GET /api/state returns valid initial board state and telemetry."""
    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()

    assert data["type"] == "gameState"
    assert data["fen"] == chess.STARTING_FEN
    assert data["turn"] == "white"
    assert data["ply"] == 0
    assert data["is_check"] is False
    assert data["is_game_over"] is False
    assert data["is_paused"] is False
    assert data["active_uncertainty"] is False
    assert isinstance(data["move_history"], list)
    assert len(data["move_history"]) == 0


def test_api_pgn_endpoint(client: TestClient, orchestrator: SystemOrchestrator) -> None:
    """Verifies GET /api/pgn returns authoritative PGN format and supports file download."""
    # Plain text preview
    res1 = client.get("/api/pgn")
    assert res1.status_code == 200
    assert "text/plain" in res1.headers["content-type"]
    assert '[Event "Chess Recording System Session"]' in res1.text

    # Download attachment
    res2 = client.get("/api/pgn?download=true")
    assert res2.status_code == 200
    assert 'attachment; filename="game.pgn"' in res2.headers.get("content-disposition", "")


def test_api_action_pause(client: TestClient, orchestrator: SystemOrchestrator) -> None:
    """Verifies POST /api/action/pause toggles pause state."""
    assert orchestrator._is_paused is False

    # Pause
    res1 = client.post("/api/action/pause")
    assert res1.status_code == 200
    assert res1.json()["is_paused"] is True
    assert orchestrator._is_paused is True

    # Resume
    res2 = client.post("/api/action/pause")
    assert res2.status_code == 200
    assert res2.json()["is_paused"] is False
    assert orchestrator._is_paused is False


def test_api_action_rollback(client: TestClient, orchestrator: SystemOrchestrator) -> None:
    """Verifies POST /api/action/rollback undoes last move."""
    # Commit move directly
    orchestrator.state_manager.manual_commit("e2e4")
    assert orchestrator.state_manager.ply == 1

    res = client.post("/api/action/rollback")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["rolled_back_move"] == "e4"
    assert data["current_ply"] == 0
    assert orchestrator.state_manager.ply == 0


def test_api_action_recalibrate(client: TestClient, orchestrator: SystemOrchestrator) -> None:
    """Verifies POST /api/action/recalibrate resets board geometry."""
    orchestrator._orientation_calibrated = True
    res = client.post("/api/action/recalibrate")
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert orchestrator._orientation_calibrated is False
    assert orchestrator._board_geometry is None


def test_api_action_resolve(client: TestClient, orchestrator: SystemOrchestrator) -> None:
    """Verifies POST /api/action/resolve arbitrates UNCERTAIN state."""
    # Resolve with valid UCI move
    res = client.post("/api/action/resolve", json={"move_uci": "e2e4"})
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert orchestrator.state_manager.ply == 1
    assert "e4" in orchestrator.state_manager.history[-1].move_san

    # Resolve with invalid move should fail with 400
    res_fail = client.post("/api/action/resolve", json={"move_uci": "invalid_move"})
    assert res_fail.status_code == 400


def test_mjpeg_streamer() -> None:
    """Tests MJPEGStreamer frame updating, placeholders, and multipart chunks."""
    streamer = MJPEGStreamer(target_fps=15, quality=75)

    # Placeholder test
    ph = streamer.get_latest_jpeg()
    assert isinstance(ph, bytes)
    assert len(ph) > 0
    assert ph.startswith(b"\xff\xd8")  # JPEG SOI header

    # Frame update test
    sample_img = np.zeros((100, 100, 3), dtype=np.uint8)
    streamer.update_frame(sample_img)
    frame = streamer.get_latest_jpeg()
    assert isinstance(frame, bytes)
    assert len(frame) > 0
    assert frame != ph

    # Frame generator test
    async def run_gen():
        chunks = []
        gen = streamer.frame_generator(fps=60)
        for _ in range(2):
            chunk = await anext(gen)
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run_gen())
    assert len(chunks) == 2
    assert chunks[0].startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")


def test_websocket_live(client: TestClient) -> None:
    """Tests WebSocket endpoint connection, initial state delivery, and ping/pong."""
    with client.websocket_connect("/ws/live") as ws:
        # Initial message must be gameState
        msg = ws.receive_json()
        assert msg["type"] == "gameState"
        assert msg["fen"] == chess.STARTING_FEN

        # Send ping
        ws.send_text("ping")
        pong = ws.receive_text()
        assert pong == "pong"


def test_cli_web_parser() -> None:
    """Verifies that the `web` subcommand is registered with correct default options."""
    parser = build_parser()
    args = parser.parse_args(["web", "--source", "camera.mp4", "--port", "8080", "--host", "127.0.0.1"])
    assert args.subcommand == "web"
    assert args.source == "camera.mp4"
    assert args.port == 8080
    assert args.host == "127.0.0.1"


def test_action_set_corners_endpoint(client: TestClient, orchestrator: SystemOrchestrator) -> None:
    """Verifies POST /api/action/set_corners manually locks valid board quadrilateral."""
    corners = [[100.0, 100.0], [500.0, 100.0], [520.0, 500.0], [80.0, 500.0]]
    response = client.post("/api/action/set_corners", json={"corners": corners})
    assert response.status_code == 200
    assert response.json() == {"success": True}

    assert orchestrator._board_geometry is not None
    assert orchestrator._board_geometry.is_valid is True
    assert len(orchestrator._board_geometry.corners) == 4

    # Invalid: fewer than 4 corners
    bad_resp = client.post("/api/action/set_corners", json={"corners": [[100.0, 100.0], [200.0, 200.0]]})
    assert bad_resp.status_code == 400

    # Invalid: collinear or non-convex
    bad_poly = client.post(
        "/api/action/set_corners",
        json={"corners": [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]},
    )
    assert bad_poly.status_code == 400


def test_state_payload_with_uncertain_candidates(client: TestClient, orchestrator: SystemOrchestrator) -> None:
    """Verifies get_state_payload serializes MoveHypothesis candidates using total_score without AttributeError."""
    from src.core.datatypes import DecisionResult, DecisionStatus, MoveHypothesis

    hypo = MoveHypothesis(
        move_uci="e2e4",
        move_san="e4",
        changed_squares=["e2", "e4"],
        score_visual=0.72,
        total_score=0.75,
    )
    orchestrator._active_uncertainty = True
    orchestrator._last_decision = DecisionResult(
        status=DecisionStatus.UNCERTAIN,
        top_hypotheses=[hypo],
        margin=0.05,
        rejection_reason="Margin below threshold",
    )

    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert data["active_uncertainty"] is True
    assert len(data["uncertain_candidates"]) == 1
    assert data["uncertain_candidates"][0]["uci"] == "e2e4"
    assert data["uncertain_candidates"][0]["score"] == 0.75


def test_state_payload_with_analysis_result(client: TestClient, orchestrator: SystemOrchestrator) -> None:
    """Verifies get_state_payload serializes AnalysisResult using eval_after fields without AttributeError."""
    from src.core.datatypes import AnalysisResult, EngineEvaluation, EngineTaskStatus, MoveQuality

    orchestrator._latest_eval = AnalysisResult(
        event_id="evt-1",
        ply=1,
        move_uci="e2e4",
        status=EngineTaskStatus.COMPLETED,
        eval_after=EngineEvaluation(eval_cp=35, depth=12, best_move_uci="e7e5"),
        quality=MoveQuality.BEST,
    )

    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert data["engine_eval"]["score_cp"] == 35
    assert data["engine_eval"]["depth"] == 12
    assert data["engine_eval"]["best_move"] == "e7e5"
    assert data["engine_eval"]["quality"] == "best"


def test_web_board_square_color_parity(client: TestClient) -> None:
    """Verifies FIDE square coloring in Web UI JS: a1 dark (f=0,r=1), h1 light (f=7,r=1)."""
    response = client.get("/static/js/app.js")
    assert response.status_code == 200
    assert "const isLight = (f + r) % 2 === 0;" in response.text


def test_state_payload_includes_uncertainty_reason_and_squares(
    client: TestClient, orchestrator: SystemOrchestrator
) -> None:
    """Verifies /api/state exposes arbiter rejection reason and affected squares for illegal moves."""
    from src.core.datatypes import DecisionResult, DecisionStatus

    orchestrator._active_uncertainty = True
    orchestrator._last_decision = DecisionResult(
        status=DecisionStatus.UNCERTAIN,
        rejection_reason="Illegal move on squares c1, f4",
        uncertain_squares=["c1", "f4"],
        top_hypotheses=[],
    )
    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert data["active_uncertainty"] is True
    assert data["uncertainty_reason"] == "Illegal move on squares c1, f4"
    assert data["uncertain_squares"] == ["c1", "f4"]
    assert data["uncertain_candidates"] == []


def test_start_paused_and_manual_calibrate_while_paused(
    orchestrator: SystemOrchestrator,
) -> None:
    """Starts paused with no auto-calibration, then calibrates manually on the latest frame."""
    import cv2
    from src.core.datatypes import Frame
    from src.vision.synthetic_board import SyntheticBoardGenerator

    app = create_web_app(orchestrator=orchestrator, start_worker=False, start_paused=True)
    client = TestClient(app)

    state = client.get("/api/state").json()
    assert state["is_paused"] is True
    assert state["fsm_state"] == "paused"

    gen = SyntheticBoardGenerator(canvas_width=640, canvas_height=480, fps=10.0)
    raw_frame: Frame = gen.render_frame(0)

    # Processing frames while paused must NOT auto-calibrate board geometry
    out_state = orchestrator.process_frame(raw_frame)
    assert out_state.board_detected is False
    assert orchestrator._board_geometry is None

    # Manual calibration during pause detects geometry on latest frame while keeping pause active
    cal_resp = client.post("/api/action/calibrate")
    assert cal_resp.status_code == 200
    cal_data = cal_resp.json()
    assert cal_data["success"] is True
    assert cal_data["board_detected"] is True
    assert orchestrator._is_paused is True


def test_sessions_new_resume_and_reset(
    client: TestClient,
    orchestrator: SystemOrchestrator,
) -> None:
    """Verifies new session creation, resuming a past session, and resetting current session."""
    first_id = orchestrator.session_id
    orchestrator.state_manager.manual_commit("e2e4")
    orchestrator.state_manager.manual_commit("e7e5")
    assert orchestrator.state_manager.ply == 2

    # Move history must include fen_after for move-by-move board scrubbing
    s0 = client.get("/api/state").json()
    assert s0["initial_fen"] == chess.STARTING_FEN
    assert len(s0["move_history"]) == 2
    assert "fen_after" in s0["move_history"][1]

    # Create a new session -> saves previous, resets board to ply 0, starts paused
    new_resp = client.post(
        "/api/sessions/new",
        json={"white_player": "Alice", "black_player": "Bob", "pause": True},
    )
    assert new_resp.status_code == 200
    second_id = new_resp.json()["session_id"]
    assert second_id != first_id
    assert orchestrator.state_manager.ply == 0
    assert orchestrator._is_paused is True

    # Commit one move in second session then reset back to starting position
    orchestrator.state_manager.manual_commit("d2d4")
    assert orchestrator.state_manager.ply == 1
    reset_resp = client.post("/api/action/reset_session")
    assert reset_resp.status_code == 200
    assert orchestrator.state_manager.ply == 0
    assert orchestrator.state_manager.current_fen == chess.STARTING_FEN

    # List sessions and resume the first session cleanly
    list_resp = client.get("/api/sessions")
    assert list_resp.status_code == 200
    sessions = {s["session_id"]: s for s in list_resp.json()["sessions"]}
    assert first_id in sessions
    assert second_id in sessions
    assert sessions[first_id]["total_plies"] == 2

    res_resp = client.post(f"/api/sessions/{first_id}/resume")
    assert res_resp.status_code == 200
    assert res_resp.json()["ply"] == 2
    assert orchestrator.session_id == first_id
    assert orchestrator.state_manager.ply == 2


def test_video_recording_upload_and_transcribe_pipeline(
    client: TestClient,
    orchestrator: SystemOrchestrator,
    tmp_path: Path,
) -> None:
    """Verifies raw video upload, download, source switching, recording stop/start, and background decoding."""
    import cv2
    from src.vision.synthetic_board import SyntheticBoardGenerator

    # 1. Start and stop raw video recording using an active frame dimension
    orchestrator._last_raw_frame = SyntheticBoardGenerator(canvas_width=320, canvas_height=240, fps=10.0).render_frame(0)
    start_rec = client.post("/api/recording/start")
    assert start_rec.status_code == 200
    assert start_rec.json()["recording"]["is_recording"] is True

    stop_rec = client.post("/api/recording/stop")
    assert stop_rec.status_code == 200
    rec_meta = stop_rec.json()["video"]
    assert Path(rec_meta["path"]).is_file()
    dl_resp = client.get(rec_meta["download_url"])
    assert dl_resp.status_code == 200

    # 2. Generate a short synthetic game MP4, upload without multipart, and transcribe
    pgn_file = Path("data/ground_truth/test_short_game.pgn")
    gen = SyntheticBoardGenerator(canvas_width=640, canvas_height=480, fps=6.0)
    if pgn_file.is_file():
        gen.set_game_from_pgn(str(pgn_file))
    vid_src = tmp_path / "sample_game.mp4"
    vw = cv2.VideoWriter(str(vid_src), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (640, 480))
    for i in range(min(30, gen.total_frames)):
        vw.write(gen.render_frame(i).image)
    vw.release()

    up_resp = client.post(
        "/api/video/upload?filename=sample_game.mp4",
        content=vid_src.read_bytes(),
    )
    assert up_resp.status_code == 200
    uploaded_path = up_resp.json()["path"]
    assert Path(uploaded_path).is_file()

    # Switch live video source to uploaded video
    sw_resp = client.post("/api/video/source", json={"source": uploaded_path})
    assert sw_resp.status_code == 200
    assert sw_resp.json()["source_type"] == "file"

    # Run background transcription and poll until completion
    tr_resp = client.post(
        "/api/video/transcribe",
        json={"video_path": uploaded_path, "white_player": "VidW", "black_player": "VidB"},
    )
    assert tr_resp.status_code == 200
    deadline = time.monotonic() + 30.0
    job_status = tr_resp.json()
    while time.monotonic() < deadline and job_status["status"] == "running":
        time.sleep(0.1)
        job_status = client.get("/api/video/transcribe/status").json()

    assert job_status["status"] == "completed"
    assert job_status["progress_pct"] == 100.0
    assert orchestrator.session_id == job_status["session_id"]

    # Devices endpoint responds cleanly even in headless test container
    dev_resp = client.get("/api/video/devices")
    assert dev_resp.status_code == 200
    assert "devices" in dev_resp.json()

    # New UI controls present in index.html & app.js
    html = client.get("/").text
    for elem_id in ("videoDeviceSelect", "btnResetSession", "btnNewSession", "btnRecordVideo", "btnCalibrateNow", "btnReplayPrev", "btnToggleUncertain"):
        assert elem_id in html
    assert "uncertain-panel" in html
    js = client.get("/static/js/app.js").text
    assert "setViewPly" in js
    assert "applyUncertainMinimizeState" in js
    assert "_notifiedCompletedJobIds" in js
    assert "loadInitialState" in js
    assert "lastActionError" in js
    assert "offsetX" in js


def test_download_blocks_non_video_and_resolve_error(
    client: TestClient, orchestrator: SystemOrchestrator
) -> None:
    """Download endpoint rejects non-video files in session dir and resolve 400 preserves state."""
    orchestrator.save_session_snapshot()
    manifest_file = orchestrator.session_dir / "manifest.json"
    assert manifest_file.is_file()
    dl_non_video = client.get(f"/api/videos/download?path={manifest_file}")
    assert dl_non_video.status_code == 404

    res_bad = client.post("/api/action/resolve", json={"move_uci": "e1e8"})
    assert res_bad.status_code == 400
    assert orchestrator.state_manager.ply == 0


