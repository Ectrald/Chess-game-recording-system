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
