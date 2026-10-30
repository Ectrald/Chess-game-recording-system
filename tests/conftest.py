"""Shared pytest fixtures for the Chess Game Recording System."""

from pathlib import Path
import numpy as np
import pytest

from src.core.config import AppConfig, load_config
from src.core.datatypes import (
    BoardGeometry,
    BoardObservation,
    CellObservation,
    Frame,
    CHESS_SQUARES,
)


@pytest.fixture
def repo_root() -> Path:
    """Returns absolute path to the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """Returns a dummy 640x640 BGR image array."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame(sample_bgr_image) -> Frame:
    """Returns a valid Frame instance."""
    return Frame(
        frame_id=42,
        timestamp=1.4,
        image=sample_bgr_image,
        width=640,
        height=640,
    )


@pytest.fixture
def sample_corners() -> list[tuple[float, float]]:
    """Returns standard 4 corner coordinates in TL, TR, BR, BL order."""
    return [
        (480.0, 200.0),
        (1440.0, 200.0),
        (1600.0, 920.0),
        (320.0, 920.0),
    ]


@pytest.fixture
def sample_geometry(sample_corners) -> BoardGeometry:
    """Returns a valid BoardGeometry instance."""
    return BoardGeometry(
        corners=sample_corners,
        homography_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        confidence=0.95,
        is_valid=True,
        orientation_calibrated=True,
    )


@pytest.fixture
def sample_board_observation() -> BoardObservation:
    """Returns a valid BoardObservation instance containing all 64 cells."""
    cells = {}
    for sq in CHESS_SQUARES:
        cells[sq] = CellObservation(
            square=sq,
            probabilities={"empty": 0.95, "wP": 0.05},
            predicted_piece="empty",
            confidence=0.95,
            is_occupied=False,
            is_occluded=False,
        )
    return BoardObservation(
        timestamp=10.0,
        frame_id=300,
        cells=cells,
        motion_detected=False,
        motion_level=0.01,
        hand_present=False,
        geometry_confidence=0.98,
    )


@pytest.fixture
def default_config(repo_root) -> AppConfig:
    """Loads and returns default system configuration."""
    cfg_path = repo_root / "configs" / "default_config.yaml"
    return load_config(cfg_path)


@pytest.fixture
def short_game_pgn_path(repo_root) -> Path:
    """Returns path to test_short_game.pgn."""
    return repo_root / "data" / "ground_truth" / "test_short_game.pgn"


@pytest.fixture
def special_moves_pgn_path(repo_root) -> Path:
    """Returns path to test_special_moves.pgn."""
    return repo_root / "data" / "ground_truth" / "test_special_moves.pgn"
