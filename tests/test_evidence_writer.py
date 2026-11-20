"""Tests for EvidenceWriter and selective retention policies."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.core.datatypes import (
    CHESS_SQUARES,
    BoardObservation,
    CellObservation,
    FSMState,
    MoveEvent,
)
from src.persistence.evidence_writer import EvidenceWriter


def make_dummy_obs(frame_id: int = 10, confidence: float = 0.95) -> BoardObservation:
    cells = {
        sq: CellObservation(
            square=sq,
            probabilities={"empty": 1.0},
            predicted_piece="empty",
            is_occupied=False,
            confidence=1.0,
        )
        for sq in CHESS_SQUARES
    }
    return BoardObservation(
        frame_id=frame_id,
        timestamp=1.0,
        cells=cells,
        fsm_state=FSMState.STABLE,
        motion_detected=False,
        motion_level=0.0,
        hand_present=False,
        geometry_confidence=confidence,
    )


def test_evidence_writer_save_image_and_metadata(tmp_path: Path):
    writer = EvidenceWriter(evidence_dir=tmp_path / "evidence", policy="selective")

    event = MoveEvent(
        ply=1,
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_uci="e2e4",
        move_san="e4",
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        confidence=0.85,
        timestamp=1.5,
        evidence_frame_id=120,
    )
    obs = make_dummy_obs(frame_id=120)
    fake_board_img = np.zeros((100, 100, 3), dtype=np.uint8)

    rel_path = writer.save_evidence(event, obs, board_image=fake_board_img)
    assert rel_path is not None
    assert "ply_0001_e2e4_frame_000120" in rel_path

    # Check that image and meta files exist
    img_files = list((tmp_path / "evidence").glob("*.jpg"))
    meta_files = list((tmp_path / "evidence").glob("*.json"))

    assert len(img_files) == 1
    assert len(meta_files) == 1

    meta_content = json.loads(meta_files[0].read_text())
    assert meta_content["ply"] == 1
    assert meta_content["move_uci"] == "e2e4"
    assert meta_content["evidence_frame_id"] == 120


def test_evidence_writer_policy_none(tmp_path: Path):
    writer = EvidenceWriter(evidence_dir=tmp_path / "evidence", policy="none")
    event = MoveEvent(
        ply=1,
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_uci="e2e4",
        move_san="e4",
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        confidence=0.85,
        timestamp=1.5,
        evidence_frame_id=120,
    )
    obs = make_dummy_obs()
    res = writer.save_evidence(event, obs, np.zeros((10, 10, 3), dtype=np.uint8))
    assert res is None
    assert len(list((tmp_path / "evidence").glob("*"))) == 0
