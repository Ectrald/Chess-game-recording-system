"""Tests for DecisionGate acceptance criteria and uncertainty filtering."""

import pytest

from src.core.config import DecisionConfig
from src.core.datatypes import (
    CHESS_SQUARES,
    BoardObservation,
    CellObservation,
    DecisionResult,
    DecisionStatus,
    FSMState,
    MoveHypothesis,
)
from src.reasoning.decision_gate import DecisionGate


def make_test_observation(
    geometry_confidence: float = 0.95,
    occluded_squares: dict[str, float] = None,
) -> BoardObservation:
    occluded_squares = occluded_squares or {}
    cells = {}
    for sq in CHESS_SQUARES:
        occ_ratio = occluded_squares.get(sq, 0.0)
        cells[sq] = CellObservation(
            square=sq,
            probabilities={"empty": 1.0},
            predicted_piece="empty",
            is_occupied=False,
            confidence=1.0,
            is_occluded=(occ_ratio > 0.0),
            occlusion_ratio=occ_ratio,
            motion_level=0.0,
        )

    return BoardObservation(
        frame_id=1,
        timestamp=0.1,
        cells=cells,
        fsm_state=FSMState.VERIFYING,
        motion_detected=False,
        motion_level=0.0,
        hand_present=bool(occluded_squares),
        geometry_confidence=geometry_confidence,
    )


def make_hypo(move_uci: str, total_score: float, changed: list[str]) -> MoveHypothesis:
    return MoveHypothesis(
        move_uci=move_uci,
        move_san=move_uci,
        changed_squares=changed,
        score_visual=total_score,
        score_temporal=1.0,
        score_geometry=1.0,
        penalty_occlusion=0.0,
        total_score=total_score,
        is_legal=True,
    )


def test_decision_gate_accepted():
    gate = DecisionGate()
    obs = make_test_observation()

    h1 = make_hypo("e2e4", 0.85, ["e2", "e4"])
    h2 = make_hypo("e2e3", 0.60, ["e2", "e3"])

    res = gate.evaluate([h1, h2], obs)
    assert res.status == DecisionStatus.ACCEPTED
    assert res.accepted_hypothesis is not None
    assert res.accepted_hypothesis.move_uci == "e2e4"
    assert res.margin == 0.25
    assert res.rejection_reason is None


def test_decision_gate_uncertain_low_score():
    gate = DecisionGate()
    obs = make_test_observation()

    h1 = make_hypo("e2e4", 0.45, ["e2", "e4"])
    h2 = make_hypo("e2e3", 0.20, ["e2", "e3"])

    res = gate.evaluate([h1, h2], obs)
    assert res.status == DecisionStatus.UNCERTAIN
    assert res.accepted_hypothesis is None
    assert "below minimum acceptance threshold" in res.rejection_reason
    assert res.uncertain_squares == ["e2", "e4"]


def test_decision_gate_uncertain_low_margin():
    gate = DecisionGate()
    obs = make_test_observation()

    # Margin = 0.80 - 0.75 = 0.05 < 0.15
    h1 = make_hypo("e2e4", 0.80, ["e2", "e4"])
    h2 = make_hypo("e2e3", 0.75, ["e2", "e3"])

    res = gate.evaluate([h1, h2], obs)
    assert res.status == DecisionStatus.UNCERTAIN
    assert res.accepted_hypothesis is None
    assert "Confidence margin" in res.rejection_reason
    assert set(res.uncertain_squares) == {"e2", "e3", "e4"}


def test_decision_gate_uncertain_critical_occlusion():
    gate = DecisionGate()
    # e4 has 0.70 occlusion ratio
    obs = make_test_observation(occluded_squares={"e4": 0.70})

    h1 = make_hypo("e2e4", 0.85, ["e2", "e4"])
    h2 = make_hypo("d2d4", 0.50, ["d2", "d4"])

    res = gate.evaluate([h1, h2], obs)
    assert res.status == DecisionStatus.UNCERTAIN
    assert res.accepted_hypothesis is None
    assert "Critical occlusion" in res.rejection_reason
    assert "e4" in res.uncertain_squares


def test_decision_gate_rejected_bad_geometry():
    gate = DecisionGate()
    # Geometry confidence = 0.40 < 0.50
    obs = make_test_observation(geometry_confidence=0.40)

    h1 = make_hypo("e2e4", 0.90, ["e2", "e4"])
    res = gate.evaluate([h1], obs)
    assert res.status == DecisionStatus.REJECTED
    assert "geometry" in res.rejection_reason.lower()


def test_decision_gate_no_candidates():
    gate = DecisionGate()
    obs = make_test_observation()
    res = gate.evaluate([], obs)
    assert res.status == DecisionStatus.NO_CANDIDATE
    assert res.accepted_hypothesis is None

