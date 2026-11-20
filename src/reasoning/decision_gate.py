"""Two-factor decision gate for candidate move verification and acceptance.

Evaluates ranked candidate hypotheses against:
1. Absolute acceptance score threshold tau_accept (default 0.55)
2. Confidence margin Delta = S(m1) - S(m2) (default 0.15)
3. Board geometry validity (S_g >= 0.50)
4. Critical occlusion limits on affected squares (tau_occ_crit < 0.50)

Reference:
- Explanatory Note, Section 2.4.4
- STAGE_4.md, Section 4.3
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.core.config import DecisionConfig
from src.core.datatypes import (
    BoardObservation,
    DecisionResult,
    DecisionStatus,
    MoveHypothesis,
)

logger = logging.getLogger(__name__)


class DecisionGate:
    """Verifies candidate move validity and resolves ambiguities or rejections."""

    def __init__(self, config: Optional[DecisionConfig] = None) -> None:
        """Initializes DecisionGate with thresholds.

        Args:
            config: DecisionConfig with min_acceptance_score, min_confidence_margin,
                    and max_occlusion_threshold.
        """
        self.config = config or DecisionConfig()

    def evaluate(
        self,
        ranked_hypotheses: List[MoveHypothesis],
        observation: BoardObservation,
    ) -> DecisionResult:
        """Evaluates ranked candidate hypotheses and decides acceptance outcome.

        Args:
            ranked_hypotheses: Candidate moves ordered descending by total_score S(m).
            observation: BoardObservation associated with this evaluation window.

        Returns:
            DecisionResult indicating ACCEPTED, UNCERTAIN, REJECTED, or NO_CANDIDATE.
        """
        if not ranked_hypotheses:
            return DecisionResult(
                status=DecisionStatus.NO_CANDIDATE,
                rejection_reason="No legal move candidates provided",
            )

        top_hypo = ranked_hypotheses[0]
        top_slice = ranked_hypotheses[:5]

        # Calculate confidence margin Delta
        if len(ranked_hypotheses) == 1:
            margin = 1.0
        else:
            second_hypo = ranked_hypotheses[1]
            margin = max(0.0, round(top_hypo.total_score - second_hypo.total_score, 4))

        # 1. Geometry stability check
        if observation.geometry_confidence < 0.50:
            return DecisionResult(
                status=DecisionStatus.REJECTED,
                rejection_reason=(
                    f"Board geometry confidence {observation.geometry_confidence:.3f} "
                    f"is below minimum acceptable threshold 0.50"
                ),
                margin=margin,
                top_hypotheses=top_slice,
                uncertain_squares=[],
            )

        # 2. Critical square occlusion check
        critically_occluded: List[str] = []
        for sq in top_hypo.changed_squares:
            cell = observation.cells.get(sq)
            if cell is not None:
                ratio = getattr(cell, "occlusion_ratio", 0.0)
                if ratio == 0.0 and cell.is_occluded:
                    ratio = 1.0
                if ratio >= self.config.max_occlusion_threshold:
                    critically_occluded.append(sq)

        if critically_occluded:
            return DecisionResult(
                status=DecisionStatus.UNCERTAIN,
                rejection_reason=(
                    f"Critical occlusion on move squares: {critically_occluded} "
                    f"(>= {self.config.max_occlusion_threshold:.2f})"
                ),
                margin=margin,
                top_hypotheses=top_slice,
                uncertain_squares=sorted(critically_occluded),
            )

        # 3. Absolute score threshold check
        if top_hypo.total_score < self.config.min_acceptance_score:
            return DecisionResult(
                status=DecisionStatus.UNCERTAIN,
                rejection_reason=(
                    f"Top candidate score {top_hypo.total_score:.3f} is below "
                    f"minimum acceptance threshold {self.config.min_acceptance_score:.3f}"
                ),
                margin=margin,
                top_hypotheses=top_slice,
                uncertain_squares=top_hypo.changed_squares,
            )

        # 4. Confidence margin check
        if len(ranked_hypotheses) > 1 and margin < self.config.min_confidence_margin:
            second_hypo = ranked_hypotheses[1]
            competing_squares = sorted(
                list(set(top_hypo.changed_squares).union(set(second_hypo.changed_squares)))
            )
            return DecisionResult(
                status=DecisionStatus.UNCERTAIN,
                rejection_reason=(
                    f"Confidence margin {margin:.3f} is below required "
                    f"delta threshold {self.config.min_confidence_margin:.3f} "
                    f"between '{top_hypo.move_uci}' ({top_hypo.total_score:.3f}) and "
                    f"'{second_hypo.move_uci}' ({second_hypo.total_score:.3f})"
                ),
                margin=margin,
                top_hypotheses=top_slice,
                uncertain_squares=competing_squares,
            )

        # All criteria satisfied: Accept candidate move
        return DecisionResult(
            status=DecisionStatus.ACCEPTED,
            accepted_hypothesis=top_hypo,
            margin=margin,
            top_hypotheses=top_slice,
            uncertain_squares=[],
        )

