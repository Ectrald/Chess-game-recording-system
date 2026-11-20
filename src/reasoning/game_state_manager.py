"""Authoritative chess game state manager.

Maintains official board state, enforces FIDE legality, provides atomic commit,
SAN generation, FEN tracking, idempotency deduplication, PGN projection,
event history journaling, rollback, and deterministic replay.

Reference:
- Explanatory Note, Section 2.4.4
- STAGE_4.md, Section 4.4
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Callable, Dict, List, Optional, Set
import chess
import chess.pgn

from src.core.datatypes import (
    CHESS_SQUARES,
    BoardObservation,
    CellObservation,
    MoveEvent,
    MoveHypothesis,
)

logger = logging.getLogger(__name__)


class GameStateManager:
    """Authoritative thread-safe manager for the official chess game position."""

    def __init__(
        self,
        initial_fen: str = chess.STARTING_FEN,
        on_move_committed: Optional[Callable[[MoveEvent], None]] = None,
    ) -> None:
        """Initializes GameStateManager with starting position.

        Args:
            initial_fen: Starting FEN string (default standard initial position).
            on_move_committed: Optional callback invoked whenever a move is committed.
        """
        self._initial_fen = initial_fen
        self._board = chess.Board(initial_fen)
        self._history: List[MoveEvent] = []
        self._committed_keys: Set[str] = set()
        self._lock = threading.RLock()
        self.on_move_committed = on_move_committed

    @property
    def current_board(self) -> chess.Board:
        """Returns an isolated copy of the current authoritative board state."""
        with self._lock:
            return self._board.copy()

    @property
    def current_fen(self) -> str:
        """Returns the current FEN string of the authoritative board."""
        with self._lock:
            return self._board.fen()

    @property
    def ply(self) -> int:
        """Returns current half-move index (0 before first move, 1 after White move 1, etc.)."""
        with self._lock:
            return len(self._history)

    @property
    def history(self) -> List[MoveEvent]:
        """Returns a shallow copy of all committed move events."""
        with self._lock:
            return list(self._history)

    def commit(
        self,
        hypothesis: MoveHypothesis,
        observation: BoardObservation,
    ) -> MoveEvent:
        """Atomically commits a verified move hypothesis to the authoritative game state.

        Performs:
        1. Thread lock acquisition
        2. FIDE legality validation
        3. Idempotency deduplication check
        4. SAN calculation on current position
        5. FEN before and FEN after recording
        6. board.push(move)
        7. MoveEvent creation and history appending
        8. Notification callback execution

        Args:
            hypothesis: Accepted candidate move hypothesis.
            observation: Settled observation providing evidence frame and timestamp.

        Returns:
            Committed MoveEvent.

        Raises:
            ValueError: If the move is not legal in the current position.
        """
        with self._lock:
            # Idempotency check: if this exact move event was already processed, return it
            if self._history:
                last_event = self._history[-1]
                candidate_key = f"{last_event.ply}_{hypothesis.move_uci}_{observation.frame_id}"
                if candidate_key in self._committed_keys:
                    logger.warning("Duplicate move event ignored (idempotent): %s", candidate_key)
                    return last_event

            move = chess.Move.from_uci(hypothesis.move_uci)
            if not self._board.is_legal(move):
                raise ValueError(
                    f"Move '{hypothesis.move_uci}' is not legal in position '{self._board.fen()}'"
                )

            next_ply = len(self._history) + 1
            idempotency_key = f"{next_ply}_{hypothesis.move_uci}_{observation.frame_id}"

            # Generate Standard Algebraic Notation before pushing the move
            move_san = self._board.san(move)
            fen_before = self._board.fen()

            # Execute move on authoritative board
            self._board.push(move)
            fen_after = self._board.fen()

            event = MoveEvent(
                event_id=uuid.uuid4().hex,
                ply=next_ply,
                fen_before=fen_before,
                move_uci=hypothesis.move_uci,
                move_san=move_san,
                fen_after=fen_after,
                confidence=hypothesis.total_score,
                timestamp=observation.timestamp,
                evidence_frame_id=observation.frame_id,
                idempotency_key=idempotency_key,
            )

            self._history.append(event)
            self._committed_keys.add(idempotency_key)

            logger.info(
                "Committed move ply=%d %s (%s) -> FEN: %s",
                event.ply,
                event.move_san,
                event.move_uci,
                event.fen_after,
            )

            if self.on_move_committed:
                try:
                    self.on_move_committed(event)
                except Exception as exc:
                    logger.error("Error in on_move_committed callback: %s", exc, exc_info=True)

            return event

    def manual_commit(
        self,
        move_uci: str,
        observation: Optional[BoardObservation] = None,
    ) -> MoveEvent:
        """Manual operator override to commit a move directly.

        Args:
            move_uci: Move in UCI notation (e.g. 'e2e4', 'e7e8q').
            observation: Optional current BoardObservation.

        Returns:
            Committed MoveEvent.
        """
        with self._lock:
            # Validate move syntax
            try:
                move = chess.Move.from_uci(move_uci)
            except Exception as e:
                raise ValueError(f"Invalid UCI move string '{move_uci}': {e}") from e

            # Create dummy observation if not provided
            if observation is None:
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
                obs = BoardObservation(
                    frame_id=0,
                    timestamp=0.0,
                    cells=cells,
                    motion_detected=False,
                    motion_level=0.0,
                    hand_present=False,
                    geometry_confidence=1.0,
                )
            else:
                obs = observation

            # Construct synthetic hypothesis
            hypo = MoveHypothesis(
                move_uci=move_uci,
                move_san=None,
                changed_squares=[chess.square_name(move.from_square), chess.square_name(move.to_square)],
                score_visual=1.0,
                score_temporal=1.0,
                score_geometry=1.0,
                penalty_occlusion=0.0,
                total_score=1.0,
                is_legal=True,
            )
            return self.commit(hypo, obs)

    def rollback(self) -> Optional[MoveEvent]:
        """Undoes the most recent committed move and restores previous board position.

        Returns:
            The reverted MoveEvent, or None if history is empty.
        """
        with self._lock:
            if not self._history:
                return None

            popped = self._history.pop()
            if popped.idempotency_key:
                self._committed_keys.discard(popped.idempotency_key)

            self._board.pop()
            logger.info("Rolled back move ply=%d (%s)", popped.ply, popped.move_san)
            return popped

    def export_pgn(self, headers: Optional[Dict[str, str]] = None) -> str:
        """Exports the recorded game history as a standard PGN string.

        Args:
            headers: Optional dictionary of PGN header tags.

        Returns:
            Full PGN text representation formatted to FIDE standards.
        """
        with self._lock:
            game = chess.pgn.Game()
            if self._initial_fen != chess.STARTING_FEN:
                game.setup(self._initial_fen)

            # Default headers
            hdrs = {
                "Event": "Recorded Game",
                "Site": "Chess Game Recording System",
                "Date": "????.??.??",
                "Round": "1",
                "White": "White Player",
                "Black": "Black Player",
                "Result": self._board.result(),
            }
            if headers:
                hdrs.update(headers)

            for key, val in hdrs.items():
                game.headers[key] = str(val)

            # Reconstruct move tree from history
            curr_node = game
            for event in self._history:
                move = chess.Move.from_uci(event.move_uci)
                curr_node = curr_node.add_main_variation(move)

            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
            return game.accept(exporter)

    def replay(self, events: List[MoveEvent]) -> chess.Board:
        """Deterministically validates and replays a sequence of MoveEvents.

        Guarantees that replaying the journal from initial_fen produces the exact
        same position as the authoritative board.

        Args:
            events: List of MoveEvents to replay.

        Returns:
            Reconstructed chess.Board state.

        Raises:
            ValueError: If any event FEN or legality fails verification.
        """
        with self._lock:
            board = chess.Board(self._initial_fen)
            for i, event in enumerate(events):
                if board.fen() != event.fen_before:
                    raise ValueError(
                        f"Replay divergence at ply {event.ply}: "
                        f"expected FEN '{event.fen_before}', but replay board is '{board.fen()}'"
                    )

                move = chess.Move.from_uci(event.move_uci)
                if not board.is_legal(move):
                    raise ValueError(f"Replay illegal move at ply {event.ply}: '{event.move_uci}'")

                board.push(move)

                if board.fen() != event.fen_after:
                    raise ValueError(
                        f"Replay post-move FEN mismatch at ply {event.ply}: "
                        f"expected '{event.fen_after}', got '{board.fen()}'"
                    )

            return board
