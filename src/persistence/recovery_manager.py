"""Crash recovery and cold-start state reconstruction manager.

Guarantees data safety against sudden application crashes or power failures:
- Scans events.jsonl for incomplete trailing writes
- Sanitizes incomplete tail bytes (truncate to last valid newline)
- Verifies cryptographic hash chain integrity
- Deterministically replays game history to restore GameStateManager
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import chess

from src.core.datatypes import MoveEvent
from src.persistence.event_repository import JSONLEventRepository
from src.reasoning.game_state_manager import GameStateManager

logger = logging.getLogger(__name__)


class RecoveryManager:
    """Manages session crash detection, file tail sanitization, and state restoration."""

    def __init__(self, session_dir: Path | str, journal_filename: str = "events.jsonl") -> None:
        """Initializes RecoveryManager for a session directory.

        Args:
            session_dir: Directory containing events.jsonl.
            journal_filename: Journal file name (default 'events.jsonl').
        """
        self.session_dir = Path(session_dir)
        self.journal_file = self.session_dir / journal_filename
        self.repository = JSONLEventRepository(self.session_dir, filename=journal_filename)

    def _truncate_journal_in_place(self, truncate_pos: int) -> None:
        """Safely truncates the journal file in-place and flushes to disk."""
        with self.journal_file.open("r+b") as f:
            f.truncate(truncate_pos)
            f.flush()
            os.fsync(f.fileno())
        self.repository._initialize_from_existing()

    def sanitize_corrupted_tail(self) -> int:
        """Detects and trims incomplete or malformed trailing lines in the journal file.

        Such incomplete lines typically occur when process execution is terminated
        abruptly (e.g. SIGKILL, power loss) mid-write.

        Returns:
            Number of truncated bytes (0 if file was clean).
        """
        if not self.journal_file.exists() or self.journal_file.stat().st_size == 0:
            return 0

        with self.journal_file.open("rb") as f:
            data = f.read()

        if not data:
            return 0

        orig_len = len(data)
        pos = orig_len

        while pos > 0:
            line_start = data.rfind(b"\n", 0, pos)
            # If the slice ends with \n, trim one trailing newline step back
            if pos == data.rfind(b"\n", 0, pos) + 1:
                line_start = data.rfind(b"\n", 0, pos - 1)
                candidate = data[line_start + 1 : pos - 1].strip()
                if not candidate:
                    pos = line_start + 1 if line_start != -1 else 0
                    continue
                try:
                    MoveEvent.model_validate_json(candidate.decode("utf-8"))
                    break
                except Exception:
                    pos = line_start + 1 if line_start != -1 else 0
            else:
                # Last line without trailing newline
                candidate = data[line_start + 1 : pos].strip()
                if not candidate:
                    pos = line_start + 1 if line_start != -1 else 0
                    continue
                pos = line_start + 1 if line_start != -1 else 0

        truncated_bytes = orig_len - pos
        if truncated_bytes > 0:
            self._truncate_journal_in_place(pos)
            logger.warning(
                "Truncated %d incomplete/corrupted tail bytes from journal %s",
                truncated_bytes,
                self.journal_file.name,
            )
        return truncated_bytes

    def recover_session(self) -> Tuple[GameStateManager, List[MoveEvent]]:
        """Recovers authoritative GameStateManager and event history from journal.

        Returns:
            Tuple of (restored_game_state_manager, list_of_recovered_events).

        Raises:
            ValueError: If journal cryptographic integrity or sequence replay fails.
        """
        # 1. Sanitize any incomplete trailing writes from crashes
        self.sanitize_corrupted_tail()

        # 2. Check cryptographic hash chain integrity
        is_valid, error_reason = self.repository.verify_integrity()
        if not is_valid:
            raise ValueError(f"Journal integrity check failed: {error_reason}")

        # 3. Read all audited events and fold rollbacks/anomalies into active move list
        all_events = self.repository.read_all_events()
        if not all_events:
            return GameStateManager(), []

        events: List[MoveEvent] = []
        for ev in all_events:
            ev_type = getattr(ev, "event_type", "move")
            if ev_type == "move":
                events.append(ev)
            elif ev_type == "rollback" and events:
                events.pop()

        # 4. Replay active moves into fresh GameStateManager
        initial_fen = all_events[0].fen_before
        state_mgr = GameStateManager(initial_fen=initial_fen)

        if events:
            final_board = state_mgr.replay(events)
            state_mgr._board = final_board
            state_mgr._history = list(events)
            state_mgr._committed_keys = {
                ev.idempotency_key for ev in events if ev.idempotency_key
            }

        logger.info(
            "Successfully recovered session with %d moves; current FEN: %s",
            len(events),
            state_mgr.current_fen,
        )
        return state_mgr, events
