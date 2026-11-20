"""Append-only JSONL event repository with cryptographic SHA-256 audit hashing.

Implements atomic append writes, OS fsync disk flushes, hash chaining integrity,
and tamper detection:
- Explanatory Note, Section 2.4.6, Table 2.4 ('Сбой приложения / Восстановление')
- ADR-011: JSONL + PGN + evidence files
- STAGE_5.md, Section 4.1
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import List, Optional, Tuple

from src.core.datatypes import MoveEvent

logger = logging.getLogger(__name__)


def compute_event_hash(event: MoveEvent, prev_hash: Optional[str] = None) -> str:
    """Computes deterministic SHA-256 hash for a MoveEvent in hash chaining order.

    Excludes the event_hash field itself to avoid recursive hashing.
    """
    raw_dict = event.model_dump(mode="json", exclude={"event_hash"})
    # Deterministic JSON canonical representation
    canonical_json = json.dumps(raw_dict, sort_keys=True, separators=(",", ":"))
    payload = f"{prev_hash or ''}:{canonical_json}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JSONLEventRepository:
    """Thread-safe append-only repository for storing MoveEvents to JSONL."""

    def __init__(
        self,
        session_dir: Path | str,
        filename: str = "events.jsonl",
        sync_to_disk: bool = True,
    ) -> None:
        """Initializes event repository within specified session directory.

        Args:
            session_dir: Directory path where events.jsonl resides.
            filename: Name of the journal file (default 'events.jsonl').
            sync_to_disk: If True, triggers os.fsync on each write for power-fail safety.
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.session_dir / filename
        self.sync_to_disk = sync_to_disk

        self._lock = threading.RLock()
        self._last_event_hash: Optional[str] = None
        self._event_count: int = 0

        # Scan existing file if already exists
        self._initialize_from_existing()

    def _initialize_from_existing(self) -> None:
        """Inspects existing file to discover last event hash and total event count."""
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            return

        with self.file_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        ev = MoveEvent.model_validate_json(stripped)
                        self._last_event_hash = ev.event_hash
                        self._event_count += 1
                    except Exception as e:
                        logger.warning("Corrupted or incomplete line during repository init: %s", e)

    @property
    def last_event_hash(self) -> Optional[str]:
        """Returns the SHA-256 hash of the most recent committed event."""
        with self._lock:
            return self._last_event_hash

    @property
    def count(self) -> int:
        """Returns the total number of events written to this repository."""
        with self._lock:
            return self._event_count

    def append_event(self, event: MoveEvent) -> MoveEvent:
        """Atomically appends a MoveEvent to the JSONL journal with SHA-256 chaining.

        Args:
            event: MoveEvent to commit to disk.

        Returns:
            MoveEvent with prev_event_hash and event_hash populated.
        """
        with self._lock:
            # Hash chaining
            event.prev_event_hash = self._last_event_hash
            event.event_hash = compute_event_hash(event, self._last_event_hash)

            json_line = event.model_dump_json() + "\n"

            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(json_line)
                if self.sync_to_disk:
                    f.flush()
                    os.fsync(f.fileno())

            self._last_event_hash = event.event_hash
            self._event_count += 1

            logger.debug(
                "Appended MoveEvent ply=%d (%s) hash=%s",
                event.ply,
                event.move_san,
                event.event_hash[:8],
            )
            return event

    def read_all_events(self) -> List[MoveEvent]:
        """Reads all valid MoveEvents from the journal.

        Returns:
            List of MoveEvents ordered chronologically.
        """
        with self._lock:
            if not self.file_path.exists():
                return []

            events: List[MoveEvent] = []
            with self.file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        events.append(MoveEvent.model_validate_json(stripped))
            return events

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verifies cryptographic hash chain and sequence integrity of the journal.

        Returns:
            (True, None) if journal is fully intact and valid.
            (False, error_reason) if any tampering or corruption is discovered.
        """
        with self._lock:
            if not self.file_path.exists() or self.file_path.stat().st_size == 0:
                return True, None

            expected_prev_hash: Optional[str] = None
            expected_ply: int = 1

            with self.file_path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue

                    try:
                        ev = MoveEvent.model_validate_json(stripped)
                    except Exception as e:
                        return False, f"Line {line_idx}: invalid JSON or contract structure: {e}"

                    if ev.ply != expected_ply:
                        return False, f"Line {line_idx}: broken ply sequence. Expected {expected_ply}, got {ev.ply}"

                    if ev.prev_event_hash != expected_prev_hash:
                        return (
                            False,
                            f"Line {line_idx} (ply {ev.ply}): prev_event_hash mismatch. "
                            f"Expected '{expected_prev_hash}', got '{ev.prev_event_hash}'",
                        )

                    recalculated_hash = compute_event_hash(ev, expected_prev_hash)
                    if ev.event_hash != recalculated_hash:
                        return (
                            False,
                            f"Line {line_idx} (ply {ev.ply}): event_hash verification failed. "
                            f"Recorded '{ev.event_hash}', recalculated '{recalculated_hash}'",
                        )

                    expected_prev_hash = ev.event_hash
                    expected_ply += 1

            return True, None

