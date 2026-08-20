"""Thread-safe operational state shared by processing and health threads."""

import threading
from datetime import datetime, timezone
from typing import Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus:
    """Maintain a consistent JSON-serializable processing status snapshot."""

    def __init__(self):
        self._lock = threading.RLock()
        self._status = {
            "state": "idle",
            "outcome": "never_run",
            "run_started_at": None,
            "run_finished_at": None,
            "current_library": None,
            "batches_processed": 0,
            "batch_limit": 0,
            "assets_processed": 0,
            "tags_assigned": 0,
            "last_successful_run": None,
            "last_error": None,
            "next_run": None,
            "skipped_runs": 0,
            "last_skip_at": None,
            "last_skip_reason": None,
        }

    def start(self, batch_limit: int) -> None:
        """Begin a new run while retaining historical schedule/skip state."""
        with self._lock:
            self._status.update(
                {
                    "state": "running",
                    "outcome": "running",
                    "run_started_at": _utc_now(),
                    "run_finished_at": None,
                    "current_library": None,
                    "batches_processed": 0,
                    "batch_limit": batch_limit,
                    "assets_processed": 0,
                    "tags_assigned": 0,
                    "last_error": None,
                }
            )

    def set_current_library(self, library_name: Optional[str]) -> None:
        with self._lock:
            self._status["current_library"] = library_name

    def update_progress(
        self,
        batches_processed: int,
        assets_processed: int,
        tags_assigned: int,
    ) -> None:
        with self._lock:
            self._status.update(
                {
                    "batches_processed": batches_processed,
                    "assets_processed": assets_processed,
                    "tags_assigned": tags_assigned,
                }
            )

    def record_error(self, error: str) -> None:
        with self._lock:
            self._status["last_error"] = error

    def record_skip(self, reason: str) -> None:
        with self._lock:
            self._status["skipped_runs"] += 1
            self._status["last_skip_at"] = _utc_now()
            self._status["last_skip_reason"] = reason

    def set_next_run(self, next_run: Optional[datetime]) -> None:
        with self._lock:
            self._status["next_run"] = (
                next_run.isoformat() if next_run is not None else None
            )

    def finish(self, outcome: str, successful: bool) -> None:
        finished_at = _utc_now()
        with self._lock:
            self._status.update(
                {
                    "state": "idle",
                    "outcome": outcome,
                    "run_finished_at": finished_at,
                    "current_library": None,
                }
            )
            if successful:
                self._status["last_successful_run"] = finished_at

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._status)
