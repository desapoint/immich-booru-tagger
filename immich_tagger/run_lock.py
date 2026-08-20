"""Cross-thread and cross-process guard for complete processing runs."""

import fcntl
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TextIO

from .config import settings


_PROCESS_LOCAL_LOCK = threading.Lock()


class ProcessingRunLock:
    """Non-blocking advisory lock shared through the persistent config mount."""

    def __init__(
        self,
        purpose: str,
        lock_path: Optional[Path] = None,
    ):
        self.purpose = purpose
        self.lock_path = lock_path or (
            Path(settings.config_dir) / ".processing-run.lock"
        )
        self._lock_file: Optional[TextIO] = None
        self._owns_local_lock = False

    def acquire(self, blocking: bool = False) -> bool:
        """Try to own the run lock without waiting by default."""
        if self._lock_file is not None:
            return False

        if not _PROCESS_LOCAL_LOCK.acquire(blocking=blocking):
            return False
        self._owns_local_lock = True
        lock_file = None

        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_file = self.lock_path.open("a+", encoding="utf-8")
            flags = fcntl.LOCK_EX
            if not blocking:
                flags |= fcntl.LOCK_NB
            try:
                fcntl.flock(lock_file.fileno(), flags)
            except BlockingIOError:
                lock_file.close()
                self._release_local_lock()
                return False

            owner = {
                "pid": os.getpid(),
                "purpose": self.purpose,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
            lock_file.seek(0)
            lock_file.truncate()
            json.dump(owner, lock_file)
            lock_file.flush()
            self._lock_file = lock_file
            return True
        except Exception:
            if lock_file is not None and not lock_file.closed:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                finally:
                    lock_file.close()
            self._release_local_lock()
            raise

    def owner_description(self) -> str:
        """Return best-effort context about the process holding the lock."""
        try:
            owner = json.loads(self.lock_path.read_text(encoding="utf-8"))
            return (
                f"pid={owner.get('pid', 'unknown')}, "
                f"mode={owner.get('purpose', 'unknown')}, "
                f"since={owner.get('acquired_at', 'unknown')}"
            )
        except Exception:
            return "owner details unavailable"

    def release(self) -> None:
        """Release ownership; an abandoned file never represents a lock."""
        lock_file = self._lock_file
        if lock_file is None:
            return

        self._lock_file = None
        try:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.flush()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
            self._release_local_lock()

    def _release_local_lock(self) -> None:
        if self._owns_local_lock:
            self._owns_local_lock = False
            _PROCESS_LOCAL_LOCK.release()

    def locked(self) -> bool:
        """Return whether this instance currently owns the lock."""
        return self._lock_file is not None
