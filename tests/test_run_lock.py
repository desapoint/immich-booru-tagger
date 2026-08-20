import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.run_lock import ProcessingRunLock


class ProcessingRunLockTests(unittest.TestCase):
    def test_second_process_cannot_acquire_held_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "processing.lock"
            first_lock = ProcessingRunLock("scheduler", lock_path)
            self.assertTrue(first_lock.acquire())
            self.addCleanup(first_lock.release)

            script = "\n".join(
                [
                    "import sys",
                    "from pathlib import Path",
                    "from immich_tagger.run_lock import ProcessingRunLock",
                    "lock = ProcessingRunLock('single', Path(sys.argv[1]))",
                    "sys.exit(1 if lock.acquire() else 0)",
                ]
            )
            result = subprocess.run(
                [sys.executable, "-c", script, str(lock_path)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("mode=scheduler", first_lock.owner_description())

    def test_stale_lock_file_does_not_block_new_owner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "processing.lock"
            lock_path.write_text("stale owner", encoding="utf-8")
            lock = ProcessingRunLock("continuous", lock_path)

            self.assertTrue(lock.acquire())
            self.assertTrue(lock.locked())
            lock.release()
            self.assertFalse(lock.locked())

    def test_lock_can_be_reacquired_after_release(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "processing.lock"
            first_lock = ProcessingRunLock("scheduler", lock_path)
            second_lock = ProcessingRunLock("single", lock_path)

            self.assertTrue(first_lock.acquire())
            first_lock.release()
            self.assertTrue(second_lock.acquire())
            second_lock.release()


if __name__ == "__main__":
    unittest.main()
