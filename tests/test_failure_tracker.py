import os
import tempfile
import unittest


os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.config import settings
from immich_tagger.failure_tracker import FailureTracker


class FailureTrackerPersistenceTests(unittest.TestCase):
    def test_default_failure_file_is_stored_under_config_directory(self):
        original_config_dir = settings.config_dir
        with tempfile.TemporaryDirectory() as config_dir:
            settings.config_dir = config_dir
            self.addCleanup(setattr, settings, "config_dir", original_config_dir)

            tracker = FailureTracker("My Library")
            tracker.record_failure("asset-1")

            self.assertEqual(
                tracker.failure_file,
                os.path.join(
                    config_dir,
                    "processing_failures_My_Library.json",
                ),
            )
            self.assertTrue(os.path.isfile(tracker.failure_file))


if __name__ == "__main__":
    unittest.main()
