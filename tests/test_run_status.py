import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from immich_tagger.run_status import RunStatus


class RunStatusTests(unittest.TestCase):
    def test_run_lifecycle_produces_json_safe_snapshot(self):
        status = RunStatus()
        timestamps = iter(
            [
                "2026-08-20T10:00:00+00:00",
                "2026-08-20T10:01:00+00:00",
            ]
        )

        with patch(
            "immich_tagger.run_status._utc_now",
            side_effect=lambda: next(timestamps),
        ):
            status.start(batch_limit=4)
            status.set_current_library("Library_1")
            status.update_progress(2, 20, 200)
            status.finish("paused", successful=True)

        snapshot = status.snapshot()
        self.assertEqual(snapshot["state"], "idle")
        self.assertEqual(snapshot["outcome"], "paused")
        self.assertEqual(
            snapshot["run_started_at"],
            "2026-08-20T10:00:00+00:00",
        )
        self.assertEqual(
            snapshot["run_finished_at"],
            "2026-08-20T10:01:00+00:00",
        )
        self.assertEqual(
            snapshot["last_successful_run"],
            snapshot["run_finished_at"],
        )
        self.assertIsNone(snapshot["current_library"])
        self.assertEqual(snapshot["batches_processed"], 2)
        self.assertEqual(snapshot["assets_processed"], 20)
        self.assertEqual(snapshot["tags_assigned"], 200)

    def test_snapshot_is_isolated_and_schedule_is_serialized(self):
        status = RunStatus()
        next_run = datetime(2026, 8, 20, 11, 0, tzinfo=timezone.utc)
        status.set_next_run(next_run)

        first_snapshot = status.snapshot()
        first_snapshot["state"] = "corrupted"

        self.assertEqual(status.snapshot()["state"], "idle")
        self.assertEqual(status.snapshot()["next_run"], next_run.isoformat())

    def test_errors_and_overlap_skips_are_retained(self):
        status = RunStatus()
        status.start(batch_limit=1)
        status.record_error("Library_1 failed")
        status.record_skip("pid=42, mode=single")
        status.finish("completed_with_errors", successful=False)

        snapshot = status.snapshot()
        self.assertEqual(snapshot["last_error"], "Library_1 failed")
        self.assertEqual(snapshot["skipped_runs"], 1)
        self.assertEqual(
            snapshot["last_skip_reason"],
            "pid=42, mode=single",
        )
        self.assertIsNone(snapshot["last_successful_run"])


if __name__ == "__main__":
    unittest.main()
