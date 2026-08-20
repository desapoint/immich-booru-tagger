import os
import threading
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch


os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.scheduler import Scheduler
from immich_tagger.config import settings


class SchedulerThreadingTests(unittest.IsolatedAsyncioTestCase):
    async def test_processing_cycle_runs_outside_the_async_event_loop(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._run_processing_cycle_sync = Mock()

        with patch(
            "immich_tagger.scheduler.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as to_thread:
            await scheduler._run_processing_cycle()

        to_thread.assert_awaited_once_with(
            scheduler._run_processing_cycle_sync
        )


class SchedulerRunControlTests(unittest.TestCase):
    def setUp(self):
        self.original_batch_limit = settings.max_batches_per_run
        self.addCleanup(
            setattr,
            settings,
            "max_batches_per_run",
            self.original_batch_limit,
        )

        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.logger = Mock()
        self.scheduler.processor = Mock()
        self.scheduler.timezone = timezone.utc
        self.scheduler.last_run_time = None
        self.scheduler._processing_lock = threading.Lock()
        self.scheduler._manage_model_retention = Mock()
        self.scheduler._next_library_index = 0
        self.current_library_index = None

        self.scheduler.processor.immich_client.library_configs = [
            {"name": "Library_1"},
            {"name": "Library_2"},
        ]
        self.scheduler.processor.immich_client.get_current_user_info.return_value = {
            "name": "Test User",
            "email": "test@example.com",
        }
        self.scheduler.processor.immich_client.switch_to_library.side_effect = (
            lambda index: setattr(self, "current_library_index", index)
        )
        self.scheduler.processor.library_metrics = {
            "Library_1": {
                "processed_assets": 0,
                "assigned_tags": 0,
            },
            "Library_2": {
                "processed_assets": 0,
                "assigned_tags": 0,
            },
        }

    def test_run_stops_at_global_batch_limit(self):
        settings.max_batches_per_run = 3

        def process_batch():
            library_name = f"Library_{self.current_library_index + 1}"
            metrics = self.scheduler.processor.library_metrics[library_name]
            metrics["processed_assets"] += 10
            metrics["assigned_tags"] += 20
            return True

        self.scheduler.processor.run_processing_cycle.side_effect = process_batch

        result = self.scheduler._run_processing_cycle_sync()

        self.assertTrue(result)
        self.assertEqual(
            self.scheduler.processor.run_processing_cycle.call_count,
            3,
        )
        self.assertEqual(
            [
                call.args[0]
                for call in self.scheduler.processor.immich_client
                .switch_to_library.call_args_list
            ],
            [0, 1, 0],
        )
        self.assertTrue(
            any(
                "Reached per-run limit of 3 batches" in call.args[0]
                for call in self.scheduler.logger.info.call_args_list
            )
        )
        self.assertEqual(self.scheduler._next_library_index, 1)
        self.assertFalse(self.scheduler._processing_lock.locked())

    def test_starting_library_rotates_when_cap_is_smaller_than_library_count(self):
        settings.max_batches_per_run = 1
        self.scheduler.processor.run_processing_cycle.return_value = True

        self.assertTrue(self.scheduler._run_processing_cycle_sync())
        self.assertTrue(self.scheduler._run_processing_cycle_sync())

        self.assertEqual(
            [
                call.args[0]
                for call in self.scheduler.processor.immich_client
                .switch_to_library.call_args_list
            ],
            [0, 1],
        )
        self.assertEqual(self.scheduler._next_library_index, 0)

    def test_library_error_does_not_consume_cap_or_block_other_library(self):
        settings.max_batches_per_run = 2

        def process_batch():
            if self.current_library_index == 0:
                raise RuntimeError("library failed")
            return True

        self.scheduler.processor.run_processing_cycle.side_effect = process_batch

        result = self.scheduler._run_processing_cycle_sync()

        self.assertTrue(result)
        self.assertEqual(
            [
                call.args[0]
                for call in self.scheduler.processor.immich_client
                .switch_to_library.call_args_list
            ],
            [0, 1, 1],
        )
        self.assertTrue(
            any(
                "Error processing library 'Library_1': library failed"
                in call.args[0]
                for call in self.scheduler.logger.error.call_args_list
            )
        )

    def test_overlapping_run_is_skipped(self):
        self.scheduler._processing_lock.acquire()
        self.addCleanup(self.scheduler._processing_lock.release)

        result = self.scheduler._run_processing_cycle_sync()

        self.assertFalse(result)
        self.scheduler.processor.run_processing_cycle.assert_not_called()
        self.scheduler.logger.warning.assert_called_once()
        self.assertIn(
            "Processing run skipped: another run is already active",
            self.scheduler.logger.warning.call_args.args[0],
        )
        self.scheduler._manage_model_retention.assert_not_called()

    def test_run_lock_is_released_if_model_retention_fails(self):
        settings.max_batches_per_run = 1
        self.scheduler.processor.run_processing_cycle.return_value = True
        self.scheduler._manage_model_retention.side_effect = RuntimeError(
            "retention failed"
        )

        with self.assertRaisesRegex(RuntimeError, "retention failed"):
            self.scheduler._run_processing_cycle_sync()

        self.assertFalse(self.scheduler._processing_lock.locked())


class SchedulerModelRetentionTests(unittest.TestCase):
    def setUp(self):
        self.original_unload = settings.unload_model_after_run
        settings.unload_model_after_run = True
        self.addCleanup(
            setattr,
            settings,
            "unload_model_after_run",
            self.original_unload,
        )

        self.now = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.logger = Mock()
        self.scheduler.processor = Mock()
        self.scheduler.processor.is_tagging_engine_loaded = True
        self.scheduler.timezone = timezone.utc

    def manage_retention_with_next_run_in(self, minutes):
        self.scheduler._get_next_run_time = Mock(
            return_value=self.now + timedelta(minutes=minutes)
        )
        with patch("immich_tagger.scheduler.datetime") as current_datetime:
            current_datetime.now.return_value = self.now
            self.scheduler._manage_model_retention()

    def test_model_stays_loaded_when_next_run_is_under_fifteen_minutes(self):
        self.manage_retention_with_next_run_in(14)

        self.scheduler.processor.unload_tagging_engine.assert_not_called()
        self.assertIn(
            "Keeping ONNX model loaded",
            self.scheduler.logger.info.call_args.args[0],
        )

    def test_model_unloads_when_next_run_is_exactly_fifteen_minutes_away(self):
        self.manage_retention_with_next_run_in(15)

        self.scheduler.processor.unload_tagging_engine.assert_called_once_with()
        self.assertIn(
            "unloading ONNX model",
            self.scheduler.logger.info.call_args.args[0],
        )

    def test_model_retention_setting_is_opt_in(self):
        settings.unload_model_after_run = False

        self.manage_retention_with_next_run_in(60)

        self.scheduler.processor.unload_tagging_engine.assert_not_called()
        self.scheduler.logger.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
