import os
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
