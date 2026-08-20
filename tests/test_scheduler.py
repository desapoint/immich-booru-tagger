import os
import unittest
from unittest.mock import AsyncMock, Mock, patch


os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.scheduler import Scheduler


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


if __name__ == "__main__":
    unittest.main()
