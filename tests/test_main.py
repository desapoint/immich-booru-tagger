import os
import unittest
from unittest.mock import AsyncMock, Mock, patch


os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.main import run_with_health_server


class MainRunLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_mode_skips_when_another_process_owns_lock(self):
        processor = Mock()
        run_lock = Mock()
        run_lock.acquire.return_value = False
        run_lock.owner_description.return_value = "pid=42, mode=scheduler"

        with (
            patch("immich_tagger.main.ProcessingRunLock", return_value=run_lock),
            patch(
                "immich_tagger.main.run_health_server_async",
                new_callable=AsyncMock,
            ),
            patch(
                "immich_tagger.main.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as to_thread,
        ):
            await run_with_health_server(processor, "single")

        run_lock.acquire.assert_called_once_with(blocking=False)
        to_thread.assert_not_awaited()
        run_lock.release.assert_not_called()

    async def test_manual_mode_releases_lock_when_processing_fails(self):
        processor = Mock()
        run_lock = Mock()
        run_lock.acquire.return_value = True

        with (
            patch("immich_tagger.main.ProcessingRunLock", return_value=run_lock),
            patch(
                "immich_tagger.main.run_health_server_async",
                new_callable=AsyncMock,
            ),
            patch(
                "immich_tagger.main.asyncio.to_thread",
                new_callable=AsyncMock,
                side_effect=RuntimeError("processing failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "processing failed"):
                await run_with_health_server(processor, "continuous")

        run_lock.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
