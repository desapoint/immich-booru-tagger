import os
import unittest
from unittest.mock import Mock, patch


os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.health_server import HealthServer


class HealthServerTests(unittest.TestCase):
    def test_health_checks_do_not_switch_the_active_library(self):
        processor = Mock()
        processor.library_metrics = {"Library_1": {"processed_assets": 2}}
        processor.get_metrics.return_value = {"assets_processed": 2}
        processor.immich_client.library_configs = [
            {"name": "Library_1", "api_key": "key-1"},
            {"name": "Library_2", "api_key": "key-2"},
        ]
        response = Mock(status_code=200)
        processor.immich_client._make_request_silent.return_value = response
        processor.immich_client.get_user_info_for_api_key.side_effect = [
            {"name": "One", "email": "one@example.com"},
            {"name": "Two", "email": "two@example.com"},
        ]
        server = HealthServer(processor)

        with patch.dict(
            os.environ,
            {"APP_VERSION": "v2.1.0", "GIT_REVISION": "abc123"},
        ):
            healthy, payload = server._collect_health_response()

        self.assertTrue(healthy)
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(payload["version"], "v2.1.0")
        self.assertEqual(
            payload["build"],
            {"version": "v2.1.0", "revision": "abc123"},
        )
        self.assertEqual(
            processor.immich_client._make_request_silent.call_args_list[0].kwargs,
            {"method": "GET", "endpoint": "/api/tags", "api_key": "key-1"},
        )
        self.assertEqual(
            processor.immich_client._make_request_silent.call_args_list[1].kwargs,
            {"method": "GET", "endpoint": "/api/tags", "api_key": "key-2"},
        )
        processor.immich_client._switch_to_library_silent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
