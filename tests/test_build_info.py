import os
import unittest
from unittest.mock import patch

from immich_tagger.build_info import get_build_info


class BuildInfoTests(unittest.TestCase):
    def test_configured_build_identity_is_reported(self):
        with patch.dict(
            os.environ,
            {"APP_VERSION": "v2.1.0", "GIT_REVISION": "abc123"},
        ):
            self.assertEqual(
                get_build_info(),
                {"version": "v2.1.0", "revision": "abc123"},
            )

    def test_empty_build_identity_uses_development_fallbacks(self):
        with patch.dict(
            os.environ,
            {"APP_VERSION": "", "GIT_REVISION": ""},
        ):
            self.assertEqual(
                get_build_info(),
                {"version": "development", "revision": "unknown"},
            )


if __name__ == "__main__":
    unittest.main()
