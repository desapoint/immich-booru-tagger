import os
import unittest


os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.logging import get_logger


class StructuredLoggingTests(unittest.TestCase):
    def test_context_keywords_are_formatted_without_reaching_logger_log(self):
        logger = get_logger("tests.structured_logging")

        with self.assertLogs("tests.structured_logging", level="INFO") as logs:
            logger.info(
                "Deleted tag",
                tag_id="tag-1",
                asset_count=2,
            )

        self.assertIn(
            "Deleted tag | tag_id='tag-1', asset_count=2",
            logs.output[0],
        )


if __name__ == "__main__":
    unittest.main()
