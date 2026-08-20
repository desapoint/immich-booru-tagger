import os
import unittest
from unittest.mock import Mock, patch


os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.models import Tag
from immich_tagger.processor import ImmichAutoTagger, ProcessorError


class FakeImmichClient:
    def __init__(self):
        self.library_configs = [
            {"name": "Library_1", "api_key": "key-1"},
            {"name": "Library_2", "api_key": "key-2"},
        ]
        self.current_library_index = 0
        self.current_library = self.library_configs[0]
        self.tag_lookups = []

    @property
    def api_key(self):
        return self.current_library["api_key"]

    @property
    def current_library_name(self):
        return self.current_library["name"]

    def switch_to_library(self, index):
        self.current_library_index = index
        self.current_library = self.library_configs[index]

    def get_or_create_tag(self, name):
        self.tag_lookups.append(self.api_key)
        return Tag(id=f"{self.api_key}-processed", name=name)


class ProcessorTests(unittest.TestCase):
    def test_processed_marker_is_resolved_per_api_key(self):
        client = FakeImmichClient()

        with (
            patch("immich_tagger.processor.ImmichClient", return_value=client),
            patch("immich_tagger.processor.create_tagging_engine", return_value=Mock()),
            patch("immich_tagger.processor.FailureTracker", return_value=Mock()),
        ):
            processor = ImmichAutoTagger()

            self.assertEqual(processor.processed_tag.id, "key-1-processed")

            client.switch_to_library(1)
            processor.set_current_library("Library_2")
            self.assertEqual(processor.processed_tag.id, "key-2-processed")

            client.switch_to_library(0)
            processor.set_current_library("Library_1")
            self.assertEqual(processor.processed_tag.id, "key-1-processed")

        self.assertEqual(client.tag_lookups, ["key-1", "key-2"])

    def test_processor_forwards_batch_limit_to_client(self):
        processor = ImmichAutoTagger.__new__(ImmichAutoTagger)
        processor.logger = Mock()
        processor.failure_tracker = None
        processor.processed_tag = Tag(id="processed-tag", name="auto:processed")
        processor.immich_client = Mock()
        processor.immich_client.current_library_name = "Library_1"
        processor.immich_client.get_unprocessed_assets.return_value = []

        assets = processor.get_unprocessed_assets(limit=17)

        self.assertEqual(assets, [])
        processor.immich_client.get_unprocessed_assets.assert_called_once_with(
            processed_tag_id="processed-tag",
            limit=17,
        )

    def test_processing_errors_propagate_instead_of_signalling_completion(self):
        processor = ImmichAutoTagger.__new__(ImmichAutoTagger)
        processor.logger = Mock()
        processor.get_unprocessed_assets = Mock(
            side_effect=ProcessorError("album lookup failed")
        )

        with self.assertRaisesRegex(ProcessorError, "album lookup failed"):
            processor.run_processing_cycle()


if __name__ == "__main__":
    unittest.main()
