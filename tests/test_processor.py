import os
import unittest
from unittest.mock import Mock, patch


os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.models import Asset, Tag, TagPrediction
from immich_tagger.processor import (
    CONTENT_RATINGS,
    ImmichAutoTagger,
    ProcessorError,
)


class FakeImmichClient:
    def __init__(self):
        self.library_configs = [
            {"name": "Library_1", "api_key": "key-1"},
            {"name": "Library_2", "api_key": "key-2"},
        ]
        self.current_library_index = 0
        self.current_library = self.library_configs[0]
        self.tag_lookups = []
        self.child_tag_lookups = []
        self.migrations = []

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
        return Tag(id=f"{self.api_key}-{name}", name=name)

    def get_all_tags(self, use_cache=True):
        return []

    def get_or_create_child_tag(self, parent, name):
        self.child_tag_lookups.append((self.api_key, parent.id, name))
        return Tag(
            id=f"{self.api_key}-rating-{name}",
            name=name,
            value=f"content-rating/{name}",
            parentId=parent.id,
        )

    def migrate_tag(self, source, destination):
        self.migrations.append((source, destination))
        return 0


class ProcessorTests(unittest.TestCase):
    def test_processed_marker_is_resolved_per_api_key(self):
        client = FakeImmichClient()

        with (
            patch("immich_tagger.processor.ImmichClient", return_value=client),
            patch("immich_tagger.processor.create_tagging_engine", return_value=Mock()),
            patch("immich_tagger.processor.FailureTracker", return_value=Mock()),
        ):
            processor = ImmichAutoTagger()

            self.assertEqual(processor.processed_tag.id, "key-1-auto:processed")

            client.switch_to_library(1)
            processor.set_current_library("Library_2")
            self.assertEqual(processor.processed_tag.id, "key-2-auto:processed")

            client.switch_to_library(0)
            processor.set_current_library("Library_1")
            self.assertEqual(processor.processed_tag.id, "key-1-auto:processed")

        self.assertEqual(client.tag_lookups, ["key-1", "key-1", "key-2", "key-2"])
        self.assertEqual(len(client.child_tag_lookups), len(CONTENT_RATINGS) * 2)

    def test_only_existing_flat_rating_tags_are_migrated(self):
        client = FakeImmichClient()
        flat_general = Tag(id="flat-general", name="general")
        unrelated = Tag(id="unrelated", name="landscape")
        client.get_all_tags = Mock(return_value=[flat_general, unrelated])

        with (
            patch("immich_tagger.processor.ImmichClient", return_value=client),
            patch("immich_tagger.processor.create_tagging_engine", return_value=Mock()),
            patch("immich_tagger.processor.FailureTracker", return_value=Mock()),
        ):
            ImmichAutoTagger()

        self.assertEqual(len(client.migrations), 1)
        source, destination = client.migrations[0]
        self.assertEqual(source.id, "flat-general")
        self.assertEqual(destination.path, "content-rating/general")

    def test_rating_predictions_use_hierarchical_tag(self):
        processor = ImmichAutoTagger.__new__(ImmichAutoTagger)
        processor.logger = Mock()
        processor.metrics = Mock()
        processor.metrics.metrics = {
            "assets_processed": 0,
            "tags_assigned": 0,
            "processing_time": 0,
            "failures": 0,
        }
        processor.processed_tag = Tag(id="processed", name="auto:processed")
        processor.immich_client = Mock()
        processor.immich_client.api_key = "key-1"
        processor.immich_client.download_asset.return_value = b"image"
        processor.immich_client.get_or_create_tags_bulk.return_value = {
            "1girl": Tag(id="one-girl", name="1girl")
        }
        processor.tagging_engine = Mock()
        processor.tagging_engine.predict_tags.return_value = [
            TagPrediction(name="general", confidence=0.9),
            TagPrediction(name="1girl", confidence=0.8),
        ]
        processor.content_rating_tags = {
            "key-1": {
                "general": Tag(
                    id="rating-general",
                    name="general",
                    value="content-rating/general",
                    parentId="rating-parent",
                )
            }
        }

        asset = Mock(spec=Asset)
        asset.id = "asset-1"
        asset.type = "IMAGE"
        asset.tags = []

        with patch("immich_tagger.processor.performance_monitor"):
            result = processor.process_asset(asset)

        self.assertTrue(result.success)
        self.assertEqual(
            result.tags_assigned,
            ["content-rating/general", "1girl"],
        )
        processor.immich_client.get_or_create_tags_bulk.assert_called_once_with(
            ["1girl"]
        )
        self.assertEqual(
            processor.immich_client.tag_single_asset.call_args_list[0].args,
            ("asset-1", ["rating-general", "one-girl"]),
        )

    def test_process_batch_uses_one_batched_model_call(self):
        processor = ImmichAutoTagger.__new__(ImmichAutoTagger)
        processor.logger = Mock()
        processor.metrics = Mock()
        processor.metrics.metrics = {
            "assets_processed": 0,
            "tags_assigned": 0,
            "processing_time": 0,
            "failures": 0,
        }
        processor.processed_tag = Tag(id="processed", name="auto:processed")
        processor.content_rating_tags = {"key-1": {}}
        processor.immich_client = Mock()
        processor.immich_client.api_key = "key-1"
        processor.immich_client.current_library_name = "Library_1"
        processor.immich_client.download_asset.side_effect = [b"one", b"two"]
        processor.immich_client.get_or_create_tags_bulk.side_effect = [
            {"tag_one": Tag(id="tag-one", name="tag_one")},
            {"tag_two": Tag(id="tag-two", name="tag_two")},
        ]
        processor.tagging_engine = Mock()
        processor.tagging_engine.predict_tags_batch.return_value = [
            [TagPrediction(name="tag_one", confidence=0.8)],
            [TagPrediction(name="tag_two", confidence=0.7)],
        ]
        processor.failure_tracker = Mock()
        processor.total_processed_assets = 0
        processor.total_assigned_tags = 0
        processor.library_metrics = {
            "Library_1": {
                "processed_assets": 0,
                "assigned_tags": 0,
                "failed_assets": 0,
            }
        }

        assets = []
        for index in range(2):
            asset = Mock(spec=Asset)
            asset.id = f"asset-{index + 1}"
            asset.type = "IMAGE"
            asset.tags = []
            asset.originalFileName = f"asset-{index + 1}.jpg"
            assets.append(asset)

        with patch("immich_tagger.processor.performance_monitor"):
            result = processor.process_batch(assets)

        self.assertEqual(result.successful, 2)
        processor.tagging_engine.predict_tags_batch.assert_called_once_with(
            [b"one", b"two"]
        )
        processor.tagging_engine.predict_tags.assert_not_called()

    def test_failed_batch_inference_retries_each_image_individually(self):
        processor = ImmichAutoTagger.__new__(ImmichAutoTagger)
        processor.logger = Mock()
        processor.metrics = Mock()
        processor.metrics.metrics = {
            "assets_processed": 0,
            "tags_assigned": 0,
            "processing_time": 0,
            "failures": 0,
        }
        processor.processed_tag = Tag(id="processed", name="auto:processed")
        processor.content_rating_tags = {"key-1": {}}
        processor.immich_client = Mock()
        processor.immich_client.api_key = "key-1"
        processor.immich_client.current_library_name = "Library_1"
        processor.immich_client.download_asset.side_effect = [b"valid", b"bad"]
        processor.immich_client.get_or_create_tags_bulk.return_value = {
            "tag_one": Tag(id="tag-one", name="tag_one")
        }
        processor.tagging_engine = Mock()
        processor.tagging_engine.predict_tags_batch.side_effect = RuntimeError(
            "invalid image in batch"
        )
        processor.tagging_engine.predict_tags.side_effect = [
            [TagPrediction(name="tag_one", confidence=0.8)],
            RuntimeError("bad image"),
        ]
        processor.failure_tracker = Mock()
        processor.failure_tracker.record_failure.return_value = True
        processor.total_processed_assets = 0
        processor.total_assigned_tags = 0
        processor.library_metrics = {
            "Library_1": {
                "processed_assets": 0,
                "assigned_tags": 0,
                "failed_assets": 0,
            }
        }

        assets = []
        for index in range(2):
            asset = Mock(spec=Asset)
            asset.id = f"asset-{index + 1}"
            asset.type = "IMAGE"
            asset.tags = []
            asset.originalFileName = f"asset-{index + 1}.jpg"
            assets.append(asset)

        with patch("immich_tagger.processor.performance_monitor"):
            result = processor.process_batch(assets)

        self.assertEqual(result.successful, 1)
        self.assertEqual(result.failed, 1)
        self.assertEqual(
            [call.args[0] for call in processor.tagging_engine.predict_tags.call_args_list],
            [b"valid", b"bad"],
        )
        processor.failure_tracker.record_failure.assert_called_once_with("asset-2")

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
