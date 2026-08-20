import os
import unittest
from unittest.mock import Mock


os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.immich_client import ImmichClient
from immich_tagger.models import Tag


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class ContentRatingTagClientTests(unittest.TestCase):
    def setUp(self):
        self.client = ImmichClient.__new__(ImmichClient)
        self.client.current_library = {
            "name": "Library_1",
            "api_key": "test-api-key",
        }
        self.client._tag_caches = {"test-api-key": {}}
        self.client._tag_cache_valid = {"test-api-key": False}
        self.client._tag_cache_timestamp = {"test-api-key": 0}
        self.client.logger = Mock()

    def test_tag_cache_distinguishes_flat_and_hierarchical_tags(self):
        self.client._make_request = Mock(
            return_value=FakeResponse(
                [
                    {"id": "flat", "name": "general", "value": "general"},
                    {
                        "id": "child",
                        "name": "general",
                        "value": "content-rating/general",
                        "parentId": "parent",
                    },
                ]
            )
        )

        self.client.get_all_tags(use_cache=True)

        self.assertEqual(self.client._tag_cache["general"].id, "flat")
        self.assertEqual(
            self.client._tag_cache["content-rating/general"].id,
            "child",
        )

    def test_child_tag_creation_uses_parent_id_and_full_path_cache(self):
        parent = Tag(id="parent", name="content-rating", value="content-rating")
        self.client.get_all_tags = Mock(return_value=[])
        self.client._make_request = Mock(
            return_value=FakeResponse(
                {
                    "id": "child",
                    "name": "general",
                    "value": "content-rating/general",
                    "parentId": "parent",
                }
            )
        )

        tag = self.client.get_or_create_child_tag(parent, "general")

        self.assertEqual(tag.path, "content-rating/general")
        self.client._make_request.assert_called_once_with(
            method="POST",
            endpoint="/api/tags",
            json_data={"name": "general", "parentId": "parent"},
        )
        self.assertIs(
            self.client._tag_cache["content-rating/general"],
            tag,
        )

    def test_migration_moves_all_associations_before_deleting_flat_tag(self):
        source = Tag(id="flat-general", name="general")
        destination = Tag(
            id="child-general",
            name="general",
            value="content-rating/general",
            parentId="parent",
        )
        self.client.get_asset_ids_with_tag = Mock(
            return_value=["asset-1", "asset-2", "asset-3"]
        )
        self.client.bulk_tag_assets = Mock()
        self.client.bulk_untag_assets = Mock()
        self.client.delete_tag = Mock()
        self.client.invalidate_tag_cache = Mock()

        migrated = self.client.migrate_tag(source, destination, batch_size=2)

        self.assertEqual(migrated, 3)
        self.assertEqual(
            self.client.bulk_tag_assets.call_args_list[0].args,
            (["asset-1", "asset-2"], ["child-general"]),
        )
        self.assertEqual(
            self.client.bulk_tag_assets.call_args_list[1].args,
            (["asset-3"], ["child-general"]),
        )
        self.assertEqual(
            self.client.bulk_untag_assets.call_args_list[0].args,
            ("flat-general", ["asset-1", "asset-2"]),
        )
        self.client.delete_tag.assert_called_once_with("flat-general")
        self.client.invalidate_tag_cache.assert_called_once_with()

    def test_migration_with_no_associations_only_deletes_flat_tag(self):
        source = Tag(id="flat-general", name="general")
        destination = Tag(
            id="child-general",
            name="general",
            value="content-rating/general",
            parentId="parent",
        )
        self.client.get_asset_ids_with_tag = Mock(return_value=[])
        self.client.bulk_tag_assets = Mock()
        self.client.bulk_untag_assets = Mock()
        self.client.delete_tag = Mock()
        self.client.invalidate_tag_cache = Mock()

        migrated = self.client.migrate_tag(source, destination)

        self.assertEqual(migrated, 0)
        self.client.bulk_tag_assets.assert_not_called()
        self.client.bulk_untag_assets.assert_not_called()
        self.client.delete_tag.assert_called_once_with("flat-general")

    def test_single_asset_untag_uses_current_bulk_endpoint(self):
        self.client.bulk_untag_assets = Mock()

        self.client.remove_tags_from_asset("asset-1", ["tag-1", "tag-2"])

        self.assertEqual(
            self.client.bulk_untag_assets.call_args_list[0].args,
            ("tag-1", ["asset-1"]),
        )
        self.assertEqual(
            self.client.bulk_untag_assets.call_args_list[1].args,
            ("tag-2", ["asset-1"]),
        )


if __name__ == "__main__":
    unittest.main()
