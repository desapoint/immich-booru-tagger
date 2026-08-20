import os
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock


os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.config import settings
from immich_tagger.immich_client import ImmichAPIError, ImmichClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def asset_payload(asset_id):
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    return {
        "id": asset_id,
        "type": "IMAGE",
        "originalPath": f"/library/{asset_id}.jpg",
        "originalFileName": f"{asset_id}.jpg",
        "fileCreatedAt": timestamp,
        "fileModifiedAt": timestamp,
        "updatedAt": timestamp,
        "checksum": asset_id,
        "ownerId": "owner-id",
        "libraryId": None,
    }


def metadata_response(items, next_page=None):
    return {
        "albums": {"items": [], "total": 0},
        "assets": {
            "items": items,
            "total": len(items),
            "nextPage": next_page,
        },
    }


class AlbumFilteringTests(unittest.TestCase):
    def setUp(self):
        self.original_target_albums = settings.target_albums
        self.addCleanup(
            setattr,
            settings,
            "target_albums",
            self.original_target_albums,
        )

        self.client = ImmichClient.__new__(ImmichClient)
        self.client.current_library = {
            "name": "Library_1",
            "api_key": "test-api-key",
        }
        self.client.logger = Mock()

    def test_processed_assets_are_excluded_when_search_results_omit_tags(self):
        settings.target_albums = "Anime"
        calls = []

        def make_request(method, endpoint, params=None, json_data=None):
            calls.append((method, endpoint, json_data))
            if endpoint == "/api/albums":
                return FakeResponse([
                    {"id": "anime-album", "albumName": "Anime"}
                ])
            if json_data.get("tagIds") == ["processed-tag"]:
                # Metadata search commonly omits the embedded tags. The ID is
                # still returned because Immich performed the tag filtering.
                return FakeResponse(metadata_response([asset_payload("done")]))
            return FakeResponse(
                metadata_response([
                    asset_payload("done"),
                    asset_payload("pending"),
                ])
            )

        self.client._make_request = make_request

        assets = self.client.get_unprocessed_assets(
            processed_tag_id="processed-tag",
            limit=10,
        )

        self.assertEqual([asset.id for asset in assets], ["pending"])
        tag_search = calls[1][2]
        self.assertEqual(tag_search["albumIds"], ["anime-album"])
        self.assertEqual(tag_search["tagIds"], ["processed-tag"])

    def test_missing_album_is_reported_while_valid_album_is_processed(self):
        settings.target_albums = "Missing, Anime"

        def make_request(method, endpoint, params=None, json_data=None):
            if endpoint == "/api/albums":
                return FakeResponse([
                    {"id": "anime-album", "albumName": "Anime"}
                ])
            if json_data.get("tagIds"):
                return FakeResponse(metadata_response([]))
            return FakeResponse(metadata_response([asset_payload("pending")]))

        self.client._make_request = make_request

        assets = self.client.get_unprocessed_assets(
            processed_tag_id="processed-tag",
            limit=10,
        )

        self.assertEqual([asset.id for asset in assets], ["pending"])
        self.client.logger.error.assert_called_once()
        self.client.logger.warning.assert_called_once()

    def test_processed_and_album_searches_are_fully_paginated(self):
        settings.target_albums = "Anime"

        def make_request(method, endpoint, params=None, json_data=None):
            if endpoint == "/api/albums":
                return FakeResponse([
                    {"id": "anime-album", "albumName": "Anime"}
                ])

            page = json_data["page"]
            if json_data.get("tagIds"):
                if page == 1:
                    return FakeResponse(
                        metadata_response([asset_payload("done-1")], next_page="2")
                    )
                return FakeResponse(metadata_response([asset_payload("done-2")]))

            if page == 1:
                return FakeResponse(
                    metadata_response([asset_payload("done-1")], next_page="2")
                )
            if page == 2:
                return FakeResponse(
                    metadata_response([asset_payload("done-2")], next_page="3")
                )
            return FakeResponse(metadata_response([asset_payload("pending")]))

        self.client._make_request = make_request

        assets = self.client.get_unprocessed_assets(
            processed_tag_id="processed-tag",
            limit=1,
        )

        self.assertEqual([asset.id for asset in assets], ["pending"])

    def test_all_missing_albums_fail_the_library_run(self):
        settings.target_albums = "Missing, Also Missing"
        self.client._make_request = Mock(
            return_value=FakeResponse([
                {"id": "anime-album", "albumName": "Anime"}
            ])
        )

        with self.assertRaisesRegex(
            ImmichAPIError,
            "None of the configured target albums were found",
        ):
            self.client.get_unprocessed_assets(
                processed_tag_id="processed-tag",
                limit=10,
            )

        self.assertEqual(self.client.logger.error.call_count, 2)

    def test_no_album_mode_forwards_the_requested_limit(self):
        settings.target_albums = ""
        self.client.get_untagged_assets = Mock(return_value=[])

        self.client.get_unprocessed_assets(
            processed_tag_id="processed-tag",
            limit=17,
        )

        self.client.get_untagged_assets.assert_called_once_with(limit=17)


if __name__ == "__main__":
    unittest.main()
