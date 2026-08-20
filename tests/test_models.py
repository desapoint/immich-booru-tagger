import unittest
from datetime import datetime, timezone

from immich_tagger.models import Asset


def asset_payload(duration):
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    return {
        "id": "asset-1",
        "type": "IMAGE",
        "originalPath": "/library/asset-1.jpg",
        "originalFileName": "asset-1.jpg",
        "fileCreatedAt": timestamp,
        "fileModifiedAt": timestamp,
        "updatedAt": timestamp,
        "checksum": "checksum",
        "ownerId": "owner-id",
        "duration": duration,
    }


class AssetModelTests(unittest.TestCase):
    def test_numeric_duration_is_normalized_to_string(self):
        self.assertEqual(Asset(**asset_payload(0)).duration, "0")
        self.assertEqual(Asset(**asset_payload(12.5)).duration, "12.5")

    def test_string_and_null_durations_remain_supported(self):
        self.assertEqual(Asset(**asset_payload("00:00:00")).duration, "00:00:00")
        self.assertIsNone(Asset(**asset_payload(None)).duration)


if __name__ == "__main__":
    unittest.main()
