import io
import os
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image


os.environ.setdefault("IMMICH_BASE_URL", "http://immich.test")
os.environ.setdefault("IMMICH_API_KEY", "test-api-key")

from immich_tagger.config import settings
from immich_tagger.tagging_engine import WD14ONNXTaggingEngine


def image_bytes(color=(255, 0, 0, 255), size=(2, 1)):
    image = Image.new("RGBA", size, color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakeONNXModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.batch_sizes = []

    def run(self, output_names, inputs):
        batch = next(iter(inputs.values()))
        self.batch_sizes.append(len(batch))
        return [np.stack([self.scores] * len(batch))]


class WD14ONNXTaggingEngineTests(unittest.TestCase):
    def setUp(self):
        self.original_confidence = settings.confidence_threshold
        self.original_character = settings.character_threshold
        settings.confidence_threshold = 0.35
        settings.character_threshold = 0.9
        self.addCleanup(
            setattr,
            settings,
            "confidence_threshold",
            self.original_confidence,
        )
        self.addCleanup(
            setattr,
            settings,
            "character_threshold",
            self.original_character,
        )

        self.engine = WD14ONNXTaggingEngine.__new__(WD14ONNXTaggingEngine)
        self.engine.logger = Mock()
        self.engine.model_target_size = 2
        self.engine.input_name = "input"
        self.engine.output_name = "output"
        self.engine.inference_batch_size = 2
        self.engine.tag_names = [
            "general",
            "sensitive",
            "questionable",
            "explicit",
            "blue_hair",
            "character_name",
        ]
        self.engine.rating_indexes = [0, 1, 2, 3]
        self.engine.general_indexes = [4]
        self.engine.character_indexes = [5]

    def test_batching_preserves_raw_names_thresholds_and_highest_rating(self):
        self.engine.model = FakeONNXModel(
            [0.1, 0.2, 0.8, 0.3, 0.7, 0.95]
        )

        results = self.engine.predict_tags_batch(
            [image_bytes(), image_bytes(), image_bytes()]
        )

        self.assertEqual(self.engine.model.batch_sizes, [2, 1])
        self.assertEqual(len(results), 3)
        self.assertEqual(
            [prediction.name for prediction in results[0]],
            ["character_name", "questionable", "blue_hair"],
        )
        inference_logs = [
            call.args[0]
            for call in self.engine.logger.info.call_args_list
        ]
        self.assertEqual(len(inference_logs), 2)
        self.assertTrue(
            inference_logs[0].startswith(
                "🧠 Inference batch 1/2 complete: 2 images in "
            )
        )
        self.assertTrue(
            inference_logs[1].startswith(
                "🧠 Inference batch 2/2 complete: 1 image in "
            )
        )

    def test_transparent_pixels_are_composited_on_white_and_converted_to_bgr(self):
        prepared = self.engine._prepare_image(
            image_bytes(color=(255, 0, 0, 0), size=(1, 1))
        )

        self.assertEqual(prepared.shape, (2, 2, 3))
        np.testing.assert_array_equal(prepared[0, 0], [255.0, 255.0, 255.0])


if __name__ == "__main__":
    unittest.main()
