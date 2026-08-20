"""Batched ONNX inference for anime-style image tagging."""

import csv
import io
import os
from typing import List

import numpy as np
from PIL import Image

from .config import settings
from .logging import get_logger
from .models import TagPrediction


MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"
RATING_CATEGORY = 9
GENERAL_CATEGORY = 0
CHARACTER_CATEGORY = 4


class TaggingEngineError(Exception):
    """Raised when the tagging model cannot load or perform inference."""


class BaseTaggingEngine:
    """Base interface for image tagging engines."""

    def __init__(self):
        self.logger = get_logger("tagging_engine")

    def predict_tags(self, image_data: bytes) -> List[TagPrediction]:
        """Predict tags for one image."""
        return self.predict_tags_batch([image_data])[0]

    def predict_tags_batch(
        self,
        images: List[bytes],
    ) -> List[List[TagPrediction]]:
        """Predict tags for multiple images."""
        raise NotImplementedError


class WD14ONNXTaggingEngine(BaseTaggingEngine):
    """Run the official WD v3 ONNX model with dynamic batching."""

    def __init__(self):
        super().__init__()
        self.model_name = settings.wd_model_repo
        self.inference_batch_size = settings.inference_batch_size
        self.logger.info(f"🤖 Loading ONNX AI model: {self.model_name}")
        self._load_model()

    def _load_model(self):
        """Download the model assets and initialize ONNX Runtime."""
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            token = os.environ.get("HF_TOKEN") or None
            common_download_args = {
                "repo_id": self.model_name,
                "cache_dir": settings.model_cache_dir,
                "token": token,
            }
            labels_path = hf_hub_download(
                filename=LABEL_FILENAME,
                **common_download_args,
            )
            model_path = hf_hub_download(
                filename=MODEL_FILENAME,
                **common_download_args,
            )

            self._load_labels(labels_path)

            session_options = ort.SessionOptions()
            if settings.onnx_intra_op_threads:
                session_options.intra_op_num_threads = settings.onnx_intra_op_threads
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            self.model = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            model_input = self.model.get_inputs()[0]
            model_output = self.model.get_outputs()[0]
            input_shape = model_input.shape

            if (
                len(input_shape) != 4
                or not isinstance(input_shape[1], int)
                or not isinstance(input_shape[2], int)
                or input_shape[1] != input_shape[2]
            ):
                raise TaggingEngineError(
                    f"Unsupported WD ONNX input shape: {input_shape}"
                )

            self.model_target_size = input_shape[1]
            self.input_name = model_input.name
            self.output_name = model_output.name
            self.logger.info(
                "✅ ONNX AI model loaded successfully "
                f"({len(self.tag_names)} labels, batch size "
                f"{self.inference_batch_size})"
            )
        except TaggingEngineError:
            raise
        except ImportError as e:
            raise TaggingEngineError(
                "ONNX dependencies are missing; install onnxruntime and "
                "huggingface-hub"
            ) from e
        except Exception as e:
            raise TaggingEngineError(f"Failed to load WD ONNX model: {e}") from e

    def _load_labels(self, labels_path: str):
        """Load label names and their WD category indexes from CSV."""
        tag_names = []
        rating_indexes = []
        general_indexes = []
        character_indexes = []

        with open(labels_path, newline="", encoding="utf-8") as labels_file:
            reader = csv.DictReader(labels_file)
            if not reader.fieldnames or not {"name", "category"}.issubset(
                reader.fieldnames
            ):
                raise TaggingEngineError(
                    "selected_tags.csv is missing name/category columns"
                )

            for index, row in enumerate(reader):
                name = row.get("name", "").strip()
                try:
                    category = int(row.get("category", ""))
                except (TypeError, ValueError) as e:
                    raise TaggingEngineError(
                        f"Invalid label category at row {index + 2}"
                    ) from e

                if not name:
                    raise TaggingEngineError(
                        f"Empty label name at row {index + 2}"
                    )

                tag_names.append(name)
                if category == RATING_CATEGORY:
                    rating_indexes.append(index)
                elif category == GENERAL_CATEGORY:
                    general_indexes.append(index)
                elif category == CHARACTER_CATEGORY:
                    character_indexes.append(index)

        if not tag_names or not rating_indexes:
            raise TaggingEngineError(
                "selected_tags.csv does not contain WD labels and ratings"
            )

        self.tag_names = tag_names
        self.rating_indexes = rating_indexes
        self.general_indexes = general_indexes
        self.character_indexes = character_indexes

    def _prepare_image(self, image_data: bytes) -> np.ndarray:
        """Match the official WD ONNX RGBA, padding, resize, and BGR path."""
        with Image.open(io.BytesIO(image_data)) as source:
            source.load()
            image = source.convert("RGBA")

        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        width, height = image.size
        max_dim = max(width, height)
        pad_left = (max_dim - width) // 2
        pad_top = (max_dim - height) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_left, pad_top))

        if max_dim != self.model_target_size:
            padded = padded.resize(
                (self.model_target_size, self.model_target_size),
                Image.Resampling.BICUBIC,
            )

        rgb = np.asarray(padded, dtype=np.float32)
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def _predictions_from_scores(
        self,
        scores: np.ndarray,
    ) -> List[TagPrediction]:
        """Apply the same WD category thresholds and highest-rating rule."""
        if len(scores) != len(self.tag_names):
            raise TaggingEngineError(
                "ONNX output label count does not match selected_tags.csv: "
                f"{len(scores)} != {len(self.tag_names)}"
            )

        predictions = []

        for index in self.general_indexes:
            confidence = float(scores[index])
            if confidence >= settings.confidence_threshold:
                predictions.append(
                    TagPrediction(
                        name=self.tag_names[index],
                        confidence=confidence,
                    )
                )

        for index in self.character_indexes:
            confidence = float(scores[index])
            if confidence >= max(
                settings.confidence_threshold,
                settings.character_threshold,
            ):
                predictions.append(
                    TagPrediction(
                        name=self.tag_names[index],
                        confidence=confidence,
                    )
                )

        rating_index = max(
            self.rating_indexes,
            key=lambda index: float(scores[index]),
        )
        rating_confidence = float(scores[rating_index])
        if rating_confidence >= settings.confidence_threshold:
            predictions.append(
                TagPrediction(
                    name=self.tag_names[rating_index],
                    confidence=rating_confidence,
                )
            )

        predictions.sort(key=lambda prediction: prediction.confidence, reverse=True)
        return predictions

    def predict_tags_batch(
        self,
        images: List[bytes],
    ) -> List[List[TagPrediction]]:
        """Evaluate images in bounded dynamic ONNX batches."""
        if not images:
            return []

        try:
            results = []
            for offset in range(0, len(images), self.inference_batch_size):
                image_batch = images[offset:offset + self.inference_batch_size]
                model_input = np.stack(
                    [self._prepare_image(image_data) for image_data in image_batch]
                )
                model_output = self.model.run(
                    [self.output_name],
                    {self.input_name: model_input},
                )[0]

                if len(model_output) != len(image_batch):
                    raise TaggingEngineError(
                        "ONNX output batch size does not match input batch size"
                    )

                results.extend(
                    self._predictions_from_scores(scores)
                    for scores in model_output
                )

            self.logger.debug(
                "WD ONNX batch predictions",
                images=len(images),
                inference_batch_size=self.inference_batch_size,
                total_predictions=sum(len(result) for result in results),
            )
            return results
        except TaggingEngineError:
            raise
        except Exception as e:
            raise TaggingEngineError(f"WD ONNX prediction failed: {e}") from e


def create_tagging_engine() -> BaseTaggingEngine:
    """Create the configured tagging engine."""
    if settings.tagging_model != "wd14":
        raise ValueError(f"Unsupported tagging model: {settings.tagging_model}")
    return WD14ONNXTaggingEngine()
