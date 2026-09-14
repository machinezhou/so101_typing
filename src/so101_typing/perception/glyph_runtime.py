from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from so101_typing.perception.glyph_baselines import (
    HOGConfig,
    HOGLinearSVMGlyphClassifier,
    augment_glyph,
)


@dataclass(frozen=True, slots=True)
class GlyphAcceptanceConfig:
    """Acceptance policy for the persisted WRIST glyph recognizer.

    `vote_fraction` is not a probability.  It measures prediction stability
    under the same small pose jitters used during training.  `similarity` is
    cosine similarity to the predicted class's HOG centroid.
    """

    min_vote_fraction: float = 7.0 / 9.0
    similarity_quantile: float = 0.05
    similarity_slack: float = 0.03
    min_similarity_floor: float = 0.70


@dataclass(frozen=True, slots=True)
class GlyphPrediction:
    label: str
    accepted: bool
    vote_fraction: float
    similarity: float
    similarity_threshold: float
    quality_score: float
    votes: dict[str, int]

    @property
    def similarity_margin(self) -> float:
        return self.similarity - self.similarity_threshold

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "accepted": self.accepted,
            "vote_fraction": self.vote_fraction,
            "similarity": self.similarity,
            "similarity_threshold": self.similarity_threshold,
            "similarity_margin": self.similarity_margin,
            "quality_score": self.quality_score,
            "votes": dict(self.votes),
        }


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.zeros_like(vector)
    return vector / norm


def _config_from_json(payload: dict) -> HOGConfig:
    tuple_fields = {"win_size", "block_size", "block_stride", "cell_size"}
    kwargs = {
        key: tuple(value) if key in tuple_fields else value
        for key, value in payload.items()
    }
    return HOGConfig(**kwargs)


class RuntimeHOGGlyphRecognizer:
    """Persistable HOG+linear-SVM recognizer with conservative rejection."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        classifier: HOGLinearSVMGlyphClassifier,
        classes: list[str],
        centroids: np.ndarray,
        similarity_thresholds: dict[str, float],
        acceptance: GlyphAcceptanceConfig | None = None,
        training_count: int = 0,
    ) -> None:
        if not classes:
            raise ValueError("classes must not be empty")
        if centroids.shape != (len(classes), classifier.descriptor_size):
            raise ValueError(
                "centroids shape must be "
                f"({len(classes)}, {classifier.descriptor_size})"
            )

        self.classifier = classifier
        self.classes = list(classes)
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.similarity_thresholds = {
            label: float(similarity_thresholds[label])
            for label in self.classes
        }
        self.acceptance = acceptance or GlyphAcceptanceConfig()
        self.training_count = int(training_count)
        self._class_to_index = {
            label: index for index, label in enumerate(self.classes)
        }

    @classmethod
    def fit(
        cls,
        labels: list[str],
        images: list[np.ndarray],
        *,
        hog_config: HOGConfig | None = None,
        acceptance: GlyphAcceptanceConfig | None = None,
        augment: bool = True,
    ) -> "RuntimeHOGGlyphRecognizer":
        if len(labels) != len(images) or not images:
            raise ValueError("labels/images must be non-empty and have equal length")

        acceptance = acceptance or GlyphAcceptanceConfig()
        classifier = HOGLinearSVMGlyphClassifier(hog_config)
        classifier.fit(labels, images, augment=augment)
        classes = sorted(set(map(str, labels)))

        normalized_features: dict[str, list[np.ndarray]] = {
            label: [] for label in classes
        }
        for label, image in zip(labels, images):
            normalized_features[str(label)].append(
                _normalize_feature(classifier.feature(image))
            )

        centroids = []
        thresholds: dict[str, float] = {}
        for label in classes:
            features = np.stack(normalized_features[label]).astype(np.float32)
            centroid = _normalize_feature(features.mean(axis=0))
            similarities = features @ centroid
            threshold = float(
                np.quantile(
                    similarities,
                    float(acceptance.similarity_quantile),
                )
                - float(acceptance.similarity_slack)
            )
            threshold = max(
                float(acceptance.min_similarity_floor),
                threshold,
            )
            centroids.append(centroid)
            thresholds[label] = min(1.0, threshold)

        return cls(
            classifier=classifier,
            classes=classes,
            centroids=np.stack(centroids).astype(np.float32),
            similarity_thresholds=thresholds,
            acceptance=acceptance,
            training_count=len(images),
        )

    def predict(self, image: np.ndarray) -> GlyphPrediction:
        variants = augment_glyph(image)
        labels = [self.classifier.predict(variant) for variant in variants]
        votes = Counter(labels)

        original_label = labels[0]
        top_count = max(votes.values())
        tied = sorted(
            label for label, count in votes.items() if count == top_count
        )
        label = original_label if original_label in tied else tied[0]
        vote_fraction = float(votes[label]) / float(len(labels))

        feature = _normalize_feature(self.classifier.feature(image))
        centroid = self.centroids[self._class_to_index[label]]
        similarity = float(feature @ centroid)
        threshold = float(self.similarity_thresholds[label])

        accepted = (
            vote_fraction >= float(self.acceptance.min_vote_fraction)
            and similarity >= threshold
        )

        # A bounded ranking score for choosing between duplicate observations.
        similarity_quality = float(
            np.clip(
                (similarity - threshold) / max(1e-6, 1.0 - threshold),
                0.0,
                1.0,
            )
        )
        quality_score = 0.65 * vote_fraction + 0.35 * similarity_quality

        return GlyphPrediction(
            label=label,
            accepted=accepted,
            vote_fraction=vote_fraction,
            similarity=similarity,
            similarity_threshold=threshold,
            quality_score=quality_score,
            votes=dict(sorted(votes.items())),
        )

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.classifier._svm is None:  # noqa: SLF001 - persisted wrapper owns it.
            raise RuntimeError("classifier is not fitted")

        svm_path = model_dir / "svm.xml"
        self.classifier._svm.save(str(svm_path))  # noqa: SLF001
        np.savez_compressed(model_dir / "centroids.npz", centroids=self.centroids)

        metadata = {
            "schema_version": self.SCHEMA_VERSION,
            "classes": self.classes,
            "hog_config": asdict(self.classifier.config),
            "acceptance": asdict(self.acceptance),
            "similarity_thresholds": self.similarity_thresholds,
            "training_count": self.training_count,
        }
        (model_dir / "model.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> "RuntimeHOGGlyphRecognizer":
        model_dir = Path(model_dir)
        metadata = json.loads(
            (model_dir / "model.json").read_text(encoding="utf-8")
        )
        if int(metadata.get("schema_version", -1)) != cls.SCHEMA_VERSION:
            raise ValueError("unsupported glyph runtime model schema")

        hog_config = _config_from_json(metadata["hog_config"])
        classifier = HOGLinearSVMGlyphClassifier(hog_config)
        classifier._svm = cv2.ml.SVM_load(str(model_dir / "svm.xml"))  # noqa: SLF001
        classifier._classes = list(metadata["classes"])  # noqa: SLF001

        with np.load(model_dir / "centroids.npz") as payload:
            centroids = payload["centroids"].astype(np.float32)

        return cls(
            classifier=classifier,
            classes=list(metadata["classes"]),
            centroids=centroids,
            similarity_thresholds={
                str(key): float(value)
                for key, value in metadata["similarity_thresholds"].items()
            },
            acceptance=GlyphAcceptanceConfig(**metadata["acceptance"]),
            training_count=int(metadata.get("training_count", 0)),
        )
