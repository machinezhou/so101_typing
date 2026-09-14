from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np


def _zscore(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    return (x - float(x.mean())) / (float(x.std()) + 1e-6)


def shifted_normalized_correlation(
    query: np.ndarray,
    template: np.ndarray,
    max_shift: int = 2,
) -> float:
    """Best normalized correlation under a small x/y translation."""

    if query.ndim != 2 or template.ndim != 2:
        raise ValueError("query and template must be grayscale images")
    if query.shape != template.shape:
        raise ValueError("query and template must have the same shape")

    shift = max(0, int(max_shift))
    q = _zscore(query)
    h, w = query.shape
    padded = cv2.copyMakeBorder(
        template,
        shift,
        shift,
        shift,
        shift,
        cv2.BORDER_REPLICATE,
    )

    best = -np.inf
    for dy in range(2 * shift + 1):
        for dx in range(2 * shift + 1):
            candidate = _zscore(padded[dy : dy + h, dx : dx + w])
            best = max(best, float(np.mean(q * candidate)))
    return float(best)


class TemplateGlyphClassifier:
    """Nearest-template baseline with a small translation search."""

    def __init__(self, max_shift: int = 2) -> None:
        self.max_shift = int(max_shift)
        self._templates: list[tuple[str, np.ndarray]] = []

    def fit(self, labels: list[str], images: list[np.ndarray]) -> None:
        if len(labels) != len(images) or not images:
            raise ValueError("labels/images must be non-empty and have equal length")
        self._templates = [
            (str(label), image.copy())
            for label, image in zip(labels, images)
        ]

    def predict_scores(self, image: np.ndarray) -> dict[str, float]:
        if not self._templates:
            raise RuntimeError("classifier is not fitted")

        scores: dict[str, float] = {}
        for label, template in self._templates:
            score = shifted_normalized_correlation(
                image,
                template,
                self.max_shift,
            )
            scores[label] = max(scores.get(label, -np.inf), score)
        return scores

    def predict(self, image: np.ndarray) -> str:
        scores = self.predict_scores(image)
        return max(scores, key=scores.get)


@dataclass(frozen=True, slots=True)
class HOGConfig:
    win_size: tuple[int, int] = (32, 32)
    block_size: tuple[int, int] = (16, 16)
    block_stride: tuple[int, int] = (8, 8)
    cell_size: tuple[int, int] = (8, 8)
    bins: int = 9
    svm_c: float = 10.0


class HOGLinearSVMGlyphClassifier:
    """OpenCV HOG + linear C-SVC baseline.

    OpenCV is used deliberately so this checkpoint does not add a scikit-learn
    dependency to the robot project.
    """

    def __init__(self, config: HOGConfig | None = None) -> None:
        self.config = config or HOGConfig()
        self._hog = cv2.HOGDescriptor(
            self.config.win_size,
            self.config.block_size,
            self.config.block_stride,
            self.config.cell_size,
            int(self.config.bins),
        )
        self._svm = None
        self._classes: list[str] = []

    @property
    def descriptor_size(self) -> int:
        return int(self._hog.getDescriptorSize())

    def feature(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError("HOG input must be grayscale")
        expected = (self.config.win_size[1], self.config.win_size[0])
        if image.shape != expected:
            raise ValueError(
                f"HOG input must have shape {expected}, got {image.shape}"
            )
        return self._hog.compute(image).reshape(-1).astype(np.float32)

    def fit(
        self,
        labels: list[str],
        images: list[np.ndarray],
        augment: bool = True,
    ) -> None:
        if len(labels) != len(images) or not images:
            raise ValueError("labels/images must be non-empty and have equal length")

        self._classes = sorted(set(map(str, labels)))
        if len(self._classes) < 2:
            raise ValueError("HOG+SVM needs at least two classes")

        class_to_id = {label: i for i, label in enumerate(self._classes)}
        features: list[np.ndarray] = []
        targets: list[int] = []

        for label, image in zip(labels, images):
            variants = augment_glyph(image) if augment else [image]
            for variant in variants:
                features.append(self.feature(variant))
                targets.append(class_to_id[str(label)])

        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setC(float(self.config.svm_c))
        svm.setTermCriteria(
            (cv2.TERM_CRITERIA_MAX_ITER, 10000, 1e-6)
        )
        svm.train(
            np.stack(features).astype(np.float32),
            cv2.ml.ROW_SAMPLE,
            np.asarray(targets, dtype=np.int32),
        )
        self._svm = svm

    def predict(self, image: np.ndarray) -> str:
        if self._svm is None:
            raise RuntimeError("classifier is not fitted")
        _, prediction = self._svm.predict(self.feature(image)[None, :])
        return self._classes[int(prediction[0, 0])]


def augment_glyph(image: np.ndarray) -> list[np.ndarray]:
    """Small pose jitter only; never arbitrary 90-degree rotations."""

    if image.ndim != 2:
        raise ValueError("augmentation input must be grayscale")

    h, w = image.shape
    center = (w / 2.0, h / 2.0)
    output = [image.copy()]

    for angle in (-5.0, -2.5, 2.5, 5.0):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        output.append(
            cv2.warpAffine(
                image,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )

    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        output.append(
            cv2.warpAffine(
                image,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )

    return output
