from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from so101_typing.perception.glyph_runtime import (
    GlyphPrediction,
    RuntimeHOGGlyphRecognizer,
)
from so101_typing.perception.glyphs import preprocess_glyph
from so101_typing.perception.keycaps import (
    KeycapCandidate,
    KeycapDetectionConfig,
    detect_keycaps,
    rectify_keycap,
)


@dataclass(frozen=True, slots=True)
class KeycapGlyphObservation:
    candidate_index: int
    center_px: tuple[float, float]
    center_norm: tuple[float, float]
    error_norm: tuple[float, float]
    bbox: tuple[int, int, int, int]
    quad: np.ndarray
    prediction: GlyphPrediction
    rectangularity: float

    def to_dict(self) -> dict:
        return {
            "candidate_index": self.candidate_index,
            "center_px": list(self.center_px),
            "center_norm": list(self.center_norm),
            "error_norm": list(self.error_norm),
            "bbox": list(self.bbox),
            "quad": np.asarray(self.quad).tolist(),
            "prediction": self.prediction.to_dict(),
            "rectangularity": self.rectangularity,
        }


@dataclass(frozen=True, slots=True)
class TargetObservation:
    target_label: str
    found: bool
    center_px: tuple[float, float] | None = None
    center_norm: tuple[float, float] | None = None
    error_norm: tuple[float, float] | None = None
    bbox: tuple[int, int, int, int] | None = None
    quad: np.ndarray | None = None
    candidate_index: int | None = None
    vote_fraction: float = 0.0
    similarity: float = 0.0
    similarity_threshold: float = 0.0
    quality_score: float = 0.0

    @classmethod
    def missing(cls, target_label: str) -> "TargetObservation":
        return cls(target_label=target_label, found=False)

    def to_dict(self) -> dict:
        return {
            "target_label": self.target_label,
            "found": self.found,
            "center_px": None if self.center_px is None else list(self.center_px),
            "center_norm": None if self.center_norm is None else list(self.center_norm),
            "error_norm": None if self.error_norm is None else list(self.error_norm),
            "bbox": None if self.bbox is None else list(self.bbox),
            "quad": None if self.quad is None else np.asarray(self.quad).tolist(),
            "candidate_index": self.candidate_index,
            "vote_fraction": self.vote_fraction,
            "similarity": self.similarity,
            "similarity_threshold": self.similarity_threshold,
            "quality_score": self.quality_score,
        }


def observe_glyph_candidates(
    frame_bgr: np.ndarray,
    recognizer: RuntimeHOGGlyphRecognizer,
    keycap_config: KeycapDetectionConfig | None = None,
) -> list[KeycapGlyphObservation]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 BGR image")

    height, width = frame_bgr.shape[:2]
    candidates = detect_keycaps(frame_bgr, keycap_config)
    observations: list[KeycapGlyphObservation] = []

    for index, candidate in enumerate(candidates):
        crop = rectify_keycap(frame_bgr, candidate, (64, 64))
        glyph = preprocess_glyph(crop)
        prediction = recognizer.predict(glyph)
        center_x, center_y = candidate.center
        center_norm = (center_x / width, center_y / height)
        error_norm = (
            (center_x - width / 2.0) / (width / 2.0),
            (center_y - height / 2.0) / (height / 2.0),
        )
        observations.append(
            KeycapGlyphObservation(
                candidate_index=index,
                center_px=(center_x, center_y),
                center_norm=center_norm,
                error_norm=error_norm,
                bbox=candidate.bbox,
                quad=candidate.quad.copy(),
                prediction=prediction,
                rectangularity=float(candidate.rectangularity),
            )
        )

    return observations


def select_target_observation(
    target_label: str,
    observations: list[KeycapGlyphObservation],
) -> TargetObservation:
    target = str(target_label).strip().upper()
    if len(target) != 1 or not target.isalpha():
        raise ValueError("target_label must be one A-Z letter")

    matches = [
        observation
        for observation in observations
        if observation.prediction.accepted
        and observation.prediction.label == target
    ]
    if not matches:
        return TargetObservation.missing(target)

    best = max(
        matches,
        key=lambda observation: (
            observation.prediction.quality_score,
            observation.prediction.similarity_margin,
            observation.rectangularity,
        ),
    )
    prediction = best.prediction
    return TargetObservation(
        target_label=target,
        found=True,
        center_px=best.center_px,
        center_norm=best.center_norm,
        error_norm=best.error_norm,
        bbox=best.bbox,
        quad=best.quad.copy(),
        candidate_index=best.candidate_index,
        vote_fraction=prediction.vote_fraction,
        similarity=prediction.similarity,
        similarity_threshold=prediction.similarity_threshold,
        quality_score=prediction.quality_score,
    )


def observe_target(
    frame_bgr: np.ndarray,
    target_label: str,
    recognizer: RuntimeHOGGlyphRecognizer,
    keycap_config: KeycapDetectionConfig | None = None,
) -> TargetObservation:
    observations = observe_glyph_candidates(
        frame_bgr,
        recognizer,
        keycap_config,
    )
    return select_target_observation(target_label, observations)


def draw_glyph_observations(
    frame_bgr: np.ndarray,
    observations: list[KeycapGlyphObservation],
    target_label: str | None = None,
) -> np.ndarray:
    result = frame_bgr.copy()
    target = None if target_label is None else target_label.strip().upper()

    for observation in observations:
        prediction = observation.prediction
        quad = np.rint(observation.quad).astype(np.int32)
        is_target = (
            target is not None
            and prediction.accepted
            and prediction.label == target
        )
        if is_target:
            color = (0, 255, 255)
            thickness = 2
        elif prediction.accepted:
            color = (0, 220, 0)
            thickness = 1
        else:
            color = (110, 110, 110)
            thickness = 1

        cv2.polylines(result, [quad], True, color, thickness, cv2.LINE_AA)
        x, y, _, _ = observation.bbox
        status = prediction.label if prediction.accepted else f"?{prediction.label}"
        text = f"{status} v{prediction.vote_fraction:.2f} s{prediction.similarity:.2f}"
        cv2.putText(
            result,
            text,
            (x, max(12, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )

    return result
