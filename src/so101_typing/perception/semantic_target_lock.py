from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np


def _centers_array(
    centers: Sequence[Sequence[float]],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        centers,
        dtype=np.float64,
    )

    if (
        array.ndim != 2
        or array.shape[1:] != (2,)
    ):
        raise ValueError(
            f"{name} must have shape (N, 2)"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} must contain only finite values"
        )

    return array


@dataclass(frozen=True, slots=True)
class SemanticTargetLockConfig:
    min_centers: int = 4
    max_pair_distance_px: float = 18.0
    inlier_residual_px: float = 4.0

    min_tracking_inliers: int = 8
    max_tracking_median_residual_px: float = 2.5
    # Maximum propagated target motion between two committed
    # visual references.
    max_translation_norm_px: float = 25.0

    # Similarity-transform safety gates.
    # Geometry fallback is allowed to model:
    #   translation + small rotation + small uniform scale.
    max_scale_change_fraction: float = 0.08
    max_rotation_deg: float = 5.0

    # Legacy direct mutual-NN matching used an 18 px pair gate.
    # Search a wider coarse translation basin first so a larger
    # frame-to-frame image jump can still enter the correct basin.
    coarse_translation_search_px: float = 35.0

    def __post_init__(self) -> None:
        if self.min_centers < 2:
            raise ValueError(
                "min_centers must be at least 2"
            )

        if self.min_tracking_inliers < 2:
            raise ValueError(
                "min_tracking_inliers must be at least 2"
            )

        for name in (
            "max_pair_distance_px",
            "inlier_residual_px",
            "max_tracking_median_residual_px",
            "max_translation_norm_px",
            "max_scale_change_fraction",
            "max_rotation_deg",
            "coarse_translation_search_px",
        ):
            value = float(
                getattr(self, name)
            )

            if (
                not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and positive"
                )


@dataclass(frozen=True, slots=True)
class KeyboardTranslationEstimate:
    translation_px: tuple[float, float]
    matches: int
    inliers: int
    median_residual_px: float
    max_residual_px: float



@dataclass(frozen=True, slots=True)
class KeyboardSimilarityEstimate:
    """Robust keyboard similarity transform.

    Row-vector convention:

        p_current = scale * (p_previous @ R) + translation
    """

    scale: float
    rotation_deg: float
    translation_px: tuple[float, float]

    matches: int
    inliers: int
    median_residual_px: float
    max_residual_px: float

    def transform_point(
        self,
        point: Sequence[float],
    ) -> tuple[float, float]:
        p = np.asarray(
            point,
            dtype=np.float64,
        )

        if (
            p.shape != (2,)
            or not np.isfinite(p).all()
        ):
            raise ValueError(
                "point must contain two finite values"
            )

        theta = math.radians(
            self.rotation_deg
        )

        c = math.cos(theta)
        s = math.sin(theta)

        row_rotation = np.asarray(
            [
                [c, s],
                [-s, c],
            ],
            dtype=np.float64,
        )

        translation = np.asarray(
            self.translation_px,
            dtype=np.float64,
        )

        result = (
            self.scale
            * (p @ row_rotation)
            + translation
        )

        return (
            float(result[0]),
            float(result[1]),
        )


@dataclass(frozen=True, slots=True)
class LockedTargetObservation:
    target_label: str
    found: bool
    center_px: tuple[float, float] | None
    source: str

    matches: int | None = None
    inliers: int | None = None
    median_residual_px: float | None = None

    scale: float | None = None
    rotation_deg: float | None = None


def estimate_keyboard_translation(
    previous_centers: Sequence[Sequence[float]],
    current_centers: Sequence[Sequence[float]],
    *,
    config: SemanticTargetLockConfig | None = None,
) -> KeyboardTranslationEstimate | None:
    cfg = (
        config
        or SemanticTargetLockConfig()
    )

    previous = _centers_array(
        previous_centers,
        name="previous_centers",
    )

    current = _centers_array(
        current_centers,
        name="current_centers",
    )

    if (
        len(previous) < cfg.min_centers
        or len(current) < cfg.min_centers
    ):
        return None

    distances = np.linalg.norm(
        previous[:, None, :]
        - current[None, :, :],
        axis=2,
    )

    previous_to_current = np.argmin(
        distances,
        axis=1,
    )

    current_to_previous = np.argmin(
        distances,
        axis=0,
    )

    shifts: list[np.ndarray] = []

    for previous_index, raw_current_index in enumerate(
        previous_to_current
    ):
        current_index = int(
            raw_current_index
        )

        if (
            int(
                current_to_previous[
                    current_index
                ]
            )
            != previous_index
        ):
            continue

        pair_distance = float(
            distances[
                previous_index,
                current_index,
            ]
        )

        if (
            pair_distance
            > cfg.max_pair_distance_px
        ):
            continue

        shifts.append(
            current[current_index]
            - previous[previous_index]
        )

    if len(shifts) < cfg.min_centers:
        return None

    shift_array = np.asarray(
        shifts,
        dtype=np.float64,
    )

    first_median = np.median(
        shift_array,
        axis=0,
    )

    residuals = np.linalg.norm(
        shift_array - first_median,
        axis=1,
    )

    inlier_mask = (
        residuals
        <= cfg.inlier_residual_px
    )

    if (
        int(np.count_nonzero(inlier_mask))
        < cfg.min_centers
    ):
        return None

    inlier_shifts = (
        shift_array[inlier_mask]
    )

    translation = np.median(
        inlier_shifts,
        axis=0,
    )

    final_residuals = np.linalg.norm(
        inlier_shifts - translation,
        axis=1,
    )

    return KeyboardTranslationEstimate(
        translation_px=(
            float(translation[0]),
            float(translation[1]),
        ),
        matches=len(shifts),
        inliers=len(inlier_shifts),
        median_residual_px=float(
            np.median(final_residuals)
        ),
        max_residual_px=float(
            np.max(final_residuals)
        ),
    )


def _mutual_matches_after_offset(
    previous: np.ndarray,
    current: np.ndarray,
    offset: np.ndarray,
    *,
    max_pair_distance_px: float,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    shifted = (
        previous
        + offset[None, :]
    )

    distances = np.linalg.norm(
        shifted[:, None, :]
        - current[None, :, :],
        axis=2,
    )

    previous_to_current = np.argmin(
        distances,
        axis=1,
    )

    current_to_previous = np.argmin(
        distances,
        axis=0,
    )

    pairs: list[tuple[int, int]] = []
    residuals: list[float] = []

    for previous_index, raw_current_index in enumerate(
        previous_to_current
    ):
        current_index = int(
            raw_current_index
        )

        if (
            int(
                current_to_previous[
                    current_index
                ]
            )
            != previous_index
        ):
            continue

        residual = float(
            distances[
                previous_index,
                current_index,
            ]
        )

        if residual > max_pair_distance_px:
            continue

        pairs.append(
            (
                previous_index,
                current_index,
            )
        )

        residuals.append(
            residual
        )

    return (
        pairs,
        np.asarray(
            residuals,
            dtype=np.float64,
        ),
    )


def _coarse_similarity_correspondences(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    config: SemanticTargetLockConfig,
) -> list[tuple[int, int]] | None:
    """Find a coarse image-motion basin before tight NN matching."""

    candidate_offsets = (
        current[:, None, :]
        - previous[None, :, :]
    ).reshape(
        -1,
        2,
    )

    best_pairs: list[
        tuple[int, int]
    ] | None = None

    best_score: tuple[
        int,
        float,
        float,
    ] | None = None

    for offset in candidate_offsets:
        offset_norm = float(
            np.linalg.norm(offset)
        )

        if (
            offset_norm
            > config.coarse_translation_search_px
        ):
            continue

        pairs, residuals = (
            _mutual_matches_after_offset(
                previous,
                current,
                offset,
                max_pair_distance_px=(
                    config.max_pair_distance_px
                ),
            )
        )

        if (
            len(pairs)
            < config.min_centers
        ):
            continue

        median_residual = float(
            np.median(residuals)
        )

        # Prefer:
        #   1. most consistent correspondences
        #   2. lowest residual
        #   3. smallest coarse translation
        score = (
            len(pairs),
            -median_residual,
            -offset_norm,
        )

        if (
            best_score is None
            or score > best_score
        ):
            best_score = score
            best_pairs = pairs

    return best_pairs


def _similarity_from_two_pairs(
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
] | None:
    source_vector = (
        source_b
        - source_a
    )

    target_vector = (
        target_b
        - target_a
    )

    source_norm = float(
        np.linalg.norm(
            source_vector
        )
    )

    target_norm = float(
        np.linalg.norm(
            target_vector
        )
    )

    if (
        source_norm <= 1e-9
        or target_norm <= 1e-9
    ):
        return None

    scale = (
        target_norm
        / source_norm
    )

    dot = float(
        np.dot(
            source_vector,
            target_vector,
        )
    )

    cross = float(
        source_vector[0]
        * target_vector[1]
        - source_vector[1]
        * target_vector[0]
    )

    angle = math.atan2(
        cross,
        dot,
    )

    c = math.cos(angle)
    ss = math.sin(angle)

    row_rotation = np.asarray(
        [
            [c, ss],
            [-ss, c],
        ],
        dtype=np.float64,
    )

    translation = (
        target_a
        - scale
        * (
            source_a
            @ row_rotation
        )
    )

    return (
        float(scale),
        row_rotation,
        translation,
    )


def _fit_similarity_least_squares(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
] | None:
    if (
        len(source) < 2
        or len(target) != len(source)
    ):
        return None

    source_mean = np.mean(
        source,
        axis=0,
    )

    target_mean = np.mean(
        target,
        axis=0,
    )

    source_centered = (
        source
        - source_mean
    )

    target_centered = (
        target
        - target_mean
    )

    denominator = float(
        np.sum(
            source_centered
            * source_centered
        )
    )

    if denominator <= 1e-9:
        return None

    covariance = (
        source_centered.T
        @ target_centered
    )

    u, singular_values, vt = (
        np.linalg.svd(
            covariance
        )
    )

    correction = np.ones(
        len(singular_values),
        dtype=np.float64,
    )

    if (
        np.linalg.det(
            u @ vt
        )
        < 0.0
    ):
        correction[-1] = -1.0

    row_rotation = (
        u
        @ np.diag(correction)
        @ vt
    )

    scale = float(
        np.sum(
            singular_values
            * correction
        )
        / denominator
    )

    if (
        not math.isfinite(scale)
        or scale <= 0.0
    ):
        return None

    translation = (
        target_mean
        - scale
        * (
            source_mean
            @ row_rotation
        )
    )

    return (
        scale,
        row_rotation,
        translation,
    )


def _apply_similarity(
    points: np.ndarray,
    *,
    scale: float,
    row_rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return (
        scale
        * (
            points
            @ row_rotation
        )
        + translation
    )


def estimate_keyboard_similarity(
    previous_centers: Sequence[Sequence[float]],
    current_centers: Sequence[Sequence[float]],
    *,
    config: SemanticTargetLockConfig | None = None,
) -> KeyboardSimilarityEstimate | None:
    """Estimate keyboard translation + rotation + uniform scale."""

    cfg = (
        config
        or SemanticTargetLockConfig()
    )

    previous = _centers_array(
        previous_centers,
        name="previous_centers",
    )

    current = _centers_array(
        current_centers,
        name="current_centers",
    )

    if (
        len(previous) < cfg.min_centers
        or len(current) < cfg.min_centers
    ):
        return None

    pairs = (
        _coarse_similarity_correspondences(
            previous,
            current,
            config=cfg,
        )
    )

    if (
        pairs is None
        or len(pairs) < cfg.min_centers
    ):
        return None

    source = np.asarray(
        [
            previous[previous_index]
            for previous_index, _
            in pairs
        ],
        dtype=np.float64,
    )

    target = np.asarray(
        [
            current[current_index]
            for _, current_index
            in pairs
        ],
        dtype=np.float64,
    )

    best_mask: np.ndarray | None = None

    best_score: tuple[
        int,
        float,
    ] | None = None

    # Small deterministic RANSAC-like consensus.
    for first in range(
        len(source)
    ):
        for second in range(
            first + 1,
            len(source),
        ):
            model = (
                _similarity_from_two_pairs(
                    source[first],
                    source[second],
                    target[first],
                    target[second],
                )
            )

            if model is None:
                continue

            (
                scale,
                row_rotation,
                translation,
            ) = model

            predicted = (
                _apply_similarity(
                    source,
                    scale=scale,
                    row_rotation=row_rotation,
                    translation=translation,
                )
            )

            residuals = np.linalg.norm(
                predicted - target,
                axis=1,
            )

            inlier_mask = (
                residuals
                <= cfg.inlier_residual_px
            )

            inlier_count = int(
                np.count_nonzero(
                    inlier_mask
                )
            )

            if inlier_count < 2:
                continue

            median_residual = float(
                np.median(
                    residuals[
                        inlier_mask
                    ]
                )
            )

            score = (
                inlier_count,
                -median_residual,
            )

            if (
                best_score is None
                or score > best_score
            ):
                best_score = score
                best_mask = inlier_mask

    if (
        best_mask is None
        or int(
            np.count_nonzero(
                best_mask
            )
        )
        < cfg.min_centers
    ):
        return None

    model = (
        _fit_similarity_least_squares(
            source[best_mask],
            target[best_mask],
        )
    )

    if model is None:
        return None

    (
        scale,
        row_rotation,
        translation,
    ) = model

    predicted = (
        _apply_similarity(
            source,
            scale=scale,
            row_rotation=row_rotation,
            translation=translation,
        )
    )

    residuals = np.linalg.norm(
        predicted - target,
        axis=1,
    )

    inlier_mask = (
        residuals
        <= cfg.inlier_residual_px
    )

    if (
        int(
            np.count_nonzero(
                inlier_mask
            )
        )
        < cfg.min_centers
    ):
        return None

    # Refit once using only accepted inliers.
    model = (
        _fit_similarity_least_squares(
            source[inlier_mask],
            target[inlier_mask],
        )
    )

    if model is None:
        return None

    (
        scale,
        row_rotation,
        translation,
    ) = model

    predicted = (
        _apply_similarity(
            source,
            scale=scale,
            row_rotation=row_rotation,
            translation=translation,
        )
    )

    residuals = np.linalg.norm(
        predicted - target,
        axis=1,
    )

    final_inlier_mask = (
        residuals
        <= cfg.inlier_residual_px
    )

    final_inliers = int(
        np.count_nonzero(
            final_inlier_mask
        )
    )

    if final_inliers < cfg.min_centers:
        return None

    final_residuals = (
        residuals[
            final_inlier_mask
        ]
    )

    rotation_deg = math.degrees(
        math.atan2(
            float(
                row_rotation[0, 1]
            ),
            float(
                row_rotation[0, 0]
            ),
        )
    )

    return KeyboardSimilarityEstimate(
        scale=float(scale),
        rotation_deg=float(
            rotation_deg
        ),
        translation_px=(
            float(translation[0]),
            float(translation[1]),
        ),
        matches=len(pairs),
        inliers=final_inliers,
        median_residual_px=float(
            np.median(
                final_residuals
            )
        ),
        max_residual_px=float(
            np.max(
                final_residuals
            )
        ),
    )


class SemanticTargetLock:
    """Carry one visually-established target identity through occlusion.

    Semantic recognition is the only operation that establishes target identity.
    Geometry may move that already-established identity, but it must never select
    a new key by proximity.
    """

    def __init__(
        self,
        target_label: str,
        *,
        config: SemanticTargetLockConfig | None = None,
    ) -> None:
        target = (
            str(target_label)
            .strip()
            .upper()
        )

        if (
            len(target) != 1
            or not target.isascii()
            or not target.isalpha()
        ):
            raise ValueError(
                "target_label must be one A-Z letter"
            )

        self.target_label = target

        self.config = (
            config
            or SemanticTargetLockConfig()
        )

        self._previous_centers: (
            np.ndarray | None
        ) = None

        self._target_center: (
            np.ndarray | None
        ) = None

    @property
    def locked(self) -> bool:
        return (
            self._previous_centers
            is not None
            and self._target_center
            is not None
        )

    @property
    def target_center_px(
        self,
    ) -> tuple[float, float] | None:
        if self._target_center is None:
            return None

        return (
            float(self._target_center[0]),
            float(self._target_center[1]),
        )

    def observe_semantic(
        self,
        *,
        target_center_px: Sequence[float],
        keycap_centers: Sequence[Sequence[float]],
    ) -> LockedTargetObservation:
        center = np.asarray(
            target_center_px,
            dtype=np.float64,
        )

        if (
            center.shape != (2,)
            or not np.isfinite(center).all()
        ):
            raise ValueError(
                "target_center_px must contain two finite values"
            )

        centers = _centers_array(
            keycap_centers,
            name="keycap_centers",
        )

        if (
            len(centers)
            < self.config.min_centers
        ):
            raise ValueError(
                "semantic lock requires enough visible keycaps"
            )

        self._target_center = (
            center.copy()
        )

        self._previous_centers = (
            centers.copy()
        )

        return LockedTargetObservation(
            target_label=self.target_label,
            found=True,
            center_px=(
                float(center[0]),
                float(center[1]),
            ),
            source="semantic",
        )

    def propagate_geometry(
        self,
        keycap_centers: Sequence[Sequence[float]],
    ) -> LockedTargetObservation:
        if not self.locked:
            return LockedTargetObservation(
                target_label=self.target_label,
                found=False,
                center_px=None,
                source="unlocked",
            )

        centers = _centers_array(
            keycap_centers,
            name="keycap_centers",
        )

        estimate = (
            estimate_keyboard_similarity(
                self._previous_centers,
                centers,
                config=self.config,
            )
        )

        if estimate is None:
            return LockedTargetObservation(
                target_label=self.target_label,
                found=False,
                center_px=None,
                source="geometry_rejected",
            )

        target_before = (
            self._target_center.copy()
        )

        predicted = np.asarray(
            estimate.transform_point(
                target_before
            ),
            dtype=np.float64,
        )

        target_motion_norm = float(
            np.linalg.norm(
                predicted
                - target_before
            )
        )

        scale_change = abs(
            estimate.scale
            - 1.0
        )

        accepted = (
            estimate.inliers
            >= self.config.min_tracking_inliers
            and (
                estimate.median_residual_px
                <= self.config.max_tracking_median_residual_px
            )
            and (
                target_motion_norm
                <= self.config.max_translation_norm_px
            )
            and (
                scale_change
                <= self.config.max_scale_change_fraction
            )
            and (
                abs(
                    estimate.rotation_deg
                )
                <= self.config.max_rotation_deg
            )
        )

        if not accepted:
            return LockedTargetObservation(
                target_label=self.target_label,
                found=False,
                center_px=None,
                source="geometry_rejected",
                matches=estimate.matches,
                inliers=estimate.inliers,
                median_residual_px=(
                    estimate.median_residual_px
                ),
                scale=estimate.scale,
                rotation_deg=(
                    estimate.rotation_deg
                ),
            )

        # Advance reference only after an accepted transform.
        self._previous_centers = (
            centers.copy()
        )

        self._target_center = (
            predicted.copy()
        )

        return LockedTargetObservation(
            target_label=self.target_label,
            found=True,
            center_px=(
                float(predicted[0]),
                float(predicted[1]),
            ),
            source="geometry",
            matches=estimate.matches,
            inliers=estimate.inliers,
            median_residual_px=(
                estimate.median_residual_px
            ),
            scale=estimate.scale,
            rotation_deg=(
                estimate.rotation_deg
            ),
        )

