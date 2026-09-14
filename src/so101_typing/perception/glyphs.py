from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class GlyphPreprocessConfig:
    """Preprocessing for WRIST key glyph recognition.

    The current frozen WRIST/tool mounting makes glyphs appear 90 degrees
    clockwise after geometric keycap rectification.  Normalize that mounting
    offset with one fixed 90-degree counter-clockwise rotation.

    Do not make the classifier invariant to arbitrary 90-degree rotations:
    doing so creates semantic ambiguity for classes such as M/W.
    """

    rotation_quadrants_ccw: int = 1
    crop_margin: int = 5
    output_size: tuple[int, int] = (32, 32)
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 4


def orient_glyph_crop(
    crop_bgr: np.ndarray,
    rotation_quadrants_ccw: int = 1,
) -> np.ndarray:
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        raise ValueError("crop_bgr must be an HxWx3 BGR image")

    turns = int(rotation_quadrants_ccw) % 4
    if turns == 0:
        return crop_bgr.copy()
    return np.ascontiguousarray(np.rot90(crop_bgr, k=turns))


def preprocess_glyph(
    crop_bgr: np.ndarray,
    config: GlyphPreprocessConfig | None = None,
) -> np.ndarray:
    """Return a normalized uint8 grayscale glyph image."""

    config = config or GlyphPreprocessConfig()
    oriented = orient_glyph_crop(
        crop_bgr,
        config.rotation_quadrants_ccw,
    )
    gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)

    margin = max(0, int(config.crop_margin))
    if margin * 2 >= min(gray.shape[:2]):
        raise ValueError("crop_margin is too large for the input crop")
    if margin:
        gray = gray[margin:-margin, margin:-margin]

    grid = max(1, int(config.clahe_grid_size))
    gray = cv2.createCLAHE(
        clipLimit=float(config.clahe_clip_limit),
        tileGridSize=(grid, grid),
    ).apply(gray)

    width, height = map(int, config.output_size)
    if width <= 1 or height <= 1:
        raise ValueError("output_size must be larger than 1x1")

    return cv2.resize(
        gray,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
