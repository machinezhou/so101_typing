from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class KeycapDetectionConfig:
    """Geometry-first wrist keycap detector.

    The Phase-1 detector used a fixed grayscale threshold.  Real wrist frames
    show that thresholding is brittle under viewpoint/exposure changes, so the
    Phase-2 foundation uses keycap border structure (CLAHE + Canny) and then
    rejects implausible rotated rectangles.
    """

    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    blur_kernel: int = 5

    canny_low_ratio: float = 0.50
    canny_high_ratio: float = 1.30
    canny_low_floor: int = 20
    canny_high_ceiling: int = 220

    morphology_kernel: int = 3

    min_area: float = 250.0
    max_area: float = 6000.0
    min_short_side: float = 14.0
    max_long_side: float = 105.0
    max_rotated_aspect_ratio: float = 1.75
    min_rectangularity: float = 0.55

    # White robot/tool surfaces can also contain rectangular edges.  This is
    # deliberately permissive so dark keycaps are kept even under exposure
    # variation.  Glyph recognition remains responsible for semantic reject.
    max_inner_mean_gray: float = 120.0
    inner_sample_scale: float = 0.72

    nms_iou_threshold: float = 0.45


@dataclass(frozen=True, slots=True)
class KeycapCandidate:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: float
    aspect_ratio: float
    rectangularity: float
    quad: np.ndarray
    rotated_aspect_ratio: float = 1.0
    inner_mean_gray: float = 0.0


def _odd_at_least_one(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right

    x0 = max(lx, rx)
    y0 = max(ly, ry)
    x1 = min(lx + lw, rx + rw)
    y1 = min(ly + lh, ry + rh)

    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection <= 0:
        return 0.0

    union = lw * lh + rw * rh - intersection
    return float(intersection) / float(max(1, union))


def _inner_mean_gray(
    gray: np.ndarray,
    quad: np.ndarray,
    scale: float,
) -> float:
    center = quad.mean(axis=0, keepdims=True)
    inner = center + (quad - center) * float(scale)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(
        mask,
        np.rint(inner).astype(np.int32),
        255,
    )

    pixels = gray[mask > 0]
    if pixels.size == 0:
        return 255.0
    return float(pixels.mean())


def detect_keycaps(
    frame_bgr: np.ndarray,
    config: KeycapDetectionConfig | None = None,
) -> list[KeycapCandidate]:
    config = config or KeycapDetectionConfig()

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 BGR image")

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    grid = max(1, int(config.clahe_grid_size))
    normalized = cv2.createCLAHE(
        clipLimit=float(config.clahe_clip_limit),
        tileGridSize=(grid, grid),
    ).apply(gray)

    blur_kernel = _odd_at_least_one(config.blur_kernel)
    normalized = cv2.GaussianBlur(
        normalized,
        (blur_kernel, blur_kernel),
        0,
    )

    median = float(np.median(normalized))
    low = int(max(config.canny_low_floor, config.canny_low_ratio * median))
    high = int(min(config.canny_high_ceiling, config.canny_high_ratio * median))
    high = max(low + 1, high)

    edges = cv2.Canny(normalized, low, high)

    kernel_size = max(1, int(config.morphology_kernel))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    raw_candidates: list[KeycapCandidate] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not config.min_area <= area <= config.max_area:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (rect_width, rect_height), _ = rect
        short_side = min(rect_width, rect_height)
        long_side = max(rect_width, rect_height)

        if short_side <= 0 or long_side <= 0:
            continue
        if short_side < config.min_short_side:
            continue
        if long_side > config.max_long_side:
            continue

        rotated_aspect_ratio = long_side / short_side
        if rotated_aspect_ratio > config.max_rotated_aspect_ratio:
            continue

        rotated_area = rect_width * rect_height
        rectangularity = area / float(max(1.0, rotated_area))
        if rectangularity < config.min_rectangularity:
            continue

        quad = cv2.boxPoints(rect).astype(np.float32)
        inner_mean = _inner_mean_gray(
            gray,
            quad,
            config.inner_sample_scale,
        )
        if inner_mean > config.max_inner_mean_gray:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue

        raw_candidates.append(
            KeycapCandidate(
                bbox=(x, y, width, height),
                center=(float(rect[0][0]), float(rect[0][1])),
                area=area,
                aspect_ratio=float(width) / float(height),
                rectangularity=rectangularity,
                quad=quad,
                rotated_aspect_ratio=rotated_aspect_ratio,
                inner_mean_gray=inner_mean,
            )
        )

    # RETR_LIST intentionally sees both sides of a key border.  NMS collapses
    # those near-duplicate rectangles while preserving neighboring keycaps.
    raw_candidates.sort(
        key=lambda candidate: (
            candidate.area,
            candidate.rectangularity,
        ),
        reverse=True,
    )

    candidates: list[KeycapCandidate] = []
    for candidate in raw_candidates:
        if any(
            _bbox_iou(candidate.bbox, kept.bbox)
            >= config.nms_iou_threshold
            for kept in candidates
        ):
            continue
        candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            candidate.center[1],
            candidate.center[0],
        )
    )
    return candidates


def order_quad_points(quad: np.ndarray) -> np.ndarray:
    """Return four points in TL, TR, BR, BL image order."""

    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def rectify_keycap(
    frame_bgr: np.ndarray,
    candidate: KeycapCandidate,
    output_size: tuple[int, int] = (64, 64),
    expand: float = 1.04,
) -> np.ndarray:
    """Perspective-normalize one candidate.

    This normalizes keycap geometry, not semantic glyph orientation.  A glyph
    recognizer may evaluate 0/90/180/270-degree rotations without using a
    pre-programmed keyboard layout.
    """

    width, height = int(output_size[0]), int(output_size[1])
    if width <= 1 or height <= 1:
        raise ValueError("output_size must be larger than 1x1")
    if expand <= 0:
        raise ValueError("expand must be positive")

    quad = np.asarray(candidate.quad, dtype=np.float32).reshape(4, 2)
    center = quad.mean(axis=0, keepdims=True)
    quad = center + (quad - center) * float(expand)
    src = order_quad_points(quad)

    dst = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        frame_bgr,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def draw_keycap_candidates(
    frame_bgr: np.ndarray,
    candidates: list[KeycapCandidate],
) -> np.ndarray:
    result = frame_bgr.copy()

    for index, candidate in enumerate(candidates):
        quad = np.rint(candidate.quad).astype(np.int32)
        cv2.polylines(
            result,
            [quad],
            True,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        center = (
            int(round(candidate.center[0])),
            int(round(candidate.center[1])),
        )
        cv2.circle(
            result,
            center,
            2,
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            result,
            str(index),
            (center[0] + 3, center[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return result
