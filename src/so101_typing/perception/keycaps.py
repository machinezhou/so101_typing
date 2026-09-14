from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class KeycapDetectionConfig:
    threshold: int = 105

    min_area: float = 250.0
    max_area: float = 9000.0

    min_aspect_ratio: float = 0.45
    max_aspect_ratio: float = 2.20

    morphology_kernel: int = 3


@dataclass(frozen=True, slots=True)
class KeycapCandidate:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: float
    aspect_ratio: float
    rectangularity: float
    quad: np.ndarray


def detect_keycaps(
    frame_bgr: np.ndarray,
    config: KeycapDetectionConfig | None = None,
) -> list[KeycapCandidate]:
    config = config or KeycapDetectionConfig()

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    _, binary = cv2.threshold(
        gray,
        config.threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )

    kernel_size = max(1, int(config.morphology_kernel))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[KeycapCandidate] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if not config.min_area <= area <= config.max_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        if width <= 0 or height <= 0:
            continue

        aspect_ratio = width / height

        if not (
            config.min_aspect_ratio
            <= aspect_ratio
            <= config.max_aspect_ratio
        ):
            continue

        rectangularity = area / float(width * height)

        if rectangularity < 0.45:
            continue

        rect = cv2.minAreaRect(contour)

        quad = cv2.boxPoints(rect).astype(np.float32)

        candidates.append(
            KeycapCandidate(
                bbox=(x, y, width, height),
                center=(
                    x + width / 2.0,
                    y + height / 2.0,
                ),
                area=area,
                aspect_ratio=aspect_ratio,
                rectangularity=rectangularity,
                quad=quad,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.center[1],
            candidate.center[0],
        )
    )

    return candidates


def draw_keycap_candidates(
    frame_bgr: np.ndarray,
    candidates: list[KeycapCandidate],
) -> np.ndarray:
    result = frame_bgr.copy()

    for index, candidate in enumerate(candidates):
        x, y, width, height = candidate.bbox

        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            1,
        )

        center = (
            int(candidate.center[0]),
            int(candidate.center[1]),
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
            (x, max(12, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return result
