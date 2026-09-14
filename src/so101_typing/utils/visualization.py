from __future__ import annotations

import cv2
import numpy as np


def fit_panel(
    image: np.ndarray,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    source_height, source_width = image.shape[:2]

    scale = min(
        width / source_width,
        height / source_height,
    )

    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    x = (width - resized_width) // 2
    y = (height - resized_height) // 2

    canvas[
        y : y + resized_height,
        x : x + resized_width,
    ] = resized

    return canvas


def label_panel(
    image: np.ndarray,
    label: str,
) -> np.ndarray:
    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 38),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        label,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


def placeholder(
    text: str,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        image,
        text,
        (30, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return image
