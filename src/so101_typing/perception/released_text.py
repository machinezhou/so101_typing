from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalCleanTextResult:
    frequency: np.ndarray
    persistent_mask: np.ndarray
    selected_mask: np.ndarray
    median_roi: np.ndarray
    clean_crop: np.ndarray
    stable_bbox_xywh: tuple[int, int, int, int]
    crop_bbox_xywh: tuple[int, int, int, int]
    components: tuple[dict, ...]
    recognition_padding_px: int


def _boxes_intersect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw <= bx
        or bx + bw <= ax
        or ay + ah <= by
        or by + bh <= ay
    )


def _horizontal_gap(
    component: dict,
    selected_x0: int,
    selected_x1: int,
) -> int:
    x0 = int(component["x"])
    x1 = x0 + int(component["w"])

    if x1 < selected_x0:
        return selected_x0 - x1
    if x0 > selected_x1:
        return x0 - selected_x1
    return 0


def build_temporal_clean_text(
    change_model,
    post_rois: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    persistence_fraction: float = 0.70,
    crop_margin_px: int = 10,
    min_component_area_px: int = 10,
    event_bbox_xywh: tuple[int, int, int, int] | list[int] | None = None,
) -> TemporalCleanTextResult:
    # Difference semantics are used only to localize persistent new foreground.
    # Recognition foreground pixels come from ORIGINAL post-event ROI pixels.
    # When the triggering event bbox is available, it anchors component
    # selection so unrelated persistent occluders elsewhere in the ROI cannot
    # enlarge the OCR crop. Synthetic white padding is added only as blank OCR
    # context; it never replaces the observed glyph pixels.
    if not post_rois:
        raise ValueError("post_rois must not be empty")
    if not 0.0 < float(persistence_fraction) <= 1.0:
        raise ValueError("persistence_fraction must be in (0, 1]")
    if int(crop_margin_px) < 0:
        raise ValueError("crop_margin_px must be >= 0")
    if int(min_component_area_px) <= 0:
        raise ValueError("min_component_area_px must be > 0")

    first_shape = np.asarray(post_rois[0]).shape
    if len(first_shape) != 3 or first_shape[2] != 3:
        raise ValueError("post_rois must contain BGR images")

    rois: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for roi in post_rois:
        image = np.asarray(roi)
        if image.shape != first_shape:
            raise ValueError("all post_rois must share one shape")

        rois.append(image)
        masks.append(
            (change_model._novel_mask(image) > 0).astype(np.uint8)  # noqa: SLF001
        )

    frequency = np.mean(np.stack(masks, axis=0), axis=0)
    persistent = (
        frequency >= float(persistence_fraction)
    ).astype(np.uint8) * 255

    persistent = cv2.morphologyEx(
        persistent,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (persistent > 0).astype(np.uint8),
        connectivity=8,
    )

    candidates: list[dict] = []

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < int(min_component_area_px):
            continue

        candidates.append(
            {
                "label": int(index),
                "x": int(stats[index, cv2.CC_STAT_LEFT]),
                "y": int(stats[index, cv2.CC_STAT_TOP]),
                "w": int(stats[index, cv2.CC_STAT_WIDTH]),
                "h": int(stats[index, cv2.CC_STAT_HEIGHT]),
                "area": area,
            }
        )

    if not candidates:
        raise RuntimeError(
            "no persistent new foreground survived temporal filtering"
        )

    recognition_padding_px = 0
    context_margin = int(crop_margin_px)

    if event_bbox_xywh is None:
        selected_components = candidates
    else:
        if len(event_bbox_xywh) != 4:
            raise ValueError("event_bbox_xywh must contain exactly four values")

        event_x, event_y, event_w, event_h = (
            int(value) for value in event_bbox_xywh
        )

        if event_w <= 0 or event_h <= 0:
            raise ValueError("event bbox width/height must be positive")

        roi_h, roi_w = first_shape[:2]
        if (
            event_x < 0
            or event_y < 0
            or event_x + event_w > roi_w
            or event_y + event_h > roi_h
        ):
            raise ValueError("event bbox must lie inside the SIDE text ROI")

        event_scale = max(event_w, event_h)

        # Event-mask geometry and released-mask geometry can move by a few
        # pixels because of antialiasing and release settling. Use a small,
        # scale-derived anchor expansion instead of exact bbox equality.
        anchor_pad = max(2, int(round(0.15 * event_scale)))
        anchor_box = (
            max(0, event_x - anchor_pad),
            max(0, event_y - anchor_pad),
            min(roi_w, event_x + event_w + anchor_pad)
            - max(0, event_x - anchor_pad),
            min(roi_h, event_y + event_h + anchor_pad)
            - max(0, event_y - anchor_pad),
        )

        anchors = [
            component
            for component in candidates
            if _boxes_intersect(
                (
                    int(component["x"]),
                    int(component["y"]),
                    int(component["w"]),
                    int(component["h"]),
                ),
                anchor_box,
            )
        ]

        if not anchors:
            raise RuntimeError(
                "no persistent released-text component overlaps the "
                "triggering SIDE event"
            )

        anchor_center_y = float(
            np.median(
                [
                    float(component["y"])
                    + 0.5 * float(component["h"])
                    for component in anchors
                ]
            )
        )

        # Keep components on the same physical text row as the event. This
        # rejects persistent arm/shadow fragments below/above the glyph while
        # still allowing disconnected parts such as an i/j dot.
        vertical_tolerance_px = max(
            8.0,
            0.75 * float(event_h),
        )
        same_line = [
            component
            for component in candidates
            if abs(
                (
                    float(component["y"])
                    + 0.5 * float(component["h"])
                )
                - anchor_center_y
            )
            <= vertical_tolerance_px
        ]

        selected_labels = {
            int(component["label"])
            for component in anchors
        }

        # Preserve repeated characters (e.g. QQ) by growing horizontally from
        # the event-anchored component through nearby same-line components.
        horizontal_gap_limit_px = max(
            8,
            int(round(1.25 * float(event_scale))),
        )

        changed = True
        while changed:
            changed = False
            currently_selected = [
                component
                for component in same_line
                if int(component["label"]) in selected_labels
            ]

            selected_x0 = min(
                int(component["x"])
                for component in currently_selected
            )
            selected_x1 = max(
                int(component["x"]) + int(component["w"])
                for component in currently_selected
            )

            for component in same_line:
                label = int(component["label"])
                if label in selected_labels:
                    continue

                if (
                    _horizontal_gap(
                        component,
                        selected_x0,
                        selected_x1,
                    )
                    <= horizontal_gap_limit_px
                ):
                    selected_labels.add(label)
                    changed = True

        selected_components = [
            component
            for component in candidates
            if int(component["label"]) in selected_labels
        ]

        # With an event anchor we no longer need a large crop margin that can
        # reach backward into the previous prefix. Keep only a tiny amount of
        # observed context, then add clean white OCR padding after the crop.
        context_margin = min(
            int(crop_margin_px),
            max(2, int(round(0.05 * float(event_scale)))),
        )
        recognition_padding_px = max(
            6,
            int(round(0.30 * float(event_scale))),
        )

    selected = np.zeros_like(persistent)

    components: list[dict] = []
    for component in selected_components:
        label = int(component["label"])
        selected[labels == label] = 255
        components.append(
            {
                "x": int(component["x"]),
                "y": int(component["y"]),
                "w": int(component["w"]),
                "h": int(component["h"]),
                "area": int(component["area"]),
            }
        )

    ys, xs = np.nonzero(selected)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError(
            "no event-anchored persistent foreground survived filtering"
        )

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    median_roi = np.median(
        np.stack(rois, axis=0).astype(np.float32),
        axis=0,
    ).astype(np.uint8)

    continuation_x0 = int(change_model.continuation_x0)

    crop_x0 = max(continuation_x0, x0 - context_margin)
    crop_y0 = max(0, y0 - context_margin)
    crop_x1 = min(median_roi.shape[1], x1 + context_margin)
    crop_y1 = min(median_roi.shape[0], y1 + context_margin)

    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        raise RuntimeError("temporal clean-text crop is empty")

    clean_crop = median_roi[
        crop_y0:crop_y1,
        crop_x0:crop_x1,
    ].copy()

    if recognition_padding_px > 0:
        clean_crop = cv2.copyMakeBorder(
            clean_crop,
            recognition_padding_px,
            recognition_padding_px,
            recognition_padding_px,
            recognition_padding_px,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    return TemporalCleanTextResult(
        frequency=frequency,
        persistent_mask=persistent,
        selected_mask=selected,
        median_roi=median_roi,
        clean_crop=clean_crop,
        stable_bbox_xywh=(x0, y0, x1 - x0, y1 - y0),
        crop_bbox_xywh=(
            crop_x0,
            crop_y0,
            crop_x1 - crop_x0,
            crop_y1 - crop_y0,
        ),
        components=tuple(components),
        recognition_padding_px=int(recognition_padding_px),
    )
