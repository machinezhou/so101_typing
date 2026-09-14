from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from so101_typing.perception.glyphs import (
    GlyphPreprocessConfig,
    orient_glyph_crop,
    preprocess_glyph,
)
from so101_typing.perception.keycaps import detect_keycaps, rectify_keycap


def normalized_center(candidate, width: int, height: int) -> np.ndarray:
    return np.asarray(
        [candidate.center[0] / width, candidate.center[1] / height],
        dtype=np.float32,
    )


def match_anchor(
    candidates,
    anchor_norm: tuple[float, float] | list[float],
    width: int,
    height: int,
    max_distance: float,
    excluded: set[int] | None = None,
) -> tuple[int, float]:
    """Match a labeled anchor to the nearest detected candidate center.

    Normalized centers are used instead of brittle candidate indices because
    OpenCV contour ordering can differ slightly across versions/platforms.
    """

    if not candidates:
        raise ValueError("no keycap candidates available")

    excluded = excluded or set()
    anchor = np.asarray(anchor_norm, dtype=np.float32)
    choices: list[tuple[float, int]] = []
    for index, candidate in enumerate(candidates):
        if index in excluded:
            continue
        distance = float(
            np.linalg.norm(normalized_center(candidate, width, height) - anchor)
        )
        choices.append((distance, index))

    if not choices:
        raise ValueError("no unmatched keycap candidates remain")

    distance, index = min(choices)
    if distance > float(max_distance):
        raise ValueError(
            f"nearest candidate is too far from anchor: "
            f"{distance:.4f} > {max_distance:.4f}"
        )
    return index, distance


def make_sheet(
    items: list[tuple[str, np.ndarray]],
    tile: int = 80,
    columns: int = 10,
) -> np.ndarray:
    rows = max(1, (len(items) + columns - 1) // columns)
    sheet = np.full((rows * tile, columns * tile, 3), 255, np.uint8)
    for i, (label, crop) in enumerate(items):
        row, col = divmod(i, columns)
        preview = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        x, y = col * tile + 8, row * tile + 8
        sheet[y : y + 64, x : x + 64] = preview
        cv2.putText(
            sheet,
            label,
            (col * tile + 2, row * tile + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the labeled Phase-2 round-2 WRIST glyph dataset."
    )
    parser.add_argument("sample_root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/perception/glyph_round2_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2_glyph_round2"),
    )
    parser.add_argument(
        "--max-anchor-distance",
        type=float,
        default=0.035,
        help="Maximum normalized image-center distance for anchor matching.",
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = payload["samples"]
    args.output.mkdir(parents=True, exist_ok=True)

    by_sample: dict[str, list[dict]] = {}
    for item in entries:
        by_sample.setdefault(str(item["sample"]), []).append(item)

    exported: list[dict] = []
    sheet_items: list[tuple[str, np.ndarray]] = []
    match_distances: list[float] = []

    for sample, sample_entries in sorted(by_sample.items()):
        image_path = args.sample_root / sample / "wrist_raw.jpg"
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise SystemExit(f"Cannot read: {image_path}")

        height, width = frame.shape[:2]
        candidates = detect_keycaps(frame)
        used: set[int] = set()

        for item in sample_entries:
            label = str(item["label"]).upper()
            try:
                index, distance = match_anchor(
                    candidates,
                    item["anchor_norm"],
                    width,
                    height,
                    args.max_anchor_distance,
                    excluded=used,
                )
            except ValueError as exc:
                raise SystemExit(
                    f"{sample} {label}: anchor match failed: {exc}"
                ) from exc

            used.add(index)
            match_distances.append(distance)
            rectified = rectify_keycap(frame, candidates[index], (64, 64))
            oriented = orient_glyph_crop(rectified, 1)
            processed = preprocess_glyph(
                rectified,
                GlyphPreprocessConfig(),
            )

            class_dir = args.output / "by_label" / label
            class_dir.mkdir(parents=True, exist_ok=True)
            anchor_x, anchor_y = map(float, item["anchor_norm"])
            stem = (
                f"{sample}__anchor_"
                f"{anchor_x:.6f}_{anchor_y:.6f}"
            )
            oriented_path = class_dir / f"{stem}.jpg"
            processed_path = class_dir / f"{stem}__gray.png"
            cv2.imwrite(str(oriented_path), oriented)
            cv2.imwrite(str(processed_path), processed)

            exported.append(
                {
                    "label": label,
                    "sample": sample,
                    "candidate_index": index,
                    "anchor_norm": [anchor_x, anchor_y],
                    "anchor_match_distance": distance,
                    "oriented_crop": oriented_path.name,
                    "processed_glyph": processed_path.name,
                }
            )
            sheet_items.append((label, oriented))

    counts = Counter(item["label"] for item in exported)
    report = {
        "count": len(exported),
        "per_label": dict(sorted(counts.items())),
        "max_anchor_match_distance": max(match_distances, default=0.0),
        "items": exported,
    }
    (args.output / "dataset.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    cv2.imwrite(
        str(args.output / "glyph_round2_sheet.jpg"),
        make_sheet(sheet_items),
    )

    print(f"[OK] exported {len(exported)} labeled round-2 glyph crops")
    print(
        "[OK] max anchor match distance: "
        f"{report['max_anchor_match_distance']:.4f}"
    )
    print(
        "Per label: "
        + " ".join(
            f"{label}:{counts.get(label, 0)}"
            for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
    )
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
