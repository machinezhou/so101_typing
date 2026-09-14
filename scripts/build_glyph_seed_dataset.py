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


def make_sheet(items: list[tuple[str, np.ndarray]], tile: int = 80, columns: int = 10) -> np.ndarray:
    if not items:
        return np.full((tile, tile * columns, 3), 255, dtype=np.uint8)

    rows = (len(items) + columns - 1) // columns
    sheet = np.full((rows * tile, columns * tile, 3), 255, dtype=np.uint8)

    for i, (label, crop) in enumerate(items):
        row, col = divmod(i, columns)
        preview = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        x, y = col * tile + 8, row * tile + 8
        sheet[y:y + 64, x:x + 64] = preview
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
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/perception/glyph_seed_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2_glyph_seed"),
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = payload["samples"]
    args.output.mkdir(parents=True, exist_ok=True)

    frame_cache: dict[str, np.ndarray] = {}
    candidate_cache = {}
    exported = []
    sheet_items: list[tuple[str, np.ndarray]] = []

    for item in entries:
        label = str(item["label"]).upper()
        sample = str(item["sample"])
        index = int(item["candidate_index"])

        image_path = args.sample_root / sample / "wrist_raw.jpg"
        if sample not in frame_cache:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise SystemExit(f"Cannot read: {image_path}")
            frame_cache[sample] = frame
            candidate_cache[sample] = detect_keycaps(frame)

        frame = frame_cache[sample]
        candidates = candidate_cache[sample]
        if not 0 <= index < len(candidates):
            raise SystemExit(
                f"{sample}: candidate {index} is out of range; "
                f"detector produced {len(candidates)} candidates"
            )

        rectified = rectify_keycap(frame, candidates[index], (64, 64))
        oriented = orient_glyph_crop(rectified, 1)
        processed = preprocess_glyph(rectified, GlyphPreprocessConfig())

        class_dir = args.output / "by_label" / label
        class_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{sample}__candidate_{index:02d}"
        oriented_path = class_dir / f"{stem}.jpg"
        processed_path = class_dir / f"{stem}__gray.png"
        cv2.imwrite(str(oriented_path), oriented)
        cv2.imwrite(str(processed_path), processed)

        exported.append(
            {
                "label": label,
                "sample": sample,
                "candidate_index": index,
                "oriented_crop": str(oriented_path),
                "processed_glyph": str(processed_path),
            }
        )
        sheet_items.append((label, oriented))

    counts = Counter(item["label"] for item in exported)
    report = {
        "count": len(exported),
        "per_label": dict(sorted(counts.items())),
        "items": exported,
    }
    (args.output / "dataset.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    cv2.imwrite(str(args.output / "glyph_seed_sheet.jpg"), make_sheet(sheet_items))

    print(f"[OK] exported {len(exported)} labeled glyph crops")
    print("Per label:")
    print(" ".join(f"{label}:{counts.get(label, 0)}" for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
