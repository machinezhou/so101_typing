from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from so101_typing.perception.keycaps import (
    detect_keycaps,
    draw_keycap_candidates,
    rectify_keycap,
)


def make_crop_sheet(
    frame: np.ndarray,
    candidates,
    tile_size: int = 80,
    columns: int = 8,
) -> np.ndarray:
    if not candidates:
        return np.full((tile_size, tile_size * columns, 3), 255, dtype=np.uint8)

    rows = (len(candidates) + columns - 1) // columns
    sheet = np.full((rows * tile_size, columns * tile_size, 3), 255, dtype=np.uint8)

    crop_size = max(16, tile_size - 16)
    for index, candidate in enumerate(candidates):
        crop = rectify_keycap(frame, candidate, (crop_size, crop_size))
        row = index // columns
        col = index % columns
        x = col * tile_size + 8
        y = row * tile_size + 8
        sheet[y : y + crop_size, x : x + crop_size] = crop
        cv2.putText(
            sheet,
            str(index),
            (col * tile_size + 2, row * tile_size + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return sheet


def find_wrist_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("wrist_raw.jpg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase2_keycap_eval"))
    args = parser.parse_args()

    images = find_wrist_images(args.input)
    if not images:
        raise SystemExit(f"No wrist_raw.jpg images found under: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    report = []

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[SKIP] cannot read {image_path}")
            continue

        candidates = detect_keycaps(frame)
        overlay = draw_keycap_candidates(frame, candidates)
        sheet = make_crop_sheet(frame, candidates)

        sample_name = image_path.parent.name if image_path.name == "wrist_raw.jpg" else image_path.stem
        sample_dir = args.output / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(sample_dir / "overlay.jpg"), overlay)
        cv2.imwrite(str(sample_dir / "key_crops.jpg"), sheet)

        item = {
            "sample": sample_name,
            "input": str(image_path),
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "index": i,
                    "center": [round(c.center[0], 2), round(c.center[1], 2)],
                    "area": round(c.area, 2),
                    "rotated_aspect_ratio": round(c.rotated_aspect_ratio, 3),
                    "rectangularity": round(c.rectangularity, 3),
                    "inner_mean_gray": round(c.inner_mean_gray, 2),
                }
                for i, c in enumerate(candidates)
            ],
        }
        report.append(item)
        print(f"[OK] {sample_name}: {len(candidates)} candidates")

    (args.output / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
