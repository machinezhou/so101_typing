from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from so101_typing.perception.glyph_runtime import (
    GlyphAcceptanceConfig,
    RuntimeHOGGlyphRecognizer,
)


def resolve_processed_image(dataset_root: Path, item: dict) -> Path:
    recorded = Path(str(item["processed_glyph"]))
    if recorded.exists():
        return recorded

    fallback = dataset_root / "by_label" / str(item["label"]) / recorded.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Cannot resolve processed glyph for {item['label']} {item['sample']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and persist the Phase-2 WRIST HOG+SVM runtime model."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/glyph_hog_svm"),
    )
    parser.add_argument("--min-vote-fraction", type=float, default=7.0 / 9.0)
    parser.add_argument("--similarity-quantile", type=float, default=0.05)
    parser.add_argument("--similarity-slack", type=float, default=0.03)
    parser.add_argument("--min-similarity-floor", type=float, default=0.70)
    args = parser.parse_args()

    payload = json.loads(
        (args.dataset_root / "dataset.json").read_text(encoding="utf-8")
    )
    labels = []
    images = []
    for item in payload["items"]:
        path = resolve_processed_image(args.dataset_root, item)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"Cannot read: {path}")
        labels.append(str(item["label"]).upper())
        images.append(image)

    acceptance = GlyphAcceptanceConfig(
        min_vote_fraction=float(args.min_vote_fraction),
        similarity_quantile=float(args.similarity_quantile),
        similarity_slack=float(args.similarity_slack),
        min_similarity_floor=float(args.min_similarity_floor),
    )
    recognizer = RuntimeHOGGlyphRecognizer.fit(
        labels,
        images,
        acceptance=acceptance,
        augment=True,
    )
    recognizer.save(args.output)

    accepted = sum(recognizer.predict(image).accepted for image in images)
    print(f"[OK] trained {len(images)} glyphs across {len(recognizer.classes)} classes")
    print(
        f"[OK] training-set acceptance sanity: {accepted}/{len(images)} "
        f"= {100.0 * accepted / len(images):.1f}%"
    )
    print(f"[OK] model: {args.output.resolve()}")
    print("[OK] files: svm.xml model.json centroids.npz")


if __name__ == "__main__":
    main()
