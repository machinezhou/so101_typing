from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def resolve_artifact(dataset_root: Path, item: dict, key: str) -> Path | None:
    value = item.get(key)
    if not value:
        return None
    recorded = Path(str(value))
    if recorded.exists():
        return recorded
    fallback = dataset_root / "by_label" / str(item["label"]) / recorded.name
    return fallback if fallback.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge labeled glyph datasets into one portable dataset."
    )
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    merged: list[dict] = []

    for dataset_root in args.datasets:
        payload = json.loads(
            (dataset_root / "dataset.json").read_text(encoding="utf-8")
        )
        for item in payload["items"]:
            label = str(item["label"]).upper()
            class_dir = args.output / "by_label" / label
            class_dir.mkdir(parents=True, exist_ok=True)

            processed_src = resolve_artifact(
                dataset_root,
                item,
                "processed_glyph",
            )
            if processed_src is None:
                raise SystemExit(
                    f"Cannot resolve processed glyph for "
                    f"{item['sample']} {label}"
                )

            stem = f"{item['sample']}__{processed_src.name}"
            processed_dst = class_dir / stem
            if processed_dst.exists():
                raise SystemExit(f"Duplicate output path: {processed_dst}")
            shutil.copy2(processed_src, processed_dst)

            new_item = dict(item)
            new_item["label"] = label
            new_item["processed_glyph"] = processed_dst.name

            oriented_src = resolve_artifact(
                dataset_root,
                item,
                "oriented_crop",
            )
            if oriented_src is not None:
                oriented_dst = class_dir / f"{item['sample']}__{oriented_src.name}"
                if oriented_dst != processed_dst:
                    shutil.copy2(oriented_src, oriented_dst)
                new_item["oriented_crop"] = oriented_dst.name

            merged.append(new_item)

    counts = Counter(item["label"] for item in merged)
    report = {
        "count": len(merged),
        "per_label": dict(sorted(counts.items())),
        "items": merged,
    }
    (args.output / "dataset.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] merged {len(merged)} glyphs")
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
