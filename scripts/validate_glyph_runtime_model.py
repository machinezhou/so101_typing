from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from so101_typing.perception.glyph_runtime import RuntimeHOGGlyphRecognizer
from so101_typing.perception.wrist_target import observe_glyph_candidates


def find_frame(sample: str, roots: list[Path]) -> Path | None:
    for root in roots:
        path = root / sample / "wrist_raw.jpg"
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Runtime wiring validation: for each source frame in dataset.json, "
            "check whether every labeled letter present in that frame is found "
            "by the persisted detector+recognizer pipeline. This is not a held-out "
            "accuracy estimate because the model was trained on the same corpus."
        )
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--sample-root",
        action="append",
        type=Path,
        required=True,
        help="May be supplied more than once, e.g. round-1 and round-2 roots.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2_runtime_validation"),
    )
    args = parser.parse_args()

    payload = json.loads(
        (args.dataset_root / "dataset.json").read_text(encoding="utf-8")
    )
    truth: dict[str, set[str]] = defaultdict(set)
    for item in payload["items"]:
        truth[str(item["sample"])].add(str(item["label"]).upper())

    recognizer = RuntimeHOGGlyphRecognizer.load(args.model)
    args.output.mkdir(parents=True, exist_ok=True)

    frames = []
    total_expected = 0
    total_found = 0
    missing_samples = []

    for sample in sorted(truth):
        image_path = find_frame(sample, args.sample_root)
        if image_path is None:
            missing_samples.append(sample)
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise SystemExit(f"Cannot read: {image_path}")
        observations = observe_glyph_candidates(frame, recognizer)
        found_labels = {
            obs.prediction.label
            for obs in observations
            if obs.prediction.accepted
        }
        expected = truth[sample]
        hit = expected & found_labels
        missing = sorted(expected - found_labels)
        total_expected += len(expected)
        total_found += len(hit)

        frames.append(
            {
                "sample": sample,
                "expected_labels": sorted(expected),
                "found_expected_labels": sorted(hit),
                "missing_expected_labels": missing,
                "candidate_count": len(observations),
                "accepted_candidate_count": sum(
                    obs.prediction.accepted for obs in observations
                ),
            }
        )
        print(
            f"[OK] {sample}: {len(hit)}/{len(expected)} expected labels found"
            + ("" if not missing else f"; missing {','.join(missing)}")
        )

    recall = total_found / total_expected if total_expected else 0.0
    report = {
        "note": (
            "Training-frame runtime wiring recall; not a held-out accuracy metric."
        ),
        "expected": total_expected,
        "found": total_found,
        "recall": recall,
        "missing_sample_directories": missing_samples,
        "frames": frames,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"[OK] runtime wiring recall: {total_found}/{total_expected} "
        f"= {100.0 * recall:.1f}%"
    )
    if missing_samples:
        print(f"[WARN] missing sample directories: {', '.join(missing_samples)}")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
