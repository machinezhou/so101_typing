from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from so101_typing.perception.glyph_runtime import RuntimeHOGGlyphRecognizer
from so101_typing.perception.wrist_target import (
    draw_glyph_observations,
    observe_glyph_candidates,
    select_target_observation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline WRIST runtime perception inspection on saved samples."
    )
    parser.add_argument("sample_root", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2_runtime_inspect"),
    )
    args = parser.parse_args()

    target = None if args.target is None else args.target.strip().upper()
    if target is not None and (len(target) != 1 or not target.isalpha()):
        raise SystemExit("--target must be one A-Z letter")

    recognizer = RuntimeHOGGlyphRecognizer.load(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"target": target, "frames": []}

    sample_dirs = sorted(path for path in args.sample_root.iterdir() if path.is_dir())
    for sample_dir in sample_dirs:
        image_path = sample_dir / "wrist_raw.jpg"
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        observations = observe_glyph_candidates(frame, recognizer)
        target_observation = (
            None
            if target is None
            else select_target_observation(target, observations)
        )
        overlay = draw_glyph_observations(frame, observations, target)
        cv2.imwrite(str(args.output / f"{sample_dir.name}.jpg"), overlay)

        accepted = [
            obs for obs in observations if obs.prediction.accepted
        ]
        print(
            f"[OK] {sample_dir.name}: {len(observations)} candidates, "
            f"{len(accepted)} accepted"
            + (
                ""
                if target_observation is None
                else f", target {target}: {'FOUND' if target_observation.found else 'missing'}"
            )
        )
        report["frames"].append(
            {
                "sample": sample_dir.name,
                "candidate_count": len(observations),
                "accepted_count": len(accepted),
                "target_observation": (
                    None
                    if target_observation is None
                    else target_observation.to_dict()
                ),
                "observations": [obs.to_dict() for obs in observations],
            }
        )

    (args.output / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
