from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_summary(run_dir: Path, dataset_episode_index: int) -> tuple[Path, dict]:
    matches: list[tuple[Path, dict]] = []
    for path in sorted(run_dir.glob("attempt_*/summary.json")):
        payload = _read_json(path)
        if not payload.get("accepted"):
            continue
        if int(payload.get("dataset_episode_index", -1)) == dataset_episode_index:
            matches.append((path, payload))
    if not matches:
        raise FileNotFoundError(
            f"No accepted summary with dataset_episode_index={dataset_episode_index} under {run_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple accepted summaries map to dataset_episode_index={dataset_episode_index}: "
            + ", ".join(str(p) for p, _ in matches)
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export any recorded episode's ARM/start baseline as an explicit robot joint-pose config. "
            "The utility has no HOME/start-class semantics; the caller decides what the pose means."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True, help="dataset_episode_index")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="robot_pose")
    args = parser.parse_args()

    summary_path, summary = _find_summary(args.run_dir, args.episode)
    action = summary.get("arm_baseline_sent_action")
    if not isinstance(action, dict) or not action:
        raise RuntimeError(f"Summary has no arm_baseline_sent_action: {summary_path}")

    payload = {
        "schema": "so101_typing.robot_joint_pose.v1",
        "name": str(args.name),
        "units": "degrees",
        "action": {str(k): float(v) for k, v in action.items()},
        "source": {
            "summary_json": str(summary_path),
            "dataset_episode_index": int(args.episode),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("pose config:", args.output)
    print("source     :", summary_path)
    print("episode    :", args.episode)
    print("motors     :", len(payload["action"]))


if __name__ == "__main__":
    main()
