from __future__ import annotations

import json
from pathlib import Path


TARGETS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_target(target: str) -> str:
    value = str(target).strip().upper()
    if value not in TARGETS:
        raise ValueError(f"Unsupported formal target: {target!r}")
    return value


def target_run_dirs(
    runs_root: Path,
    target: str,
) -> list[Path]:
    target = normalize_target(target)
    runs = sorted(
        path
        for path in Path(runs_root).glob(f"run_*_{target.lower()}")
        if path.is_dir()
    )
    if not runs:
        raise FileNotFoundError(
            f"No formal Phase 6 runs for target={target} found under {runs_root}"
        )
    return runs


def target_qc_dir(
    dataset_base: Path,
    target: str,
) -> Path:
    target = normalize_target(target)
    return Path(dataset_base) / "target_qc" / target.lower()


def collect_target_episode_sources(
    runs_root: Path,
    target: str,
) -> dict:
    """Collect every accepted episode for one target across all formal runs."""

    target = normalize_target(target)
    locations: set[tuple[str, str]] = set()
    episodes: dict[int, dict] = {}
    source_runs: list[str] = []

    for run_dir in target_run_dirs(runs_root, target):
        run_summary_path = run_dir / "run_summary.json"
        if not run_summary_path.exists():
            continue

        run_summary = _read_json(run_summary_path)
        run_target = str(run_summary["target"]).strip().upper()

        if run_target != target:
            raise RuntimeError(
                "Run target mismatch: "
                f"expected={target}, actual={run_target}, run={run_dir}"
            )

        locations.add(
            (
                str(run_summary["repo_id"]),
                str(run_summary["dataset_root"]),
            )
        )
        source_runs.append(str(run_dir))

        for summary_path in sorted(run_dir.glob("attempt_*/summary.json")):
            summary = _read_json(summary_path)

            if (
                not summary.get("accepted")
                or "dataset_episode_index" not in summary
            ):
                continue

            idx = int(summary["dataset_episode_index"])

            if idx in episodes:
                raise RuntimeError(
                    "Duplicate accepted dataset_episode_index "
                    f"{idx}: {episodes[idx]['summary_json']} and {summary_path}"
                )

            episodes[idx] = {
                "dataset_episode_index": idx,
                "episode_number": int(summary.get("episode_number", -1)),
                "run_dir": str(run_dir),
                "summary_json": str(summary_path),
            }

    if not source_runs:
        raise RuntimeError(
            f"No usable run_summary.json found for target={target}"
        )

    if len(locations) != 1:
        raise RuntimeError(
            "Target runs do not resolve to exactly one shared dataset: "
            f"{sorted(locations)}"
        )

    repo_id, dataset_root = next(iter(locations))

    return {
        "target": target,
        "repo_id": repo_id,
        "dataset_root": dataset_root,
        "run_dirs": source_runs,
        "episodes": [
            episodes[idx]
            for idx in sorted(episodes)
        ],
    }


def has_completed_validation(
    summary_path: Path,
) -> bool:
    """PASS/FAIL completes ground truth; invalid/aborted trials may retry."""

    validation_path = (
        Path(summary_path).parent
        / "takeover_validation.json"
    )

    if not validation_path.exists():
        return False

    payload = _read_json(validation_path)

    for trial in reversed(payload.get("trials", [])):
        status = str(trial.get("status", "UNKNOWN"))

        if status in {"PASS", "FAIL"}:
            return True

        if status in {
            "INVALID_REPLAY",
            "ABORTED",
            "UNKNOWN",
        }:
            continue

    return False


def should_auto_retry_validation(
    summary_path: Path,
) -> bool:
    """Return True only when default batch scheduling should auto-retry."""

    validation_path = (
        Path(summary_path).parent
        / "takeover_validation.json"
    )

    if not validation_path.exists():
        return True

    payload = _read_json(validation_path)

    for trial in reversed(payload.get("trials", [])):
        status = str(trial.get("status", "UNKNOWN"))

        if status in {
            "PASS",
            "FAIL",
            "INVALID_REPLAY",
        }:
            return False

        if status in {
            "ABORTED",
            "UNKNOWN",
        }:
            continue

    return True


def load_target_qc_candidate_entries(
    dataset_base: Path,
    target: str,
) -> list[dict]:
    target = normalize_target(target)
    path = target_qc_dir(dataset_base, target) / "qc_candidates.json"

    if not path.exists():
        raise RuntimeError(
            "Target-level replay candidate list not found: "
            f"{path}. Run scripts/phase6_episode_qc.py --target {target} first."
        )

    payload = _read_json(path)
    file_target = str(payload.get("target", "")).strip().upper()

    if file_target != target:
        raise RuntimeError(
            "Target-level QC target mismatch: "
            f"requested={target}, file={file_target}"
        )

    raw_entries = payload.get("episodes")
    if raw_entries is None:
        raise RuntimeError(
            "Target-level candidate file has no episode source locations. "
            "Re-run target-level QC with the current code."
        )

    entries: list[dict] = []
    seen: set[int] = set()

    for item in raw_entries:
        idx = int(item["dataset_episode_index"])

        if idx in seen:
            raise RuntimeError(
                f"Duplicate target-level QC candidate: {idx}"
            )
        seen.add(idx)

        summary_path = Path(item["summary_json"])
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)

        entries.append(
            {
                "dataset_episode_index": idx,
                "episode_number": int(item.get("episode_number", -1)),
                "run_dir": str(item["run_dir"]),
                "summary_json": str(summary_path),
            }
        )

    entries.sort(
        key=lambda item: int(item["dataset_episode_index"])
    )
    return entries
