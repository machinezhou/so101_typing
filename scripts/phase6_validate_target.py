from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QC_SCRIPT = ROOT / "scripts" / "phase6_episode_qc.py"
TAKEOVER_SCRIPT = (
    ROOT / "scripts" / "phase6_replay_takeover_validate.py"
)
FORMAL_DATASET_BASE = (
    ROOT / "artifacts" / "phase6_act_dataset" / "keyboard_v1"
)


def _ansi(text: str, code: str) -> str:
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ:
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(text: str) -> str:
    return _ansi(text, "1;32")


def _yellow(text: str) -> str:
    return _ansi(text, "1;33")


def _red(text: str) -> str:
    return _ansi(text, "1;31")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _qc_status_snapshot(target: str) -> dict[int, str]:
    path = (
        FORMAL_DATASET_BASE
        / "target_qc"
        / target.lower()
        / "qc_report.json"
    )
    if not path.exists():
        return {}

    try:
        payload = _read_json(path)
    except Exception:
        return {}

    return {
        int(item["dataset_episode_index"]): str(item["status"])
        for item in payload.get("episodes", [])
        if (
            "dataset_episode_index" in item
            and "status" in item
        )
    }


def _historical_takeover_status(summary_json: str) -> str:
    validation_path = (
        Path(summary_json).parent
        / "takeover_validation.json"
    )

    if not validation_path.exists():
        return "-"

    try:
        payload = _read_json(validation_path)
    except Exception:
        return "UNKNOWN"

    fallback = "-"

    for trial in reversed(payload.get("trials", [])):
        status = str(trial.get("status", "UNKNOWN"))

        if status in {
            "PASS",
            "FAIL",
            "INVALID_REPLAY",
        }:
            return status

        if fallback == "-" and status in {
            "ABORTED",
            "UNKNOWN",
        }:
            fallback = status

    return fallback


def _print_combined_summary(
    *,
    target: str,
    previous_qc_statuses: dict[int, str],
    current_batch_results: dict[int, str],
) -> None:
    qc_report_path = (
        FORMAL_DATASET_BASE
        / "target_qc"
        / target.lower()
        / "qc_report.json"
    )

    if not qc_report_path.exists():
        return

    try:
        qc_report = _read_json(qc_report_path)
    except Exception as exc:
        print()
        print(
            "FINAL QC + TAKEOVER SUMMARY unavailable:",
            f"{type(exc).__name__}: {exc}",
        )
        return

    episodes = list(qc_report.get("episodes", []))
    if not episodes:
        return

    print()
    print("=" * 72)
    print(
        f"FINAL QC + TAKEOVER SUMMARY — TARGET {target}"
    )
    print("=" * 72)
    print(
        f"{'dataset_ep':<12}"
        f"{'QC':<10}"
        f"{'TAKEOVER':<20}"
        "scope"
    )
    print("-" * 72)

    # Final-summary emphasis rule:
    #   - only genuinely current results are colored;
    #   - every historical row is plain/default terminal color.
    #
    # On the first target-level QC run there may be no previous target QC
    # snapshot yet. In that case, treating every old episode as "QC THIS RUN"
    # is misleading, so only episodes from the latest source run are current.
    latest_run_name = None
    run_names = []
    for item in episodes:
        summary_json = str(item.get("summary_json", ""))
        if summary_json:
            try:
                run_names.append(Path(summary_json).parents[1].name)
            except IndexError:
                pass
    if run_names:
        latest_run_name = max(run_names)

    has_previous_qc_snapshot = bool(previous_qc_statuses)

    for item in episodes:
        idx = int(item["dataset_episode_index"])
        qc_status = str(item.get("status", "UNKNOWN"))

        summary_json = str(item.get("summary_json", ""))
        source_run_name = None
        if summary_json:
            try:
                source_run_name = Path(summary_json).parents[1].name
            except IndexError:
                source_run_name = None

        if has_previous_qc_snapshot:
            qc_is_current = (
                previous_qc_statuses.get(idx) != qc_status
            )
        else:
            qc_is_current = (
                latest_run_name is not None
                and source_run_name == latest_run_name
            )

        if idx in current_batch_results:
            takeover_status = current_batch_results[idx]
            scope = "TAKEOVER THIS RUN"
        else:
            takeover_status = _historical_takeover_status(
                summary_json
            )
            scope = "QC THIS RUN" if qc_is_current else "history"

        line = (
            f"{idx:<12}"
            f"{qc_status:<10}"
            f"{takeover_status:<20}"
            f"{scope}"
        )

        if idx in current_batch_results:
            if takeover_status == "PASS":
                print(_green(line))
            else:
                print(_red(line))
        elif qc_is_current:
            if qc_status == "PASS":
                print(_green(line))
            elif qc_status == "REVIEW":
                print(_yellow(line))
            else:
                print(_red(line))
        else:
            print(line)

    print("-" * 72)

    if current_batch_results:
        counts: dict[str, int] = {}
        for status in current_batch_results.values():
            counts[status] = counts.get(status, 0) + 1

        ordered = (
            "PASS",
            "FAIL",
            "INVALID_REPLAY",
            "ABORTED",
            "UNKNOWN",
        )
        count_line = "    ".join(
            f"{status}: {counts[status]}"
            for status in ordered
            if status in counts
        )
        print("this-run takeover:", count_line)
    else:
        print("this-run takeover: no newly scheduled episode")

    qc_non_pass = [
        item
        for item in episodes
        if str(item.get("status")) != "PASS"
    ]
    if qc_non_pass:
        print(
            "QC non-pass raw:",
            "  ".join(
                f"{int(item['dataset_episode_index']):03d}"
                f"({item['status']})"
                for item in qc_non_pass
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Formal Phase 6 target validation: run target-level cross-run QC, "
            "then validate every pending QC-PASS episode by physical replay "
            "and WRIST takeover."
        )
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Formal A-Z target.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="After QC, debug/repeat one QC-PASS dataset episode.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="After QC, re-run all QC-PASS episodes including completed ones.",
    )
    parser.add_argument("--recovery-home", default=None)
    parser.add_argument("--robot-port", default=None)
    parser.add_argument("--restore-step-deg", type=float, default=None)
    parser.add_argument(
        "--max-replay-sent-diff-deg",
        type=float,
        default=None,
    )
    parser.add_argument("--required-passes", type=int, default=None)
    args = parser.parse_args()

    target = str(args.target).strip().upper()
    if target not in tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        raise ValueError("Formal Phase 6 target must be A-Z")

    if args.episode is not None and args.all:
        raise RuntimeError(
            "--episode and --all are mutually exclusive"
        )

    print("=" * 72)
    print("PHASE 6 TARGET VALIDATION")
    print("=" * 72)
    print("target :", target)
    print("flow   : target-level QC -> pending physical takeover")
    print()

    qc_cmd = [
        sys.executable,
        str(QC_SCRIPT),
        "--target",
        target,
    ]

    previous_qc_statuses = _qc_status_snapshot(
        target
    )

    print("[1/2] AUTOMATIC QC")
    qc_result = subprocess.run(
        qc_cmd,
        cwd=ROOT,
    )

    if qc_result.returncode != 0:
        raise SystemExit(qc_result.returncode)

    takeover_cmd = [
        sys.executable,
        str(TAKEOVER_SCRIPT),
        "--target",
        target,
    ]

    if args.episode is not None:
        takeover_cmd += [
            "--episode",
            str(args.episode),
        ]

    if args.all:
        takeover_cmd.append("--all")

    optional = (
        ("--recovery-home", args.recovery_home),
        ("--robot-port", args.robot_port),
        ("--restore-step-deg", args.restore_step_deg),
        (
            "--max-replay-sent-diff-deg",
            args.max_replay_sent_diff_deg,
        ),
        ("--required-passes", args.required_passes),
    )

    for flag, value in optional:
        if value is not None:
            takeover_cmd += [
                flag,
                str(value),
            ]

    batch_summary_path = (
        FORMAL_DATASET_BASE
        / "target_qc"
        / target.lower()
        / "takeover_batch_summary.json"
    )
    batch_summary_mtime_before = (
        batch_summary_path.stat().st_mtime_ns
        if batch_summary_path.exists()
        else None
    )

    print()
    print("[2/2] PHYSICAL REPLAY -> WRIST TAKEOVER")
    result = subprocess.run(
        takeover_cmd,
        cwd=ROOT,
    )

    current_batch_results: dict[int, str] = {}

    if batch_summary_path.exists():
        batch_summary_mtime_after = (
            batch_summary_path.stat().st_mtime_ns
        )

        if (
            batch_summary_mtime_before is None
            or batch_summary_mtime_after
            != batch_summary_mtime_before
        ):
            try:
                batch_summary = _read_json(
                    batch_summary_path
                )
                current_batch_results = {
                    int(item["dataset_episode_index"]): str(
                        item.get("status", "UNKNOWN")
                    )
                    for item in batch_summary.get(
                        "results",
                        [],
                    )
                }
            except Exception:
                current_batch_results = {}

    _print_combined_summary(
        target=target,
        previous_qc_statuses=previous_qc_statuses,
        current_batch_results=current_batch_results,
    )

    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
