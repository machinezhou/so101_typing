from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QC_SCRIPT = ROOT / "scripts" / "phase6_episode_qc.py"
TAKEOVER_SCRIPT = (
    ROOT / "scripts" / "phase6_replay_takeover_validate.py"
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

    print()
    print("[2/2] PHYSICAL REPLAY -> WRIST TAKEOVER")
    result = subprocess.run(
        takeover_cmd,
        cwd=ROOT,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
