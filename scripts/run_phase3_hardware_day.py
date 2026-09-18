from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-session Phase-3 hardware workflow: direct WRIST p_tip calibration, "
            "software preflight, explicit operator confirmation, then one staged-servo primitive."
        )
    )
    parser.add_argument("--target", default="G")
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--leader-port", default="/dev/ttyACM1")
    parser.add_argument("--robot-id", default="lawson_follower_arm")
    parser.add_argument("--leader-id", default="lawson_leader_arm")
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=Path("configs/cameras/wrist.yaml"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/models/glyph_hog_svm"),
    )
    parser.add_argument(
        "--tool-reference",
        type=Path,
        default=Path("calibration/tool_reference.json"),
    )
    parser.add_argument(
        "--image-jacobian",
        type=Path,
        default=Path("calibration/image_jacobian.json"),
    )
    parser.add_argument("--picker", choices=("browser", "terminal"), default="browser")
    parser.add_argument("--tip-samples", type=int, default=5)
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Pass through to the direct-tip calibration browser picker.",
    )
    args = parser.parse_args()

    target = args.target.strip().upper()
    if len(target) != 1 or not ("A" <= target <= "Z"):
        raise ValueError("--target must be A-Z")

    python = sys.executable
    calibrate = [
        python,
        "scripts/calibrate_tool_reference.py",
        "--camera-config",
        str(args.camera_config),
        "--output",
        str(args.tool_reference),
        "--samples",
        str(args.tip_samples),
        "--picker",
        args.picker,
    ]
    if args.no_open_browser:
        calibrate.append("--no-open-browser")

    common = [
        "--target",
        target,
        "--camera-config",
        str(args.camera_config),
        "--model-dir",
        str(args.model_dir),
        "--tool-reference",
        str(args.tool_reference),
        "--image-jacobian",
        str(args.image_jacobian),
    ]
    preflight = [python, "scripts/run_phase3_staged_servo.py", *common, "--dry-run"]
    hardware = [
        python,
        "scripts/run_phase3_staged_servo.py",
        *common,
        "--robot-port",
        args.robot_port,
        "--leader-port",
        args.leader_port,
        "--robot-id",
        args.robot_id,
        "--leader-id",
        args.leader_id,
    ]

    print("===== PHASE 3 HARDWARE-DAY WORKFLOW =====")
    print("This workflow does not run ACT, screen OCR, or Phase-5 press-success logic.")
    print(
        "Start with leader/follower at the normal zero/home pose and the fixed "
        "tool/camera geometry."
    )
    print("Step 1 opens only the WRIST camera; it does not connect or command the robot.")

    _run(calibrate)
    _run(preflight)

    print()
    print("===== OPERATOR SAFETY CONFIRMATION =====")
    print("The next command will connect follower + leader once at the normal zero/home pose.")
    print("It will then enter in-process teleoperation for you to move to a safe local hover.")
    print("Autonomy remains bounded to Phase-3 XY alignment and the small staged-Z budget.")
    confirmation = input("Type RUN PHASE3 to continue, or anything else to stop: ").strip()
    if confirmation != "RUN PHASE3":
        print("Stopped before robot connection.")
        return

    _run(hardware)
    print()
    print("One staged primitive run is complete, not full Phase-3 acceptance.")
    print(
        "README acceptance still requires repeated G alignment from several initial offsets "
        "and two cross-keyboard transfer sanity checks."
    )


if __name__ == "__main__":
    main()
