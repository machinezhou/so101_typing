from __future__ import annotations

import argparse
import json
from pathlib import Path

from so101_typing.control.image_jacobian import (
    REQUESTED_CARTESIAN_DELTA,
    ImageJacobianCalibration,
)


DEFAULT_CANDIDATE = Path(
    "artifacts/image_jacobian_calibration/candidate_image_jacobian.json"
)
DEFAULT_SESSION = Path("artifacts/image_jacobian_calibration/session.json")
DEFAULT_OUTPUT = Path("calibration/image_jacobian.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a reviewed H3.2 command-space image-Jacobian candidate "
            "to the canonical Phase-3 calibration artifact."
        )
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--confirm-reviewed",
        action="store_true",
        help=(
            "Required acknowledgement that direction consistency, residuals, "
            "conditioning, and sample artifacts were reviewed."
        ),
    )
    args = parser.parse_args()

    if not args.confirm_reviewed:
        raise RuntimeError(
            "Refusing to promote an unreviewed candidate. Re-run with "
            "--confirm-reviewed only after reviewing the H3.2 session/artifacts."
        )

    candidate = ImageJacobianCalibration.load(args.candidate)
    if candidate is None:
        raise RuntimeError(f"Candidate is uncalibrated: {args.candidate}")
    if candidate.input_semantics != REQUESTED_CARTESIAN_DELTA:
        raise RuntimeError("Candidate does not use requested Cartesian command semantics")

    session_data = json.loads(args.session.read_text(encoding="utf-8"))
    if (
        session_data.get("protocol")
        != "fixed_anchor_conditioned_cartesian_paired_xy_perturbations_v3"
    ):
        raise RuntimeError(
            "Session is not a conditioned fixed-anchor H3.2 v3 calibration session"
        )

    conditioning = session_data.get("conditioning")
    if (
        not isinstance(conditioning, dict)
        or conditioning.get("converged") is not True
    ):
        raise RuntimeError(
            "Session did not pass adaptive conditioning"
        )

    if (
        conditioning.get("excluded_from_jacobian_fit")
        is not True
    ):
        raise RuntimeError(
            "Conditioning samples were not explicitly excluded from J fit"
        )
    if session_data.get("input_semantics") != REQUESTED_CARTESIAN_DELTA:
        raise RuntimeError("Session input semantics do not match the command-space contract")
    if session_data.get("image_jacobian") != candidate.to_dict():
        raise RuntimeError("Session Jacobian does not exactly match the candidate artifact")

    formal_samples = session_data.get("samples")
    if (
        not isinstance(formal_samples, list)
        or len(formal_samples) != candidate.sample_count
    ):
        raise RuntimeError(
            "Formal sample count does not match the candidate"
        )

    if any(
        sample.get("sample_role") != "calibration"
        for sample in formal_samples
    ):
        raise RuntimeError(
            "Session formal sample set contains non-calibration samples"
        )
    acceptance = session_data.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
        raise RuntimeError("Session did not pass automatic H3.2 acceptance gates")
    if acceptance.get("errors"):
        raise RuntimeError("Session acceptance record contains errors")

    direction = session_data.get("direction_consistency")
    if not isinstance(direction, dict) or not all(axis in direction for axis in ("x", "y")):
        raise RuntimeError("Session is missing X/Y direction-consistency diagnostics")
    for axis in ("x", "y"):
        axis_report = direction[axis]
        if not isinstance(axis_report, dict) or axis_report.get("opposition_cosine") is None:
            raise RuntimeError(f"Session has unusable {axis.upper()} direction diagnostics")

    candidate.save(args.output)
    print("===== H3.2 PROMOTED =====")
    print(f"candidate = {args.candidate.resolve()}")
    print(f"session   = {args.session.resolve()}")
    print(f"canonical = {args.output.resolve()}")
    print("Next gate: closed-loop XY visual-servo hardware validation; still no Z press.")


if __name__ == "__main__":
    main()
