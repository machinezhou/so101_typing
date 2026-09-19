from __future__ import annotations

import json
import time
from pathlib import Path

from so101_typing.adapters.cameras import (
    CameraSpec,
    ThreadedOpenCVCamera,
)
from so101_typing.perception.screen_ocr import (
    TesseractScreenLineOCR,
)
from so101_typing.perception.screen_rectify import (
    ScreenCalibration,
)
from so101_typing.supervisor.verification import (
    ScreenVerificationStatus,
    verify_screen_texts,
)


CAMERA_CONFIG = Path(
    "configs/cameras/screen.yaml"
)
SCREEN_CALIBRATION = Path(
    "calibration/screen_homography.json"
)

ARTIFACT_DIR = Path(
    "artifacts/phase4_screen_verification"
)

PREFIX = "KEYPRESS"
EXPECTED_CHAR = "G"
WRONG_CHAR = "H"

FRAME_COUNT = 9
MIN_VOTE_FRACTION = 0.60
MAX_PREFIX_DISTANCE = 2


def wait_next_frame(
    camera: ThreadedOpenCVCamera,
    *,
    after_frame_id: int,
    timeout_s: float = 3.0,
):
    deadline = (
        time.monotonic()
        + timeout_s
    )

    while time.monotonic() < deadline:
        frame = camera.latest(
            copy_image=True
        )

        if (
            frame is not None
            and frame.frame_id
            > after_frame_id
        ):
            return frame

        time.sleep(0.01)

    raise RuntimeError(
        "Timed out waiting for a fresh SIDE frame"
    )


def capture_verification_burst(
    *,
    camera,
    calibration,
    ocr,
    stage_name,
):
    observations = []
    records = []
    last_frame_id = -1

    print()
    print(
        f"===== CAPTURE {stage_name} ====="
    )

    for index in range(
        1,
        FRAME_COUNT + 1,
    ):
        frame = wait_next_frame(
            camera,
            after_frame_id=last_frame_id,
        )

        last_frame_id = int(
            frame.frame_id
        )

        rectified = calibration.rectify(
            frame.image
        )

        roi = calibration.crop_text_roi(
            rectified
        )

        if roi is None:
            raise RuntimeError(
                "screen text ROI is missing"
            )

        observation = ocr.recognize(
            roi
        )

        observations.append(
            observation.normalized_text
        )

        records.append(
            {
                "frame_id": int(
                    frame.frame_id
                ),
                "frame_age_ms": float(
                    frame.frame_age_ms
                ),
                "ocr_confidence": float(
                    observation.mean_confidence
                ),
                "text": (
                    observation.normalized_text
                ),
            }
        )

        print()
        print(
            f"[{stage_name}] "
            f"{index}/{FRAME_COUNT} "
            f"frame={frame.frame_id} "
            f"confidence="
            f"{observation.mean_confidence:.1f}"
        )

        print(
            observation.normalized_text
            if observation.normalized_text
            else "<EMPTY>"
        )

    result = verify_screen_texts(
        observations,
        confirmed_prefix=PREFIX,
        expected_char=EXPECTED_CHAR,
        min_vote_fraction=(
            MIN_VOTE_FRACTION
        ),
        max_prefix_distance=(
            MAX_PREFIX_DISTANCE
        ),
    )

    print()
    print(
        f"===== {stage_name} RESULT ====="
    )

    print(
        f"status          : {result.status}"
    )
    print(
        f"success votes   : "
        f"{result.success_votes}/{result.total_frames}"
    )
    print(
        f"no-change votes : "
        f"{result.no_change_votes}/{result.total_frames}"
    )
    print(
        f"wrong votes     : "
        f"{result.wrong_votes}/{result.total_frames}"
    )
    print(
        f"wrong character : "
        f"{result.wrong_character}"
    )
    print(
        f"uncertain votes : "
        f"{result.uncertain_votes}/{result.total_frames}"
    )
    print(
        f"winning fraction: "
        f"{result.vote_fraction:.3f}"
    )

    return result, records


def require_status(
    actual,
    expected,
    stage,
):
    if actual.status != expected:
        print()
        print(
            f"[{stage}] NOT ACCEPTED: "
            f"expected {expected}, "
            f"got {actual.status}"
        )

        return False

    print()
    print(
        f"[{stage}] PASS: {actual.status}"
    )

    return True


def main() -> None:
    print("=" * 72)
    print(
        "PHASE 4.3 — LIVE FOUR-STATE "
        "SCREEN VERIFIER"
    )
    print("=" * 72)
    print(
        "NO ROBOT CONNECTION — "
        "SIDE CAMERA ONLY"
    )
    print()
    print(
        f"confirmed prefix : {PREFIX}"
    )
    print(
        f"expected char    : {EXPECTED_CHAR}"
    )
    print(
        f"wrong test char  : {WRONG_CHAR}"
    )

    calibration = ScreenCalibration.load(
        SCREEN_CALIBRATION
    )

    if calibration is None:
        raise RuntimeError(
            "screen calibration is missing"
        )

    spec = CameraSpec.from_yaml(
        CAMERA_CONFIG
    )

    ocr = TesseractScreenLineOCR(
        language="eng",
        scale=3.0,
        clahe_clip_limit=2.0,
        dark_threshold=185,
    )

    camera = ThreadedOpenCVCamera(
        spec
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {}

    try:
        camera.start()
        time.sleep(0.3)

        print()
        print("=" * 72)
        print("STEP 1 — NO CHANGE")
        print("=" * 72)
        print()
        print(
            "On the MacBook, put this exact text "
            "on ONE line inside the typing ROI:"
        )
        print()
        print(f"    {PREFIX}")
        print()
        print(
            "The caret should be immediately after "
            "the final S."
        )
        input(
            "When ready, press ENTER here "
            "to capture NO_CHANGE: "
        )

        no_change, records = (
            capture_verification_burst(
                camera=camera,
                calibration=calibration,
                ocr=ocr,
                stage_name="NO_CHANGE",
            )
        )

        payload["no_change"] = records

        pass_no_change = require_status(
            no_change,
            ScreenVerificationStatus.CONFIRMED_NO_CHANGE,
            "NO_CHANGE",
        )

        print()
        print("=" * 72)
        print("STEP 2 — EXPECTED CHARACTER")
        print("=" * 72)
        print()
        print(
            "On the MacBook append exactly ONE G, "
            "so the line becomes:"
        )
        print()
        print(f"    {PREFIX}{EXPECTED_CHAR}")
        print()

        input(
            "When ready, press ENTER here "
            "to capture SUCCESS: "
        )

        success, records = (
            capture_verification_burst(
                camera=camera,
                calibration=calibration,
                ocr=ocr,
                stage_name="SUCCESS",
            )
        )

        payload["success"] = records

        pass_success = require_status(
            success,
            ScreenVerificationStatus.CONFIRMED_SUCCESS,
            "SUCCESS",
        )

        print()
        print("=" * 72)
        print("STEP 3 — WRONG CHARACTER")
        print("=" * 72)
        print()
        print(
            "On the MacBook replace the final G "
            "with ONE H, so the line becomes:"
        )
        print()
        print(f"    {PREFIX}{WRONG_CHAR}")
        print()

        input(
            "When ready, press ENTER here "
            "to capture WRONG: "
        )

        wrong, records = (
            capture_verification_burst(
                camera=camera,
                calibration=calibration,
                ocr=ocr,
                stage_name="WRONG",
            )
        )

        payload["wrong"] = records

        pass_wrong = require_status(
            wrong,
            ScreenVerificationStatus.CONFIRMED_WRONG,
            "WRONG",
        )

        summary = {
            "prefix": PREFIX,
            "expected_char": EXPECTED_CHAR,
            "wrong_test_char": WRONG_CHAR,
            "no_change_status": str(
                no_change.status
            ),
            "success_status": str(
                success.status
            ),
            "wrong_status": str(
                wrong.status
            ),
            "wrong_character": (
                wrong.wrong_character
            ),
            "pass_no_change": (
                pass_no_change
            ),
            "pass_success": (
                pass_success
            ),
            "pass_wrong": (
                pass_wrong
            ),
        }

        payload["summary"] = summary

        output = (
            ARTIFACT_DIR
            / "validation.json"
        )

        output.write_text(
            json.dumps(
                payload,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 72)
        print("PHASE 4.3 SUMMARY")
        print("=" * 72)

        print(
            "NO_CHANGE :",
            no_change.status,
        )
        print(
            "SUCCESS   :",
            success.status,
        )
        print(
            "WRONG     :",
            wrong.status,
            "observed wrong char=",
            wrong.wrong_character,
        )

        if (
            pass_no_change
            and pass_success
            and pass_wrong
        ):
            print()
            print(
                "[PASS] Live screen verifier "
                "distinguished all three "
                "authoritative outcomes."
            )
        else:
            print()
            print(
                "[NOT YET ACCEPTED] At least one "
                "live outcome needs refinement."
            )

        print()
        print("UNCERTAIN behavior is covered by unit tests.")
        print(
            f"Saved: {output}"
        )

    finally:
        camera.stop()


if __name__ == "__main__":
    main()
