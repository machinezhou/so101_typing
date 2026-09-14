from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from so101_typing.adapters.cameras import (
    CameraSpec,
    ThreadedOpenCVCamera,
)

from so101_typing.perception.keycaps import (
    detect_keycaps,
    draw_keycap_candidates,
)

from so101_typing.perception.screen_rectify import (
    ScreenCalibration,
)

from so101_typing.perception.top_mask import (
    TopMaskConfig,
    analyze_and_mask,
)

from so101_typing.utils.visualization import (
    fit_panel,
    label_panel,
    placeholder,
)


CAMERA_CONFIGS = {
    "top": "configs/cameras/top.yaml",
    "wrist": "configs/cameras/wrist.yaml",
    "side": "configs/cameras/screen.yaml",
}


def wait_for_frames(
    cameras: dict[str, ThreadedOpenCVCamera],
    timeout_s: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        if all(
            camera.latest() is not None
            for camera in cameras.values()
        ):
            return

        time.sleep(0.02)

    missing = [
        name
        for name, camera in cameras.items()
        if camera.latest() is None
    ]

    raise RuntimeError(
        f"No initial frames from cameras: {missing}"
    )


def make_mosaic(
    top_raw: np.ndarray,
    top_masked: np.ndarray,
    wrist_overlay: np.ndarray,
    side_raw: np.ndarray,
    side_rectified: np.ndarray | None,
    text_roi: np.ndarray | None,
    labels: dict[str, str],
) -> np.ndarray:
    panels = [
        label_panel(
            fit_panel(top_raw),
            labels["top_raw"],
        ),

        label_panel(
            fit_panel(top_masked),
            labels["top_masked"],
        ),

        label_panel(
            fit_panel(wrist_overlay),
            labels["wrist"],
        ),

        label_panel(
            fit_panel(side_raw),
            labels["side"],
        ),

        label_panel(
            fit_panel(side_rectified)
            if side_rectified is not None
            else placeholder(
                "SIDE homography not calibrated"
            ),
            "SIDE rectified",
        ),

        label_panel(
            fit_panel(text_roi)
            if text_roi is not None
            else placeholder(
                "SIDE text ROI not calibrated"
            ),
            "SIDE text ROI",
        ),
    ]

    return np.vstack(
        [
            np.hstack(panels[0:2]),
            np.hstack(panels[2:4]),
            np.hstack(panels[4:6]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--output",
        default="artifacts/camera_sanity",
    )

    parser.add_argument(
        "--snapshot-period",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--fps-pass",
        type=float,
        default=27.0,
        help="Practical lower bound for intended 30 FPS stream",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cameras: dict[str, ThreadedOpenCVCamera] = {}

    try:
        for logical_name, config_path in CAMERA_CONFIGS.items():
            spec = CameraSpec.from_yaml(config_path)

            camera = ThreadedOpenCVCamera(spec)
            camera.start()

            cameras[logical_name] = camera

            print(
                f"[OPEN] {logical_name:<5} "
                f"index={spec.index_or_path} "
                f"requested={spec.width}x{spec.height}@{spec.fps} "
                f"fourcc={spec.fourcc}"
            )

        wait_for_frames(cameras)

        top_mask_config = TopMaskConfig.load(
            "calibration/top_screen_mask.json"
        )

        screen_calibration = ScreenCalibration.load(
            "calibration/screen_homography.json"
        )

        start = time.monotonic()
        next_snapshot = start

        final_report = {}

        while True:
            now = time.monotonic()

            if now - start >= args.duration:
                break

            top = cameras["top"].latest(copy_image=True)
            wrist = cameras["wrist"].latest(copy_image=True)
            side = cameras["side"].latest(copy_image=True)

            if top is None or wrist is None or side is None:
                time.sleep(0.01)
                continue

            top_masked, leakage = analyze_and_mask(
                top.image,
                top_mask_config,
            )

            candidates = detect_keycaps(
                wrist.image
            )

            wrist_overlay = draw_keycap_candidates(
                wrist.image,
                candidates,
            )

            side_rectified = None
            text_roi = None

            if screen_calibration is not None:
                side_rectified = screen_calibration.rectify(
                    side.image
                )

                text_roi = screen_calibration.crop_text_roi(
                    side_rectified
                )

            if now < next_snapshot:
                time.sleep(0.01)
                continue

            next_snapshot = now + args.snapshot_period

            fps = {
                name: camera.measured_fps
                for name, camera in cameras.items()
            }

            labels = {
                "top_raw": (
                    f"TOP raw | {fps['top']:.1f} FPS "
                    f"| id={top.frame_id}"
                ),
                "top_masked": (
                    "TOP mask | "
                    f"configured={leakage['configured']}"
                ),
                "wrist": (
                    f"WRIST | {fps['wrist']:.1f} FPS "
                    f"| candidates={len(candidates)}"
                ),
                "side": (
                    f"SIDE raw | {fps['side']:.1f} FPS "
                    f"| id={side.frame_id}"
                ),
            }

            mosaic = make_mosaic(
                top_raw=top.image,
                top_masked=top_masked,
                wrist_overlay=wrist_overlay,
                side_raw=side.image,
                side_rectified=side_rectified,
                text_roi=text_roi,
                labels=labels,
            )

            cv2.imwrite(
                str(output_dir / "latest_mosaic.jpg"),
                mosaic,
            )

            cv2.imwrite(
                str(output_dir / "top_raw.jpg"),
                top.image,
            )

            cv2.imwrite(
                str(output_dir / "wrist_raw.jpg"),
                wrist.image,
            )

            cv2.imwrite(
                str(output_dir / "wrist_candidates.jpg"),
                wrist_overlay,
            )

            cv2.imwrite(
                str(output_dir / "side_raw.jpg"),
                side.image,
            )

            if side_rectified is not None:
                cv2.imwrite(
                    str(output_dir / "side_rectified.jpg"),
                    side_rectified,
                )

            if text_roi is not None:
                cv2.imwrite(
                    str(output_dir / "side_text_roi.jpg"),
                    text_roi,
                )

            final_report = {
                "elapsed_s": now - start,

                "cameras": {
                    name: {
                        "index_or_path": camera.spec.index_or_path,
                        "requested": {
                            "width": camera.spec.width,
                            "height": camera.spec.height,
                            "fps": camera.spec.fps,
                            "fourcc": camera.spec.fourcc,
                        },
                        "actual": camera.actual_properties(),
                        "measured_fps": camera.measured_fps,
                        "read_errors": camera.read_errors,
                    }
                    for name, camera in cameras.items()
                },

                "frames": {
                    "top": {
                        "frame_id": top.frame_id,
                        "capture_timestamp": top.capture_timestamp,
                        "processing_timestamp": top.processing_timestamp,
                        "frame_age_ms": top.frame_age_ms,
                    },
                    "wrist": {
                        "frame_id": wrist.frame_id,
                        "capture_timestamp": wrist.capture_timestamp,
                        "processing_timestamp": wrist.processing_timestamp,
                        "frame_age_ms": wrist.frame_age_ms,
                    },
                    "side": {
                        "frame_id": side.frame_id,
                        "capture_timestamp": side.capture_timestamp,
                        "processing_timestamp": side.processing_timestamp,
                        "frame_age_ms": side.frame_age_ms,
                    },
                },

                "wrist_keycap_candidates": len(candidates),

                "top_screen_mask": leakage,

                "screen_calibrated": (
                    screen_calibration is not None
                ),
            }

            with (
                output_dir / "report.json"
            ).open("w", encoding="utf-8") as handle:
                json.dump(
                    final_report,
                    handle,
                    indent=2,
                )

            print(
                "[STATUS] "
                f"TOP={fps['top']:.1f} "
                f"WRIST={fps['wrist']:.1f} "
                f"SIDE={fps['side']:.1f} FPS | "
                f"keycaps={len(candidates)} | "
                f"screen_calibrated="
                f"{screen_calibration is not None}"
            )

        if not final_report:
            raise RuntimeError(
                "No camera report was produced"
            )

        failing = [
            name
            for name, camera in cameras.items()
            if camera.measured_fps < args.fps_pass
        ]

        print()
        print("===== PHASE 1 CAMERA SANITY SUMMARY =====")

        for name, camera in cameras.items():
            print(
                f"{name.upper():<6} "
                f"{camera.measured_fps:6.2f} FPS "
                f"errors={camera.read_errors}"
            )

        print(
            f"TOP mask configured: "
            f"{top_mask_config.polygon is not None}"
        )

        print(
            f"SIDE homography configured: "
            f"{screen_calibration is not None}"
        )

        if failing:
            print(
                "FPS CHECK: FAIL/INVESTIGATE -> "
                + ", ".join(failing)
            )
        else:
            print("FPS CHECK: PASS")

        print(
            f"Artifacts: {output_dir.resolve()}"
        )

    finally:
        for camera in cameras.values():
            camera.stop()


if __name__ == "__main__":
    main()
