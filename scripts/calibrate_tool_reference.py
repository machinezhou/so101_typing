from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from so101_typing.adapters.cameras import CameraSpec, ThreadedOpenCVCamera
from so101_typing.control.tool_reference import ToolReferenceCalibration
from so101_typing.perception.glyph_runtime import RuntimeHOGGlyphRecognizer
from so101_typing.perception.wrist_target import observe_target


def _discover_model_dir(explicit: str | None) -> Path:
    if explicit is not None:
        model_dir = Path(explicit)
        _validate_model_dir(model_dir)
        return model_dir

    candidates: list[Path] = []
    for root in (Path("artifacts"), Path("checkpoints")):
        if not root.exists():
            continue
        for metadata in root.rglob("model.json"):
            model_dir = metadata.parent
            if (model_dir / "svm.xml").is_file() and (model_dir / "centroids.npz").is_file():
                candidates.append(model_dir)

    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise RuntimeError(
            "No persisted glyph runtime model found under artifacts/ or checkpoints/. "
            "Pass --model-dir explicitly."
        )
    rendered = "\n".join(f"  - {path}" for path in unique)
    raise RuntimeError(
        "Multiple persisted glyph runtime models were found. Pass --model-dir explicitly:\n"
        f"{rendered}"
    )


def _validate_model_dir(model_dir: Path) -> None:
    required = ("model.json", "svm.xml", "centroids.npz")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Invalid glyph model directory {model_dir}: missing {', '.join(missing)}"
        )


def _resolve_voice_backend(requested: str) -> str | None:
    if requested == "off":
        return None
    if requested != "auto":
        path = shutil.which(requested)
        if path is None:
            raise RuntimeError(f"Requested voice backend {requested!r} was not found in PATH")
        return path

    for candidate in ("spd-say", "espeak-ng", "espeak"):
        path = shutil.which(candidate)
        if path is not None:
            return path
    raise RuntimeError(
        "No speech backend found. Install/provide one of: spd-say, espeak-ng, espeak. "
        "Use --voice-backend off only if spoken guidance is not needed."
    )


def _speak(backend: str | None, text: str) -> None:
    print(f"[VOICE] {text}", flush=True)
    if backend is None:
        return
    executable = Path(backend).name
    if executable == "spd-say":
        command = [backend, "-w", text]
    else:
        command = [backend, "-s", "165", text]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Speech backend {executable!r} failed with exit code {result.returncode}"
        )


def _spoken_countdown(backend: str | None, seconds: float) -> None:
    count = math.ceil(seconds)
    for remaining in range(count, 0, -1):
        started = time.monotonic()
        _speak(backend, str(remaining))
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, 1.0 - elapsed))


def _wait_for_initial_frame(
    camera: ThreadedOpenCVCamera,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if camera.latest() is not None:
            return
        time.sleep(0.01)
    raise RuntimeError("No initial WRIST frame arrived before timeout")


def _capture_burst(
    camera: ThreadedOpenCVCamera,
    recognizer: RuntimeHOGGlyphRecognizer,
    target_label: str,
    *,
    frame_count: int,
    max_frame_age_ms: float,
    timeout_s: float,
    after_frame_id: int | None,
) -> tuple[np.ndarray, list[dict], np.ndarray, int]:
    centers: list[tuple[float, float]] = []
    observations: list[dict] = []
    last_image: np.ndarray | None = None
    last_frame_id = -1 if after_frame_id is None else after_frame_id
    deadline = time.monotonic() + timeout_s

    while len(centers) < frame_count and time.monotonic() < deadline:
        frame = camera.latest(copy_image=True)
        if frame is None or frame.frame_id <= last_frame_id:
            time.sleep(0.005)
            continue
        last_frame_id = frame.frame_id

        now = time.monotonic()
        age_ms = max(
            frame.frame_age_ms,
            max(0.0, (now - frame.capture_timestamp) * 1000.0),
        )
        if age_ms > max_frame_age_ms:
            continue

        observation = observe_target(frame.image, target_label, recognizer)
        if not observation.found or observation.center_px is None:
            continue

        center = (float(observation.center_px[0]), float(observation.center_px[1]))
        centers.append(center)
        observations.append(
            {
                "frame_id": int(frame.frame_id),
                "capture_timestamp": float(frame.capture_timestamp),
                "frame_age_ms": float(age_ms),
                "center_px": [center[0], center[1]],
                "vote_fraction": float(observation.vote_fraction),
                "similarity": float(observation.similarity),
                "similarity_threshold": float(observation.similarity_threshold),
                "quality_score": float(observation.quality_score),
            }
        )
        last_image = frame.image

    if len(centers) < frame_count or last_image is None:
        raise RuntimeError(
            f"Only acquired {len(centers)}/{frame_count} accepted fresh observations of "
            f"target {target_label!r} within {timeout_s:.1f}s"
        )

    array = np.asarray(centers, dtype=np.float64)
    return np.median(array, axis=0), observations, last_image, last_frame_id


def _draw_sample_preview(
    image: np.ndarray,
    target_label: str,
    center: np.ndarray,
    sample_index: int,
) -> np.ndarray:
    preview = image.copy()
    u, v = np.rint(center).astype(int)
    cv2.drawMarker(
        preview,
        (int(u), int(v)),
        (0, 255, 255),
        cv2.MARKER_CROSS,
        24,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"sample={sample_index} target={target_label} center=({center[0]:.1f},{center[1]:.1f})",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def _validate_args(args: argparse.Namespace) -> None:
    target = args.target.strip().upper()
    if len(target) != 1 or target < "A" or target > "Z":
        raise ValueError("--target must be one ASCII A-Z letter")
    if args.samples < 3:
        raise ValueError("--samples must be >= 3")
    if args.burst_frames < 1:
        raise ValueError("--burst-frames must be >= 1")
    for name in (
        "max_frame_age_ms",
        "capture_timeout_s",
        "initial_timeout_s",
        "first_prepare_seconds",
        "prepare_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and > 0")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hands-free spoken H3.1 tool-reference calibration for torque-off manual teach. "
            "This script never commands robot motion."
        )
    )
    parser.add_argument("--target", default="G")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--burst-frames", type=int, default=5)
    parser.add_argument("--camera-config", default="configs/cameras/wrist.yaml")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output", default="calibration/tool_reference.json")
    parser.add_argument("--artifact-dir", default="artifacts/tool_reference_calibration")
    parser.add_argument("--max-frame-age-ms", type=float, default=100.0)
    parser.add_argument("--capture-timeout-s", type=float, default=5.0)
    parser.add_argument("--initial-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--first-prepare-seconds",
        type=float,
        default=20.0,
        help="Spoken countdown for walking to the follower and making the first alignment.",
    )
    parser.add_argument(
        "--prepare-seconds",
        type=float,
        default=10.0,
        help="Spoken countdown for each realignment after the first capture.",
    )
    parser.add_argument(
        "--voice-test",
        action="store_true",
        help="Speak one headphone test phrase and exit before opening the camera.",
    )
    parser.add_argument(
        "--voice-backend",
        choices=("auto", "spd-say", "espeak-ng", "espeak", "off"),
        default="auto",
        help="Speech backend. auto tries spd-say, espeak-ng, then espeak.",
    )
    args = parser.parse_args()
    _validate_args(args)

    target_label = args.target.strip().upper()
    voice_backend = _resolve_voice_backend(args.voice_backend)
    if args.voice_test:
        _speak(
            voice_backend,
            "Audio check. You should hear this clearly in your headphones.",
        )
        return

    model_dir = _discover_model_dir(args.model_dir)
    recognizer = RuntimeHOGGlyphRecognizer.load(model_dir)
    spec = CameraSpec.from_yaml(args.camera_config)
    if spec.name != "wrist":
        raise ValueError(
            f"tool-reference calibration requires camera name 'wrist', got {spec.name!r}"
        )

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    camera = ThreadedOpenCVCamera(spec)
    samples: list[list[float]] = []
    sample_records: list[dict] = []
    last_frame_id: int | None = None

    print("===== PHASE 3 H3.1 SPOKEN MANUAL-TEACH CALIBRATION =====")
    print(f"WRIST config   : {args.camera_config}")
    print(f"glyph model    : {model_dir}")
    print(f"target         : {target_label}")
    print(f"manual teaches : {args.samples}")
    print(f"burst/teach    : {args.burst_frames} fresh accepted frames")
    print(f"voice backend  : {voice_backend or 'off'}")
    print()
    print("SAFETY: NO robot commands, NO Z/downward motion, NO key press.")
    print("Stop teleoperation first and confirm follower torque is off.")
    print("After launch, no keyboard interaction is required.")
    print("Keep headphones on and follow the spoken prompts beside the follower.")
    print()

    try:
        camera.start()
        _wait_for_initial_frame(camera, args.initial_timeout_s)
        print(
            f"[OPEN] wrist index={spec.index_or_path} "
            f"requested={spec.width}x{spec.height}@{spec.fps} fourcc={spec.fourcc}"
        )

        _speak(
            voice_backend,
            "Tool reference calibration starting. Move to the follower. "
            f"Keep the pencil above the center of key {target_label}. Do not press the key.",
        )

        for index in range(1, args.samples + 1):
            print()
            print(f"--- manual teach {index}/{args.samples} ---")
            if index == 1:
                prepare_seconds = float(args.first_prepare_seconds)
                _speak(
                    voice_backend,
                    f"Sample {index} of {args.samples}. Align the pencil tip over the center "
                    f"of key {target_label}. First capture in "
                    f"{math.ceil(prepare_seconds)} seconds.",
                )
            else:
                prepare_seconds = float(args.prepare_seconds)
                _speak(
                    voice_backend,
                    f"Sample {index} of {args.samples}. Realign the pencil tip over the center "
                    f"of key {target_label}. Capture in {math.ceil(prepare_seconds)} seconds.",
                )

            _spoken_countdown(voice_backend, prepare_seconds)
            _speak(voice_backend, "Hold still. Capturing now.")

            while True:
                try:
                    center, burst, image, last_frame_id = _capture_burst(
                        camera,
                        recognizer,
                        target_label,
                        frame_count=args.burst_frames,
                        max_frame_age_ms=args.max_frame_age_ms,
                        timeout_s=args.capture_timeout_s,
                        after_frame_id=last_frame_id,
                    )
                    break
                except RuntimeError as exc:
                    print(f"[RETRY] {exc}")
                    _speak(
                        voice_backend,
                        f"I could not capture key {target_label}. Keep the pencil safely above "
                        "the key and hold still. Retrying in five seconds.",
                    )
                    _spoken_countdown(voice_backend, 5.0)
                    _speak(voice_backend, "Hold still. Capturing now.")

            samples.append([float(center[0]), float(center[1])])
            burst_array = np.asarray([entry["center_px"] for entry in burst], dtype=np.float64)
            burst_std = np.std(burst_array, axis=0, ddof=0)
            sample_records.append(
                {
                    "sample_index": index,
                    "center_px": [float(center[0]), float(center[1])],
                    "burst_std_px": [float(burst_std[0]), float(burst_std[1])],
                    "observations": burst,
                }
            )
            preview = _draw_sample_preview(image, target_label, center, index)
            preview_path = artifact_dir / f"sample_{index:02d}.jpg"
            if not cv2.imwrite(str(preview_path), preview):
                raise RuntimeError(f"Failed to write preview image: {preview_path}")
            print(
                f"[CAPTURED] center=({center[0]:.2f}, {center[1]:.2f}) px "
                f"burst_std=({burst_std[0]:.2f}, {burst_std[1]:.2f}) px"
            )

            if index < args.samples:
                _speak(
                    voice_backend,
                    "Captured. Move the pencil clearly away from the key now.",
                )
            else:
                _speak(
                    voice_backend,
                    "Final sample captured. Calibration collection is complete. "
                    "You may move the pencil away.",
                )

        calibration = ToolReferenceCalibration.from_samples(
            samples,
            camera_name=spec.name,
            image_size=(spec.width, spec.height),
        )
        sample_array = np.asarray(samples, dtype=np.float64)
        delta = sample_array - np.asarray(calibration.center, dtype=np.float64)
        radial = np.linalg.norm(delta, axis=1)
        max_radial_deviation_px = float(np.max(radial))
        rms_radial_deviation_px = float(np.sqrt(np.mean(np.square(radial))))

        calibration.save(args.output)
        session = {
            "schema_version": 1,
            "target_label_used_for_provenance": target_label,
            "note": "p* is a tool/camera reference, not a target-key-specific reference.",
            "camera_config": str(args.camera_config),
            "model_dir": str(model_dir),
            "output": str(args.output),
            "protocol": "torque_off_spoken_manual_teach",
            "independent_manual_teaches": int(args.samples),
            "burst_frames_per_teach": int(args.burst_frames),
            "first_prepare_seconds": float(args.first_prepare_seconds),
            "prepare_seconds": float(args.prepare_seconds),
            "voice_backend": Path(voice_backend).name if voice_backend is not None else "off",
            "max_frame_age_ms": float(args.max_frame_age_ms),
            "samples": sample_records,
            "tool_reference": calibration.to_dict(),
            "repeatability": {
                "std_u_px": float(calibration.std_u_px),
                "std_v_px": float(calibration.std_v_px),
                "rms_radial_deviation_px": rms_radial_deviation_px,
                "max_radial_deviation_px": max_radial_deviation_px,
            },
            "camera": {
                "measured_fps": float(camera.measured_fps),
                "read_errors": int(camera.read_errors),
                "actual_properties": camera.actual_properties(),
            },
        }
        report_path = artifact_dir / "session.json"
        report_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")

        print()
        print("===== H3.1 RESULT =====")
        print(f"p*                    = ({calibration.u:.3f}, {calibration.v:.3f}) px")
        print(f"std_u_px              = {calibration.std_u_px:.3f}")
        print(f"std_v_px              = {calibration.std_v_px:.3f}")
        print(f"rms_radial_dev_px     = {rms_radial_deviation_px:.3f}")
        print(f"max_radial_dev_px     = {max_radial_deviation_px:.3f}")
        print(f"wrist_fps             = {camera.measured_fps:.2f}")
        print(f"wrist_read_errors     = {camera.read_errors}")
        print(f"calibration           = {Path(args.output).resolve()}")
        print(f"session artifacts     = {artifact_dir.resolve()}")
        print(
            "NOTE: between-teach spread includes manual alignment uncertainty; "
            "burst_std measures within-pose perception jitter."
        )
        print("REVIEW REQUIRED: do not proceed to Jacobian motion until H3.1 is reviewed.")
        _speak(
            voice_backend,
            "H three point one is finished. Return to the computer and review the result.",
        )
    except KeyboardInterrupt:
        print("\n[ABORTED] No completed calibration result was accepted.")
        raise SystemExit(130) from None
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
