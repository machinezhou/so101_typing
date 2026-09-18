from __future__ import annotations

import argparse
import base64
import json
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from so101_typing.adapters.cameras import CameraSpec, ThreadedOpenCVCamera
from so101_typing.control.tool_reference import ToolReferenceCalibration, WRIST_TOOL_TIP


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _wait_for_initial_frame(camera: ThreadedOpenCVCamera, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if camera.latest() is not None:
            return
        time.sleep(0.01)
    raise RuntimeError("No initial WRIST frame arrived before timeout")


def _fresh_frame(
    camera: ThreadedOpenCVCamera,
    *,
    after_frame_id: int | None,
    timeout_s: float,
):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = camera.latest(copy_image=True)
        if frame is None:
            time.sleep(0.01)
            continue
        if after_frame_id is not None and frame.frame_id <= after_frame_id:
            time.sleep(0.01)
            continue
        return frame
    raise RuntimeError("No fresh WRIST frame arrived before timeout")


def _click_html(jpeg_b64: str, sample_index: int, sample_count: int) -> bytes:
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SO-101 pencil-tip calibration</title>
<style>
body {{ font-family: sans-serif; background:#111; color:#eee; text-align:center; margin:20px; }}
img {{ width:min(1280px,95vw); height:auto; max-height:80vh; cursor:crosshair; }}
#msg {{ margin:12px; font-size:18px; }}
</style></head>
<body>
<h2>WRIST pencil-tip calibration — sample {sample_index}/{sample_count}</h2>
<div id="msg">Click the physical pencil tip once. Do not click the G/key center.</div>
<img id="frame" src="data:image/jpeg;base64,{jpeg_b64}">
<script>
const img = document.getElementById('frame');
let sent = false;
img.addEventListener('click', async (ev) => {{
  if (sent) return;
  const r = img.getBoundingClientRect();
  const u = (ev.clientX - r.left) * img.naturalWidth / r.width;
  const v = (ev.clientY - r.top) * img.naturalHeight / r.height;
  sent = true;
  document.getElementById('msg').textContent =
    `Captured tip: (${{u.toFixed(2)}}, ${{v.toFixed(2)}}) px`;
  const endpoint = `/click?u=${{encodeURIComponent(u)}}&v=${{encodeURIComponent(v)}}`;
  await fetch(endpoint, {{method:'POST'}});
}});
</script></body></html>"""
    return html.encode("utf-8")


def _pick_point_in_browser(
    image: np.ndarray,
    *,
    sample_index: int,
    sample_count: int,
    timeout_s: float,
    open_browser: bool,
) -> tuple[float, float]:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("Failed to encode WRIST frame for browser picker")
    jpeg_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    result: queue.Queue[tuple[float, float]] = queue.Queue(maxsize=1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/":
                self.send_error(404)
                return
            payload = _click_html(jpeg_b64, sample_index, sample_count)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/click":
                self.send_error(404)
                return
            values = parse_qs(parsed.query)
            try:
                point = (float(values["u"][0]), float(values["v"][0]))
            except (KeyError, IndexError, ValueError):
                self.send_error(400)
                return
            if result.empty():
                result.put(point)
            payload = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[CLICK] Open {url} and click the PHYSICAL pencil tip once.", flush=True)
    if open_browser:
        webbrowser.open(url, new=1)
    try:
        point = result.get(timeout=timeout_s)
    except queue.Empty as exc:
        raise RuntimeError(
            f"Timed out after {timeout_s:.0f}s waiting for pencil-tip click"
        ) from exc
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    height, width = image.shape[:2]
    u, v = point
    if not (0.0 <= u < width and 0.0 <= v < height):
        raise RuntimeError(f"Clicked point is outside the original image: {(u, v)}")
    return point



def _pick_point_in_terminal(
    image: np.ndarray,
    *,
    raw_path: Path,
) -> tuple[float, float]:
    height, width = image.shape[:2]
    print(
        f"[COORDINATE PICKER] Inspect {raw_path.resolve()} and enter the PHYSICAL "
        "pencil-tip pixel as: u v",
        flush=True,
    )
    while True:
        try:
            raw = input("p_tip u v> ").strip().replace(",", " ")
        except EOFError as exc:
            raise RuntimeError("Terminal coordinate picker received EOF") from exc
        parts = raw.split()
        if len(parts) != 2:
            print("Enter exactly two numbers: u v", flush=True)
            continue
        try:
            u, v = (float(parts[0]), float(parts[1]))
        except ValueError:
            print("Coordinates must be numeric.", flush=True)
            continue
        if not np.isfinite([u, v]).all():
            print("Coordinates must be finite.", flush=True)
            continue
        if not (0.0 <= u < width and 0.0 <= v < height):
            print(
                f"Point must be inside the original {width}x{height} image.",
                flush=True,
            )
            continue
        return (u, v)

def _draw_tip_preview(image: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    preview = image.copy()
    pixel = tuple(np.rint(np.asarray(point)).astype(int))
    cv2.drawMarker(preview, pixel, (0, 255, 255), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
    cv2.putText(
        preview,
        f"physical pencil tip ({point[0]:.1f}, {point[1]:.1f})",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directly calibrate the physical pencil-tip projection in the WRIST image."
    )
    parser.add_argument("--camera-config", type=Path, default=Path("configs/cameras/wrist.yaml"))
    parser.add_argument("--output", type=Path, default=Path("calibration/tool_reference.json"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/tool_tip_calibration"),
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--initial-timeout-s", type=float, default=5.0)
    parser.add_argument("--frame-timeout-s", type=float, default=5.0)
    parser.add_argument("--click-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-std-px", type=float, default=3.0)
    parser.add_argument(
        "--picker",
        choices=("browser", "terminal"),
        default="browser",
        help=(
            "Use the localhost browser click picker, or enter u/v coordinates in the "
            "terminal after inspecting each saved raw frame."
        ),
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Print the localhost picker URL without trying to open a browser automatically.",
    )
    args = parser.parse_args()

    if args.samples < 3:
        raise ValueError("--samples must be >= 3")
    spec = CameraSpec.from_yaml(args.camera_config)
    if spec.name != "wrist":
        raise ValueError(f"Expected WRIST camera config, got {spec.name!r}")

    run_dir = args.artifact_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    old_calibration = None
    if args.output.exists():
        old_calibration = json.loads(args.output.read_text(encoding="utf-8"))

    session = {
        "protocol": "direct_wrist_physical_pencil_tip_v2",
        "camera_name": spec.name,
        "image_size": [spec.width, spec.height],
        "output": str(args.output),
        "previous_canonical": old_calibration,
        "picker": args.picker,
        "samples": [],
        "status": "starting",
    }
    session_path = run_dir / "session.json"
    _write_json(session_path, session)

    camera = ThreadedOpenCVCamera(spec)
    points: list[tuple[float, float]] = []
    last_frame_id: int | None = None
    try:
        print("===== DIRECT WRIST PENCIL-TIP CALIBRATION =====", flush=True)
        print("No robot/leader is opened and no robot command is sent.", flush=True)
        print("Click the physical pencil tip itself on every sample.", flush=True)
        camera.start()
        _wait_for_initial_frame(camera, args.initial_timeout_s)
        print(f"[CAMERA] WRIST ready ~{camera.measured_fps:.1f} FPS", flush=True)

        for index in range(1, args.samples + 1):
            frame = _fresh_frame(
                camera,
                after_frame_id=last_frame_id,
                timeout_s=args.frame_timeout_s,
            )
            last_frame_id = int(frame.frame_id)
            height, width = frame.image.shape[:2]
            if (width, height) != (spec.width, spec.height):
                raise RuntimeError(
                    "WRIST frame geometry mismatch during p_tip calibration: "
                    f"actual={(width, height)} configured={(spec.width, spec.height)}"
                )
            raw_path = run_dir / f"sample_{index:02d}_raw.jpg"
            if not cv2.imwrite(str(raw_path), frame.image):
                raise RuntimeError(f"Failed to write {raw_path}")

            if args.picker == "browser":
                point = _pick_point_in_browser(
                    frame.image,
                    sample_index=index,
                    sample_count=args.samples,
                    timeout_s=args.click_timeout_s,
                    open_browser=not args.no_open_browser,
                )
            else:
                point = _pick_point_in_terminal(frame.image, raw_path=raw_path)
            points.append(point)
            preview = _draw_tip_preview(frame.image, point)
            preview_path = run_dir / f"sample_{index:02d}_tip.jpg"
            if not cv2.imwrite(str(preview_path), preview):
                raise RuntimeError(f"Failed to write {preview_path}")
            session["samples"].append(
                {
                    "index": index,
                    "frame_id": int(frame.frame_id),
                    "capture_timestamp": float(frame.capture_timestamp),
                    "tip_px": [float(point[0]), float(point[1])],
                    "raw_image": str(raw_path),
                    "preview_image": str(preview_path),
                }
            )
            _write_json(session_path, session)
            print(f"[SAMPLE {index}/{args.samples}] p_tip=({point[0]:.2f}, {point[1]:.2f}) px")

        calibration = ToolReferenceCalibration.from_samples(
            points,
            camera_name=spec.name,
            image_size=(spec.width, spec.height),
            reference_kind=WRIST_TOOL_TIP,
            method=f"direct_manual_click_{args.picker}",
        )
        if max(calibration.std_u_px, calibration.std_v_px) > args.max_std_px:
            raise RuntimeError(
                "Pencil-tip clicks were not repeatable enough: "
                f"std=({calibration.std_u_px:.2f}, {calibration.std_v_px:.2f}) px, "
                f"limit={args.max_std_px:.2f} px. Repeat calibration more carefully."
            )
        calibration.save(args.output)
        session["status"] = "accepted"
        session["calibration"] = calibration.to_dict()
        session["repeatability"] = {
            "std_u_px": calibration.std_u_px,
            "std_v_px": calibration.std_v_px,
            "rms_radial_deviation_px": calibration.rms_radial_deviation_px,
            "max_radial_deviation_px": calibration.max_radial_deviation_px,
        }
        _write_json(session_path, session)
        print()
        print("===== TIP CALIBRATION ACCEPTED =====")
        print(
            f"p_tip=({calibration.u:.2f}, {calibration.v:.2f}) px; "
            f"std=({calibration.std_u_px:.2f}, {calibration.std_v_px:.2f}) px; "
            f"radial_rms={calibration.rms_radial_deviation_px:.2f}px; "
            f"radial_max={calibration.max_radial_deviation_px:.2f}px"
        )
        print(f"canonical: {args.output.resolve()}")
        print(f"artifacts: {run_dir.resolve()}")
    except Exception:
        session["status"] = "failed"
        _write_json(session_path, session)
        raise
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
