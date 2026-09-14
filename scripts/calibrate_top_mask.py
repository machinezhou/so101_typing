from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import webbrowser

import cv2
import numpy as np

from so101_typing.adapters.cameras import CameraSpec, ThreadedOpenCVCamera
from so101_typing.perception.top_mask import TopMaskConfig, analyze_and_mask


DEFAULT_CAMERA_CONFIG = "configs/cameras/top.yaml"
DEFAULT_CALIBRATION = "calibration/top_screen_mask.json"
DEFAULT_ARTIFACT_DIR = "artifacts/top_mask_calibration"
DEFAULT_BRIGHTNESS_THRESHOLD = 245


def encode_jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return encoded.tobytes()


def load_input_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read input image: {path}")
    return image


def capture_top_frame(
    camera_config: str,
    timeout_s: float,
    warmup_s: float,
) -> np.ndarray:
    spec = CameraSpec.from_yaml(camera_config)
    camera = ThreadedOpenCVCamera(spec)
    camera.start()

    try:
        deadline = time.monotonic() + timeout_s
        first_frame_time: float | None = None

        while time.monotonic() < deadline:
            frame = camera.latest(copy_image=True)
            if frame is None:
                time.sleep(0.02)
                continue

            if first_frame_time is None:
                first_frame_time = time.monotonic()

            if time.monotonic() - first_frame_time >= warmup_s:
                return frame.image

            time.sleep(0.02)

        raise RuntimeError(
            f"No warmed-up frame received from {spec.name!r} "
            f"within {timeout_s:.1f}s"
        )
    finally:
        camera.stop()


def normalize_polygon(raw_points: object) -> np.ndarray:
    points = np.asarray(raw_points, dtype=np.float64)
    if points.shape != (4, 2):
        raise ValueError(
            "Exactly four display corners are required: TL -> TR -> BR -> BL"
        )
    return np.rint(points).astype(np.int32)


def draw_polygon_overlay(
    frame: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    overlay = frame.copy()
    cv2.polylines(
        overlay,
        [polygon.reshape((-1, 1, 2))],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    labels = ("TL", "TR", "BR", "BL")
    for label, (x, y) in zip(labels, polygon, strict=True):
        cv2.circle(
            overlay,
            (int(x), int(y)),
            6,
            (0, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


@dataclass(slots=True)
class CalibrationState:
    frame: np.ndarray
    calibration_path: Path
    artifact_dir: Path
    brightness_threshold: int

    def build_config(self, raw_points: object) -> TopMaskConfig:
        polygon = normalize_polygon(raw_points)
        config = TopMaskConfig(
            polygon=polygon,
            brightness_threshold=self.brightness_threshold,
        )
        config.validate_for_frame(self.frame)
        return config

    def preview(self, raw_points: object) -> tuple[np.ndarray, dict]:
        config = self.build_config(raw_points)
        masked, stats = analyze_and_mask(self.frame, config)
        return masked, stats

    def save(self, raw_points: object) -> tuple[TopMaskConfig, dict]:
        config = self.build_config(raw_points)
        masked, stats = analyze_and_mask(self.frame, config)
        config.save(self.calibration_path)

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.artifact_dir / "top_raw.jpg"), self.frame)
        cv2.imwrite(
            str(self.artifact_dir / "top_screen_polygon.jpg"),
            draw_polygon_overlay(self.frame, config.polygon),
        )
        cv2.imwrite(str(self.artifact_dir / "top_masked.jpg"), masked)
        (self.artifact_dir / "top_mask_stats.json").write_text(
            json.dumps(stats, indent=2) + "\n",
            encoding="utf-8",
        )
        return config, stats


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO-101 TOP Screen Mask Calibration</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { margin: 0; background: #111; color: #eee; }
main { max-width: 1450px; margin: 0 auto; padding: 20px; }
h1 { margin-top: 0; font-size: 24px; }
.instructions { line-height: 1.5; color: #ccc; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.panel { background: #1b1b1b; border: 1px solid #333; border-radius: 10px; padding: 14px; }
canvas, img { display: block; width: 100%; height: auto; background: #000; }
canvas { cursor: crosshair; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0; }
button { padding: 9px 14px; border: 0; border-radius: 7px; cursor: pointer; font-weight: 600; }
button:disabled { cursor: not-allowed; opacity: 0.45; }
#status { min-height: 1.5em; font-family: ui-monospace, monospace; white-space: pre-wrap; }
.good { color: #8ee28e; }
.bad { color: #ff8d8d; }
.small { color: #aaa; font-size: 13px; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
<h1>SO-101 TOP Screen Mask Calibration</h1>
<p class="instructions">
Click four points around the <b>visible MacBook display region</b> in this exact order:
<b>TL → TR → BR → BL</b>. If the display is clipped by the TOP image boundary, use the intersections between the visible display and the image boundary; the physical screen corners do not need to be visible.
The goal is only to remove screen information from TOP while preserving as much robot/keyboard workspace as possible.
</p>
<div class="controls">
<button id="reset">Reset points</button>
<button id="preview" disabled>Preview mask</button>
<button id="save" disabled>Save calibration</button>
</div>
<div id="status">Loading TOP frame…</div>
<div class="grid">
  <section class="panel">
    <h2>1. TOP raw</h2>
    <canvas id="rawCanvas"></canvas>
    <p class="small">Points: <span id="pointText">0 / 4</span></p>
  </section>
  <section class="panel">
    <h2>2. Mask preview</h2>
    <img id="previewImage" alt="Mask preview">
    <pre id="stats" class="small"></pre>
  </section>
</div>
</main>
<script>
const labels = ["TL", "TR", "BR", "BL"];
const canvas = document.getElementById("rawCanvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const pointText = document.getElementById("pointText");
const previewButton = document.getElementById("preview");
const saveButton = document.getElementById("save");
const resetButton = document.getElementById("reset");
const previewImage = document.getElementById("previewImage");
const statsEl = document.getElementById("stats");
let image = new Image();
let points = [];
let previewReady = false;

function setStatus(message, good=true) {
  statusEl.textContent = message;
  statusEl.className = good ? "good" : "bad";
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);
  if (points.length > 0) {
    ctx.strokeStyle = "#00ff66";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    if (points.length === 4) ctx.closePath();
    ctx.stroke();
  }
  points.forEach((p, i) => {
    ctx.fillStyle = "#ff3333";
    ctx.beginPath();
    ctx.arc(p[0], p[1], 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#ff3333";
    ctx.font = "bold 18px sans-serif";
    ctx.fillText(labels[i], p[0] + 9, p[1] - 9);
  });
  pointText.textContent = `${points.length} / 4`;
  previewButton.disabled = points.length !== 4;
  saveButton.disabled = !previewReady;
}

image.onload = () => {
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  redraw();
  setStatus(`TOP frame loaded: ${canvas.width}x${canvas.height}. Click TL → TR → BR → BL.`);
};
image.src = "/frame.jpg";

canvas.addEventListener("click", event => {
  if (points.length >= 4) return;
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * canvas.width / rect.width;
  const y = (event.clientY - rect.top) * canvas.height / rect.height;
  points.push([Math.round(x), Math.round(y)]);
  previewReady = false;
  previewImage.removeAttribute("src");
  statsEl.textContent = "";
  redraw();
});

resetButton.addEventListener("click", () => {
  points = [];
  previewReady = false;
  previewImage.removeAttribute("src");
  statsEl.textContent = "";
  redraw();
  setStatus("Points reset. Click TL → TR → BR → BL.");
});

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

previewButton.addEventListener("click", async () => {
  try {
    const body = await postJson("/preview", {points});
    previewImage.src = `data:image/jpeg;base64,${body.image}`;
    statsEl.textContent = JSON.stringify(body.stats, null, 2);
    previewReady = true;
    redraw();
    setStatus("Preview ready. Verify only the display is blacked out and useful workspace remains visible.");
  } catch (error) {
    previewReady = false;
    redraw();
    setStatus(error.message, false);
  }
});

saveButton.addEventListener("click", async () => {
  try {
    const body = await postJson("/save", {points});
    statsEl.textContent = JSON.stringify(body.stats, null, 2);
    setStatus(`Saved: ${body.path}`);
  } catch (error) {
    setStatus(error.message, false);
  }
});
</script>
</body>
</html>
"""


class CalibrationHandler(BaseHTTPRequestHandler):
    state: CalibrationState

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(
        self,
        payload: dict,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self._send_bytes(
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("Invalid request body")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON request must be an object")
        return data

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_bytes(
                HTML_TEMPLATE.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if self.path == "/frame.jpg":
            self._send_bytes(encode_jpeg(self.state.frame), "image/jpeg")
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            request = self._read_json()
            points = request.get("points")

            if self.path == "/preview":
                masked, stats = self.state.preview(points)
                self._send_json(
                    {
                        "image": base64.b64encode(encode_jpeg(masked)).decode("ascii"),
                        "stats": stats,
                    }
                )
                return

            if self.path == "/save":
                _, stats = self.state.save(points)
                self._send_json(
                    {
                        "path": str(self.state.calibration_path),
                        "stats": stats,
                    }
                )
                return

            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively calibrate the fixed TOP MacBook display mask."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Existing TOP frame. If omitted, capture one from the TOP camera.",
    )
    parser.add_argument("--camera-config", default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_CALIBRATION))
    parser.add_argument("--artifact-dir", type=Path, default=Path(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--brightness-threshold", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.input is not None:
        frame = load_input_image(args.input)
        print(f"[INPUT] Using existing TOP frame: {args.input}")
    else:
        frame = capture_top_frame(args.camera_config, args.timeout, args.warmup)
        print(f"[INPUT] Captured TOP frame from: {args.camera_config}")

    threshold = DEFAULT_BRIGHTNESS_THRESHOLD
    if args.output.exists():
        try:
            threshold = TopMaskConfig.load(args.output).brightness_threshold
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if args.brightness_threshold is not None:
        threshold = args.brightness_threshold

    state = CalibrationState(
        frame=frame,
        calibration_path=args.output,
        artifact_dir=args.artifact_dir,
        brightness_threshold=threshold,
    )

    handler_class = type(
        "BoundCalibrationHandler",
        (CalibrationHandler,),
        {"state": state},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_class)
    url = f"http://{args.host}:{args.port}/"

    print(f"[FRAME] {frame.shape[1]}x{frame.shape[0]}")
    print(f"[THRESHOLD] {threshold}")
    print()
    print("===== TOP SCREEN MASK CALIBRATION =====")
    print(f"Open: {url}")
    print("1) Mark visible display polygon: TL -> TR -> BR -> BL")
    print("2) Preview the black software mask")
    print("3) Save calibration/top_screen_mask.json")
    print("4) Ctrl+C here when finished")

    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n[STOP] Calibration server stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
