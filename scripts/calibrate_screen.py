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
from so101_typing.perception.screen_rectify import ScreenCalibration


DEFAULT_CAMERA_CONFIG = "configs/cameras/screen.yaml"
DEFAULT_CALIBRATION = "calibration/screen_homography.json"
DEFAULT_ARTIFACT_DIR = "artifacts/screen_calibration"
DEFAULT_OUTPUT_SIZE = (1280, 800)


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


def capture_side_frame(
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


def normalize_source_points(raw_points: object) -> np.ndarray:
    points = np.asarray(raw_points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("Exactly four source points are required")
    return points


def validate_points_inside_frame(
    points: np.ndarray,
    frame: np.ndarray,
) -> None:
    height, width = frame.shape[:2]
    xs = points[:, 0]
    ys = points[:, 1]

    if (
        np.any(xs < 0)
        or np.any(xs > width - 1)
        or np.any(ys < 0)
        or np.any(ys > height - 1)
    ):
        raise ValueError("All source points must stay inside the SIDE frame")


def draw_source_overlay(
    frame: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    overlay = frame.copy()
    integer_points = np.rint(points).astype(np.int32)
    cv2.polylines(
        overlay,
        [integer_points.reshape((-1, 1, 2))],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    labels = ("TL", "TR", "BR", "BL")
    for label, (x, y) in zip(labels, integer_points, strict=True):
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


def draw_roi_overlay(
    rectified: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    overlay = rectified.copy()
    x, y, width, height = roi
    cv2.rectangle(
        overlay,
        (x, y),
        (x + width - 1, y + height - 1),
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    return overlay


@dataclass(slots=True)
class CalibrationState:
    frame: np.ndarray
    output_size: tuple[int, int]
    calibration_path: Path
    artifact_dir: Path
    rectified: np.ndarray | None = None
    source_points: np.ndarray | None = None

    def preview(self, raw_points: object) -> np.ndarray:
        points = normalize_source_points(raw_points)
        validate_points_inside_frame(points, self.frame)

        calibration = ScreenCalibration(
            source_points=points,
            output_size=self.output_size,
            text_roi=None,
        )
        rectified = calibration.rectify(self.frame)

        self.source_points = points
        self.rectified = rectified
        return rectified

    def save(
        self,
        raw_points: object,
        raw_roi: object,
    ) -> ScreenCalibration:
        points = normalize_source_points(raw_points)
        validate_points_inside_frame(points, self.frame)

        if not isinstance(raw_roi, list) or len(raw_roi) != 4:
            raise ValueError("A text ROI [x, y, width, height] is required")

        roi = tuple(int(value) for value in raw_roi)
        calibration = ScreenCalibration(
            source_points=points,
            output_size=self.output_size,
            text_roi=roi,
        )
        rectified = calibration.rectify(self.frame)
        text_roi = calibration.crop_text_roi(rectified)
        if text_roi is None or text_roi.size == 0:
            raise ValueError("The selected text ROI is empty")

        calibration.save(self.calibration_path)

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(self.artifact_dir / "side_raw.jpg"),
            self.frame,
        )
        cv2.imwrite(
            str(self.artifact_dir / "side_source_points.jpg"),
            draw_source_overlay(self.frame, points),
        )
        cv2.imwrite(
            str(self.artifact_dir / "side_rectified.jpg"),
            rectified,
        )
        cv2.imwrite(
            str(self.artifact_dir / "side_rectified_roi.jpg"),
            draw_roi_overlay(rectified, roi),
        )
        cv2.imwrite(
            str(self.artifact_dir / "side_text_roi.jpg"),
            text_roi,
        )

        self.source_points = points
        self.rectified = rectified
        return calibration


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO-101 SIDE Screen Calibration</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { margin: 0; background: #111; color: #eee; }
main { max-width: 1500px; margin: 0 auto; padding: 20px; }
h1 { margin-top: 0; font-size: 24px; }
.instructions { line-height: 1.5; color: #ccc; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.panel { background: #1b1b1b; border: 1px solid #333; border-radius: 10px; padding: 14px; }
canvas { display: block; width: 100%; height: auto; background: #000; cursor: crosshair; }
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
<h1>SO-101 SIDE Screen Calibration</h1>
<p class="instructions">
Step 1: click the four corners of the <b>illuminated display area</b> in this exact order:
<b>TL → TR → BR → BL</b>. Do not use the outer laptop bezel corners.
Then generate the rectified preview. Step 2: drag a rectangle over the fixed typing/text region
inside the rectified screen. Finally save the calibration.
</p>
<div class="controls">
<button id="resetPoints">Reset screen corners</button>
<button id="preview" disabled>Generate rectified preview</button>
<button id="resetRoi" disabled>Reset text ROI</button>
<button id="save" disabled>Save calibration</button>
</div>
<div id="status">Loading SIDE frame…</div>
<div class="grid">
  <section class="panel">
    <h2>1. SIDE raw</h2>
    <canvas id="rawCanvas"></canvas>
    <p class="small">Points: <span id="pointText">0 / 4</span></p>
  </section>
  <section class="panel">
    <h2>2. Rectified screen + text ROI</h2>
    <canvas id="rectCanvas" width="__OUT_W__" height="__OUT_H__"></canvas>
    <p class="small">ROI: <span id="roiText">not selected</span></p>
  </section>
</div>
</main>
<script>
const pointLabels = ["TL", "TR", "BR", "BL"];
const rawCanvas = document.getElementById("rawCanvas");
const rawCtx = rawCanvas.getContext("2d");
const rectCanvas = document.getElementById("rectCanvas");
const rectCtx = rectCanvas.getContext("2d");
const previewButton = document.getElementById("preview");
const saveButton = document.getElementById("save");
const resetRoiButton = document.getElementById("resetRoi");
const statusBox = document.getElementById("status");
const pointText = document.getElementById("pointText");
const roiText = document.getElementById("roiText");

let rawImage = new Image();
let rectImage = new Image();
let points = [];
let roi = null;
let dragStart = null;
let dragging = false;

function setStatus(text, good = null) {
  statusBox.textContent = text;
  statusBox.className = good === true ? "good" : good === false ? "bad" : "";
}

function canvasPoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.round((event.clientX - rect.left) * canvas.width / rect.width),
    y: Math.round((event.clientY - rect.top) * canvas.height / rect.height),
  };
}

function drawRaw() {
  rawCtx.clearRect(0, 0, rawCanvas.width, rawCanvas.height);
  rawCtx.drawImage(rawImage, 0, 0, rawCanvas.width, rawCanvas.height);
  if (points.length > 1) {
    rawCtx.strokeStyle = "#00ff66";
    rawCtx.lineWidth = 2;
    rawCtx.beginPath();
    rawCtx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; ++i) rawCtx.lineTo(points[i][0], points[i][1]);
    if (points.length === 4) rawCtx.closePath();
    rawCtx.stroke();
  }
  points.forEach((point, index) => {
    rawCtx.fillStyle = "#ff3535";
    rawCtx.beginPath();
    rawCtx.arc(point[0], point[1], 6, 0, Math.PI * 2);
    rawCtx.fill();
    rawCtx.fillStyle = "#ffefef";
    rawCtx.font = "bold 16px sans-serif";
    rawCtx.fillText(pointLabels[index], point[0] + 9, point[1] - 9);
  });
  pointText.textContent = `${points.length} / 4`;
  previewButton.disabled = points.length !== 4;
}

function drawRectified() {
  rectCtx.clearRect(0, 0, rectCanvas.width, rectCanvas.height);
  if (rectImage.complete && rectImage.naturalWidth) {
    rectCtx.drawImage(rectImage, 0, 0, rectCanvas.width, rectCanvas.height);
  }
  if (roi) {
    rectCtx.strokeStyle = "#00ff66";
    rectCtx.lineWidth = 4;
    rectCtx.strokeRect(roi[0], roi[1], roi[2], roi[3]);
  }
}

rawImage.onload = () => {
  rawCanvas.width = rawImage.naturalWidth;
  rawCanvas.height = rawImage.naturalHeight;
  drawRaw();
  setStatus("SIDE frame loaded. Click TL → TR → BR → BL.");
};
rawImage.onerror = () => setStatus("Failed to load SIDE frame.", false);
rawImage.src = "/frame.jpg";

rawCanvas.addEventListener("click", (event) => {
  if (points.length >= 4) return;
  const p = canvasPoint(event, rawCanvas);
  points.push([p.x, p.y]);
  roi = null;
  saveButton.disabled = true;
  resetRoiButton.disabled = true;
  drawRaw();
});

document.getElementById("resetPoints").addEventListener("click", () => {
  points = [];
  roi = null;
  rectImage = new Image();
  rectCtx.clearRect(0, 0, rectCanvas.width, rectCanvas.height);
  saveButton.disabled = true;
  resetRoiButton.disabled = true;
  roiText.textContent = "not selected";
  drawRaw();
  setStatus("Screen corners reset. Click TL → TR → BR → BL.");
});

previewButton.addEventListener("click", async () => {
  try {
    setStatus("Computing homography…");
    const response = await fetch("/api/preview", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({source_points: points}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "preview failed");

    roi = null;
    roiText.textContent = "not selected";
    saveButton.disabled = true;
    resetRoiButton.disabled = false;
    rectImage = new Image();
    rectImage.onload = () => {
      drawRectified();
      setStatus("Rectification ready. Drag the fixed typing/text ROI on the right image.", true);
    };
    rectImage.src = `/rectified.jpg?t=${Date.now()}`;
  } catch (error) {
    setStatus(`Preview error: ${error.message}`, false);
  }
});

rectCanvas.addEventListener("pointerdown", (event) => {
  if (!rectImage.naturalWidth) return;
  rectCanvas.setPointerCapture(event.pointerId);
  dragStart = canvasPoint(event, rectCanvas);
  dragging = true;
  roi = null;
  saveButton.disabled = true;
});

rectCanvas.addEventListener("pointermove", (event) => {
  if (!dragging || !dragStart) return;
  const p = canvasPoint(event, rectCanvas);
  const x = Math.max(0, Math.min(dragStart.x, p.x));
  const y = Math.max(0, Math.min(dragStart.y, p.y));
  const x2 = Math.min(rectCanvas.width, Math.max(dragStart.x, p.x));
  const y2 = Math.min(rectCanvas.height, Math.max(dragStart.y, p.y));
  roi = [x, y, Math.max(1, x2 - x), Math.max(1, y2 - y)];
  drawRectified();
});

rectCanvas.addEventListener("pointerup", (event) => {
  if (!dragging) return;
  dragging = false;
  rectCanvas.releasePointerCapture(event.pointerId);
  if (!roi || roi[2] < 2 || roi[3] < 2) {
    roi = null;
    roiText.textContent = "not selected";
    saveButton.disabled = true;
    drawRectified();
    return;
  }
  roiText.textContent = `[${roi.join(", ")}]`;
  saveButton.disabled = false;
  drawRectified();
});

resetRoiButton.addEventListener("click", () => {
  roi = null;
  roiText.textContent = "not selected";
  saveButton.disabled = true;
  drawRectified();
});

saveButton.addEventListener("click", async () => {
  try {
    setStatus("Saving calibration…");
    const response = await fetch("/api/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({source_points: points, text_roi: roi}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "save failed");
    setStatus(
      `Saved ${data.calibration_path}\n` +
      `Artifacts: ${data.artifact_dir}\n` +
      "You can close this page and stop the script with Ctrl+C.",
      true,
    );
  } catch (error) {
    setStatus(`Save error: ${error.message}`, false);
  }
});
</script>
</body>
</html>
"""


def make_handler(state: CalibrationState) -> type[BaseHTTPRequestHandler]:
    class CalibrationHandler(BaseHTTPRequestHandler):
        server_version = "SO101ScreenCalibration/1.0"

        def log_message(self, format: str, *args: object) -> None:
            print(f"[HTTP] {self.address_string()} - {format % args}")

        def send_bytes(
            self,
            body: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(
            self,
            payload: dict,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_bytes(body, "application/json; charset=utf-8", status)

        def read_json(self) -> dict:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 1_000_000:
                raise ValueError("Invalid request body length")

            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON request must be an object")
            return payload

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/?"):
                width, height = state.output_size
                html = (
                    HTML_TEMPLATE
                    .replace("__OUT_W__", str(width))
                    .replace("__OUT_H__", str(height))
                )
                self.send_bytes(
                    html.encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return

            if self.path.startswith("/frame.jpg"):
                self.send_bytes(encode_jpeg(state.frame), "image/jpeg")
                return

            if self.path.startswith("/rectified.jpg"):
                if state.rectified is None:
                    self.send_json(
                        {"error": "Rectified preview is not available yet"},
                        HTTPStatus.NOT_FOUND,
                    )
                    return
                self.send_bytes(encode_jpeg(state.rectified), "image/jpeg")
                return

            if self.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return

            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                payload = self.read_json()

                if self.path == "/api/preview":
                    rectified = state.preview(payload.get("source_points"))
                    self.send_json(
                        {
                            "ok": True,
                            "width": int(rectified.shape[1]),
                            "height": int(rectified.shape[0]),
                        }
                    )
                    return

                if self.path == "/api/save":
                    calibration = state.save(
                        payload.get("source_points"),
                        payload.get("text_roi"),
                    )
                    self.send_json(
                        {
                            "ok": True,
                            "calibration_path": str(
                                state.calibration_path.resolve()
                            ),
                            "artifact_dir": str(state.artifact_dir.resolve()),
                            "calibration": calibration.to_dict(),
                        }
                    )
                    return

                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(
                    {"error": str(exc)},
                    HTTPStatus.BAD_REQUEST,
                )
            except Exception as exc:
                print(f"[ERROR] {type(exc).__name__}: {exc}")
                self.send_json(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return CalibrationHandler


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive browser-based SIDE screen homography and "
            "text-ROI calibration"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Use an existing SIDE raw image instead of opening the camera. "
            "Recommended after an accepted camera_sanity run."
        ),
    )
    parser.add_argument(
        "--camera-config",
        default=DEFAULT_CAMERA_CONFIG,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(DEFAULT_CALIBRATION),
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(DEFAULT_ARTIFACT_DIR),
    )
    parser.add_argument("--output-width", type=int, default=DEFAULT_OUTPUT_SIZE[0])
    parser.add_argument("--output-height", type=int, default=DEFAULT_OUTPUT_SIZE[1])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera-timeout", type=float, default=5.0)
    parser.add_argument("--camera-warmup", type=float, default=1.0)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not try to open the calibration page automatically",
    )
    args = parser.parse_args()

    if args.output_width <= 0 or args.output_height <= 0:
        raise SystemExit("output width/height must be positive")

    if args.input is not None:
        frame = load_input_image(args.input)
        print(f"[INPUT] Using existing SIDE frame: {args.input.resolve()}")
    else:
        print(f"[CAMERA] Capturing SIDE frame from {args.camera_config}")
        frame = capture_side_frame(
            camera_config=args.camera_config,
            timeout_s=args.camera_timeout,
            warmup_s=args.camera_warmup,
        )

    height, width = frame.shape[:2]
    print(f"[FRAME] {width}x{height}")

    state = CalibrationState(
        frame=frame,
        output_size=(args.output_width, args.output_height),
        calibration_path=args.calibration,
        artifact_dir=args.artifacts,
    )

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(state),
    )
    url = f"http://{args.host}:{args.port}/"

    print()
    print("===== SIDE SCREEN CALIBRATION =====")
    print(f"Open: {url}")
    print("1) Click screen corners: TL -> TR -> BR -> BL")
    print("2) Generate rectified preview")
    print("3) Drag the fixed typing/text ROI")
    print("4) Save calibration")
    print("5) Ctrl+C here when finished")
    print()

    if not args.no_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n[STOP] Calibration server stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
