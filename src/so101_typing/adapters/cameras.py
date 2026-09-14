from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
import subprocess
import time

import cv2
import yaml

from so101_typing.contracts import CameraFrame


@dataclass(frozen=True, slots=True)
class CameraSpec:
    name: str
    index_or_path: int | str
    width: int
    height: int
    fps: int
    fourcc: str

    v4l2_controls: dict[str, int] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CameraSpec":
        path = Path(path)

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        return cls(**data)


class ThreadedOpenCVCamera:
    """
    Independent reader thread per physical camera.

    This avoids serial blocking reads artificially reducing the measured
    FPS when three 30 FPS cameras are active simultaneously.
    """

    def __init__(self, spec: CameraSpec):
        self.spec = spec

        self._capture: cv2.VideoCapture | None = None
        self._thread: Thread | None = None

        self._stop = Event()
        self._lock = Lock()

        self._latest: CameraFrame | None = None
        self._frame_id = 0

        self._capture_times: deque[float] = deque(maxlen=180)
        self._read_errors = 0

    def _apply_v4l2_controls(self) -> None:
        controls = self.spec.v4l2_controls

        if not controls:
            return

        index_or_path = self.spec.index_or_path

        if isinstance(index_or_path, int):
            device = f"/dev/video{index_or_path}"
        elif (
            isinstance(index_or_path, str)
            and index_or_path.startswith("/dev/video")
        ):
            device = index_or_path
        else:
            raise RuntimeError(
                f"Camera {self.spec.name!r} has V4L2 controls "
                f"configured, but index_or_path="
                f"{index_or_path!r} is not a V4L2 device"
            )

        assignments = ",".join(
            f"{name}={value}"
            for name, value in controls.items()
        )

        try:
            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    device,
                    "--set-ctrl",
                    assignments,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "v4l2-ctl is required for configured "
                "camera controls"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (
                exc.stderr.strip()
                or exc.stdout.strip()
                or f"exit code {exc.returncode}"
            )

            raise RuntimeError(
                f"Failed to apply V4L2 controls for "
                f"{self.spec.name!r} on {device}: {detail}"
            ) from exc

    def start(self) -> None:
        if self._thread is not None:
            return

        self._apply_v4l2_controls()

        capture = cv2.VideoCapture(self.spec.index_or_path)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Unable to open camera {self.spec.name!r} "
                f"at {self.spec.index_or_path!r}"
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        capture.set(cv2.CAP_PROP_FPS, self.spec.fps)

        if self.spec.fourcc:
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*self.spec.fourcc),
            )

        self._capture = capture
        self._stop.clear()

        self._thread = Thread(
            target=self._reader,
            name=f"camera-{self.spec.name}",
            daemon=True,
        )
        self._thread.start()

    def _reader(self) -> None:
        assert self._capture is not None

        while not self._stop.is_set():
            ok, image = self._capture.read()

            capture_timestamp = time.monotonic()

            if not ok or image is None:
                with self._lock:
                    self._read_errors += 1

                time.sleep(0.005)
                continue

            frame = CameraFrame(
                camera_name=self.spec.name,
                frame_id=self._frame_id,
                capture_timestamp=capture_timestamp,
                processing_timestamp=capture_timestamp,
                image=image,
            )

            with self._lock:
                self._latest = frame
                self._capture_times.append(capture_timestamp)

            self._frame_id += 1

    def latest(
        self,
        copy_image: bool = False,
    ) -> CameraFrame | None:
        with self._lock:
            frame = self._latest

            if frame is None:
                return None

            image = (
                frame.image.copy()
                if copy_image
                else frame.image
            )

            processing_timestamp = time.monotonic()

            return CameraFrame(
                camera_name=frame.camera_name,
                frame_id=frame.frame_id,
                capture_timestamp=frame.capture_timestamp,
                processing_timestamp=processing_timestamp,
                image=image,
            )

    @property
    def measured_fps(self) -> float:
        with self._lock:
            timestamps = list(self._capture_times)

        if len(timestamps) < 2:
            return 0.0

        elapsed = timestamps[-1] - timestamps[0]

        if elapsed <= 0:
            return 0.0

        return (len(timestamps) - 1) / elapsed

    @property
    def read_errors(self) -> int:
        with self._lock:
            return self._read_errors

    def actual_properties(self) -> dict:
        if self._capture is None:
            return {}

        return {
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_property": float(self._capture.get(cv2.CAP_PROP_FPS)),
            "fourcc_int": int(self._capture.get(cv2.CAP_PROP_FOURCC)),
        }

    def stop(self) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None
