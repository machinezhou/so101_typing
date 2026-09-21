from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
import time

import cv2
import numpy as np

from so101_typing.perception.screen_ocr import (
    detect_screen_text_lines,
)


@dataclass(frozen=True, slots=True)
class FastScreenChangeConfig:
    """Conservative press-event detector for the SIDE screen camera.

    The detector does not recognize a character. It answers only:

        did persistent new foreground ink appear after the confirmed prefix?

    Cursor blink is absorbed into the multi-frame baseline envelope.
    """

    dark_threshold: int = 185
    baseline_dilate_px: int = 5
    continuation_overlap_px: int = 14
    min_novel_pixels: int = 45
    min_component_area_px: int = 18

    # A very strong single-frame continuation glyph can trigger the safety
    # release immediately. Ordinary changes still require normal
    # consecutive-frame confirmation.
    strong_min_novel_pixels: int = 300
    strong_min_component_area_px: int = 250

    required_consecutive_frames: int = 2
    max_frame_age_ms: float = 120.0
    poll_interval_s: float = 0.005


@dataclass(frozen=True, slots=True)
class FastScreenChangeObservation:
    changed: bool
    strong_changed: bool
    triggered: bool
    consecutive_frames: int
    novel_pixels: int
    max_component_area_px: int
    continuation_x0: int
    bbox_xywh: tuple[int, int, int, int] | None


class FastScreenChangeModel:
    def __init__(
        self,
        *,
        baseline_safe_mask: np.ndarray,
        continuation_x0: int,
        config: FastScreenChangeConfig | None = None,
    ) -> None:
        cfg = config or FastScreenChangeConfig()

        mask = np.asarray(baseline_safe_mask)
        if mask.ndim != 2:
            raise ValueError("baseline_safe_mask must be a 2-D mask")

        self.config = cfg
        self.baseline_safe_mask = mask.astype(bool, copy=True)
        self.continuation_x0 = int(continuation_x0)
        self._streak = 0
        self._triggered = False

    @classmethod
    def from_baseline_rois(
        cls,
        rois: list[np.ndarray] | tuple[np.ndarray, ...],
        *,
        config: FastScreenChangeConfig | None = None,
    ) -> "FastScreenChangeModel":
        cfg = config or FastScreenChangeConfig()

        if len(rois) < 2:
            raise ValueError("at least two baseline ROI frames are required")

        shape = rois[0].shape[:2]
        union = np.zeros(shape, dtype=np.uint8)
        line_right_edges: list[int] = []

        for roi in rois:
            if roi.shape[:2] != shape:
                raise ValueError("all baseline ROI frames must share one shape")

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            dark = (gray < cfg.dark_threshold).astype(np.uint8)
            union = cv2.bitwise_or(union, dark)

            boxes = detect_screen_text_lines(
                roi,
                dark_threshold=cfg.dark_threshold,
            )
            if boxes:
                box = max(boxes, key=lambda item: item.width)
                line_right_edges.append(box.x + box.width - 1)

        if not line_right_edges:
            columns = np.flatnonzero(np.count_nonzero(union, axis=0) >= 2)
            if columns.size == 0:
                raise RuntimeError("baseline screen ROI contains no detectable text")
            baseline_right = int(columns[-1])
        else:
            baseline_right = int(round(float(np.median(line_right_edges))))

        continuation_x0 = max(
            0,
            baseline_right - int(cfg.continuation_overlap_px),
        )

        radius = max(0, int(cfg.baseline_dilate_px))
        if radius:
            size = radius * 2 + 1
            kernel = np.ones((size, size), dtype=np.uint8)
            union = cv2.dilate(union, kernel, iterations=1)

        safe = union.astype(bool)

        return cls(
            baseline_safe_mask=safe,
            continuation_x0=continuation_x0,
            config=cfg,
        )

    def _novel_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        if roi_bgr.shape[:2] != self.baseline_safe_mask.shape:
            raise ValueError("screen ROI shape changed after baseline")

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        dark = gray < self.config.dark_threshold
        novel = dark & ~self.baseline_safe_mask
        novel[:, : self.continuation_x0] = False

        mask = (novel.astype(np.uint8) * 255)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
        )
        return mask

    def observe(self, roi_bgr: np.ndarray) -> FastScreenChangeObservation:
        mask = self._novel_mask(roi_bgr)

        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8),
            connectivity=8,
        )

        novel_pixels = int(np.count_nonzero(mask))
        max_area = 0
        bbox = None

        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area > max_area:
                max_area = area
                bbox = (
                    int(stats[index, cv2.CC_STAT_LEFT]),
                    int(stats[index, cv2.CC_STAT_TOP]),
                    int(stats[index, cv2.CC_STAT_WIDTH]),
                    int(stats[index, cv2.CC_STAT_HEIGHT]),
                )

        changed = (
            novel_pixels >= self.config.min_novel_pixels
            and max_area >= self.config.min_component_area_px
        )

        strong_changed = (
            novel_pixels >= self.config.strong_min_novel_pixels
            and max_area >= self.config.strong_min_component_area_px
        )

        if changed:
            self._streak += 1
        else:
            self._streak = 0

        if (
            strong_changed
            or self._streak >= self.config.required_consecutive_frames
        ):
            self._triggered = True

        return FastScreenChangeObservation(
            changed=changed,
            strong_changed=strong_changed,
            triggered=self._triggered,
            consecutive_frames=self._streak,
            novel_pixels=novel_pixels,
            max_component_area_px=max_area,
            continuation_x0=self.continuation_x0,
            bbox_xywh=bbox,
        )

    def extract_novel_crop(
        self,
        roi_bgr: np.ndarray,
        *,
        margin_px: int = 10,
    ) -> tuple[np.ndarray | None, FastScreenChangeObservation]:
        # Do not mutate the persistent streak while extracting an OCR crop.
        old_streak = self._streak
        old_triggered = self._triggered
        observation = self.observe(roi_bgr)
        self._streak = old_streak
        self._triggered = old_triggered

        mask = self._novel_mask(roi_bgr)
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            return None, observation

        margin = max(0, int(margin_px))
        x0 = max(self.continuation_x0, int(xs.min()) - margin)
        x1 = min(roi_bgr.shape[1], int(xs.max()) + margin + 1)
        y0 = max(0, int(ys.min()) - margin)
        y1 = min(roi_bgr.shape[0], int(ys.max()) + margin + 1)

        if x1 <= x0 or y1 <= y0:
            return None, observation

        return roi_bgr[y0:y1, x0:x1].copy(), observation


class FastSidePressEventWatcher:
    """Background SIDE watcher.

    It never commands the robot.  It only latches a high-priority event.
    The foreground control thread owns the actual release/retract command.
    """

    def __init__(
        self,
        *,
        side_camera,
        calibration,
        artifact_dir: str | Path,
        config: FastScreenChangeConfig | None = None,
    ) -> None:
        self.side_camera = side_camera
        self.calibration = calibration
        self.artifact_dir = Path(artifact_dir)
        self.config = config or FastScreenChangeConfig()

        self.model: FastScreenChangeModel | None = None
        self._trigger = Event()
        self._stop = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._details: dict | None = None

    @property
    def triggered(self) -> bool:
        return self._trigger.is_set()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._details or {})

    def _roi_from_frame(self, frame) -> np.ndarray | None:
        rectified = self.calibration.rectify(frame.image)
        return self.calibration.crop_text_roi(rectified)

    def arm(self, *, baseline_duration_s: float = 1.4) -> dict:
        if self._thread is not None:
            raise RuntimeError("SIDE event watcher is already armed")

        deadline = time.monotonic() + float(baseline_duration_s)
        last_frame_id = -1
        rois: list[np.ndarray] = []
        first_id = None
        last_id = None

        while time.monotonic() < deadline:
            frame = self.side_camera.latest(copy_image=True)
            if frame is None or frame.frame_id <= last_frame_id:
                time.sleep(self.config.poll_interval_s)
                continue

            last_frame_id = int(frame.frame_id)
            if frame.frame_age_ms > self.config.max_frame_age_ms:
                continue

            roi = self._roi_from_frame(frame)
            if roi is None:
                continue

            rois.append(roi.copy())
            first_id = frame.frame_id if first_id is None else first_id
            last_id = frame.frame_id

        if len(rois) < 8:
            raise RuntimeError(
                "SIDE event baseline did not collect enough fresh frames: "
                f"{len(rois)}"
            )

        self.model = FastScreenChangeModel.from_baseline_rois(
            rois,
            config=self.config,
        )

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        safe = self.model.baseline_safe_mask.astype(np.uint8) * 255
        cv2.imwrite(str(self.artifact_dir / "baseline_safe_mask.png"), safe)
        cv2.imwrite(str(self.artifact_dir / "baseline_last_roi.png"), rois[-1])

        self._stop.clear()
        self._trigger.clear()
        self._thread = Thread(
            target=self._run,
            name="side-press-event-watcher",
            daemon=True,
        )
        self._thread.start()

        return {
            "baseline_frames": len(rois),
            "first_frame_id": first_id,
            "last_frame_id": last_id,
            "continuation_x0": self.model.continuation_x0,
        }

    def _run(self) -> None:
        assert self.model is not None
        last_frame_id = -1

        while not self._stop.is_set() and not self._trigger.is_set():
            frame = self.side_camera.latest(copy_image=True)
            if frame is None or frame.frame_id <= last_frame_id:
                time.sleep(self.config.poll_interval_s)
                continue

            last_frame_id = int(frame.frame_id)
            if frame.frame_age_ms > self.config.max_frame_age_ms:
                continue

            roi = self._roi_from_frame(frame)
            if roi is None:
                continue

            observation = self.model.observe(roi)

            if observation.triggered:
                crop, _ = self.model.extract_novel_crop(roi)

                cv2.imwrite(
                    str(self.artifact_dir / "event_full_frame.png"),
                    frame.image,
                )
                cv2.imwrite(
                    str(self.artifact_dir / "event_text_roi.png"),
                    roi,
                )
                if crop is not None:
                    cv2.imwrite(
                        str(self.artifact_dir / "event_novel_crop.png"),
                        crop,
                    )

                with self._lock:
                    self._details = {
                        "frame_id": int(frame.frame_id),
                        "capture_timestamp": float(frame.capture_timestamp),
                        "novel_pixels": observation.novel_pixels,
                        "max_component_area_px": observation.max_component_area_px,
                        "strong_changed": observation.strong_changed,
                        "consecutive_frames": observation.consecutive_frames,
                        "continuation_x0": observation.continuation_x0,
                        "bbox_xywh": observation.bbox_xywh,
                    }

                self._trigger.set()
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
