from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
import io
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ScreenOCRConfig:
    language: str = "eng"
    psm: int = 6
    scale: float = 2.0
    clahe_clip_limit: float = 2.0
    min_word_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise ValueError("language must be non-empty")

        if self.psm < 0:
            raise ValueError("psm must be non-negative")

        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and positive")

        if (
            not math.isfinite(self.clahe_clip_limit)
            or self.clahe_clip_limit <= 0.0
        ):
            raise ValueError(
                "clahe_clip_limit must be finite and positive"
            )

        if not math.isfinite(self.min_word_confidence):
            raise ValueError(
                "min_word_confidence must be finite"
            )


@dataclass(frozen=True, slots=True)
class ScreenOCRObservation:
    raw_text: str
    normalized_text: str
    mean_confidence: float
    word_count: int
    processed_image: np.ndarray


@dataclass(frozen=True, slots=True)
class ScreenOCRConsensus:
    stable: bool
    text: str | None
    votes: int
    total: int
    vote_fraction: float
    mean_confidence: float


def normalize_ocr_text(text: str) -> str:
    """Conservative normalization.

    Do not rewrite visually ambiguous characters such as O/0 or I/1.
    Verification must never manufacture the expected character.
    """

    normalized_lines: list[str] = []

    for raw_line in str(text).splitlines():
        line = " ".join(raw_line.strip().split())

        if line:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def preprocess_screen_roi(
    roi_bgr: np.ndarray,
    *,
    config: ScreenOCRConfig | None = None,
) -> np.ndarray:
    if config is None:
        config = ScreenOCRConfig()

    image = np.asarray(roi_bgr)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "screen ROI must be a BGR image with shape HxWx3"
        )

    if image.size == 0:
        raise ValueError("screen ROI must not be empty")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    scaled = cv2.resize(
        gray,
        None,
        fx=config.scale,
        fy=config.scale,
        interpolation=cv2.INTER_CUBIC,
    )

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(8, 8),
    )

    return clahe.apply(scaled)


def parse_tesseract_tsv(
    tsv_text: str,
    *,
    min_word_confidence: float = 0.0,
) -> tuple[str, float, int]:
    reader = csv.DictReader(
        io.StringIO(tsv_text),
        delimiter="\t",
    )

    lines: dict[
        tuple[int, int, int, int],
        list[str],
    ] = {}

    confidences: list[float] = []

    for row in reader:
        # Tesseract TSV level=5 is a recognized word.  Page/block/line
        # metadata rows must never leak into screen text.
        if row.get("level") != "5":
            continue

        word = (row.get("text") or "").strip()

        if not word:
            continue

        try:
            confidence = float(
                row.get("conf", "-1")
            )
        except ValueError:
            continue

        if confidence < min_word_confidence:
            continue

        try:
            key = (
                int(row.get("page_num", "0")),
                int(row.get("block_num", "0")),
                int(row.get("par_num", "0")),
                int(row.get("line_num", "0")),
            )
        except ValueError:
            continue

        lines.setdefault(key, []).append(word)

        if confidence >= 0.0:
            confidences.append(confidence)

    ordered_lines = [
        " ".join(lines[key])
        for key in sorted(lines)
        if lines[key]
    ]

    raw_text = "\n".join(ordered_lines)
    normalized = normalize_ocr_text(raw_text)

    mean_confidence = (
        float(np.mean(confidences))
        if confidences
        else 0.0
    )

    return (
        normalized,
        mean_confidence,
        len(confidences),
    )


class TesseractScreenOCR:
    def __init__(
        self,
        config: ScreenOCRConfig | None = None,
    ) -> None:
        self.config = config or ScreenOCRConfig()

        executable = shutil.which("tesseract")

        if executable is None:
            raise RuntimeError(
                "tesseract executable was not found"
            )

        self.executable = executable

    def recognize(
        self,
        roi_bgr: np.ndarray,
    ) -> ScreenOCRObservation:
        processed = preprocess_screen_roi(
            roi_bgr,
            config=self.config,
        )

        with tempfile.TemporaryDirectory() as directory:
            input_path = (
                Path(directory)
                / "screen_ocr_input.png"
            )

            if not cv2.imwrite(
                str(input_path),
                processed,
            ):
                raise RuntimeError(
                    "failed to write temporary OCR image"
                )

            completed = subprocess.run(
                [
                    self.executable,
                    str(input_path),
                    "stdout",
                    "--oem",
                    "1",
                    "--psm",
                    str(self.config.psm),
                    "-l",
                    self.config.language,
                    "tsv",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        if completed.returncode != 0:
            detail = (
                completed.stderr.strip()
                or f"return code {completed.returncode}"
            )

            raise RuntimeError(
                f"Tesseract OCR failed: {detail}"
            )

        normalized, confidence, word_count = (
            parse_tesseract_tsv(
                completed.stdout,
                min_word_confidence=(
                    self.config.min_word_confidence
                ),
            )
        )

        return ScreenOCRObservation(
            raw_text=normalized,
            normalized_text=normalized,
            mean_confidence=confidence,
            word_count=word_count,
            processed_image=processed,
        )


def consensus_screen_text(
    observations: list[ScreenOCRObservation],
    *,
    min_vote_fraction: float = 0.60,
    min_mean_confidence: float = 60.0,
) -> ScreenOCRConsensus:
    if not observations:
        raise ValueError(
            "at least one OCR observation is required"
        )

    if not 0.0 < min_vote_fraction <= 1.0:
        raise ValueError(
            "min_vote_fraction must be in (0, 1]"
        )

    texts = [
        observation.normalized_text
        for observation in observations
        if observation.normalized_text
    ]

    if not texts:
        return ScreenOCRConsensus(
            stable=False,
            text=None,
            votes=0,
            total=len(observations),
            vote_fraction=0.0,
            mean_confidence=0.0,
        )

    counts = Counter(texts)

    winner, votes = counts.most_common(1)[0]

    winner_observations = [
        observation
        for observation in observations
        if observation.normalized_text == winner
    ]

    vote_fraction = (
        votes / len(observations)
    )

    mean_confidence = float(
        np.mean(
            [
                observation.mean_confidence
                for observation in winner_observations
            ]
        )
    )

    stable = (
        vote_fraction >= min_vote_fraction
        and mean_confidence >= min_mean_confidence
    )

    return ScreenOCRConsensus(
        stable=stable,
        text=winner,
        votes=votes,
        total=len(observations),
        vote_fraction=vote_fraction,
        mean_confidence=mean_confidence,
    )


# ===== PHASE4 LINE-LEVEL OCR =====


@dataclass(frozen=True, slots=True)
class ScreenTextLineBox:
    x: int
    y: int
    width: int
    height: int


def detect_screen_text_lines(
    roi_bgr: np.ndarray,
    *,
    dark_threshold: int = 185,
    min_row_ink_px: int = 8,
    max_row_gap_px: int = 5,
    min_line_height_px: int = 8,
    max_line_height_px: int = 90,
    x_margin_px: int = 12,
    y_margin_px: int = 8,
) -> tuple[ScreenTextLineBox, ...]:
    """Find dark text-line bands inside the already-rectified typing ROI.

    This deliberately performs geometry only.  It does not know the expected
    character or confirmed prefix and therefore cannot bias verification toward
    the desired result.
    """

    image = np.asarray(roi_bgr)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "screen ROI must be a BGR image with shape HxWx3"
        )

    if image.size == 0:
        raise ValueError("screen ROI must not be empty")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = gray.shape

    threshold = int(dark_threshold)

    if not 0 <= threshold <= 255:
        raise ValueError(
            "dark_threshold must be in [0, 255]"
        )

    # Absolute dark-pixel threshold is intentional here.  The calibrated
    # MacBook text is dark on a bright screen, while the large illumination
    # gradient across the screen must not become foreground as it can with a
    # whole-ROI Otsu threshold.
    dark = gray < threshold

    row_ink = np.count_nonzero(
        dark,
        axis=1,
    )

    required_row_ink = max(
        int(min_row_ink_px),
        int(round(width * 0.005)),
    )

    active_rows = np.flatnonzero(
        row_ink >= required_row_ink
    )

    if active_rows.size == 0:
        return ()

    groups: list[tuple[int, int]] = []

    start = int(active_rows[0])
    previous = start

    for raw_y in active_rows[1:]:
        y = int(raw_y)

        if y - previous <= int(max_row_gap_px):
            previous = y
            continue

        groups.append(
            (start, previous)
        )

        start = y
        previous = y

    groups.append(
        (start, previous)
    )

    boxes: list[ScreenTextLineBox] = []

    for raw_y0, raw_y1 in groups:
        raw_height = (
            raw_y1 - raw_y0 + 1
        )

        if (
            raw_height < min_line_height_px
            or raw_height > max_line_height_px
        ):
            continue

        band = dark[
            raw_y0 : raw_y1 + 1,
            :
        ]

        column_ink = np.count_nonzero(
            band,
            axis=0,
        )

        active_columns = np.flatnonzero(
            column_ink >= 2
        )

        if active_columns.size == 0:
            continue

        raw_x0 = int(
            active_columns[0]
        )

        raw_x1 = int(
            active_columns[-1]
        )

        x0 = max(
            0,
            raw_x0 - int(x_margin_px),
        )

        x1 = min(
            width - 1,
            raw_x1 + int(x_margin_px),
        )

        y0 = max(
            0,
            raw_y0 - int(y_margin_px),
        )

        y1 = min(
            height - 1,
            raw_y1 + int(y_margin_px),
        )

        boxes.append(
            ScreenTextLineBox(
                x=x0,
                y=y0,
                width=x1 - x0 + 1,
                height=y1 - y0 + 1,
            )
        )

    return tuple(boxes)


class TesseractScreenLineOCR:
    """OCR the rectified typing ROI one detected text line at a time.

    Whole-screen PSM 6 is intentionally avoided.  A short typing line inside a
    large mostly-empty ROI caused layout noise to dominate Tesseract.  Each
    detected physical line is instead recognized with PSM 7.

    The whitelist describes the typing task's character domain; it never
    contains or depends on the expected character for a particular press.
    """

    DEFAULT_WHITELIST = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
    )

    def __init__(
        self,
        *,
        language: str = "eng",
        scale: float = 3.0,
        clahe_clip_limit: float = 2.0,
        dark_threshold: int = 185,
        whitelist: str = DEFAULT_WHITELIST,
    ) -> None:
        executable = shutil.which(
            "tesseract"
        )

        if executable is None:
            raise RuntimeError(
                "tesseract executable was not found"
            )

        if not language.strip():
            raise ValueError(
                "language must be non-empty"
            )

        if (
            not math.isfinite(scale)
            or scale <= 0.0
        ):
            raise ValueError(
                "scale must be finite and positive"
            )

        if (
            not math.isfinite(
                clahe_clip_limit
            )
            or clahe_clip_limit <= 0.0
        ):
            raise ValueError(
                "clahe_clip_limit must be finite and positive"
            )

        if not whitelist:
            raise ValueError(
                "whitelist must be non-empty"
            )

        self.executable = executable
        self.language = language
        self.scale = float(scale)
        self.clahe_clip_limit = float(
            clahe_clip_limit
        )
        self.dark_threshold = int(
            dark_threshold
        )
        self.whitelist = str(
            whitelist
        )

    def _prepare_line(
        self,
        line_bgr: np.ndarray,
    ) -> np.ndarray:
        gray = cv2.cvtColor(
            line_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        enlarged = cv2.resize(
            gray,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_CUBIC,
        )

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(8, 8),
        )

        return clahe.apply(
            enlarged
        )

    def _recognize_line(
        self,
        line_bgr: np.ndarray,
    ) -> tuple[str, float, int]:
        processed = self._prepare_line(
            line_bgr
        )

        with tempfile.TemporaryDirectory() as directory:
            input_path = (
                Path(directory)
                / "screen_line.png"
            )

            if not cv2.imwrite(
                str(input_path),
                processed,
            ):
                raise RuntimeError(
                    "failed to write temporary line OCR image"
                )

            completed = subprocess.run(
                [
                    self.executable,
                    str(input_path),
                    "stdout",
                    "--oem",
                    "1",
                    "--psm",
                    "7",
                    "-l",
                    self.language,
                    "-c",
                    (
                        "tessedit_char_whitelist="
                        + self.whitelist
                    ),
                    "tsv",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        if completed.returncode != 0:
            detail = (
                completed.stderr.strip()
                or (
                    "return code "
                    f"{completed.returncode}"
                )
            )

            raise RuntimeError(
                f"Tesseract line OCR failed: {detail}"
            )

        return parse_tesseract_tsv(
            completed.stdout,
            min_word_confidence=0.0,
        )

    def recognize(
        self,
        roi_bgr: np.ndarray,
    ) -> ScreenOCRObservation:
        boxes = detect_screen_text_lines(
            roi_bgr,
            dark_threshold=self.dark_threshold,
        )

        recognized_lines: list[str] = []
        confidence_sum = 0.0
        confidence_weight = 0
        total_words = 0

        for box in boxes:
            crop = roi_bgr[
                box.y : box.y + box.height,
                box.x : box.x + box.width,
            ]

            (
                text,
                mean_confidence,
                word_count,
            ) = self._recognize_line(
                crop
            )

            compact = normalize_ocr_text(
                text
            )

            if not compact:
                continue

            # PSM 7 should produce one line.  Flatten any accidental line break
            # rather than allowing it to create a fake second screen line.
            compact = "".join(
                compact.splitlines()
            )

            recognized_lines.append(
                compact
            )

            total_words += int(
                word_count
            )

            if word_count > 0:
                confidence_sum += (
                    float(mean_confidence)
                    * int(word_count)
                )

                confidence_weight += int(
                    word_count
                )

        combined = "\n".join(
            recognized_lines
        )

        mean_confidence = (
            confidence_sum
            / confidence_weight
            if confidence_weight
            else 0.0
        )

        # Keep a deterministic diagnostic image in the same observation
        # contract used by the original whole-ROI baseline.
        diagnostic = preprocess_screen_roi(
            roi_bgr,
            config=ScreenOCRConfig(
                scale=2.0,
                clahe_clip_limit=2.0,
            ),
        )

        return ScreenOCRObservation(
            raw_text=combined,
            normalized_text=combined,
            mean_confidence=float(
                mean_confidence
            ),
            word_count=int(
                total_words
            ),
            processed_image=diagnostic,
        )

class TesseractSingleCharacterOCR(TesseractScreenLineOCR):
    """Recognize exactly one released key continuation character.

    Unlike the stable whole-line OCR, lowercase letters are intentionally
    allowed here because a physical keyboard press in the editor produces a
    lowercase character.  PSM 10 treats the crop as one character.
    """

    DEFAULT_WHITELIST = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
    )

    def __init__(
        self,
        *,
        language: str = "eng",
        scale: float = 4.0,
        clahe_clip_limit: float = 2.0,
        dark_threshold: int = 185,
        whitelist: str = DEFAULT_WHITELIST,
    ) -> None:
        super().__init__(
            language=language,
            scale=scale,
            clahe_clip_limit=clahe_clip_limit,
            dark_threshold=dark_threshold,
            whitelist=whitelist,
        )

    def _recognize_line(
        self,
        line_bgr: np.ndarray,
    ) -> tuple[str, float, int]:
        processed = self._prepare_line(line_bgr)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "screen_character.png"

            if not cv2.imwrite(str(input_path), processed):
                raise RuntimeError("failed to write temporary character OCR image")

            completed = subprocess.run(
                [
                    self.executable,
                    str(input_path),
                    "stdout",
                    "--oem",
                    "1",
                    "--psm",
                    "10",
                    "-l",
                    self.language,
                    "-c",
                    "tessedit_char_whitelist=" + self.whitelist,
                    "tsv",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"return code {completed.returncode}"
            raise RuntimeError(f"Tesseract character OCR failed: {detail}")

        return parse_tesseract_tsv(
            completed.stdout,
            min_word_confidence=0.0,
        )

    def recognize_character(
        self,
        crop_bgr: np.ndarray,
    ) -> tuple[str | None, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0

        text, confidence, _ = self._recognize_line(crop_bgr)
        compact = "".join(ch for ch in normalize_ocr_text(text) if ch.isalnum())

        if len(compact) != 1:
            return None, float(confidence)

        return compact.upper(), float(confidence)
