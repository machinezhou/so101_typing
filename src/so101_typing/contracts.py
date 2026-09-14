from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CameraFrame:
    camera_name: str
    frame_id: int
    capture_timestamp: float
    processing_timestamp: float
    image: np.ndarray

    @property
    def frame_age_ms(self) -> float:
        return max(
            0.0,
            (self.processing_timestamp - self.capture_timestamp) * 1000.0,
        )
