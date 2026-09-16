from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TargetBurstConfig:
    """Robust fixed-pose target-center measurement."""

    frame_count: int = 5
    min_inliers: int = 4
    cluster_radius_px: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.frame_count, bool) or not isinstance(
            self.frame_count, int
        ):
            raise TypeError("frame_count must be an integer")

        if self.frame_count < 1:
            raise ValueError("frame_count must be >= 1")

        if isinstance(self.min_inliers, bool) or not isinstance(
            self.min_inliers, int
        ):
            raise TypeError("min_inliers must be an integer")

        if not 1 <= self.min_inliers <= self.frame_count:
            raise ValueError(
                "min_inliers must satisfy "
                "1 <= min_inliers <= frame_count"
            )

        radius = float(self.cluster_radius_px)

        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(
                "cluster_radius_px must be finite and positive"
            )

        object.__setattr__(
            self,
            "cluster_radius_px",
            radius,
        )


@dataclass(frozen=True, slots=True)
class TargetBurstResult:
    accepted: bool
    center_px: tuple[float, float] | None
    inlier_indices: tuple[int, ...]
    outlier_indices: tuple[int, ...]
    inlier_count: int
    total_count: int
    max_inlier_radius_px: float | None


def robust_target_center(
    centers_px: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: TargetBurstConfig | None = None,
) -> TargetBurstResult:
    """Estimate a target center from one fixed-pose observation burst.

    This function is intentionally motion-agnostic.  It is for several
    fresh observations acquired while the robot is stationary.

    Consensus is found from pairwise image-space proximity, then the
    accepted center is the coordinate-wise median of the consensus
    inliers.
    """

    cfg = config or TargetBurstConfig()

    points = np.asarray(
        centers_px,
        dtype=np.float64,
    )

    if points.shape != (cfg.frame_count, 2):
        raise ValueError(
            "centers_px must have shape "
            f"({cfg.frame_count}, 2)"
        )

    if not np.isfinite(points).all():
        raise ValueError(
            "centers_px must contain only finite values"
        )

    delta = (
        points[:, None, :]
        - points[None, :, :]
    )

    distances = np.linalg.norm(
        delta,
        axis=2,
    )

    neighbor_masks = (
        distances
        <= cfg.cluster_radius_px
    )

    neighbor_counts = np.sum(
        neighbor_masks,
        axis=1,
    )

    # Prefer the seed with the largest consensus.
    # For ties, prefer the seed with the smallest
    # total distance to its consensus neighbors.
    best_seed = None
    best_key = None

    for index in range(cfg.frame_count):
        mask = neighbor_masks[index]
        count = int(neighbor_counts[index])

        local_distance_sum = float(
            np.sum(
                distances[index, mask]
            )
        )

        key = (
            count,
            -local_distance_sum,
            -index,
        )

        if best_key is None or key > best_key:
            best_key = key
            best_seed = index

    assert best_seed is not None

    seed_mask = neighbor_masks[
        best_seed
    ]

    seed_indices = np.flatnonzero(
        seed_mask
    )

    if len(seed_indices) < cfg.min_inliers:
        return TargetBurstResult(
            accepted=False,
            center_px=None,
            inlier_indices=(),
            outlier_indices=tuple(
                range(cfg.frame_count)
            ),
            inlier_count=0,
            total_count=cfg.frame_count,
            max_inlier_radius_px=None,
        )

    provisional_center = np.median(
        points[seed_indices],
        axis=0,
    )

    radial = np.linalg.norm(
        points - provisional_center,
        axis=1,
    )

    final_mask = (
        radial
        <= cfg.cluster_radius_px
    )

    final_indices = np.flatnonzero(
        final_mask
    )

    if len(final_indices) < cfg.min_inliers:
        return TargetBurstResult(
            accepted=False,
            center_px=None,
            inlier_indices=tuple(
                int(i)
                for i in final_indices
            ),
            outlier_indices=tuple(
                int(i)
                for i in np.flatnonzero(
                    ~final_mask
                )
            ),
            inlier_count=len(final_indices),
            total_count=cfg.frame_count,
            max_inlier_radius_px=(
                None
                if len(final_indices) == 0
                else float(
                    np.max(
                        radial[final_indices]
                    )
                )
            ),
        )

    center = np.median(
        points[final_indices],
        axis=0,
    )

    final_radial = np.linalg.norm(
        points - center,
        axis=1,
    )

    # Re-evaluate once around the final robust center.
    final_mask = (
        final_radial
        <= cfg.cluster_radius_px
    )

    final_indices = np.flatnonzero(
        final_mask
    )

    outlier_indices = np.flatnonzero(
        ~final_mask
    )

    if len(final_indices) < cfg.min_inliers:
        return TargetBurstResult(
            accepted=False,
            center_px=None,
            inlier_indices=tuple(
                int(i)
                for i in final_indices
            ),
            outlier_indices=tuple(
                int(i)
                for i in outlier_indices
            ),
            inlier_count=len(final_indices),
            total_count=cfg.frame_count,
            max_inlier_radius_px=(
                None
                if len(final_indices) == 0
                else float(
                    np.max(
                        final_radial[
                            final_indices
                        ]
                    )
                )
            ),
        )

    center = np.median(
        points[final_indices],
        axis=0,
    )

    final_radial = np.linalg.norm(
        points - center,
        axis=1,
    )

    return TargetBurstResult(
        accepted=True,
        center_px=(
            float(center[0]),
            float(center[1]),
        ),
        inlier_indices=tuple(
            int(i)
            for i in final_indices
        ),
        outlier_indices=tuple(
            int(i)
            for i in outlier_indices
        ),
        inlier_count=len(final_indices),
        total_count=cfg.frame_count,
        max_inlier_radius_px=float(
            np.max(
                final_radial[
                    final_indices
                ]
            )
        ),
    )
