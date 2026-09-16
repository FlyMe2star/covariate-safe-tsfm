"""Leakage-safe protocol helpers shared by data and model notebooks."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OriginPartition:
    """Chronological calibration and sealed-evaluation origin indices."""

    calibration: tuple[int, ...]
    evaluation: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.calibration or not self.evaluation:
            raise ValueError("both partitions must be non-empty")
        if max(self.calibration) >= min(self.evaluation):
            raise ValueError("every calibration origin must precede evaluation origins")


def temporal_origin_partition(
    num_origins: int,
    calibration_fraction: float = 0.60,
    minimum_per_partition: int = 2,
) -> OriginPartition:
    """Split already chronological origins without shuffling.

    The calibration size uses floor rounding, then is clipped to retain the requested
    minimum number of origins on both sides.
    """
    if minimum_per_partition < 1:
        raise ValueError("minimum_per_partition must be positive")
    if num_origins < 2 * minimum_per_partition:
        raise ValueError("num_origins is too small for the requested partition minimum")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must lie strictly between zero and one")

    calibration_size = math.floor(num_origins * calibration_fraction)
    calibration_size = max(minimum_per_partition, calibration_size)
    calibration_size = min(num_origins - minimum_per_partition, calibration_size)
    return OriginPartition(
        calibration=tuple(range(calibration_size)),
        evaluation=tuple(range(calibration_size, num_origins)),
    )
