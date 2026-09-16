import pytest

from covsafe.protocol import temporal_origin_partition


@pytest.mark.parametrize(
    ("num_origins", "expected_calibration", "expected_evaluation"),
    [
        (5, (0, 1, 2), (3, 4)),
        (10, tuple(range(6)), tuple(range(6, 10))),
        (20, tuple(range(12)), tuple(range(12, 20))),
    ],
)
def test_temporal_origin_partition(
    num_origins: int,
    expected_calibration: tuple[int, ...],
    expected_evaluation: tuple[int, ...],
) -> None:
    partition = temporal_origin_partition(num_origins)
    assert partition.calibration == expected_calibration
    assert partition.evaluation == expected_evaluation


def test_temporal_partition_rejects_insufficient_origins() -> None:
    with pytest.raises(ValueError, match="too small"):
        temporal_origin_partition(3, minimum_per_partition=2)
