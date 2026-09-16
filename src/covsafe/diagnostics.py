"""Numerical definitions for the pre-registered P0 gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


def _as_finite_1d(values: npt.ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def relative_gain(
    baseline_loss: npt.ArrayLike,
    candidate_loss: npt.ArrayLike,
    epsilon: float = 1e-12,
) -> FloatArray:
    """Compute relative loss reduction; positive values mean the candidate is better."""
    baseline = np.asarray(baseline_loss, dtype=np.float64)
    candidate = np.asarray(candidate_loss, dtype=np.float64)
    if baseline.shape != candidate.shape:
        raise ValueError("baseline_loss and candidate_loss must have identical shapes")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("loss arrays must be finite")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return (baseline - candidate) / np.maximum(np.abs(baseline), epsilon)


def harm_rate(
    target_only_loss: npt.ArrayLike,
    all_covariate_loss: npt.ArrayLike,
    harmful_relative_loss: float = 0.05,
) -> float:
    """Fraction of units where all covariates increase loss by at least the threshold."""
    if harmful_relative_loss <= 0:
        raise ValueError("harmful_relative_loss must be positive")
    gain = relative_gain(target_only_loss, all_covariate_loss)
    if gain.size == 0:
        raise ValueError("loss arrays must not be empty")
    return float(np.mean(gain <= -harmful_relative_loss))


def finite_oracle_headroom(
    target_only_loss: npt.ArrayLike,
    all_covariate_loss: npt.ArrayLike,
    candidate_losses: npt.ArrayLike,
) -> FloatArray:
    """Per-unit headroom of the finite candidate oracle over the stronger constant policy.

    ``candidate_losses`` has shape ``(num_candidates, num_units)`` and must contain the
    two constant policies if they are intended to be part of the oracle.
    """
    target = _as_finite_1d(target_only_loss, "target_only_loss")
    all_covariates = _as_finite_1d(all_covariate_loss, "all_covariate_loss")
    if target.shape != all_covariates.shape:
        raise ValueError("target_only_loss and all_covariate_loss must match")
    candidates = np.asarray(candidate_losses, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != target.size:
        raise ValueError("candidate_losses must have shape (num_candidates, num_units)")
    if candidates.shape[0] == 0 or not np.isfinite(candidates).all():
        raise ValueError("candidate_losses must be non-empty and finite")
    stronger_constant = np.minimum(target, all_covariates)
    oracle = np.min(candidates, axis=0)
    return relative_gain(stronger_constant, oracle)


def binary_auroc(labels: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Compute AUROC from average ranks, including exact handling of tied scores."""
    y = np.asarray(labels)
    score = _as_finite_1d(scores, "scores")
    if y.ndim != 1 or y.shape != score.shape:
        raise ValueError("labels and scores must be one-dimensional with identical shapes")
    if not np.isin(y, [0, 1, False, True]).all():
        raise ValueError("labels must be binary")
    y = y.astype(np.int8)
    num_positive = int(y.sum())
    num_negative = int(y.size - num_positive)
    if num_positive == 0 or num_negative == 0:
        raise ValueError("AUROC requires both positive and negative labels")

    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        end = start + 1
        while end < score.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[y == 1].sum())
    u_statistic = positive_rank_sum - num_positive * (num_positive + 1) / 2.0
    return u_statistic / (num_positive * num_negative)


@dataclass(frozen=True)
class ModelEvidence:
    """P0 aggregate evidence for one frozen backbone."""

    harm_rate: float
    mean_oracle_headroom: float
    utility_sign_auroc: float

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")


@dataclass(frozen=True)
class GateThresholds:
    minimum_harm_rate: float = 0.20
    minimum_mean_oracle_headroom: float = 0.05
    minimum_utility_sign_auroc: float = 0.65
    minimum_backbones_passing: int = 2


@dataclass(frozen=True)
class GateReport:
    passed: bool
    per_backbone: Mapping[str, Mapping[str, bool]]
    passing_backbones: tuple[str, ...]


def evaluate_p0_gates(
    evidence: Mapping[str, ModelEvidence],
    thresholds: GateThresholds | None = None,
) -> GateReport:
    """Evaluate all pre-registered gates independently for every supplied backbone."""
    thresholds = thresholds or GateThresholds()
    if not evidence:
        raise ValueError("evidence must contain at least one backbone")
    if thresholds.minimum_backbones_passing < 1:
        raise ValueError("minimum_backbones_passing must be positive")

    per_backbone: dict[str, dict[str, bool]] = {}
    passing: list[str] = []
    for backbone, values in evidence.items():
        checks = {
            "harm_rate": values.harm_rate >= thresholds.minimum_harm_rate,
            "oracle_headroom": (
                values.mean_oracle_headroom >= thresholds.minimum_mean_oracle_headroom
            ),
            "utility_sign_auroc": (
                values.utility_sign_auroc >= thresholds.minimum_utility_sign_auroc
            ),
        }
        per_backbone[backbone] = checks
        if all(checks.values()):
            passing.append(backbone)

    return GateReport(
        passed=len(passing) >= thresholds.minimum_backbones_passing,
        per_backbone=per_backbone,
        passing_backbones=tuple(sorted(passing)),
    )
