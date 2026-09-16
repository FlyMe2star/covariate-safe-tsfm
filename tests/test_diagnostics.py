import numpy as np
import pytest

from covsafe.diagnostics import (
    GateThresholds,
    ModelEvidence,
    binary_auroc,
    evaluate_p0_gates,
    finite_oracle_headroom,
    harm_rate,
    relative_gain,
)


def test_relative_gain_sign_and_scale() -> None:
    actual = relative_gain([10.0, 10.0], [8.0, 12.0])
    np.testing.assert_allclose(actual, [0.2, -0.2])


def test_harm_rate_uses_preregistered_threshold() -> None:
    target = [100.0, 100.0, 100.0, 100.0]
    all_covariates = [106.0, 105.0, 104.0, 120.0]
    assert harm_rate(target, all_covariates, harmful_relative_loss=0.05) == 0.75


def test_finite_oracle_headroom_uses_stronger_constant() -> None:
    target = [10.0, 12.0]
    all_covariates = [9.0, 15.0]
    candidates = [[10.0, 12.0], [9.0, 15.0], [8.0, 9.0]]
    actual = finite_oracle_headroom(target, all_covariates, candidates)
    np.testing.assert_allclose(actual, [1.0 / 9.0, 0.25])


def test_binary_auroc_perfect_and_tied() -> None:
    assert binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auroc([0, 1], [0.5, 0.5]) == 0.5


def test_binary_auroc_rejects_one_class() -> None:
    with pytest.raises(ValueError, match="both positive and negative"):
        binary_auroc([1, 1], [0.1, 0.2])


def test_gate_requires_two_complete_backbones() -> None:
    evidence = {
        "chronos_2": ModelEvidence(0.25, 0.06, 0.70),
        "timesfm_3": ModelEvidence(0.30, 0.07, 0.64),
    }
    report = evaluate_p0_gates(evidence, GateThresholds())
    assert report.passed is False
    assert report.passing_backbones == ("chronos_2",)
    assert report.per_backbone["timesfm_3"]["utility_sign_auroc"] is False


def test_gate_passes_at_exact_boundaries() -> None:
    boundary = ModelEvidence(0.20, 0.05, 0.65)
    report = evaluate_p0_gates({"chronos_2": boundary, "timesfm_3": boundary})
    assert report.passed is True
