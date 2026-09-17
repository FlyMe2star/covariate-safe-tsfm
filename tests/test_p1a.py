from pathlib import Path

import numpy as np
import pytest

from covsafe.p0c import tie_aware_binary_auroc
from covsafe.p1a import (
    EXPECTED_P1A_CONFIG_HASH,
    _weighted_auroc_from_cluster_counts,
    cluster_bootstrap_auroc,
    evaluate_continuation_gate,
    load_frozen_p1a,
    summarize_applicability_group,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rows(scores_labels):
    rows = []
    for origin, (score, label) in enumerate(scores_labels):
        for policy in ("p1", "p2"):
            rows.append(
                {
                    "item_id": f"item-{origin // 2}",
                    "target": "y",
                    "origin_index": origin,
                    "policy_id": policy,
                    "historical_utility_score": score,
                    "label_positive": label,
                }
            )
    return rows


def test_frozen_p1a_config_loads():
    config, decision = load_frozen_p1a(REPO_ROOT)
    assert EXPECTED_P1A_CONFIG_HASH != "TO_BE_FROZEN"
    assert config["scope"]["calibration_origins_only"] is True
    assert decision["gate"]["passed"] is False


def test_cluster_bootstrap_is_deterministic_and_perfect():
    rows = _rows([(0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)])
    first = cluster_bootstrap_auroc(rows, replicates=200, seed=7)
    second = cluster_bootstrap_auroc(rows, replicates=200, seed=7)
    assert first == second
    assert first["point_auroc"] == pytest.approx(1.0)
    assert first["one_sided_lower_bound"] == pytest.approx(1.0)
    assert first["cluster_count"] == 4


def test_weighted_cluster_auroc_matches_explicit_cluster_duplication():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.4, 0.9], dtype=np.float64)
    cluster_ids = np.asarray([0, 0, 1, 2], dtype=np.int64)
    _unique, tie_groups = np.unique(scores, return_inverse=True)
    counts = np.asarray([2, 0, 1], dtype=np.int64)
    weighted = _weighted_auroc_from_cluster_counts(
        labels, tie_groups, cluster_ids, counts
    )
    expanded_labels = [0, 1, 0, 1, 1]
    expanded_scores = [0.1, 0.4, 0.1, 0.4, 0.9]
    assert weighted == pytest.approx(
        tie_aware_binary_auroc(expanded_labels, expanded_scores)
    )


def test_group_requires_point_and_confidence_thresholds():
    rows = _rows([(0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)])
    result = summarize_applicability_group(
        rows,
        replicates=200,
        seed=11,
        lower_quantile=0.05,
        minimum_valid_replicate_fraction=0.5,
        minimum_scored_rows=4,
        point_auroc_at_least=0.60,
        lower_bound_strictly_greater_than=0.50,
    )
    assert result["eligible"] is True
    assert all(result["checks"].values())


def test_continuation_requires_three_groups_and_each_backbone():
    groups = {
        "chronos_2": {"a": {"eligible": True}, "b": {"eligible": False}},
        "timesfm_3": {"a": {"eligible": True}, "b": {"eligible": True}},
    }
    result = evaluate_continuation_gate(
        groups,
        minimum_eligible_groups_total=3,
        minimum_eligible_tasks_per_backbone=1,
    )
    assert result["passed"] is True
    groups["chronos_2"]["a"]["eligible"] = False
    result = evaluate_continuation_gate(
        groups,
        minimum_eligible_groups_total=3,
        minimum_eligible_tasks_per_backbone=1,
    )
    assert result["passed"] is False
