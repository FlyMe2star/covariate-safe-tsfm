import math

import pytest

from covsafe.p0b import (
    enumerate_candidate_policies,
    rank_historical_covariates,
    summarize_oracle_records,
)


def test_candidate_generator_deduplicates_equivalent_epf_policies() -> None:
    task = {
        "known_dynamic_columns": ["B", "A"],
        "past_dynamic_columns": [],
    }
    ranking = [
        {"column": "A", "score": 0.9},
        {"column": "B", "score": 0.5},
    ]

    policies = enumerate_candidate_policies(task, ranking)

    assert [policy.policy_id for policy in policies] == [
        "target_only",
        "all_dynamic",
        "single_known_future:A",
        "single_known_future:B",
    ]
    all_dynamic = policies[1]
    assert "correlation_top_2" in all_dynamic.aliases
    assert "correlation_top_4" in all_dynamic.aliases
    assert "leave_one_out:B" in policies[2].aliases


def test_historical_ranking_excludes_zero_variance_pairs() -> None:
    past_data = [
        {
            "target": [1.0, 2.0, 3.0, 4.0],
            "informative": [4.0, 3.0, 2.0, 1.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
        }
    ]

    ranking = rank_historical_covariates(
        past_data,
        target_columns=["target"],
        dynamic_columns=["constant", "informative"],
    )

    assert ranking[0]["column"] == "informative"
    assert ranking[0]["score"] == pytest.approx(1.0)
    assert ranking[1]["column"] == "constant"
    assert ranking[1]["score"] is None


def _baseline(task: str, variant: str, sql: float) -> dict[str, object]:
    return {
        "dataset_config": task,
        "origin_index": 0,
        "item_id": "item",
        "target": "target",
        "variant": variant,
        "SQL": sql,
    }


def _candidate(task: str, policy: str, sql: float) -> dict[str, object]:
    return {
        "dataset_config": task,
        "origin_index": 0,
        "item_id": "item",
        "target": "target",
        "policy_id": policy,
        "SQL": sql,
    }


def test_oracle_headroom_uses_stronger_constant_and_task_macro() -> None:
    baselines: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for task in ("a", "b", "c", "d"):
        baselines.extend(
            [
                _baseline(task, "target_only", 1.0),
                _baseline(task, "all_dynamic", 1.1),
            ]
        )
        candidates.append(_candidate(task, "subset", 0.9))

    report = summarize_oracle_records(baselines, candidates)

    assert report["task_macro_mean_oracle_headroom"] == pytest.approx(0.1)
    assert report["unit_micro_mean_oracle_headroom"] == pytest.approx(0.1)
    assert report["h2_gate_passed"] is True
    assert report["per_task"]["a"]["oracle_winner_counts"] == {"subset": 1}


def test_oracle_excludes_nonfinite_constant_pair() -> None:
    baselines: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for task in ("a", "b", "c", "d"):
        baselines.extend(
            [
                _baseline(task, "target_only", math.nan if task == "a" else 1.0),
                _baseline(task, "all_dynamic", 1.0),
            ]
        )
        candidates.append(_candidate(task, "subset", 0.9))

    with pytest.raises(ValueError, match="no finite units"):
        summarize_oracle_records(baselines, candidates)
