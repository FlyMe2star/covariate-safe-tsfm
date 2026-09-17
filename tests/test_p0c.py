import pytest

from covsafe.p0c import (
    build_prequential_score_records,
    summarize_predictability_records,
    tie_aware_binary_auroc,
)


def _row(origin: int, policy: str, sql: float) -> dict[str, object]:
    return {
        "backbone": "model",
        "dataset_config": "task",
        "origin_index": origin,
        "cutoff": str(origin),
        "policy_id": policy,
        "item_id": "item",
        "target": "target",
        "SQL": sql,
    }


def test_prequential_score_uses_only_strictly_earlier_origins() -> None:
    records = []
    for origin, candidate_sql in enumerate((0.8, 1.2, 0.7)):
        records.extend(
            [
                _row(origin, "target_only", 1.0),
                _row(origin, "candidate", candidate_sql),
            ]
        )

    scored, diagnostics = build_prequential_score_records(records)

    assert len(scored) == 2
    assert scored[0]["origin_index"] == 1
    assert scored[0]["historical_utility_score"] == pytest.approx(0.2)
    assert scored[0]["realized_relative_utility"] == pytest.approx(-0.2)
    assert scored[0]["label_positive"] == 0
    assert scored[0]["latest_prior_origin_index"] == 0
    assert scored[1]["latest_prior_origin_index"] == 1
    assert diagnostics["task"]["excluded_insufficient_finite_history"] == 1


def test_tie_aware_binary_auroc_gives_half_credit_for_ties() -> None:
    value = tie_aware_binary_auroc([0, 0, 1, 1], [0.0, 1.0, 1.0, 2.0])
    assert value == pytest.approx(0.875)


def test_predictability_summary_uses_equal_task_macro() -> None:
    tasks = ["a", "b", "c", "d"]
    records = []
    for task in tasks:
        records.extend(
            {
                "dataset_config": task,
                "historical_utility_score": float(label),
                "label_positive": label,
            }
            for label in ([0] * 10 + [1] * 10)
        )

    report = summarize_predictability_records(records, required_tasks=tasks)

    assert report["task_macro_auroc"] == pytest.approx(1.0)
    assert report["unit_micro_auroc"] == pytest.approx(1.0)
    assert report["all_tasks_valid"] is True
    assert report["h3_precursor_gate_passed"] is True


def test_invalid_task_is_not_silently_removed_from_macro() -> None:
    tasks = ["a", "b", "c", "d"]
    records = []
    for task in tasks[:-1]:
        records.extend(
            {
                "dataset_config": task,
                "historical_utility_score": float(label),
                "label_positive": label,
            }
            for label in ([0] * 10 + [1] * 10)
        )

    report = summarize_predictability_records(records, required_tasks=tasks)

    assert report["task_macro_auroc"] is None
    assert report["all_tasks_valid"] is False
    assert report["h3_precursor_gate_passed"] is False
    assert report["per_task"]["d"]["invalid_reason"] == "insufficient_scored_rows"
