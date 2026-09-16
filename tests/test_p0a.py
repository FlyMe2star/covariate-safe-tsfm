import math

import pytest

from covsafe.p0a import summarize_harm_records


def _row(task: str, variant: str, sql: float, unit: int = 0) -> dict[str, object]:
    return {
        "dataset_config": task,
        "variant": variant,
        "origin_index": unit,
        "item_id": f"item-{unit}",
        "target": "target",
        "SQL": sql,
    }


def test_task_macro_prevents_large_task_from_dominating_gate() -> None:
    records: list[dict[str, object]] = []
    for task in ("small_a", "small_b", "small_c"):
        records.extend([_row(task, "target_only", 1.0), _row(task, "all_dynamic", 1.1)])
    for unit in range(100):
        records.extend(
            [
                _row("large", "target_only", 1.0, unit),
                _row("large", "all_dynamic", 0.9, unit),
            ]
        )

    report = summarize_harm_records(records)

    assert report["task_macro_harm_rate"] == 0.75
    assert report["unit_micro_harm_rate"] == pytest.approx(3 / 103)
    assert report["h1_gate_passed"] is True


def test_nonfinite_pairs_are_counted_and_excluded_symmetrically() -> None:
    records: list[dict[str, object]] = []
    for task in ("a", "b", "c", "d"):
        records.extend(
            [
                _row(task, "target_only", 1.0),
                _row(task, "all_dynamic", 1.1),
                _row(task, "target_only", math.nan, 1),
                _row(task, "all_dynamic", 2.0, 1),
            ]
        )

    report = summarize_harm_records(records)

    assert report["finite_pair_count"] == 4
    assert report["excluded_nonfinite_pair_count"] == 4
    assert report["task_macro_harm_rate"] == 1.0


def test_pairing_rejects_missing_variant_unit() -> None:
    records: list[dict[str, object]] = []
    for task in ("a", "b", "c", "d"):
        records.extend([_row(task, "target_only", 1.0), _row(task, "all_dynamic", 1.1)])
    records.pop()

    with pytest.raises(ValueError, match="variant keys do not match"):
        summarize_harm_records(records)
