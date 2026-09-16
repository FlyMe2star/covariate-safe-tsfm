import pytest

from covsafe.chronos2_smoke import make_task_fields, select_finite_metrics


@pytest.fixture
def smoke_config() -> dict:
    return {
        "variants": ["target_only", "all_dynamic"],
        "dataset": {
            "dataset_path": "datasets",
            "dataset_config": "example",
            "horizon": 24,
            "seasonality": 24,
            "target": None,
            "known_dynamic_columns": ["future_a", "future_b"],
            "past_dynamic_columns": ["past_a"],
        },
        "metrics": {
            "primary": "SQL",
            "quantile_levels": [0.1, 0.5, 0.9],
        },
    }


def test_target_only_task_removes_covariates(smoke_config: dict) -> None:
    fields = make_task_fields(smoke_config, "target_only", cutoff=-48)
    assert fields["known_dynamic_columns"] == []
    assert fields["past_dynamic_columns"] == []
    assert fields["initial_cutoff"] == -48


def test_all_dynamic_task_preserves_declared_covariates(smoke_config: dict) -> None:
    fields = make_task_fields(smoke_config, "all_dynamic", cutoff=-48)
    assert fields["known_dynamic_columns"] == ["future_a", "future_b"]
    assert fields["past_dynamic_columns"] == ["past_a"]


def test_undeclared_variant_is_rejected(smoke_config: dict) -> None:
    with pytest.raises(ValueError, match="undeclared variant"):
        make_task_fields(smoke_config, "oracle", cutoff=-48)


def test_metric_selection_requires_finite_values() -> None:
    summary = {"SQL": 1.0, "WQL": 2.0, "MASE": 3.0, "WAPE": 4.0}
    assert select_finite_metrics(summary) == summary
    summary["SQL"] = float("nan")
    with pytest.raises(ValueError, match="SQL is not finite"):
        select_finite_metrics(summary)
