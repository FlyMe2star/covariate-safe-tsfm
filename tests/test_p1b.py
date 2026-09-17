from pathlib import Path

import pytest

from covsafe.config import load_yaml
from covsafe.p0b import CandidatePolicy
from covsafe.p1b import (
    EXPECTED_P1B_CONFIG_HASH,
    build_router_score_records,
    load_frozen_p1b,
    logical_policy_order,
    select_constant_policies,
)
from covsafe.p1b_eval import build_origin_decisions, evaluation_origin_indices

REPO_ROOT = Path(__file__).resolve().parents[1]


def _one_task_parent():
    parent = load_yaml(REPO_ROOT / "configs/diagnostic/covariate_utility_p0.yaml")
    return {**parent, "data": {**parent["data"], "tasks": [parent["data"]["tasks"][0]]}}


def test_frozen_p1b_loads_with_owner_approval():
    config, p0b, parent, p0a = load_frozen_p1b(REPO_ROOT)
    assert EXPECTED_P1B_CONFIG_HASH == "2b8b8284ae21"
    assert config["owner_approval"] == "APPROVE_P1B"
    assert p0b["candidate_policies"]["fixed"] == ["target_only", "all_dynamic"]
    assert len(parent["data"]["tasks"]) == 4
    assert len(p0a["tasks"]) == 4


def test_evaluation_origins_are_exact_calibration_complement():
    _config, _p0b, parent, p0a = load_frozen_p1b(REPO_ROOT)
    assert evaluation_origin_indices(parent, p0a, "epf_np") == list(range(12, 20))
    assert evaluation_origin_indices(parent, p0a, "rohlik_orders_1D") == [3, 4]
    assert evaluation_origin_indices(parent, p0a, "rossmann_1D") == [6, 7, 8, 9]
    assert evaluation_origin_indices(parent, p0a, "solar_with_weather_1H") == list(
        range(12, 20)
    )


def test_constant_selection_uses_complete_logical_policy_set():
    _config, p0b, _parent, _p0a = load_frozen_p1b(REPO_ROOT)
    parent = _one_task_parent()
    task = "epf_np"
    order = logical_policy_order(parent, p0b, task)
    rows = []
    for origin in (0, 1):
        for index, policy in enumerate(order):
            loss = 1.0 + index / 100.0
            if policy == "single_known_future:Grid load forecast":
                loss = 0.8
            rows.append(
                {
                    "backbone": "chronos_2",
                    "dataset_config": task,
                    "origin_index": origin,
                    "item_id": "x",
                    "target": "y",
                    "policy_id": policy,
                    "SQL": loss,
                }
            )
    result = select_constant_policies(
        rows, parent=parent, p0b=p0b, backbone="chronos_2"
    )
    assert result[task]["fullset_best_policy_id"] == (
        "single_known_future:Grid load forecast"
    )
    assert result[task]["binary_best_policy_id"] == "target_only"
    assert result[task]["eligible_complete_finite_policy_count"] == len(order)


def test_router_score_uses_all_calibration_observations_with_recency():
    _config, p0b, _parent, _p0a = load_frozen_p1b(REPO_ROOT)
    parent = _one_task_parent()
    rows = []
    for origin, target_sql, candidate_sql in ((0, 1.0, 1.2), (1, 1.0, 0.8)):
        base = {
            "backbone": "chronos_2",
            "dataset_config": "epf_np",
            "origin_index": origin,
            "item_id": "x",
            "target": "y",
        }
        rows.append({**base, "policy_id": "target_only", "SQL": target_sql})
        rows.append({**base, "policy_id": "all_dynamic", "SQL": candidate_sql})
    scores = build_router_score_records(
        rows,
        parent=parent,
        p0b=p0b,
        backbone="chronos_2",
        half_life_origins=1.0,
        epsilon=1.0e-12,
    )
    assert len(scores) == 1
    assert scores[0]["historical_utility_score"] == pytest.approx(1.0 / 15.0)
    assert scores[0]["latest_relative_utility"] == pytest.approx(0.2)


def test_inapplicable_group_falls_back_while_ungated_router_routes():
    config, _p0b, _parent, _p0a = load_frozen_p1b(REPO_ROOT)
    state = {
        "constant_policies": {
            "chronos_2": {
                "epf_np": {
                    "fullset_best_policy_id": "all_dynamic",
                    "binary_best_policy_id": "target_only",
                }
            }
        }
    }
    scores = [
        {
            "backbone": "chronos_2",
            "dataset_config": "epf_np",
            "item_id": "x",
            "target": "y",
            "policy_id": "single_known_future:Grid load forecast",
            "policy_order_index": 2,
            "historical_utility_score": 0.3,
            "latest_relative_utility": 0.2,
        }
    ]
    policies = [
        CandidatePolicy("target_only", (), ()),
        CandidatePolicy(
            "all_dynamic", ("Grid load forecast", "Wind power forecast"), ()
        ),
        CandidatePolicy("single_known_future:Grid load forecast", ("Grid load forecast",), ()),
    ]
    decisions = build_origin_decisions(
        config=config,
        state_report=state,
        score_records=scores,
        backbone="chronos_2",
        dataset_config="epf_np",
        origin_index=12,
        cutoff="cutoff",
        units=[("x", "y")],
        policies=policies,
    )
    by_method = {row["method"]: row for row in decisions}
    assert by_method["ungated_ewm_router"]["evaluated_policy_id"] == (
        "single_known_future:Grid load forecast"
    )
    assert by_method["applicability_gated_router"]["logical_policy_id"] == (
        "all_dynamic"
    )
    assert by_method["applicability_gated_router"]["route_mode"] == (
        "applicability_constant_fallback"
    )
