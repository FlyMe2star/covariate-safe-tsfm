import pytest

from covsafe.p1b_analysis import (
    group_macro_bootstrap,
    join_routed_losses,
    paired_relative_gains,
)
from covsafe.p1b_eval import ROUTED_METHODS


def _decision(method, policy):
    return {
        "backbone": "b",
        "dataset_config": "d",
        "origin_index": 1,
        "cutoff": "c",
        "item_id": "i",
        "target": "y",
        "method": method,
        "logical_policy_id": policy,
        "evaluated_policy_id": policy,
        "route_mode": "fixed",
    }


def _loss(policy, sql):
    return {
        "backbone": "b",
        "dataset_config": "d",
        "origin_index": 1,
        "cutoff": "c",
        "item_id": "i",
        "target": "y",
        "policy_id": policy,
        "SQL": sql,
        "WQL": sql,
        "MASE": sql,
        "WAPE": sql,
    }


def test_join_routes_and_adds_finite_oracle():
    policies = {
        "target_only": "p0",
        "all_dynamic": "p1",
        "binary_best_constant": "p0",
        "fullset_best_constant": "p0",
        "last_origin_winner": "p1",
        "ungated_ewm_router": "p1",
        "applicability_gated_router": "p1",
    }
    assert tuple(policies) == ROUTED_METHODS
    decisions = [_decision(method, policy) for method, policy in policies.items()]
    selected = join_routed_losses(decisions, [_loss("p0", 1.0), _loss("p1", 0.8)])
    assert len(selected) == len(ROUTED_METHODS) + 1
    oracle = [row for row in selected if row["method"] == "finite_candidate_oracle"]
    assert oracle[0]["SQL"] == pytest.approx(0.8)


def test_join_rejects_nonfinite_selected_primary_metric():
    policies = {method: "p0" for method in ROUTED_METHODS}
    decisions = [_decision(method, policy) for method, policy in policies.items()]
    with pytest.raises(RuntimeError, match="non-finite SQL"):
        join_routed_losses(decisions, [_loss("p0", float("nan"))])


def test_paired_group_macro_bootstrap_is_deterministic():
    selected = []
    for origin, gated, baseline in ((1, 0.8, 1.0), (2, 0.9, 1.0)):
        for method, loss in (
            ("applicability_gated_router", gated),
            ("fullset_best_constant", baseline),
        ):
            selected.append(
                {
                    "backbone": "b",
                    "dataset_config": "d",
                    "origin_index": origin,
                    "item_id": "i",
                    "target": "y",
                    "method": method,
                    "SQL": loss,
                }
            )
    gains = paired_relative_gains(
        selected,
        method="applicability_gated_router",
        comparator="fullset_best_constant",
    )
    first = group_macro_bootstrap(gains, replicates=100, seed=42)
    second = group_macro_bootstrap(gains, replicates=100, seed=42)
    assert first == second
    assert first["equal_group_macro_mean"] == pytest.approx(0.15)
    assert first["one_sided_95pct_lower_bound"] > 0.0
