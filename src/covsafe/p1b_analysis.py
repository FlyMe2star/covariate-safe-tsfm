"""Predeclared paired aggregation for completed sealed P1b artifacts."""

from __future__ import annotations

import csv
import json
import math
import platform
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from covsafe.fev_smoke import git_commit
from covsafe.p1b import (
    EXPECTED_P1B_CONFIG_HASH,
    atomic_csv,
    atomic_json,
    inventory_sha256,
    load_frozen_p1b,
    scientific_code_hash,
    sha256_file,
)
from covsafe.p1b_eval import (
    DECISION_FIELDS,
    LOSS_FIELDS,
    ROUTED_METHODS,
    completed_sealed_origin,
    decision_paths,
    evaluation_origin_indices,
    load_router_state,
    loss_paths,
)

SELECTED_FIELDS = (
    "backbone",
    "dataset_config",
    "origin_index",
    "cutoff",
    "item_id",
    "target",
    "method",
    "logical_policy_id",
    "evaluated_policy_id",
    "route_mode",
    "SQL",
    "WQL",
    "MASE",
    "WAPE",
)


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise RuntimeError(f"unexpected CSV schema: {path}")
        return [dict(row) for row in reader]


def read_verified_sealed_artifacts(
    repo_root: str | Path,
    state_root: str | Path,
    evaluation_root: str | Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
    str,
]:
    """Read every expected decision and loss row after integrity verification."""
    root = Path(repo_root).resolve()
    destination = Path(evaluation_root)
    config, _p0b, parent, p0a = load_frozen_p1b(root)
    _state, _scores, state_path = load_router_state(root, state_root)
    state_sha256 = sha256_file(state_path)
    code_hash = scientific_code_hash(root)
    decisions: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    inventory_paths: list[Path] = []
    for backbone in config["scope"]["frozen_zero_shot_backbones"]:
        for task in config["scope"]["tasks"]:
            for origin in evaluation_origin_indices(parent, p0a, task):
                if completed_sealed_origin(
                    destination,
                    backbone=backbone,
                    dataset_config=task,
                    origin_index=origin,
                    scientific_code_sha256=code_hash,
                    router_state_sha256=state_sha256,
                ) is None:
                    raise RuntimeError(f"missing sealed unit: {backbone}/{task}/{origin}")
                decision_csv, decision_json = decision_paths(
                    destination, backbone, task, origin
                )
                loss_csv, loss_json = loss_paths(
                    destination, backbone, task, origin
                )
                inventory_paths.extend(
                    [decision_csv, decision_json, loss_csv, loss_json]
                )
                for raw in _read_csv(decision_csv, DECISION_FIELDS):
                    row: dict[str, Any] = dict(raw)
                    row["origin_index"] = int(row["origin_index"])
                    decisions.append(row)
                for raw in _read_csv(loss_csv, LOSS_FIELDS):
                    row = dict(raw)
                    row["origin_index"] = int(row["origin_index"])
                    for metric in ("SQL", "WQL", "MASE", "WAPE"):
                        row[metric] = float(row[metric])
                    losses.append(row)
    return (
        config,
        decisions,
        losses,
        state_sha256,
        inventory_sha256(destination, inventory_paths),
    )


def join_routed_losses(
    decisions: Iterable[dict[str, Any]],
    losses: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join frozen decisions to losses without consulting unselected alternatives."""

    def loss_key(row: dict[str, Any]) -> tuple[str, str, int, str, str, str]:
        return (
            str(row["backbone"]),
            str(row["dataset_config"]),
            int(row["origin_index"]),
            str(row["item_id"]),
            str(row["target"]),
            str(row["policy_id"]),
        )

    loss_index: dict[tuple[str, str, int, str, str, str], dict[str, Any]] = {}
    by_unit: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in losses:
        key = loss_key(row)
        if key in loss_index:
            raise RuntimeError(f"duplicate sealed loss row: {key}")
        loss_index[key] = dict(row)
        by_unit[key[:-1]].append(dict(row))

    selected: list[dict[str, Any]] = []
    methods_by_unit: dict[tuple[str, str, int, str, str], set[str]] = defaultdict(set)
    for decision in decisions:
        unit = (
            str(decision["backbone"]),
            str(decision["dataset_config"]),
            int(decision["origin_index"]),
            str(decision["item_id"]),
            str(decision["target"]),
        )
        method = str(decision["method"])
        if method in methods_by_unit[unit]:
            raise RuntimeError(f"duplicate routed method decision: {unit}/{method}")
        methods_by_unit[unit].add(method)
        key = (*unit, str(decision["evaluated_policy_id"]))
        if key not in loss_index:
            raise RuntimeError(f"decision has no matching sealed loss: {key}")
        loss = loss_index[key]
        if not math.isfinite(float(loss["SQL"])):
            raise RuntimeError(
                "selected policy produced non-finite SQL; frozen protocol forbids exclusion"
            )
        selected.append(
            {
                "backbone": unit[0],
                "dataset_config": unit[1],
                "origin_index": unit[2],
                "cutoff": str(decision["cutoff"]),
                "item_id": unit[3],
                "target": unit[4],
                "method": method,
                "logical_policy_id": str(decision["logical_policy_id"]),
                "evaluated_policy_id": str(decision["evaluated_policy_id"]),
                "route_mode": str(decision["route_mode"]),
                **{metric: float(loss[metric]) for metric in ("SQL", "WQL", "MASE", "WAPE")},
            }
        )
    expected_methods = set(ROUTED_METHODS)
    for unit, observed in methods_by_unit.items():
        if observed != expected_methods:
            raise RuntimeError(f"routed method coverage mismatch: {unit}")

    for unit, unit_losses in by_unit.items():
        finite = [row for row in unit_losses if math.isfinite(float(row["SQL"]))]
        if not finite:
            raise RuntimeError(f"finite oracle has no finite policy: {unit}")
        winner = min(finite, key=lambda row: (float(row["SQL"]), str(row["policy_id"])))
        selected.append(
            {
                "backbone": unit[0],
                "dataset_config": unit[1],
                "origin_index": unit[2],
                "cutoff": str(winner["cutoff"]),
                "item_id": unit[3],
                "target": unit[4],
                "method": "finite_candidate_oracle",
                "logical_policy_id": str(winner["policy_id"]),
                "evaluated_policy_id": str(winner["policy_id"]),
                "route_mode": "diagnostic_oracle_not_deployable",
                **{metric: float(winner[metric]) for metric in ("SQL", "WQL", "MASE", "WAPE")},
            }
        )
    return selected


def paired_relative_gains(
    selected: Iterable[dict[str, Any]],
    *,
    method: str,
    comparator: str,
    epsilon: float = 1.0e-12,
) -> dict[tuple[str, str], list[float]]:
    """Compute paired unit gains grouped by fixed task-backbone group."""
    methods: dict[str, dict[tuple[str, str, int, str, str], float]] = defaultdict(dict)
    for row in selected:
        name = str(row["method"])
        if name not in {method, comparator}:
            continue
        key = (
            str(row["backbone"]),
            str(row["dataset_config"]),
            int(row["origin_index"]),
            str(row["item_id"]),
            str(row["target"]),
        )
        if key in methods[name]:
            raise RuntimeError(f"duplicate selected loss: {name}/{key}")
        methods[name][key] = float(row["SQL"])
    if set(methods[method]) != set(methods[comparator]):
        raise RuntimeError(f"paired unit mismatch: {method} vs {comparator}")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key in sorted(methods[method]):
        candidate = methods[method][key]
        baseline = methods[comparator][key]
        if not math.isfinite(candidate) or not math.isfinite(baseline):
            continue
        grouped[(key[0], key[1])].append(
            (baseline - candidate) / max(abs(baseline), epsilon)
        )
    return dict(grouped)


def group_macro_bootstrap(
    grouped_values: dict[tuple[str, str], list[float]],
    *,
    replicates: int,
    seed: int,
    lower_quantile: float = 0.05,
) -> dict[str, Any]:
    """Equal-weight fixed groups while resampling paired units within each group."""
    if not grouped_values:
        raise ValueError("no groups for bootstrap")
    ordered = sorted(grouped_values)
    arrays = [np.asarray(grouped_values[group], dtype=np.float64) for group in ordered]
    if any(values.size == 0 or not np.isfinite(values).all() for values in arrays):
        raise ValueError("bootstrap groups must contain finite values")
    point_groups = {
        f"{backbone}/{task}": float(np.mean(values))
        for (backbone, task), values in zip(ordered, arrays, strict=True)
    }
    point = float(np.mean(list(point_groups.values())))
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        group_means = [
            float(np.mean(values[rng.integers(0, values.size, size=values.size)]))
            for values in arrays
        ]
        estimates[replicate] = float(np.mean(group_means))
    return {
        "group_means": point_groups,
        "group_unit_counts": {
            f"{backbone}/{task}": int(values.size)
            for (backbone, task), values in zip(ordered, arrays, strict=True)
        },
        "equal_group_macro_mean": point,
        "unit_micro_mean": float(np.mean(np.concatenate(arrays))),
        "bootstrap_replicates": replicates,
        "one_sided_95pct_lower_bound": float(np.quantile(estimates, lower_quantile)),
        "bootstrap_median": float(np.quantile(estimates, 0.5)),
        "bootstrap_upper_95pct": float(np.quantile(estimates, 0.95)),
    }


def summarize_method_metrics(
    selected: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[
            (str(row["method"]), str(row["backbone"]), str(row["dataset_config"]))
        ].append(row)
    result: dict[str, dict[str, Any]] = defaultdict(dict)
    for (method, backbone, task), rows in sorted(grouped.items()):
        metrics = {}
        for metric in ("SQL", "WQL", "MASE", "WAPE"):
            values = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
            metrics[metric] = float(np.mean(values)) if values else None
        result[method][f"{backbone}/{task}"] = {
            "unit_count": len(rows),
            "metrics": metrics,
        }
    return dict(result)


def routing_summary(selected: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in selected if row["method"] == "applicability_gated_router"]
    total = len(rows)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["route_mode"])] += 1
    return {
        "unit_count": total,
        "route_mode_counts": dict(sorted(counts.items())),
        "dynamic_positive_score_fraction": (
            counts["dynamic_positive_score"] / total if total else None
        ),
        "applicability_constant_fallback_fraction": (
            counts["applicability_constant_fallback"] / total if total else None
        ),
        "target_default_fraction": counts["target_default"] / total if total else None,
    }


def run_p1b_analysis(
    repo_root: str | Path,
    state_root: str | Path,
    evaluation_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run the frozen fixed-sequence aggregate analysis exactly once."""
    root = Path(repo_root).resolve()
    destination = Path(output_root)
    report_path = destination / "reports" / "p1b_sealed_analysis.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("config_hash") != EXPECTED_P1B_CONFIG_HASH
            or existing.get("scientific_code_sha256") != scientific_code_hash(root)
        ):
            raise RuntimeError("existing P1b analysis is stale or altered")
        _, _, _, state_sha256, evaluation_inventory_sha256 = (
            read_verified_sealed_artifacts(root, state_root, evaluation_root)
        )
        selected_artifact = existing.get("selected_loss_artifact", {})
        selected_path = destination / str(selected_artifact.get("relative_path", ""))
        valid = (
            existing.get("router_state_sha256") == state_sha256
            and existing.get("sealed_evaluation_inventory_sha256")
            == evaluation_inventory_sha256
            and selected_path.is_file()
            and selected_artifact.get("sha256") == sha256_file(selected_path)
        )
        if not valid:
            raise RuntimeError("existing P1b analysis inputs or outputs were altered")
        return existing
    (
        config,
        decisions,
        losses,
        state_sha256,
        evaluation_inventory_sha256,
    ) = read_verified_sealed_artifacts(root, state_root, evaluation_root)
    selected = join_routed_losses(decisions, losses)
    selected_path = destination / "results" / "p1b_selected_losses.csv"
    selected_sha256 = atomic_csv(
        selected_path,
        sorted(
            selected,
            key=lambda row: (
                row["backbone"],
                row["dataset_config"],
                row["origin_index"],
                row["item_id"],
                row["target"],
                row["method"],
            ),
        ),
        SELECTED_FIELDS,
    )
    replicates = int(config["aggregation"]["bootstrap_replicates"])
    seed = int(config["seed"])
    step_1 = group_macro_bootstrap(
        paired_relative_gains(
            selected,
            method="applicability_gated_router",
            comparator="fullset_best_constant",
        ),
        replicates=replicates,
        seed=seed,
    )
    step_1_passed = step_1["one_sided_95pct_lower_bound"] > 0.0
    step_2 = group_macro_bootstrap(
        paired_relative_gains(
            selected,
            method="applicability_gated_router",
            comparator="ungated_ewm_router",
        ),
        replicates=replicates,
        seed=seed + 1,
    )
    step_2_passed = bool(
        step_1_passed and step_2["one_sided_95pct_lower_bound"] > 0.0
    )

    expected_groups = {
        f"{backbone}/{task}"
        for backbone in config["scope"]["frozen_zero_shot_backbones"]
        for task in config["scope"]["tasks"]
    }
    for comparison, observed in (
        ("step 1", set(step_1["group_means"])),
        ("step 2", set(step_2["group_means"])),
    ):
        if observed != expected_groups:
            raise RuntimeError(f"{comparison} group coverage mismatch")

    step_1_groups = step_1["group_means"]
    per_backbone = {
        backbone: float(
            np.mean(
                [
                    value
                    for group, value in step_1_groups.items()
                    if group.startswith(f"{backbone}/")
                ]
            )
        )
        for backbone in config["scope"]["frozen_zero_shot_backbones"]
    }
    safety_passed = all(value >= 0.0 for value in per_backbone.values())
    ungated_vs_constant = group_macro_bootstrap(
        paired_relative_gains(
            selected,
            method="ungated_ewm_router",
            comparator="fullset_best_constant",
        ),
        replicates=replicates,
        seed=seed + 2,
    )
    gated_worst = min(step_1["group_means"].values())
    ungated_worst = min(ungated_vs_constant["group_means"].values())

    report = {
        "experiment": config["experiment"],
        "schema_version": 1,
        "result_status": "paper_eligible",
        "config_hash": EXPECTED_P1B_CONFIG_HASH,
        "git_commit": git_commit(root),
        "scientific_code_sha256": scientific_code_hash(root),
        "router_state_sha256": state_sha256,
        "sealed_evaluation_inventory_sha256": evaluation_inventory_sha256,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "accelerator": "cpu",
        },
        "counts": {
            "decision_rows": len(decisions),
            "exhaustive_policy_loss_rows": len(losses),
            "selected_method_loss_rows": len(selected),
        },
        "selected_loss_artifact": {
            "relative_path": str(selected_path.relative_to(destination)),
            "row_count": len(selected),
            "sha256": selected_sha256,
        },
        "method_metrics_by_group": summarize_method_metrics(selected),
        "routing": routing_summary(selected),
        "fixed_sequence": {
            "step_1_gated_vs_fullset_best_constant": {
                **step_1,
                "passed": step_1_passed,
            },
            "step_2_gated_vs_ungated": {
                **step_2,
                "inferentially_tested": step_1_passed,
                "passed": step_2_passed,
            },
        },
        "per_backbone_safety": {
            "task_macro_point_gain_vs_fullset_best_constant": per_backbone,
            "passed": safety_passed,
        },
        "worst_group_downside": {
            "gated_minimum_group_mean_gain_vs_constant": gated_worst,
            "ungated_minimum_group_mean_gain_vs_constant": ungated_worst,
            "gated_no_worse_than_ungated": gated_worst >= ungated_worst,
            "inferential_status": "descriptive_only_eight_fixed_groups",
        },
        "decision": {
            "paper_success_gate_passed": bool(
                step_1_passed and step_2_passed and safety_passed
            ),
            "step_1_passed": step_1_passed,
            "step_2_passed": step_2_passed,
            "per_backbone_safety_passed": safety_passed,
            "next_action": (
                "reproducibility_audit_then_claim_backfill"
                if step_1_passed and step_2_passed and safety_passed
                else "retain_bounded_result_and_reassess_paper_viability"
            ),
        },
    }
    atomic_json(report_path, report)
    return report
