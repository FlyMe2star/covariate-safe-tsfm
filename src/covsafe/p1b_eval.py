"""Leakage-safe decision and artifact helpers for sealed P1b evaluation."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from covsafe.p0b import CandidatePolicy, TaskContext
from covsafe.p1b import (
    EXPECTED_P1B_CONFIG_HASH,
    ROUTER_SCORE_FIELDS,
    atomic_csv,
    atomic_json,
    scientific_code_hash,
    sha256_file,
)
from covsafe.protocol import temporal_origin_partition

DECISION_FIELDS = (
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
    "historical_score",
)
LOSS_FIELDS = (
    "backbone",
    "dataset_config",
    "origin_index",
    "cutoff",
    "policy_id",
    "item_id",
    "target",
    "SQL",
    "WQL",
    "MASE",
    "WAPE",
)
ROUTED_METHODS = (
    "target_only",
    "all_dynamic",
    "binary_best_constant",
    "fullset_best_constant",
    "last_origin_winner",
    "ungated_ewm_router",
    "applicability_gated_router",
)


def load_router_state(
    repo_root: str | Path,
    state_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """Verify the frozen state report and score artifact against current code."""
    report_path = Path(state_root) / "reports" / "p1b_router_state.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing P1b router-state report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["config_hash"] != EXPECTED_P1B_CONFIG_HASH:
        raise RuntimeError("P1b router-state configuration hash mismatch")
    if report["scientific_code_sha256"] != scientific_code_hash(repo_root):
        raise RuntimeError("P1b router-state scientific-code hash mismatch")
    if report["scope"]["sealed_evaluation_origins_instantiated"] is not False:
        raise RuntimeError("router-state report indicates sealed access")
    if report["scope"]["model_inference_performed"] is not False:
        raise RuntimeError("router-state report unexpectedly used model inference")
    if report["authorization"]["router_state_frozen"] is not True:
        raise RuntimeError("router state is not frozen")

    artifact = report["router_score_artifact"]
    score_path = Path(state_root) / artifact["relative_path"]
    if sha256_file(score_path) != artifact["sha256"]:
        raise RuntimeError("P1b router-score artifact checksum mismatch")
    with score_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROUTER_SCORE_FIELDS:
            raise RuntimeError("unexpected P1b router-score schema")
        scores: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for name in (
                "policy_order_index",
                "finite_observation_count",
                "first_origin_index",
                "latest_origin_index",
            ):
                row[name] = int(row[name])
            for name in ("historical_utility_score", "latest_relative_utility"):
                row[name] = float(row[name])
            scores.append(row)
    if len(scores) != int(artifact["row_count"]):
        raise RuntimeError("P1b router-score row count mismatch")
    return report, scores, report_path


def evaluation_origin_indices(
    parent: dict[str, Any],
    p0a: dict[str, Any],
    dataset_config: str,
) -> list[int]:
    """Return the untouched complement of the frozen calibration partition."""
    matches = [
        task
        for task in parent["data"]["tasks"]
        if task["dataset_config"] == dataset_config
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one task for {dataset_config}")
    task = matches[0]
    partition = temporal_origin_partition(
        int(task["num_windows"]),
        calibration_fraction=float(parent["data"]["calibration_fraction"]),
        minimum_per_partition=int(parent["data"]["minimum_origins_per_partition"]),
    )
    if tuple(p0a["tasks"][dataset_config]["calibration_origin_indices"]) != (
        partition.calibration
    ):
        raise RuntimeError(f"calibration partition drift for {dataset_config}")
    return list(partition.evaluation)


def origin_unit_keys(
    context: TaskContext,
    origin_index: int,
) -> list[tuple[str, str]]:
    """Enumerate item-target keys from pre-origin input data only."""
    window = context.reference_task.get_window(origin_index, num_proc=1)
    past_data, _known_future_data = window.get_input_data()
    id_column = context.reference_task.id_column
    units = {
        (str(row[id_column]), str(target))
        for row in past_data
        for target in window.target_columns
    }
    if not units:
        raise RuntimeError("sealed input window contains no item-target units")
    return sorted(units)


def logical_to_retained_policy(
    policies: Iterable[CandidatePolicy],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for policy in policies:
        for logical in (policy.policy_id, *policy.aliases):
            if logical in mapping:
                raise RuntimeError(f"logical policy appears more than once: {logical}")
            mapping[logical] = policy.policy_id
    return mapping


def score_index(
    records: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in records:
        key = (
            str(row["backbone"]),
            str(row["dataset_config"]),
            str(row["item_id"]),
            str(row["target"]),
            str(row["policy_id"]),
        )
        if key in index:
            raise RuntimeError(f"duplicate router score: {key}")
        index[key] = dict(row)
    return index


def _dynamic_choice(
    *,
    backbone: str,
    task: str,
    item_id: str,
    target: str,
    retained_policy_ids: list[str],
    scores: dict[tuple[str, str, str, str, str], dict[str, Any]],
    score_field: str,
) -> tuple[str, float, str]:
    candidates: list[tuple[float, int, str]] = []
    for policy_id in retained_policy_ids:
        if policy_id == "target_only":
            continue
        row = scores.get((backbone, task, item_id, target, policy_id))
        if row is None:
            continue
        score = float(row[score_field])
        if math.isfinite(score) and score > 0.0:
            candidates.append((score, int(row["policy_order_index"]), policy_id))
    if not candidates:
        return "target_only", 0.0, "target_default"
    score, _order, policy_id = min(
        candidates, key=lambda value: (-value[0], value[1], value[2])
    )
    return policy_id, score, "dynamic_positive_score"


def build_origin_decisions(
    *,
    config: dict[str, Any],
    state_report: dict[str, Any],
    score_records: Iterable[dict[str, Any]],
    backbone: str,
    dataset_config: str,
    origin_index: int,
    cutoff: str,
    units: Iterable[tuple[str, str]],
    policies: Iterable[CandidatePolicy],
) -> list[dict[str, Any]]:
    """Create every routing decision without accepting any current-origin loss."""
    policy_list = list(policies)
    retained_ids = [policy.policy_id for policy in policy_list]
    mapping = logical_to_retained_policy(policy_list)
    if "target_only" not in mapping or "all_dynamic" not in mapping:
        raise RuntimeError("sealed policy set is missing a fixed anchor")
    scores = score_index(score_records)
    constants = state_report["constant_policies"][backbone][dataset_config]
    fullset_logical = str(constants["fullset_best_policy_id"])
    binary_logical = str(constants["binary_best_policy_id"])
    if fullset_logical not in mapping or binary_logical not in mapping:
        raise RuntimeError("frozen constant policy is unavailable at sealed origin")
    applicable = bool(config["applicability_map"][backbone][dataset_config])

    decisions: list[dict[str, Any]] = []
    for item_id, target in units:
        ewm_policy, ewm_score, ewm_mode = _dynamic_choice(
            backbone=backbone,
            task=dataset_config,
            item_id=item_id,
            target=target,
            retained_policy_ids=retained_ids,
            scores=scores,
            score_field="historical_utility_score",
        )
        last_policy, last_score, last_mode = _dynamic_choice(
            backbone=backbone,
            task=dataset_config,
            item_id=item_id,
            target=target,
            retained_policy_ids=retained_ids,
            scores=scores,
            score_field="latest_relative_utility",
        )
        method_choices = {
            "target_only": ("target_only", mapping["target_only"], "fixed", None),
            "all_dynamic": ("all_dynamic", mapping["all_dynamic"], "fixed", None),
            "binary_best_constant": (
                binary_logical,
                mapping[binary_logical],
                "constant",
                None,
            ),
            "fullset_best_constant": (
                fullset_logical,
                mapping[fullset_logical],
                "constant",
                None,
            ),
            "last_origin_winner": (
                last_policy,
                last_policy,
                last_mode,
                last_score,
            ),
            "ungated_ewm_router": (
                ewm_policy,
                ewm_policy,
                ewm_mode,
                ewm_score,
            ),
            "applicability_gated_router": (
                (ewm_policy if applicable else fullset_logical),
                (ewm_policy if applicable else mapping[fullset_logical]),
                (ewm_mode if applicable else "applicability_constant_fallback"),
                (ewm_score if applicable else None),
            ),
        }
        if tuple(method_choices) != ROUTED_METHODS:
            raise RuntimeError("routed method ordering drift")
        for method, (logical, retained, mode, score) in method_choices.items():
            decisions.append(
                {
                    "backbone": backbone,
                    "dataset_config": dataset_config,
                    "origin_index": origin_index,
                    "cutoff": cutoff,
                    "item_id": item_id,
                    "target": target,
                    "method": method,
                    "logical_policy_id": logical,
                    "evaluated_policy_id": retained,
                    "route_mode": mode,
                    "historical_score": "" if score is None else score,
                }
            )
    return decisions


def decision_paths(
    output_root: str | Path,
    backbone: str,
    dataset_config: str,
    origin_index: int,
) -> tuple[Path, Path]:
    root = Path(output_root) / "decisions" / backbone / dataset_config
    stem = f"origin_{origin_index:03d}"
    return root / f"{stem}.csv", root / f"{stem}.json"


def loss_paths(
    output_root: str | Path,
    backbone: str,
    dataset_config: str,
    origin_index: int,
) -> tuple[Path, Path]:
    root = Path(output_root) / "sealed_units" / backbone / dataset_config
    stem = f"origin_{origin_index:03d}"
    return root / f"{stem}.csv", root / f"{stem}.json"


def ensure_decision_artifact(
    output_root: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically create or verify a pre-inference sealed decision artifact."""
    csv_path, sidecar_path = decision_paths(
        output_root,
        metadata["backbone"],
        metadata["dataset_config"],
        int(metadata["origin_index"]),
    )
    if csv_path.exists() and sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        valid = (
            sidecar.get("completed") is True
            and sidecar.get("config_hash") == EXPECTED_P1B_CONFIG_HASH
            and sidecar.get("scientific_code_sha256")
            == metadata["scientific_code_sha256"]
            and sidecar.get("router_state_sha256") == metadata["router_state_sha256"]
            and sidecar.get("csv_sha256") == sha256_file(csv_path)
        )
        if not valid:
            raise RuntimeError(f"stale or altered decision artifact: {sidecar_path}")
        return sidecar
    if csv_path.exists() or sidecar_path.exists():
        raise RuntimeError("partial sealed decision artifact requires manual inspection")

    csv_sha256 = atomic_csv(csv_path, records, DECISION_FIELDS)
    sidecar = {
        **metadata,
        "schema_version": 1,
        "result_status": "paper_eligible",
        "config_hash": EXPECTED_P1B_CONFIG_HASH,
        "decision_created_before_model_inference": True,
        "sealed_outcomes_used": False,
        "completed": True,
        "row_count": len(records),
        "csv_name": csv_path.name,
        "csv_sha256": csv_sha256,
    }
    atomic_json(sidecar_path, sidecar)
    return sidecar


def write_loss_artifact(
    output_root: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically write exhaustive policy losses after decisions are frozen."""
    csv_path, sidecar_path = loss_paths(
        output_root,
        metadata["backbone"],
        metadata["dataset_config"],
        int(metadata["origin_index"]),
    )
    if sidecar_path.exists() or csv_path.exists():
        raise RuntimeError("refusing to overwrite an existing sealed loss artifact")
    csv_sha256 = atomic_csv(csv_path, records, LOSS_FIELDS)
    sidecar = {
        **metadata,
        "schema_version": 1,
        "result_status": "paper_eligible",
        "config_hash": EXPECTED_P1B_CONFIG_HASH,
        "completed": True,
        "row_count": len(records),
        "finite_sql_count": sum(math.isfinite(float(row["SQL"])) for row in records),
        "csv_name": csv_path.name,
        "csv_sha256": csv_sha256,
    }
    atomic_json(sidecar_path, sidecar)
    return sidecar


def completed_sealed_origin(
    output_root: str | Path,
    *,
    backbone: str,
    dataset_config: str,
    origin_index: int,
    scientific_code_sha256: str,
    router_state_sha256: str,
) -> dict[str, Any] | None:
    """Verify both pre-outcome decisions and post-inference losses for resume."""
    decision_csv, decision_json = decision_paths(
        output_root, backbone, dataset_config, origin_index
    )
    loss_csv, loss_json = loss_paths(output_root, backbone, dataset_config, origin_index)
    paths = (decision_csv, decision_json, loss_csv, loss_json)
    if not any(path.exists() for path in paths):
        return None
    if not all(path.exists() for path in paths):
        decision_only = (
            decision_csv.exists()
            and decision_json.exists()
            and not loss_csv.exists()
            and not loss_json.exists()
        )
        if decision_only:
            return None
        raise RuntimeError("partial sealed origin artifact requires manual inspection")
    decision = json.loads(decision_json.read_text(encoding="utf-8"))
    loss = json.loads(loss_json.read_text(encoding="utf-8"))
    shared_valid = all(
        sidecar.get("completed") is True
        and sidecar.get("config_hash") == EXPECTED_P1B_CONFIG_HASH
        and sidecar.get("scientific_code_sha256") == scientific_code_sha256
        and sidecar.get("router_state_sha256") == router_state_sha256
        for sidecar in (decision, loss)
    )
    valid = (
        shared_valid
        and decision.get("csv_sha256") == sha256_file(decision_csv)
        and loss.get("csv_sha256") == sha256_file(loss_csv)
        and loss.get("decision_csv_sha256") == decision.get("csv_sha256")
    )
    if not valid:
        raise RuntimeError("sealed origin integrity verification failed")
    return loss
