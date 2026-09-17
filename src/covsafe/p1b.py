"""Frozen router-state construction and shared P1b artifact helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from covsafe.config import canonical_config_hash, load_yaml
from covsafe.fev_smoke import git_commit
from covsafe.p0b import (
    CSV_FIELDS,
    EXPECTED_P0B_CONFIG_HASH,
    completed_origin,
    load_frozen_p0b,
    origin_paths,
    read_candidate_records,
)
from covsafe.p0b import scientific_code_hash as p0b_scientific_code_hash
from covsafe.p1a import EXPECTED_P1A_CONFIG_HASH
from covsafe.p1a import scientific_code_hash as p1a_scientific_code_hash

EXPECTED_P1B_CONFIG_HASH = "2b8b8284ae21"
P1B_CONFIG_PATH = Path("configs/evaluation/applicability_router_p1b.yaml")
P1A_DECISION_PATH = Path("evidence/screening/p1a_applicability_decision.yaml")
ROUTER_SCORE_FIELDS = (
    "backbone",
    "dataset_config",
    "item_id",
    "target",
    "policy_id",
    "policy_order_index",
    "finite_observation_count",
    "first_origin_index",
    "latest_origin_index",
    "historical_utility_score",
    "latest_relative_utility",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(
    path: Path,
    records: Iterable[dict[str, Any]],
    fieldnames: Iterable[str],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)
    return sha256_file(path)


def load_frozen_p1b(
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the owner-approved P1b contract and all frozen ancestors."""
    root = Path(repo_root).resolve()
    config = load_yaml(root / P1B_CONFIG_PATH)
    observed_hash = canonical_config_hash(config)
    if observed_hash != EXPECTED_P1B_CONFIG_HASH:
        raise RuntimeError(
            f"P1b config hash mismatch: {observed_hash} != {EXPECTED_P1B_CONFIG_HASH}"
        )
    if config["status"] != "frozen_before_router_state_and_sealed_evaluation":
        raise RuntimeError("P1b contract is not frozen")
    if config["owner_approval"] != "APPROVE_P1B":
        raise RuntimeError("P1b owner approval is missing")
    scope = config["scope"]
    if not scope["calibration_origins_only_for_method_construction"]:
        raise RuntimeError("P1b method construction must remain calibration-only")
    if scope["sealed_evaluation_origins_instantiated_before_freeze"]:
        raise RuntimeError("P1b contract records pre-freeze sealed access")
    if scope["update_router_with_sealed_outcomes"]:
        raise RuntimeError("P1b must not update from sealed outcomes")

    p0b, parent, p0a = load_frozen_p0b(root)
    if canonical_config_hash(p0b) != config["candidate_policies"][
        "inherit_from_p0b_config_hash"
    ]:
        raise RuntimeError("P1b candidate-set ancestor mismatch")
    decision = load_yaml(root / P1A_DECISION_PATH)
    if decision["config_hash"] != config["prerequisite_p1a_config_hash"]:
        raise RuntimeError("P1a decision configuration hash mismatch")
    if decision["gate"]["passed"] is not True:
        raise RuntimeError("P1a decision did not authorize P1b")
    if decision["decision_outcome"]["sealed_evaluation_authorized"] is not False:
        raise RuntimeError("P1a decision unexpectedly authorizes sealed evaluation")
    observed_map = {
        backbone: {
            task: bool(values["eligible"])
            for task, values in tasks.items()
        }
        for backbone, tasks in decision["groups"].items()
    }
    if observed_map != config["applicability_map"]:
        raise RuntimeError("P1b applicability map does not match P1a")
    return config, p0b, parent, p0a


def verify_private_p1a_report(
    repo_root: str | Path,
    p1a_output_root: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Verify the full private P1a report against the committed compact decision."""
    root = Path(repo_root).resolve()
    report_path = Path(p1a_output_root) / "reports" / "p1a_applicability_audit.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing private P1a report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    decision = load_yaml(root / P1A_DECISION_PATH)
    if report["config_hash"] != EXPECTED_P1A_CONFIG_HASH:
        raise RuntimeError("private P1a configuration hash mismatch")
    expected_code_hash = p1a_scientific_code_hash(root)
    if report["scientific_code_sha256"] != expected_code_hash:
        raise RuntimeError("private P1a scientific-code hash mismatch")
    if decision["scientific_code_sha256"] != expected_code_hash:
        raise RuntimeError("committed P1a scientific-code hash mismatch")
    if report["scope"]["sealed_evaluation_origins_instantiated"] is not False:
        raise RuntimeError("private P1a report indicates sealed evaluation access")
    if report["scope"]["model_inference_performed"] is not False:
        raise RuntimeError("private P1a report unexpectedly used model inference")
    if report["continuation_gate"]["passed"] is not True:
        raise RuntimeError("private P1a continuation gate did not pass")

    for backbone, tasks in config["applicability_map"].items():
        for task, expected_eligible in tasks.items():
            private = report["groups"][backbone][task]
            archived = decision["groups"][backbone][task]
            checks = (
                math.isclose(
                    float(private["bootstrap"]["point_auroc"]),
                    float(archived["point_auroc"]),
                    abs_tol=1.0e-12,
                ),
                math.isclose(
                    float(private["bootstrap"]["one_sided_lower_bound"]),
                    float(archived["lower_bound"]),
                    abs_tol=1.0e-12,
                ),
                bool(private["eligible"]) is bool(expected_eligible),
            )
            if not all(checks):
                raise RuntimeError(f"private P1a group mismatch: {backbone}/{task}")
    return report


def logical_policy_order(
    parent: dict[str, Any],
    p0b: dict[str, Any],
    dataset_config: str,
) -> list[str]:
    """Reconstruct the predeclared logical policy order before deduplication."""
    matches = [
        task
        for task in parent["data"]["tasks"]
        if task["dataset_config"] == dataset_config
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one task specification for {dataset_config}")
    task = matches[0]
    known = sorted(task["known_dynamic_columns"])
    all_dynamic = sorted(known + list(task["past_dynamic_columns"]))
    order = ["target_only", "all_dynamic"]
    order.extend(f"single_known_future:{column}" for column in known)
    order.extend(f"leave_one_out:{column}" for column in all_dynamic)
    order.extend(
        f"correlation_top_{value}"
        for value in p0b["candidate_policies"]["historical_correlation_top_k"]
    )
    if len(order) != len(set(order)):
        raise RuntimeError(f"duplicate logical policy ID for {dataset_config}")
    return order


def _read_origin_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise RuntimeError(f"unexpected P0b CSV schema: {path}")
        records: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            row["origin_index"] = int(row["origin_index"])
            for metric in ("SQL", "WQL", "MASE", "WAPE"):
                row[metric] = float(row[metric])
            records.append(row)
    return records


def reexpand_calibration_aliases(
    p0b_output_root: str | Path,
    p0a: dict[str, Any],
    backbone: str,
    p0b_code_hash: str,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Re-expand logical aliases solely for complete-coverage constant selection."""
    source_root = Path(p0b_output_root)
    expanded: list[dict[str, Any]] = []
    inventory_paths: list[Path] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    for task, expected in p0a["tasks"].items():
        for origin_index in expected["calibration_origin_indices"]:
            sidecar = completed_origin(
                source_root,
                backbone=backbone,
                dataset_config=task,
                origin_index=origin_index,
                config_hash=EXPECTED_P0B_CONFIG_HASH,
                scientific_code_sha256=p0b_code_hash,
            )
            if sidecar is None:
                raise RuntimeError(f"missing P0b unit: {backbone}/{task}/{origin_index}")
            csv_path, sidecar_path = origin_paths(
                source_root, backbone, task, origin_index
            )
            inventory_paths.extend([csv_path, sidecar_path])
            alias_map: dict[str, list[str]] = {}
            for policy in sidecar["unique_policies"]:
                retained = str(policy["policy_id"])
                alias_map[retained] = [
                    retained, *[str(alias) for alias in policy.get("aliases", [])]
                ]
            if set(alias_map) != set(sidecar["evaluated_policy_ids"]):
                raise RuntimeError("P0b retained policy metadata mismatch")
            for row in _read_origin_records(csv_path):
                retained = str(row["policy_id"])
                if retained not in alias_map:
                    raise RuntimeError(f"missing alias metadata for {retained}")
                for logical_id in alias_map[retained]:
                    key = (
                        task,
                        origin_index,
                        str(row["item_id"]),
                        str(row["target"]),
                        logical_id,
                    )
                    if key in seen:
                        raise RuntimeError(f"duplicate expanded logical policy row: {key}")
                    seen.add(key)
                    expanded.append(
                        {
                            **row,
                            "policy_id": logical_id,
                            "retained_policy_id": retained,
                        }
                    )
    return expanded, inventory_paths


def select_constant_policies(
    expanded_records: Iterable[dict[str, Any]],
    *,
    parent: dict[str, Any],
    p0b: dict[str, Any],
    backbone: str,
) -> dict[str, Any]:
    """Select full-set and binary constants with complete calibration coverage."""
    rows = [dict(row) for row in expanded_records]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["backbone"] != backbone:
            raise ValueError("constant selection received another backbone")
        by_task[str(row["dataset_config"])].append(row)

    result: dict[str, Any] = {}
    for task in p0a_task_order(parent):
        task_rows = by_task[task]
        order = logical_policy_order(parent, p0b, task)
        order_index = {policy: index for index, policy in enumerate(order)}
        target_rows = [row for row in task_rows if row["policy_id"] == "target_only"]
        units = {
            (int(row["origin_index"]), str(row["item_id"]), str(row["target"]))
            for row in target_rows
        }
        if not units:
            raise RuntimeError(f"no calibration units for {backbone}/{task}")
        losses: dict[str, dict[tuple[int, str, str], float]] = defaultdict(dict)
        for row in task_rows:
            policy_id = str(row["policy_id"])
            key = (int(row["origin_index"]), str(row["item_id"]), str(row["target"]))
            if key in losses[policy_id]:
                raise RuntimeError(f"duplicate constant-selection row: {task}/{policy_id}/{key}")
            losses[policy_id][key] = float(row["SQL"])

        statistics: dict[str, dict[str, Any]] = {}
        eligible: list[tuple[float, int, str]] = []
        for policy_id in order:
            policy_losses = losses.get(policy_id, {})
            complete = set(policy_losses) == units
            all_finite = complete and all(math.isfinite(value) for value in policy_losses.values())
            mean_sql = (
                sum(policy_losses.values()) / len(policy_losses) if all_finite else None
            )
            statistics[policy_id] = {
                "complete_unit_coverage": complete,
                "all_sql_finite": all_finite,
                "unit_count": len(policy_losses),
                "mean_sql": mean_sql,
            }
            if mean_sql is not None:
                eligible.append((mean_sql, order_index[policy_id], policy_id))
        if not eligible:
            raise RuntimeError(f"no complete finite constant policy for {backbone}/{task}")
        _, _, fullset = min(eligible)
        binary_candidates = [
            entry
            for entry in eligible
            if entry[2] in {"target_only", "all_dynamic"}
        ]
        if len(binary_candidates) != 2:
            raise RuntimeError(f"binary constants lack complete coverage for {backbone}/{task}")
        _, _, binary = min(binary_candidates)
        result[task] = {
            "fullset_best_policy_id": fullset,
            "fullset_best_mean_sql": statistics[fullset]["mean_sql"],
            "binary_best_policy_id": binary,
            "binary_best_mean_sql": statistics[binary]["mean_sql"],
            "calibration_unit_count": len(units),
            "logical_policy_count": len(order),
            "eligible_complete_finite_policy_count": len(eligible),
            "policy_statistics": statistics,
        }
    return result


def p0a_task_order(parent: dict[str, Any]) -> list[str]:
    return [str(task["dataset_config"]) for task in parent["data"]["tasks"]]


def build_router_score_records(
    records: Iterable[dict[str, Any]],
    *,
    parent: dict[str, Any],
    p0b: dict[str, Any],
    backbone: str,
    half_life_origins: float,
    epsilon: float,
) -> list[dict[str, Any]]:
    """Freeze per-series retained-policy scores from every calibration outcome."""
    if half_life_origins <= 0:
        raise ValueError("half_life_origins must be positive")
    rows = [dict(row) for row in records]
    target_sql: dict[tuple[str, int, str, str], float] = {}
    by_series_policy: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["backbone"] != backbone:
            raise ValueError("router scoring received another backbone")
        task = str(row["dataset_config"])
        origin = int(row["origin_index"])
        item_id = str(row["item_id"])
        target = str(row["target"])
        policy_id = str(row["policy_id"])
        key = (task, origin, item_id, target)
        if policy_id == "target_only":
            if key in target_sql:
                raise RuntimeError(f"duplicate target-only calibration row: {key}")
            target_sql[key] = float(row["SQL"])
        else:
            by_series_policy[(task, item_id, target, policy_id)].append(row)

    order_by_task = {
        task: {
            policy: index
            for index, policy in enumerate(logical_policy_order(parent, p0b, task))
        }
        for task in p0a_task_order(parent)
    }
    scores: list[dict[str, Any]] = []
    for (task, item_id, target, policy_id), policy_rows in sorted(
        by_series_policy.items()
    ):
        observations: list[tuple[int, float]] = []
        for row in sorted(policy_rows, key=lambda value: int(value["origin_index"])):
            origin = int(row["origin_index"])
            comparator_key = (task, origin, item_id, target)
            if comparator_key not in target_sql:
                raise RuntimeError(f"missing target-only comparator: {comparator_key}")
            target_loss = target_sql[comparator_key]
            candidate_loss = float(row["SQL"])
            if not math.isfinite(target_loss) or not math.isfinite(candidate_loss):
                continue
            utility = (target_loss - candidate_loss) / max(abs(target_loss), epsilon)
            observations.append((origin, utility))
        if not observations:
            continue
        reference_origin = max(origin for origin, _ in observations) + 1
        weights = [
            0.5 ** ((reference_origin - origin) / half_life_origins)
            for origin, _ in observations
        ]
        score = sum(
            weight * utility
            for weight, (_, utility) in zip(weights, observations, strict=True)
        ) / sum(weights)
        if not math.isfinite(score):
            raise RuntimeError("finite calibration utilities produced a non-finite score")
        if policy_id not in order_by_task[task]:
            raise RuntimeError(f"unknown retained policy ID: {task}/{policy_id}")
        scores.append(
            {
                "backbone": backbone,
                "dataset_config": task,
                "item_id": item_id,
                "target": target,
                "policy_id": policy_id,
                "policy_order_index": order_by_task[task][policy_id],
                "finite_observation_count": len(observations),
                "first_origin_index": min(origin for origin, _ in observations),
                "latest_origin_index": max(origin for origin, _ in observations),
                "historical_utility_score": score,
                "latest_relative_utility": max(observations, key=lambda pair: pair[0])[1],
            }
        )
    return scores


def inventory_sha256(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    resolved_root = root.resolve()
    for path in sorted({value.resolve() for value in paths}, key=str):
        relative = path.relative_to(resolved_root)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def scientific_code_hash(repo_root: str | Path) -> str:
    """Hash every committed file controlling P1b state or sealed evaluation."""
    root = Path(repo_root).resolve()
    paths = [
        P1B_CONFIG_PATH,
        P1A_DECISION_PATH,
        Path("configs/diagnostic/applicability_gate_p1a.yaml"),
        Path("configs/diagnostic/covariate_oracle_p0b.yaml"),
        Path("configs/diagnostic/covariate_utility_p0.yaml"),
        Path("evidence/screening/p0b_cross_backbone_decision.yaml"),
        Path("src/covsafe/config.py"),
        Path("src/covsafe/diagnostics.py"),
        Path("src/covsafe/fev_smoke.py"),
        Path("src/covsafe/p0a.py"),
        Path("src/covsafe/p0b.py"),
        Path("src/covsafe/p1a.py"),
        Path("src/covsafe/p1b.py"),
        Path("src/covsafe/protocol.py"),
        Path("src/covsafe/timesfm3_smoke.py"),
    ]
    optional = [
        Path("src/covsafe/p1b_eval.py"),
        Path("src/covsafe/chronos2_p1b.py"),
        Path("src/covsafe/timesfm3_p1b.py"),
        Path("src/covsafe/p1b_analysis.py"),
    ]
    paths.extend(path for path in optional if (root / path).exists())
    digest = hashlib.sha256()
    for relative in sorted(paths, key=str):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_p1b_router_state(
    repo_root: str | Path,
    p0b_output_root: str | Path,
    p1a_output_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build the immutable calibration-only router state without sealed access."""
    root = Path(repo_root).resolve()
    source_root = Path(p0b_output_root)
    destination = Path(output_root)
    config, p0b, parent, p0a = load_frozen_p1b(root)
    existing_report_path = destination / "reports" / "p1b_router_state.json"
    existing_score_path = destination / "state" / "series_policy_scores.csv"
    if existing_report_path.exists():
        existing = json.loads(existing_report_path.read_text(encoding="utf-8"))
        artifact = existing["router_score_artifact"]
        valid = (
            existing.get("config_hash") == EXPECTED_P1B_CONFIG_HASH
            and existing.get("scientific_code_sha256") == scientific_code_hash(root)
            and existing.get("scope", {}).get("sealed_evaluation_origins_instantiated")
            is False
            and existing_score_path.exists()
            and artifact.get("sha256") == sha256_file(existing_score_path)
        )
        if not valid:
            raise RuntimeError("existing P1b router state is stale or altered")
        return existing
    if existing_score_path.exists():
        raise RuntimeError("partial P1b router state requires manual inspection")
    p1a_report = verify_private_p1a_report(root, p1a_output_root, config)

    all_scores: list[dict[str, Any]] = []
    constants: dict[str, Any] = {}
    p0b_hashes: dict[str, str] = {}
    inventory_paths: list[Path] = []
    for backbone in config["scope"]["frozen_zero_shot_backbones"]:
        code_hash = p0b_scientific_code_hash(root, backbone)
        p0b_hashes[backbone] = code_hash
        retained_records = read_candidate_records(
            source_root, p0a, backbone, code_hash
        )
        expanded_records, paths = reexpand_calibration_aliases(
            source_root, p0a, backbone, code_hash
        )
        inventory_paths.extend(paths)
        constants[backbone] = select_constant_policies(
            expanded_records,
            parent=parent,
            p0b=p0b,
            backbone=backbone,
        )
        all_scores.extend(
            build_router_score_records(
                retained_records,
                parent=parent,
                p0b=p0b,
                backbone=backbone,
                half_life_origins=float(
                    config["historical_utility_router"]["half_life_origins"]
                ),
                epsilon=float(config["historical_utility_router"]["epsilon"]),
            )
        )

    score_path = destination / "state" / "series_policy_scores.csv"
    score_sha256 = atomic_csv(
        score_path,
        sorted(
            all_scores,
            key=lambda row: (
                row["backbone"],
                row["dataset_config"],
                row["item_id"],
                row["target"],
                row["policy_order_index"],
            ),
        ),
        ROUTER_SCORE_FIELDS,
    )
    report = {
        "experiment": config["experiment"],
        "schema_version": config["schema_version"],
        "result_status": "screening_only",
        "evidence_role": "frozen_method_state_not_paper_result",
        "config_hash": EXPECTED_P1B_CONFIG_HASH,
        "git_commit": git_commit(root),
        "scientific_code_sha256": scientific_code_hash(root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "accelerator": "cpu",
        },
        "scope": {
            "calibration_origins_only": True,
            "sealed_evaluation_origins_instantiated": False,
            "model_inference_performed": False,
            "router_updates_from_sealed_outcomes": False,
        },
        "applicability_map": config["applicability_map"],
        "constant_policies": constants,
        "router_score_artifact": {
            "relative_path": str(score_path.relative_to(destination)),
            "row_count": len(all_scores),
            "sha256": score_sha256,
        },
        "source_artifacts": {
            "p1a_report_sha256": sha256_file(
                Path(p1a_output_root) / "reports" / "p1a_applicability_audit.json"
            ),
            "p1a_scientific_code_sha256": p1a_report["scientific_code_sha256"],
            "p0b_scientific_code_sha256": p0b_hashes,
            "p0b_calibration_inventory_sha256": inventory_sha256(
                source_root, inventory_paths
            ),
            "p0b_calibration_file_count": len(set(inventory_paths)),
        },
        "authorization": {
            "router_state_frozen": True,
            "sealed_evaluation_authorized_after_owner_review": True,
            "next_action": "review_state_then_run_backbone_specific_sealed_notebooks",
        },
    }
    atomic_json(existing_report_path, report)
    return report
