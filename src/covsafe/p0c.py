"""Calibration-only prequential predictability screen for P0c."""

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
    P0_CONFIG_PATH,
    P0B_CONFIG_PATH,
    load_frozen_p0b,
    read_candidate_records,
)
from covsafe.p0b import scientific_code_hash as p0b_scientific_code_hash

EXPECTED_P0C_CONFIG_HASH = "00293b538504"
P0C_CONFIG_PATH = Path("configs/diagnostic/covariate_predictability_p0c.yaml")
P0B_DECISION_PATH = Path("evidence/screening/p0b_cross_backbone_decision.yaml")
SCORE_CSV_FIELDS = (
    "backbone",
    "dataset_config",
    "origin_index",
    "cutoff",
    "item_id",
    "target",
    "policy_id",
    "prior_observation_count",
    "latest_prior_origin_index",
    "historical_utility_score",
    "realized_relative_utility",
    "label_positive",
)


def load_frozen_p0c(
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load P0c and verify its parent and passed P0b prerequisite."""
    root = Path(repo_root).resolve()
    config = load_yaml(root / P0C_CONFIG_PATH)
    observed_hash = canonical_config_hash(config)
    if observed_hash != EXPECTED_P0C_CONFIG_HASH:
        raise RuntimeError(
            f"P0c config hash mismatch: {observed_hash} != {EXPECTED_P0C_CONFIG_HASH}"
        )
    scope = config["scope"]
    if not scope["calibration_origins_only"]:
        raise RuntimeError("P0c must remain calibration-only")
    if scope["sealed_evaluation_origins_instantiated"]:
        raise RuntimeError("sealed evaluation origins must not be instantiated")
    if scope["model_inference_performed"]:
        raise RuntimeError("P0c must remain analysis-only")

    p0b, parent, p0a = load_frozen_p0b(root)
    if canonical_config_hash(p0b) != config["prerequisite_p0b_config_hash"]:
        raise RuntimeError("prerequisite P0b configuration hash mismatch")
    if canonical_config_hash(parent) != config["parent_p0_config_hash"]:
        raise RuntimeError("parent P0 configuration hash mismatch")
    if (
        config["historical_utility"]["half_life_origins"]
        != parent["historical_utility"]["half_life_origins"]
    ):
        raise RuntimeError("P0c half-life does not match the parent P0 contract")
    if (
        config["gate"]["minimum_task_macro_auroc_per_backbone"]
        != parent["gates"]["minimum_utility_sign_auroc"]
    ):
        raise RuntimeError("P0c AUROC gate does not match the parent P0 contract")
    parent_tasks = [task["dataset_config"] for task in parent["data"]["tasks"]]
    if config["inputs"]["required_tasks"] != parent_tasks:
        raise RuntimeError("P0c tasks do not match the parent P0 contract")
    if config["inputs"]["required_backbones"] != list(p0b["backbones"]):
        raise RuntimeError("P0c backbones do not match P0b")

    decision = load_yaml(root / P0B_DECISION_PATH)
    if decision["config_hash"] != config["prerequisite_p0b_config_hash"]:
        raise RuntimeError("P0b decision configuration hash mismatch")
    if decision["gate"]["passed"] is not True:
        raise RuntimeError("P0b did not authorize P0c")
    return config, p0b, parent, p0a


def _relative_utility(target_sql: float, candidate_sql: float, epsilon: float) -> float:
    return (target_sql - candidate_sql) / max(abs(target_sql), epsilon)


def build_prequential_score_records(
    records: Iterable[dict[str, Any]],
    *,
    half_life_origins: float = 3.0,
    minimum_prior_observations: int = 1,
    epsilon: float = 1.0e-12,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Score each candidate using only the same policy's strictly earlier utilities."""
    if half_life_origins <= 0:
        raise ValueError("half_life_origins must be positive")
    if minimum_prior_observations < 1:
        raise ValueError("minimum_prior_observations must be at least one")

    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("no P0b records")
    backbones = {str(row["backbone"]) for row in rows}
    if len(backbones) != 1:
        raise ValueError(f"expected one backbone, found {sorted(backbones)}")

    def full_key(row: dict[str, Any]) -> tuple[str, int, str, str, str]:
        return (
            str(row["dataset_config"]),
            int(row["origin_index"]),
            str(row["item_id"]),
            str(row["target"]),
            str(row["policy_id"]),
        )

    by_key: dict[tuple[str, int, str, str, str], dict[str, Any]] = {}
    by_series_policy: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    target_sql: dict[tuple[str, int, str, str], float] = {}
    for row in rows:
        key = full_key(row)
        if key in by_key:
            raise ValueError(f"duplicate P0b record: {key}")
        by_key[key] = row
        task, origin, item_id, target, policy_id = key
        by_series_policy[(task, item_id, target, policy_id)].append(row)
        if policy_id == "target_only":
            target_sql[(task, origin, item_id, target)] = float(row["SQL"])

    diagnostics: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "candidate_rows_seen": 0,
            "excluded_nonfinite_current_pair": 0,
            "excluded_insufficient_finite_history": 0,
            "scored_rows": 0,
        }
    )
    scored: list[dict[str, Any]] = []
    for series_key, candidate_rows in sorted(by_series_policy.items()):
        task, item_id, target, policy_id = series_key
        if policy_id == "target_only":
            continue
        ordered = sorted(candidate_rows, key=lambda row: int(row["origin_index"]))
        for current in ordered:
            counts = diagnostics[task]
            counts["candidate_rows_seen"] += 1
            current_origin = int(current["origin_index"])
            current_target_key = (task, current_origin, item_id, target)
            if current_target_key not in target_sql:
                raise ValueError(f"missing target-only comparator: {current_target_key}")
            current_target_sql = target_sql[current_target_key]
            current_candidate_sql = float(current["SQL"])
            if not math.isfinite(current_target_sql) or not math.isfinite(
                current_candidate_sql
            ):
                counts["excluded_nonfinite_current_pair"] += 1
                continue

            prior: list[tuple[int, float]] = []
            for past in ordered:
                past_origin = int(past["origin_index"])
                if past_origin >= current_origin:
                    break
                past_target_key = (task, past_origin, item_id, target)
                if past_target_key not in target_sql:
                    raise ValueError(f"missing target-only comparator: {past_target_key}")
                past_target_sql = target_sql[past_target_key]
                past_candidate_sql = float(past["SQL"])
                if not math.isfinite(past_target_sql) or not math.isfinite(
                    past_candidate_sql
                ):
                    continue
                prior.append(
                    (
                        past_origin,
                        _relative_utility(past_target_sql, past_candidate_sql, epsilon),
                    )
                )
            if len(prior) < minimum_prior_observations:
                counts["excluded_insufficient_finite_history"] += 1
                continue

            weights = [
                0.5 ** ((current_origin - past_origin) / half_life_origins)
                for past_origin, _ in prior
            ]
            weight_sum = sum(weights)
            score = sum(
                weight * utility for weight, (_, utility) in zip(weights, prior, strict=True)
            ) / weight_sum
            realized = _relative_utility(
                current_target_sql, current_candidate_sql, epsilon
            )
            if not math.isfinite(score) or not math.isfinite(realized):
                raise RuntimeError("finite inputs produced non-finite P0c utility")
            latest_prior_origin = max(origin for origin, _ in prior)
            if latest_prior_origin >= current_origin:
                raise RuntimeError("prequential history includes current or future origin")
            scored.append(
                {
                    "backbone": str(current["backbone"]),
                    "dataset_config": task,
                    "origin_index": current_origin,
                    "cutoff": str(current["cutoff"]),
                    "item_id": item_id,
                    "target": target,
                    "policy_id": policy_id,
                    "prior_observation_count": len(prior),
                    "latest_prior_origin_index": latest_prior_origin,
                    "historical_utility_score": score,
                    "realized_relative_utility": realized,
                    "label_positive": int(realized > 0.0),
                }
            )
            counts["scored_rows"] += 1
    return scored, dict(sorted(diagnostics.items()))


def tie_aware_binary_auroc(labels: Iterable[int], scores: Iterable[float]) -> float:
    """Compute AUROC using average ranks, with half credit for score ties."""
    pairs = [(float(score), int(label)) for label, score in zip(labels, scores, strict=True)]
    if not pairs:
        raise ValueError("AUROC requires at least one row")
    if any(label not in {0, 1} for _, label in pairs):
        raise ValueError("AUROC labels must be binary")
    if any(not math.isfinite(score) for score, _ in pairs):
        raise ValueError("AUROC scores must be finite")
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both label classes")

    ordered = sorted(pairs, key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def summarize_predictability_records(
    records: Iterable[dict[str, Any]],
    *,
    required_tasks: Iterable[str],
    minimum_scored_rows_per_task: int = 20,
    minimum_task_macro_auroc: float = 0.65,
) -> dict[str, Any]:
    """Compute the frozen task-macro H3 precursor screen for one backbone."""
    rows = [dict(row) for row in records]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["dataset_config"])].append(row)

    per_task: dict[str, dict[str, Any]] = {}
    valid_aurocs: list[float] = []
    all_tasks_valid = True
    for task in required_tasks:
        task_rows = by_task.get(task, [])
        labels = [int(row["label_positive"]) for row in task_rows]
        scores = [float(row["historical_utility_score"]) for row in task_rows]
        positives = sum(labels)
        negatives = len(labels) - positives
        reason: str | None = None
        auroc: float | None = None
        if len(task_rows) < minimum_scored_rows_per_task:
            reason = "insufficient_scored_rows"
        elif positives == 0 or negatives == 0:
            reason = "single_label_class"
        else:
            auroc = tie_aware_binary_auroc(labels, scores)
            valid_aurocs.append(auroc)
        valid = reason is None
        all_tasks_valid = all_tasks_valid and valid
        per_task[task] = {
            "scored_rows": len(task_rows),
            "positive_labels": positives,
            "negative_labels": negatives,
            "positive_fraction": positives / len(task_rows) if task_rows else None,
            "auroc": auroc,
            "valid_for_gate": valid,
            "invalid_reason": reason,
        }

    unexpected_tasks = sorted(set(by_task) - set(required_tasks))
    if unexpected_tasks:
        raise ValueError(f"unexpected P0c tasks: {unexpected_tasks}")
    macro = sum(valid_aurocs) / len(valid_aurocs) if all_tasks_valid else None
    all_labels = [int(row["label_positive"]) for row in rows]
    all_scores = [float(row["historical_utility_score"]) for row in rows]
    micro = (
        tie_aware_binary_auroc(all_labels, all_scores)
        if all_labels and 0 < sum(all_labels) < len(all_labels)
        else None
    )
    return {
        "aggregation_primary": "task_macro_historical_utility_sign_auroc",
        "per_task": per_task,
        "task_macro_auroc": macro,
        "unit_micro_auroc": micro,
        "scored_row_count": len(rows),
        "minimum_scored_rows_per_task": minimum_scored_rows_per_task,
        "minimum_task_macro_auroc": minimum_task_macro_auroc,
        "all_tasks_valid": all_tasks_valid,
        "h3_precursor_gate_passed": bool(
            all_tasks_valid and macro is not None and macro >= minimum_task_macro_auroc
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_score_csv(path: Path, records: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)
    return _sha256(path)


def scientific_code_hash(repo_root: str | Path) -> str:
    """Hash files that determine P0c inputs, scores, and aggregation."""
    root = Path(repo_root).resolve()
    paths = [
        P0C_CONFIG_PATH,
        P0B_CONFIG_PATH,
        P0_CONFIG_PATH,
        P0B_DECISION_PATH,
        Path("src/covsafe/config.py"),
        Path("src/covsafe/diagnostics.py"),
        Path("src/covsafe/p0b.py"),
        Path("src/covsafe/p0c.py"),
    ]
    digest = hashlib.sha256()
    for relative in sorted(paths, key=str):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_p0c(
    repo_root: str | Path,
    p0b_output_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run the deterministic CPU-only P0c screen over validated P0b artifacts."""
    root = Path(repo_root).resolve()
    destination = Path(output_root)
    config, _p0b, _parent, p0a = load_frozen_p0c(root)
    source_commit = git_commit(root)
    code_hash = scientific_code_hash(root)
    required_tasks = config["inputs"]["required_tasks"]
    settings = config["historical_utility"]
    aggregation = config["aggregation"]
    gate = config["gate"]

    all_scores: list[dict[str, Any]] = []
    backbone_reports: dict[str, Any] = {}
    prerequisite_hashes: dict[str, str] = {}
    for backbone in config["inputs"]["required_backbones"]:
        p0b_hash = p0b_scientific_code_hash(root, backbone)
        prerequisite_hashes[backbone] = p0b_hash
        source_records = read_candidate_records(
            p0b_output_root,
            p0a,
            backbone,
            p0b_hash,
        )
        scores, exclusions = build_prequential_score_records(
            source_records,
            half_life_origins=float(settings["half_life_origins"]),
            minimum_prior_observations=int(
                settings["minimum_prior_finite_observations"]
            ),
            epsilon=float(settings["epsilon"]),
        )
        summary = summarize_predictability_records(
            scores,
            required_tasks=required_tasks,
            minimum_scored_rows_per_task=int(
                aggregation["minimum_scored_rows_per_task"]
            ),
            minimum_task_macro_auroc=float(
                gate["minimum_task_macro_auroc_per_backbone"]
            ),
        )
        backbone_reports[backbone] = {
            "prerequisite_p0b_scientific_code_sha256": p0b_hash,
            "exclusions": exclusions,
            "h3_precursor_screen": summary,
        }
        all_scores.extend(scores)

    observed_passing = sum(
        report["h3_precursor_screen"]["h3_precursor_gate_passed"]
        for report in backbone_reports.values()
    )
    score_path = destination / "scores" / "p0c_prequential_scores.csv"
    score_sha256 = _write_score_csv(
        score_path,
        sorted(
            all_scores,
            key=lambda row: (
                row["backbone"],
                row["dataset_config"],
                row["origin_index"],
                row["item_id"],
                row["target"],
                row["policy_id"],
            ),
        ),
    )
    report = {
        "experiment": config["experiment"],
        "schema_version": config["schema_version"],
        "result_status": "screening_only",
        "config_hash": EXPECTED_P0C_CONFIG_HASH,
        "parent_p0_config_hash": config["parent_p0_config_hash"],
        "prerequisite_p0b_config_hash": config["prerequisite_p0b_config_hash"],
        "git_commit": source_commit,
        "scientific_code_sha256": code_hash,
        "prerequisite_p0b_scientific_code_sha256": prerequisite_hashes,
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
            "full_p0_h3_computed": False,
        },
        "backbones": backbone_reports,
        "cross_backbone_gate": {
            "minimum_task_macro_auroc_per_backbone": gate[
                "minimum_task_macro_auroc_per_backbone"
            ],
            "minimum_backbones_passing": gate["minimum_backbones_passing"],
            "observed_backbones_passing": observed_passing,
            "passed": observed_passing >= int(gate["minimum_backbones_passing"]),
            "next_if_pass": gate["next_if_pass"],
            "next_if_fail": gate["next_if_fail"],
        },
        "score_artifact": {
            "relative_path": str(score_path.relative_to(destination)),
            "row_count": len(all_scores),
            "sha256": score_sha256,
        },
    }
    _atomic_json(destination / "reports" / "p0c_predictability_screen.json", report)
    return report
