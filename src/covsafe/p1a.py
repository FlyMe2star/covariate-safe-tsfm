"""Calibration-only confidence audit for task-backbone applicability."""

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

import numpy as np

from covsafe.config import canonical_config_hash, load_yaml
from covsafe.fev_smoke import git_commit
from covsafe.p0c import (
    EXPECTED_P0C_CONFIG_HASH,
    SCORE_CSV_FIELDS,
    load_frozen_p0c,
    tie_aware_binary_auroc,
)
from covsafe.p0c import scientific_code_hash as p0c_scientific_code_hash

EXPECTED_P1A_CONFIG_HASH = "6fe503a9d399"
P1A_CONFIG_PATH = Path("configs/diagnostic/applicability_gate_p1a.yaml")
P0C_DECISION_PATH = Path("evidence/screening/p0c_predictability_decision.yaml")


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


def load_frozen_p1a(repo_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load P1a and verify that it descends from the archived failed P0c screen."""
    root = Path(repo_root).resolve()
    config = load_yaml(root / P1A_CONFIG_PATH)
    observed_hash = canonical_config_hash(config)
    if observed_hash != EXPECTED_P1A_CONFIG_HASH:
        raise RuntimeError(
            f"P1a config hash mismatch: {observed_hash} != {EXPECTED_P1A_CONFIG_HASH}"
        )
    scope = config["scope"]
    if not scope["calibration_origins_only"]:
        raise RuntimeError("P1a must remain calibration-only")
    if scope["sealed_evaluation_origins_instantiated"]:
        raise RuntimeError("sealed evaluation origins must not be instantiated")
    if scope["model_inference_performed"]:
        raise RuntimeError("P1a must remain analysis-only")

    p0c, _p0b, _parent, _p0a = load_frozen_p0c(root)
    if canonical_config_hash(p0c) != config["prerequisite_p0c_config_hash"]:
        raise RuntimeError("P0c prerequisite configuration hash mismatch")
    observed_p0c_code_hash = p0c_scientific_code_hash(root)
    if observed_p0c_code_hash != config["prerequisite_p0c_scientific_code_sha256"]:
        raise RuntimeError("P0c prerequisite scientific-code hash mismatch")

    decision = load_yaml(root / P0C_DECISION_PATH)
    if decision["config_hash"] != EXPECTED_P0C_CONFIG_HASH:
        raise RuntimeError("archived P0c decision configuration hash mismatch")
    if decision["scientific_code_sha256"] != observed_p0c_code_hash:
        raise RuntimeError("archived P0c scientific-code hash mismatch")
    if decision["gate"]["passed"] is not False:
        raise RuntimeError("P1a requires the archived failed P0c gate")
    outcome = decision["decision_outcome"]
    if outcome["sealed_evaluation_authorized"] is not False:
        raise RuntimeError("archived decision unexpectedly authorizes sealed evaluation")
    return config, decision


def _weighted_auroc_from_cluster_counts(
    labels: np.ndarray,
    tie_groups: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_counts: np.ndarray,
) -> float | None:
    """Compute AUROC after applying a bootstrap multiplicity to each cluster."""
    row_weights = cluster_counts[cluster_ids]
    positives_by_tie = np.bincount(
        tie_groups, weights=row_weights * labels, minlength=int(tie_groups.max()) + 1
    )
    negatives_by_tie = np.bincount(
        tie_groups,
        weights=row_weights * (1 - labels),
        minlength=int(tie_groups.max()) + 1,
    )
    positive_total = float(positives_by_tie.sum())
    negative_total = float(negatives_by_tie.sum())
    if positive_total == 0.0 or negative_total == 0.0:
        return None
    negatives_below = np.cumsum(negatives_by_tie) - negatives_by_tie
    numerator = float(
        np.sum(positives_by_tie * (negatives_below + 0.5 * negatives_by_tie))
    )
    return numerator / (positive_total * negative_total)


def cluster_bootstrap_auroc(
    records: Iterable[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    lower_quantile: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap AUROC while preserving policy rows within forecast-origin clusters."""
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("bootstrap requires at least one row")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < lower_quantile < 0.5:
        raise ValueError("lower_quantile must be between zero and one half")

    labels = np.asarray([int(row["label_positive"]) for row in rows], dtype=np.int8)
    scores = np.asarray(
        [float(row["historical_utility_score"]) for row in rows], dtype=np.float64
    )
    if not np.isfinite(scores).all():
        raise ValueError("bootstrap scores must be finite")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("bootstrap labels must be binary")
    point = tie_aware_binary_auroc(labels.tolist(), scores.tolist())

    cluster_lookup: dict[tuple[str, str, int], int] = {}
    cluster_ids_list: list[int] = []
    for row in rows:
        key = (str(row["item_id"]), str(row["target"]), int(row["origin_index"]))
        if key not in cluster_lookup:
            cluster_lookup[key] = len(cluster_lookup)
        cluster_ids_list.append(cluster_lookup[key])
    cluster_ids = np.asarray(cluster_ids_list, dtype=np.int64)
    cluster_count = len(cluster_lookup)
    _unique_scores, tie_groups = np.unique(scores, return_inverse=True)

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        multiplicities = np.bincount(sampled, minlength=cluster_count)
        estimate = _weighted_auroc_from_cluster_counts(
            labels, tie_groups, cluster_ids, multiplicities
        )
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        raise RuntimeError("no valid bootstrap AUROC replicate")
    values = np.asarray(estimates, dtype=np.float64)
    return {
        "point_auroc": point,
        "cluster_count": cluster_count,
        "requested_replicates": replicates,
        "valid_replicates": len(estimates),
        "valid_replicate_fraction": len(estimates) / replicates,
        "one_sided_lower_bound": float(np.quantile(values, lower_quantile)),
        "bootstrap_median": float(np.quantile(values, 0.5)),
        "bootstrap_upper_95pct": float(np.quantile(values, 0.95)),
    }


def summarize_applicability_group(
    records: Iterable[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    lower_quantile: float,
    minimum_valid_replicate_fraction: float,
    minimum_scored_rows: int,
    point_auroc_at_least: float,
    lower_bound_strictly_greater_than: float,
) -> dict[str, Any]:
    """Apply the frozen confidence and eligibility rules to one task-backbone group."""
    rows = [dict(row) for row in records]
    labels = [int(row["label_positive"]) for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    preconditions = {
        "minimum_scored_rows_passed": len(rows) >= minimum_scored_rows,
        "both_label_classes_passed": positives > 0 and negatives > 0,
    }
    if not all(preconditions.values()):
        return {
            "scored_rows": len(rows),
            "positive_labels": positives,
            "negative_labels": negatives,
            "preconditions": preconditions,
            "bootstrap": None,
            "checks": {
                "minimum_valid_replicate_fraction_passed": False,
                "point_auroc_passed": False,
                "lower_bound_passed": False,
            },
            "eligible": False,
        }

    bootstrap = cluster_bootstrap_auroc(
        rows,
        replicates=replicates,
        seed=seed,
        lower_quantile=lower_quantile,
    )
    checks = {
        "minimum_valid_replicate_fraction_passed": (
            bootstrap["valid_replicate_fraction"] >= minimum_valid_replicate_fraction
        ),
        "point_auroc_passed": bootstrap["point_auroc"] >= point_auroc_at_least,
        "lower_bound_passed": (
            bootstrap["one_sided_lower_bound"]
            > lower_bound_strictly_greater_than
        ),
    }
    return {
        "scored_rows": len(rows),
        "positive_labels": positives,
        "negative_labels": negatives,
        "preconditions": preconditions,
        "bootstrap": bootstrap,
        "checks": checks,
        "eligible": all(preconditions.values()) and all(checks.values()),
    }


def evaluate_continuation_gate(
    groups: dict[str, dict[str, dict[str, Any]]],
    *,
    minimum_eligible_groups_total: int,
    minimum_eligible_tasks_per_backbone: int,
) -> dict[str, Any]:
    """Require nontrivial coverage and representation from every backbone."""
    eligible_by_backbone = {
        backbone: sorted(task for task, result in tasks.items() if result["eligible"])
        for backbone, tasks in groups.items()
    }
    total = sum(len(tasks) for tasks in eligible_by_backbone.values())
    per_backbone_passed = {
        backbone: len(tasks) >= minimum_eligible_tasks_per_backbone
        for backbone, tasks in eligible_by_backbone.items()
    }
    checks = {
        "minimum_eligible_groups_total_passed": total >= minimum_eligible_groups_total,
        "minimum_eligible_tasks_per_backbone_passed": all(
            per_backbone_passed.values()
        ),
    }
    return {
        "eligible_groups_total": total,
        "eligible_tasks_by_backbone": eligible_by_backbone,
        "per_backbone_minimum_passed": per_backbone_passed,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _read_and_verify_p0c_scores(
    p0c_root: Path,
    config: dict[str, Any],
    decision: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    report_path = p0c_root / config["inputs"]["p0c_report_relative_path"]
    if not report_path.exists():
        raise FileNotFoundError(f"missing P0c report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["config_hash"] != config["prerequisite_p0c_config_hash"]:
        raise RuntimeError("private P0c report configuration hash mismatch")
    if report["scientific_code_sha256"] != config[
        "prerequisite_p0c_scientific_code_sha256"
    ]:
        raise RuntimeError("private P0c report scientific-code hash mismatch")
    if report["cross_backbone_gate"]["passed"] is not False:
        raise RuntimeError("private P0c report does not contain the expected failed gate")
    if report["scope"]["sealed_evaluation_origins_instantiated"] is not False:
        raise RuntimeError("private P0c report indicates sealed evaluation access")
    if report["scope"]["model_inference_performed"] is not False:
        raise RuntimeError("private P0c report unexpectedly indicates model inference")

    artifact = report["score_artifact"]
    expected_relative = config["inputs"]["p0c_score_relative_path"]
    if artifact["relative_path"] != expected_relative:
        raise RuntimeError("P0c score relative path mismatch")
    score_path = p0c_root / expected_relative
    observed_sha256 = _sha256(score_path)
    expected_sha256 = decision["score_artifact"]["sha256"]
    if artifact["sha256"] != expected_sha256 or observed_sha256 != expected_sha256:
        raise RuntimeError("P0c score artifact checksum mismatch")

    with score_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SCORE_CSV_FIELDS:
            raise RuntimeError("unexpected P0c score CSV schema")
        rows = [dict(row) for row in reader]
    if len(rows) != int(artifact["row_count"]):
        raise RuntimeError("P0c score row count mismatch")
    return rows, report, score_path


def scientific_code_hash(repo_root: str | Path) -> str:
    """Hash files that determine P1a inputs, confidence intervals, and gates."""
    root = Path(repo_root).resolve()
    paths = [
        P1A_CONFIG_PATH,
        P0C_DECISION_PATH,
        Path("configs/diagnostic/covariate_predictability_p0c.yaml"),
        Path("src/covsafe/config.py"),
        Path("src/covsafe/p0c.py"),
        Path("src/covsafe/p1a.py"),
    ]
    digest = hashlib.sha256()
    for relative in sorted(paths, key=str):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_p1a(
    repo_root: str | Path,
    p0c_output_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run the deterministic CPU-only P1a applicability audit."""
    root = Path(repo_root).resolve()
    destination = Path(output_root)
    config, decision = load_frozen_p1a(root)
    rows, p0c_report, score_path = _read_and_verify_p0c_scores(
        Path(p0c_output_root), config, decision
    )

    required_backbones = list(config["inputs"]["required_backbones"])
    required_tasks = list(config["inputs"]["required_tasks"])
    allowed_groups = {
        (backbone, task) for backbone in required_backbones for task in required_tasks
    }
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["backbone"]), str(row["dataset_config"]))
        if key not in allowed_groups:
            raise RuntimeError(f"unexpected P0c score group: {key}")
        by_group[key].append(row)
    if set(by_group) != allowed_groups:
        missing = sorted(allowed_groups - set(by_group))
        raise RuntimeError(f"missing P0c score groups: {missing}")

    bootstrap = config["bootstrap"]
    eligibility = config["group_eligibility"]
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    group_index = 0
    for backbone in required_backbones:
        groups[backbone] = {}
        for task in required_tasks:
            result = summarize_applicability_group(
                by_group[(backbone, task)],
                replicates=int(bootstrap["replicates"]),
                seed=int(config["seed"]) + group_index,
                lower_quantile=float(bootstrap["lower_quantile"]),
                minimum_valid_replicate_fraction=float(
                    bootstrap["minimum_valid_replicate_fraction"]
                ),
                minimum_scored_rows=int(eligibility["minimum_scored_rows"]),
                point_auroc_at_least=float(eligibility["point_auroc_at_least"]),
                lower_bound_strictly_greater_than=float(
                    eligibility[
                        "one_sided_95pct_lower_bound_strictly_greater_than"
                    ]
                ),
            )
            archived_point = float(
                decision["backbones"][backbone]["per_task_auroc"][task]
            )
            observed_point = result["bootstrap"]["point_auroc"]
            if not math.isclose(observed_point, archived_point, abs_tol=1.0e-12):
                raise RuntimeError(
                    f"P0c point AUROC mismatch for {backbone}/{task}: "
                    f"{observed_point} != {archived_point}"
                )
            groups[backbone][task] = result
            group_index += 1

    continuation = config["continuation_gate"]
    gate = evaluate_continuation_gate(
        groups,
        minimum_eligible_groups_total=int(
            continuation["minimum_eligible_groups_total"]
        ),
        minimum_eligible_tasks_per_backbone=int(
            continuation["minimum_eligible_tasks_per_backbone"]
        ),
    )
    gate["minimum_eligible_groups_total"] = continuation[
        "minimum_eligible_groups_total"
    ]
    gate["minimum_eligible_tasks_per_backbone"] = continuation[
        "minimum_eligible_tasks_per_backbone"
    ]
    gate["next_action"] = (
        continuation["next_if_pass"] if gate["passed"] else continuation["next_if_fail"]
    )

    report = {
        "experiment": config["experiment"],
        "schema_version": config["schema_version"],
        "result_status": "screening_only",
        "evidence_role": config["evidence_role"],
        "config_hash": EXPECTED_P1A_CONFIG_HASH,
        "prerequisite_p0c_config_hash": config["prerequisite_p0c_config_hash"],
        "prerequisite_p0c_scientific_code_sha256": config[
            "prerequisite_p0c_scientific_code_sha256"
        ],
        "git_commit": git_commit(root),
        "scientific_code_sha256": scientific_code_hash(root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "accelerator": "cpu",
        },
        "scope": {
            "calibration_origins_only": True,
            "sealed_evaluation_origins_instantiated": False,
            "model_inference_performed": False,
        },
        "thresholds": {
            "point_auroc_at_least": eligibility["point_auroc_at_least"],
            "one_sided_95pct_lower_bound_strictly_greater_than": eligibility[
                "one_sided_95pct_lower_bound_strictly_greater_than"
            ],
            "bootstrap_replicates": bootstrap["replicates"],
            "resampling_unit": bootstrap["resampling_unit"],
        },
        "groups": groups,
        "continuation_gate": gate,
        "source_artifacts": {
            "p0c_report_sha256": _sha256(
                Path(p0c_output_root)
                / config["inputs"]["p0c_report_relative_path"]
            ),
            "p0c_score_relative_path": str(
                score_path.relative_to(Path(p0c_output_root))
            ),
            "p0c_score_sha256": _sha256(score_path),
            "p0c_score_row_count": len(rows),
            "p0c_report_git_commit": p0c_report["git_commit"],
        },
    }
    _atomic_json(destination / "reports" / "p1a_applicability_audit.json", report)
    return report
