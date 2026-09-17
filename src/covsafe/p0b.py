"""Finite-candidate oracle screening helpers for calibration-only P0b."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from covsafe.config import canonical_config_hash, load_yaml
from covsafe.diagnostics import relative_gain
from covsafe.fev_smoke import git_commit
from covsafe.p0a import (
    load_frozen_p0a,
    read_backbone_records,
)
from covsafe.p0a import scientific_code_hash as p0a_scientific_code_hash

EXPECTED_P0B_CONFIG_HASH = "bfd5bae251a5"
P0B_CONFIG_PATH = Path("configs/diagnostic/covariate_oracle_p0b.yaml")
P0_CONFIG_PATH = Path("configs/diagnostic/covariate_utility_p0.yaml")
P0A_CONFIG_PATH = Path("configs/diagnostic/covariate_harm_p0a.yaml")
P0A_DECISION_PATH = Path("evidence/screening/p0a_cross_backbone_decision.yaml")
CSV_FIELDS = (
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


@dataclass
class CandidatePolicy:
    """One unique selected-column set plus any deduplicated policy aliases."""

    policy_id: str
    known_dynamic_columns: tuple[str, ...]
    past_dynamic_columns: tuple[str, ...]
    aliases: list[str] = field(default_factory=list)

    @property
    def column_key(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self.known_dynamic_columns, self.past_dynamic_columns

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "known_dynamic_columns": list(self.known_dynamic_columns),
            "past_dynamic_columns": list(self.past_dynamic_columns),
            "aliases": list(self.aliases),
        }


@dataclass
class TaskContext:
    """A prepared FEV dataset reused by every P0b policy and origin."""

    reference_task: Any
    task_spec: dict[str, Any]
    expected: dict[str, Any]


def load_frozen_p0b(
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load P0b, parent P0, and prerequisite P0a with frozen decision checks."""
    root = Path(repo_root).resolve()
    config = load_yaml(root / P0B_CONFIG_PATH)
    observed_hash = canonical_config_hash(config)
    if observed_hash != EXPECTED_P0B_CONFIG_HASH:
        raise RuntimeError(
            f"P0b config hash mismatch: {observed_hash} != {EXPECTED_P0B_CONFIG_HASH}"
        )
    if not config["scope"]["calibration_origins_only"]:
        raise RuntimeError("P0b must remain calibration-only")
    if config["scope"]["sealed_evaluation_origins_instantiated"]:
        raise RuntimeError("sealed evaluation origins must not be instantiated")

    parent = load_yaml(root / P0_CONFIG_PATH)
    if canonical_config_hash(parent) != config["parent_p0_config_hash"]:
        raise RuntimeError("parent P0 configuration hash mismatch")
    parent_candidates = parent["candidate_policies"]
    candidate_config = config["candidate_policies"]
    expected_candidate_settings = {
        "fixed": parent_candidates["fixed"],
        "enumerate_single_known_future": parent_candidates[
            "enumerate_single_known_future"
        ],
        "enumerate_leave_one_out": parent_candidates["enumerate_leave_one_out"],
        "historical_correlation_top_k": parent_candidates["historical_correlation_top_k"],
        "deduplicate_equivalent_column_sets": parent_candidates[
            "deduplicate_equivalent_policies"
        ],
    }
    for key, expected_value in expected_candidate_settings.items():
        if candidate_config[key] != expected_value:
            raise RuntimeError(f"P0b candidate setting does not match parent P0: {key}")
    if config["metrics"]["primary"] != parent["metrics"]["primary"]:
        raise RuntimeError("P0b primary metric does not match parent P0")
    if (
        config["gate"]["minimum_task_macro_mean_oracle_headroom_per_backbone"]
        != parent["gates"]["minimum_mean_oracle_headroom"]
    ):
        raise RuntimeError("P0b oracle gate does not match parent P0")
    p0a, _ = load_frozen_p0a(root)
    if canonical_config_hash(p0a) != config["prerequisite_p0a_config_hash"]:
        raise RuntimeError("prerequisite P0a configuration hash mismatch")
    decision = load_yaml(root / P0A_DECISION_PATH)
    if decision["config_hash"] != config["prerequisite_p0a_config_hash"]:
        raise RuntimeError("P0a decision configuration hash mismatch")
    if decision["gate"]["passed"] is not True:
        raise RuntimeError("P0a did not authorize P0b")
    return config, parent, p0a


def _parent_task(parent: dict[str, Any], dataset_config: str) -> dict[str, Any]:
    matches = [
        task
        for task in parent["data"]["tasks"]
        if task["dataset_config"] == dataset_config
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one parent task for {dataset_config}, found {len(matches)}")
    return matches[0]


def _task_fields(
    parent: dict[str, Any],
    task_spec: dict[str, Any],
    *,
    num_windows: int,
    known_dynamic_columns: Iterable[str],
    past_dynamic_columns: Iterable[str],
    initial_cutoff: int | str | None = None,
    window_step_size: int | str | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "dataset_path": parent["data"]["dataset_path"],
        "dataset_config": task_spec["dataset_config"],
        "horizon": task_spec["horizon"],
        "num_windows": num_windows,
        "seasonality": task_spec["seasonality"],
        "known_dynamic_columns": sorted(known_dynamic_columns),
        "past_dynamic_columns": sorted(past_dynamic_columns),
        "static_columns": [],
        "eval_metric": "SQL",
        "extra_metrics": [
            {"name": "WQL", "epsilon": 1.0},
            "MASE",
            {"name": "WAPE", "epsilon": 1.0},
        ],
        "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }
    if task_spec.get("target") is not None:
        fields["target"] = task_spec["target"]
    cutoff = initial_cutoff
    if cutoff is None and "initial_cutoff" in task_spec:
        cutoff = task_spec["initial_cutoff"]
    if cutoff is not None:
        fields["initial_cutoff"] = cutoff
    if window_step_size is not None:
        fields["window_step_size"] = window_step_size
    return fields


def prepare_task_context(
    parent: dict[str, Any],
    p0a: dict[str, Any],
    dataset_config: str,
) -> TaskContext:
    """Download and preprocess a task once for reuse across all P0b policies."""
    import fev

    task_spec = _parent_task(parent, dataset_config)
    reference = fev.Task(
        **_task_fields(
            parent,
            task_spec,
            num_windows=task_spec["num_windows"],
            known_dynamic_columns=task_spec["known_dynamic_columns"],
            past_dynamic_columns=task_spec["past_dynamic_columns"],
        )
    )
    dataset = reference.load_full_dataset(num_proc=1)
    expected = p0a["tasks"][dataset_config]
    if reference._dataset_fingerprint != expected["expected_fingerprint"]:
        raise RuntimeError(f"dataset fingerprint mismatch for {dataset_config}")
    if len(dataset) != expected["expected_item_count"]:
        raise RuntimeError(f"dataset item-count mismatch for {dataset_config}")
    return TaskContext(reference_task=reference, task_spec=task_spec, expected=expected)


def make_origin_task(
    parent: dict[str, Any],
    context: TaskContext,
    origin_index: int,
    policy: CandidatePolicy,
) -> Any:
    """Build a one-window FEV task and attach the already prepared dataset."""
    import fev

    reference = context.reference_task
    task = fev.Task(
        **_task_fields(
            parent,
            context.task_spec,
            num_windows=1,
            known_dynamic_columns=policy.known_dynamic_columns,
            past_dynamic_columns=policy.past_dynamic_columns,
            initial_cutoff=reference.cutoffs[origin_index],
            window_step_size=reference.window_step_size,
        )
    )
    task._full_dataset = reference._full_dataset
    task._freq = reference._freq
    task._dataset_fingerprint = reference._dataset_fingerprint
    return task


def _numeric_sequence(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    try:
        return raw.astype(np.float64)
    except (TypeError, ValueError):
        flattened = raw.reshape(-1).tolist()
        observed = sorted({str(value) for value in flattened if value is not None})
        encoding = {value: index for index, value in enumerate(observed)}
        encoded = [
            np.nan if value is None else float(encoding[str(value)])
            for value in flattened
        ]
        return np.asarray(encoded, dtype=np.float64).reshape(raw.shape)


def rank_historical_covariates(
    past_data: Iterable[dict[str, Any]],
    *,
    target_columns: Iterable[str],
    dynamic_columns: Iterable[str],
    minimum_paired_observations: int = 3,
) -> list[dict[str, Any]]:
    """Rank covariates by median absolute within-item historical correlation."""
    rows = list(past_data)
    ranking: list[dict[str, Any]] = []
    for covariate in sorted(dynamic_columns):
        correlations: list[float] = []
        for row in rows:
            covariate_values = _numeric_sequence(row[covariate]).reshape(-1)
            for target in target_columns:
                target_values = _numeric_sequence(row[target]).reshape(-1)
                length = min(target_values.size, covariate_values.size)
                x = target_values[-length:]
                y = covariate_values[-length:]
                finite = np.isfinite(x) & np.isfinite(y)
                if int(finite.sum()) < minimum_paired_observations:
                    continue
                x_valid = x[finite]
                y_valid = y[finite]
                if np.std(x_valid) == 0 or np.std(y_valid) == 0:
                    continue
                value = float(abs(np.corrcoef(x_valid, y_valid)[0, 1]))
                if math.isfinite(value):
                    correlations.append(value)
        score = float(np.median(correlations)) if correlations else None
        ranking.append(
            {
                "column": covariate,
                "score": score,
                "valid_item_target_pairs": len(correlations),
            }
        )
    ranking.sort(
        key=lambda entry: (
            entry["score"] is None,
            -(entry["score"] if entry["score"] is not None else 0.0),
            entry["column"],
        )
    )
    for rank, entry in enumerate(ranking, start=1):
        entry["rank"] = rank
    return ranking


def enumerate_candidate_policies(
    task_spec: dict[str, Any],
    correlation_ranking: list[dict[str, Any]],
    top_k_values: Iterable[int] = (1, 2, 4),
) -> list[CandidatePolicy]:
    """Generate the ordered finite set and deduplicate equivalent column sets."""
    known = tuple(sorted(task_spec["known_dynamic_columns"]))
    past = tuple(sorted(task_spec["past_dynamic_columns"]))
    all_dynamic = tuple(sorted(known + past))
    policies: list[CandidatePolicy] = []
    by_columns: dict[tuple[tuple[str, ...], tuple[str, ...]], CandidatePolicy] = {}

    def add(policy_id: str, selected: Iterable[str]) -> None:
        selected_set = set(selected)
        selected_known = tuple(column for column in known if column in selected_set)
        selected_past = tuple(column for column in past if column in selected_set)
        key = selected_known, selected_past
        if key in by_columns:
            by_columns[key].aliases.append(policy_id)
            return
        policy = CandidatePolicy(policy_id, selected_known, selected_past)
        by_columns[key] = policy
        policies.append(policy)

    add("target_only", [])
    add("all_dynamic", all_dynamic)
    for column in known:
        add(f"single_known_future:{column}", [column])
    for column in all_dynamic:
        add(f"leave_one_out:{column}", [name for name in all_dynamic if name != column])

    ranked_columns = [str(entry["column"]) for entry in correlation_ranking]
    for requested_k in top_k_values:
        clipped_k = min(int(requested_k), len(ranked_columns))
        add(f"correlation_top_{requested_k}", ranked_columns[:clipped_k])
    return policies


def historical_policy_set(
    config: dict[str, Any],
    context: TaskContext,
    origin_index: int,
) -> tuple[list[CandidatePolicy], list[dict[str, Any]]]:
    """Compute the origin-safe correlation ranking and unique candidate policies."""
    window = context.reference_task.get_window(origin_index, num_proc=1)
    past_data, _ = window.get_input_data()
    ranking = rank_historical_covariates(
        past_data,
        target_columns=window.target_columns,
        dynamic_columns=(window.known_dynamic_columns + window.past_dynamic_columns),
        minimum_paired_observations=int(
            config["historical_correlation"]["minimum_paired_observations"]
        ),
    )
    policies = enumerate_candidate_policies(
        context.task_spec,
        ranking,
        config["candidate_policies"]["historical_correlation_top_k"],
    )
    return policies, ranking


def normalize_policy_score_records(
    scores: Any,
    *,
    task: Any,
    backbone: str,
    dataset_config: str,
    origin_index: int,
    policy_id: str,
) -> list[dict[str, Any]]:
    """Convert one policy's FEV scores to stable CSV-safe records."""
    records: list[dict[str, Any]] = []
    cutoff = str(task.cutoffs[0])
    for raw in scores.to_dict(orient="records"):
        row: dict[str, Any] = {
            "backbone": backbone,
            "dataset_config": dataset_config,
            "origin_index": origin_index,
            "cutoff": cutoff,
            "policy_id": policy_id,
            "item_id": str(raw[task.id_column]),
            "target": str(raw["target"]),
        }
        for metric in ("SQL", "WQL", "MASE", "WAPE"):
            row[metric] = float(raw[metric])
        records.append(row)
    return records


def json_safe_metric_summary(summary: dict[str, Any]) -> dict[str, float | None]:
    """Select P0b metrics while representing undefined secondary values as null."""
    selected: dict[str, float | None] = {}
    for metric in ("SQL", "WQL", "MASE", "WAPE"):
        value = float(summary[metric])
        selected[metric] = value if math.isfinite(value) else None
    if selected["SQL"] is None:
        raise ValueError("aggregate SQL must be finite for a completed P0b policy")
    return selected


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


def origin_paths(
    output_root: str | Path,
    backbone: str,
    dataset_config: str,
    origin_index: int,
) -> tuple[Path, Path]:
    root = Path(output_root) / "units" / backbone / dataset_config
    stem = f"origin_{origin_index:03d}"
    return root / f"{stem}.csv", root / f"{stem}.json"


def write_origin_artifact(
    output_root: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically save all additional policies for one task-origin."""
    csv_path, sidecar_path = origin_paths(
        output_root,
        metadata["backbone"],
        metadata["dataset_config"],
        int(metadata["origin_index"]),
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(csv_path)
    sidecar = {
        **metadata,
        "completed": True,
        "row_count": len(records),
        "finite_sql_count": sum(math.isfinite(float(row["SQL"])) for row in records),
        "csv_sha256": _sha256(csv_path),
        "csv_name": csv_path.name,
    }
    _atomic_json(sidecar_path, sidecar)
    return sidecar


def completed_origin(
    output_root: str | Path,
    *,
    backbone: str,
    dataset_config: str,
    origin_index: int,
    config_hash: str,
    scientific_code_sha256: str,
) -> dict[str, Any] | None:
    """Return an integrity-checked origin sidecar, or ``None`` for rerun."""
    csv_path, sidecar_path = origin_paths(
        output_root, backbone, dataset_config, origin_index
    )
    if not csv_path.exists() or not sidecar_path.exists():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    valid = (
        sidecar.get("completed") is True
        and sidecar.get("config_hash") == config_hash
        and sidecar.get("scientific_code_sha256") == scientific_code_sha256
        and sidecar.get("csv_sha256") == _sha256(csv_path)
    )
    return sidecar if valid else None


def read_candidate_records(
    output_root: str | Path,
    p0a: dict[str, Any],
    backbone: str,
    scientific_code_sha256: str,
) -> list[dict[str, Any]]:
    """Read all expected integrity-checked additional-policy records."""
    records: list[dict[str, Any]] = []
    for dataset_config, expected in p0a["tasks"].items():
        for origin_index in expected["calibration_origin_indices"]:
            sidecar = completed_origin(
                output_root,
                backbone=backbone,
                dataset_config=dataset_config,
                origin_index=origin_index,
                config_hash=EXPECTED_P0B_CONFIG_HASH,
                scientific_code_sha256=scientific_code_sha256,
            )
            if sidecar is None:
                raise RuntimeError(
                    f"missing or stale P0b unit: {backbone}/{dataset_config}/{origin_index}"
                )
            csv_path, _ = origin_paths(
                output_root, backbone, dataset_config, origin_index
            )
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                for raw in csv.DictReader(handle):
                    row: dict[str, Any] = dict(raw)
                    row["origin_index"] = int(row["origin_index"])
                    for metric in ("SQL", "WQL", "MASE", "WAPE"):
                        row[metric] = float(row[metric])
                    records.append(row)
    return records


def read_p0a_baselines(
    repo_root: str | Path,
    p0a_output_root: str | Path,
    p0a: dict[str, Any],
    backbone: str,
) -> list[dict[str, Any]]:
    """Reuse P0a constants only after their scientific code and CSV hashes verify."""
    code_hash = p0a_scientific_code_hash(repo_root, backbone)
    return read_backbone_records(
        p0a_output_root,
        p0a,
        backbone,
        code_hash,
    )


def summarize_oracle_records(
    baseline_records: Iterable[dict[str, Any]],
    candidate_records: Iterable[dict[str, Any]],
    *,
    minimum_task_macro_headroom: float = 0.05,
) -> dict[str, Any]:
    """Compute the frozen finite-oracle headroom report."""

    def unit_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
        return (
            str(row["dataset_config"]),
            int(row["origin_index"]),
            str(row["item_id"]),
            str(row["target"]),
        )
    baselines: dict[str, dict[tuple[str, int, str, str], float]] = {
        "target_only": {},
        "all_dynamic": {},
    }
    for row in baseline_records:
        variant = str(row["variant"])
        if variant not in baselines:
            continue
        key = unit_key(row)
        if key in baselines[variant]:
            raise ValueError(f"duplicate baseline unit for {variant}: {key}")
        baselines[variant][key] = float(row["SQL"])
    if set(baselines["target_only"]) != set(baselines["all_dynamic"]):
        raise ValueError("P0a baseline keys do not match")

    candidates: dict[tuple[str, int, str, str], dict[str, float]] = defaultdict(dict)
    candidate_policy_ids: dict[str, set[str]] = defaultdict(set)
    for row in candidate_records:
        key = unit_key(row)
        policy_id = str(row["policy_id"])
        if policy_id in candidates[key]:
            raise ValueError(f"duplicate candidate unit for {policy_id}: {key}")
        candidates[key][policy_id] = float(row["SQL"])
        candidate_policy_ids[key[0]].add(policy_id)

    baseline_keys = set(baselines["target_only"])
    if baseline_keys != set(candidates):
        missing = len(baseline_keys - set(candidates))
        extra = len(set(candidates) - baseline_keys)
        raise ValueError(f"candidate keys do not match baselines: missing={missing}, extra={extra}")

    per_task_values: dict[str, list[float]] = defaultdict(list)
    per_task_excluded: Counter[str] = Counter()
    per_task_positive: Counter[str] = Counter()
    per_task_winners: dict[str, Counter[str]] = defaultdict(Counter)
    all_headroom: list[float] = []
    for key in sorted(baselines["target_only"]):
        dataset_config = key[0]
        target_loss = baselines["target_only"][key]
        all_loss = baselines["all_dynamic"][key]
        if not math.isfinite(target_loss) or not math.isfinite(all_loss):
            per_task_excluded[dataset_config] += 1
            continue
        finite_losses = {
            "target_only": target_loss,
            "all_dynamic": all_loss,
            **{
                policy_id: loss
                for policy_id, loss in candidates.get(key, {}).items()
                if math.isfinite(loss)
            },
        }
        constant_reference = min(target_loss, all_loss)
        winner, oracle_loss = min(finite_losses.items(), key=lambda item: (item[1], item[0]))
        headroom = float(relative_gain([constant_reference], [oracle_loss])[0])
        if headroom < -1e-12:
            raise RuntimeError("oracle headroom cannot be negative when constants are included")
        headroom = max(0.0, headroom)
        per_task_values[dataset_config].append(headroom)
        per_task_positive[dataset_config] += int(headroom > 1e-12)
        per_task_winners[dataset_config][winner] += 1
        all_headroom.append(headroom)

    candidate_tasks = set(candidate_policy_ids)
    finite_tasks = set(per_task_values)
    if finite_tasks != candidate_tasks:
        missing_tasks = sorted(candidate_tasks - finite_tasks)
        raise ValueError(f"no finite units for tasks: {missing_tasks}")
    if len(per_task_values) != 4:
        raise ValueError("P0b report must contain exactly four tasks")
    per_task: dict[str, dict[str, Any]] = {}
    task_means: list[float] = []
    for dataset_config in sorted(per_task_values):
        values = per_task_values[dataset_config]
        mean_headroom = float(np.mean(values))
        task_means.append(mean_headroom)
        per_task[dataset_config] = {
            "finite_units": len(values),
            "excluded_nonfinite_constant_pairs": per_task_excluded[dataset_config],
            "mean_oracle_headroom": mean_headroom,
            "median_oracle_headroom": float(np.median(values)),
            "positive_headroom_fraction": per_task_positive[dataset_config] / len(values),
            "unique_additional_policy_ids": len(candidate_policy_ids[dataset_config]),
            "oracle_winner_counts": dict(sorted(per_task_winners[dataset_config].items())),
        }
    task_macro = float(np.mean(task_means))
    return {
        "aggregation_primary": "task_macro_mean_oracle_headroom",
        "per_task": per_task,
        "task_macro_mean_oracle_headroom": task_macro,
        "unit_micro_mean_oracle_headroom": float(np.mean(all_headroom)),
        "finite_unit_count": len(all_headroom),
        "excluded_nonfinite_constant_pair_count": sum(per_task_excluded.values()),
        "minimum_task_macro_mean_oracle_headroom": minimum_task_macro_headroom,
        "h2_gate_passed": task_macro >= minimum_task_macro_headroom,
    }


def write_backbone_report(
    repo_root: str | Path,
    p0a_output_root: str | Path,
    output_root: str | Path,
    *,
    config: dict[str, Any],
    p0a: dict[str, Any],
    backbone: str,
    source_git_commit: str,
    scientific_code_sha256: str,
    model_metadata: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Verify P0a, then compute H2 entirely from same-hardware P0b policies."""
    read_p0a_baselines(repo_root, p0a_output_root, p0a, backbone)
    all_records = read_candidate_records(
        output_root,
        p0a,
        backbone,
        scientific_code_sha256,
    )
    baselines: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for row in all_records:
        if row["policy_id"] in {"target_only", "all_dynamic"}:
            baselines.append({**row, "variant": row["policy_id"]})
        else:
            candidates.append(row)
    result = summarize_oracle_records(
        baselines,
        candidates,
        minimum_task_macro_headroom=float(
            config["gate"]["minimum_task_macro_mean_oracle_headroom_per_backbone"]
        ),
    )
    report = {
        "experiment": config["experiment"],
        "schema_version": config["schema_version"],
        "result_status": "screening_only",
        "config_hash": EXPECTED_P0B_CONFIG_HASH,
        "parent_p0_config_hash": config["parent_p0_config_hash"],
        "prerequisite_p0a_config_hash": config["prerequisite_p0a_config_hash"],
        "git_commit": source_git_commit,
        "scientific_code_sha256": scientific_code_sha256,
        "backbone": backbone,
        "model": model_metadata,
        "runtime": runtime,
        "scope": {
            "calibration_origins_only": True,
            "sealed_evaluation_origins_instantiated": False,
            "utility_predictability_computed": False,
            "full_p0_gate_computed": False,
        },
        "h2_screen": result,
    }
    report_path = Path(output_root) / "reports" / f"{backbone}_p0b_oracle_screen.json"
    _atomic_json(report_path, report)
    return report


def current_git_commit(repo_root: str | Path) -> str:
    return git_commit(Path(repo_root).resolve())


def scientific_code_hash(repo_root: str | Path, backbone: str) -> str:
    """Hash all files that can change P0b inference, ranking, or aggregation."""
    root = Path(repo_root).resolve()
    common = [
        P0B_CONFIG_PATH,
        P0_CONFIG_PATH,
        P0A_CONFIG_PATH,
        P0A_DECISION_PATH,
        Path("src/covsafe/config.py"),
        Path("src/covsafe/diagnostics.py"),
        Path("src/covsafe/fev_smoke.py"),
        Path("src/covsafe/p0a.py"),
        Path("src/covsafe/p0b.py"),
        Path("src/covsafe/protocol.py"),
    ]
    model_specific = {
        "chronos_2": [Path("src/covsafe/chronos2_p0b.py")],
        "timesfm_3": [
            Path("src/covsafe/timesfm3_p0b.py"),
            Path("src/covsafe/timesfm3_smoke.py"),
        ],
    }
    if backbone not in model_specific:
        raise ValueError(f"unknown P0b backbone: {backbone}")
    digest = hashlib.sha256()
    for relative in sorted(common + model_specific[backbone], key=str):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
