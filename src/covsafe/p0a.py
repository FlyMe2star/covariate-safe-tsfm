"""Shared execution and analysis helpers for the calibration-only P0a screen."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from covsafe.config import canonical_config_hash, load_yaml
from covsafe.diagnostics import relative_gain
from covsafe.fev_smoke import git_commit
from covsafe.protocol import temporal_origin_partition

EXPECTED_P0A_CONFIG_HASH = "902c15e63f71"
P0A_CONFIG_PATH = Path("configs/diagnostic/covariate_harm_p0a.yaml")
PARENT_P0_CONFIG_PATH = Path("configs/diagnostic/covariate_utility_p0.yaml")
CSV_FIELDS = (
    "backbone",
    "dataset_config",
    "variant",
    "origin_index",
    "cutoff",
    "item_id",
    "target",
    "SQL",
    "WQL",
    "MASE",
    "WAPE",
)


def load_frozen_p0a(repo_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load P0a and its parent P0 configuration, enforcing both frozen hashes."""
    root = Path(repo_root).resolve()
    config = load_yaml(root / P0A_CONFIG_PATH)
    observed_hash = canonical_config_hash(config)
    if observed_hash != EXPECTED_P0A_CONFIG_HASH:
        raise RuntimeError(
            f"P0a config hash mismatch: {observed_hash} != {EXPECTED_P0A_CONFIG_HASH}"
        )
    if not config["scope"]["calibration_origins_only"]:
        raise RuntimeError("P0a must remain calibration-only")
    if config["scope"]["sealed_evaluation_origins_instantiated"]:
        raise RuntimeError("sealed evaluation origins must not be instantiated")
    if config["scope"]["candidate_subset_search_enabled"]:
        raise RuntimeError("candidate subset search is forbidden in P0a")

    parent = load_yaml(root / PARENT_P0_CONFIG_PATH)
    parent_hash = canonical_config_hash(parent)
    if parent_hash != config["parent_p0_config_hash"]:
        raise RuntimeError(
            f"parent P0 hash mismatch: {parent_hash} != {config['parent_p0_config_hash']}"
        )
    return config, parent


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
    variant: str,
    *,
    num_windows: int,
    initial_cutoff: int | str | None = None,
    window_step_size: int | str | None = None,
) -> dict[str, Any]:
    if variant not in {"target_only", "all_dynamic"}:
        raise ValueError(f"undeclared P0a variant: {variant}")
    include_covariates = variant == "all_dynamic"
    fields: dict[str, Any] = {
        "dataset_path": parent["data"]["dataset_path"],
        "dataset_config": task_spec["dataset_config"],
        "horizon": task_spec["horizon"],
        "num_windows": num_windows,
        "seasonality": task_spec["seasonality"],
        "known_dynamic_columns": (
            task_spec["known_dynamic_columns"] if include_covariates else []
        ),
        "past_dynamic_columns": (
            task_spec["past_dynamic_columns"] if include_covariates else []
        ),
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


def make_calibration_task(
    config: dict[str, Any],
    parent: dict[str, Any],
    dataset_config: str,
    variant: str,
) -> Any:
    """Build a FEV task containing only the preregistered calibration windows."""
    import fev

    task_spec = _parent_task(parent, dataset_config)
    declared = config["tasks"][dataset_config]["calibration_origin_indices"]
    partition = temporal_origin_partition(
        task_spec["num_windows"],
        calibration_fraction=parent["data"]["calibration_fraction"],
        minimum_per_partition=parent["data"]["minimum_origins_per_partition"],
    )
    if tuple(declared) != partition.calibration:
        raise RuntimeError(f"calibration-origin mismatch for {dataset_config}")

    full_task = fev.Task(
        **_task_fields(
            parent,
            task_spec,
            "all_dynamic",
            num_windows=task_spec["num_windows"],
        )
    )
    calibration_fields = _task_fields(
        parent,
        task_spec,
        variant,
        num_windows=len(declared),
        initial_cutoff=full_task.initial_cutoff,
        window_step_size=full_task.window_step_size,
    )
    return fev.Task(**calibration_fields)


def normalize_score_records(
    scores: Any,
    *,
    task: Any,
    backbone: str,
    dataset_config: str,
    variant: str,
    origin_indices: list[int],
) -> list[dict[str, Any]]:
    """Convert FEV's per-item DataFrame to stable, JSON/CSV-safe records."""
    if len(task.cutoffs) != len(origin_indices):
        raise RuntimeError("cutoff and origin counts differ")
    cutoff_by_window = dict(enumerate(task.cutoffs))
    origin_by_window = dict(enumerate(origin_indices))
    records: list[dict[str, Any]] = []
    for raw in scores.to_dict(orient="records"):
        window = int(raw["window"])
        row: dict[str, Any] = {
            "backbone": backbone,
            "dataset_config": dataset_config,
            "variant": variant,
            "origin_index": origin_by_window[window],
            "cutoff": str(cutoff_by_window[window]),
            "item_id": str(raw[task.id_column]),
            "target": str(raw["target"]),
        }
        for metric in ("SQL", "WQL", "MASE", "WAPE"):
            row[metric] = float(raw[metric])
        records.append(row)
    return records


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


def unit_paths(
    output_root: str | Path,
    backbone: str,
    dataset_config: str,
    variant: str,
) -> tuple[Path, Path]:
    unit_root = Path(output_root) / "units" / backbone
    stem = f"{dataset_config}__{variant}"
    return unit_root / f"{stem}.csv", unit_root / f"{stem}.json"


def write_unit_artifact(
    output_root: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically persist one task-variant resume unit and its integrity sidecar."""
    csv_path, sidecar_path = unit_paths(
        output_root,
        metadata["backbone"],
        metadata["dataset_config"],
        metadata["variant"],
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


def completed_unit(
    output_root: str | Path,
    *,
    backbone: str,
    dataset_config: str,
    variant: str,
    config_hash: str,
    scientific_code_sha256: str,
) -> dict[str, Any] | None:
    """Return a validated completed sidecar, or ``None`` when rerun is required."""
    csv_path, sidecar_path = unit_paths(output_root, backbone, dataset_config, variant)
    if not csv_path.exists() or not sidecar_path.exists():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    expected = (
        sidecar.get("completed") is True
        and sidecar.get("config_hash") == config_hash
        and sidecar.get("scientific_code_sha256") == scientific_code_sha256
        and sidecar.get("csv_sha256") == _sha256(csv_path)
    )
    return sidecar if expected else None


def read_backbone_records(
    output_root: str | Path,
    config: dict[str, Any],
    backbone: str,
    scientific_code_sha256: str,
) -> list[dict[str, Any]]:
    """Read every validated P0a unit for one backbone."""
    records: list[dict[str, Any]] = []
    for dataset_config in config["tasks"]:
        for variant in config["scope"]["variants"]:
            sidecar = completed_unit(
                output_root,
                backbone=backbone,
                dataset_config=dataset_config,
                variant=variant,
                config_hash=EXPECTED_P0A_CONFIG_HASH,
                scientific_code_sha256=scientific_code_sha256,
            )
            if sidecar is None:
                raise RuntimeError(f"missing or stale unit: {backbone}/{dataset_config}/{variant}")
            csv_path, _ = unit_paths(output_root, backbone, dataset_config, variant)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                for raw in csv.DictReader(handle):
                    row: dict[str, Any] = dict(raw)
                    row["origin_index"] = int(row["origin_index"])
                    for metric in ("SQL", "WQL", "MASE", "WAPE"):
                        row[metric] = float(row[metric])
                    records.append(row)
    return records


def summarize_harm_records(
    records: Iterable[dict[str, Any]],
    *,
    harmful_relative_loss: float = 0.05,
    minimum_task_macro_harm_rate: float = 0.20,
) -> dict[str, Any]:
    """Pair the two variants and compute the frozen P0a macro/micro screen."""
    by_variant: dict[str, dict[tuple[str, int, str, str], float]] = {
        "target_only": {},
        "all_dynamic": {},
    }
    for row in records:
        variant = str(row["variant"])
        if variant not in by_variant:
            raise ValueError(f"unexpected variant: {variant}")
        key = (
            str(row["dataset_config"]),
            int(row["origin_index"]),
            str(row["item_id"]),
            str(row["target"]),
        )
        if key in by_variant[variant]:
            raise ValueError(f"duplicate unit for {variant}: {key}")
        by_variant[variant][key] = float(row["SQL"])

    target_keys = set(by_variant["target_only"])
    all_keys = set(by_variant["all_dynamic"])
    if target_keys != all_keys:
        missing_target = len(all_keys - target_keys)
        missing_all = len(target_keys - all_keys)
        raise ValueError(
            f"variant keys do not match: missing_target={missing_target}, missing_all={missing_all}"
        )
    if not target_keys:
        raise ValueError("no paired P0a records")

    per_task: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "paired_units": 0,
            "finite_pairs": 0,
            "excluded_nonfinite_pairs": 0,
            "harmful_units": 0,
        }
    )
    all_gains: list[float] = []
    for key in sorted(target_keys):
        dataset_config = key[0]
        task_counts = per_task[dataset_config]
        task_counts["paired_units"] += 1
        target_sql = by_variant["target_only"][key]
        all_sql = by_variant["all_dynamic"][key]
        if not math.isfinite(target_sql) or not math.isfinite(all_sql):
            task_counts["excluded_nonfinite_pairs"] += 1
            continue
        gain = float(relative_gain([target_sql], [all_sql])[0])
        task_counts["finite_pairs"] += 1
        task_counts["harmful_units"] += int(gain <= -harmful_relative_loss)
        all_gains.append(gain)

    task_rates: list[float] = []
    for dataset_config, counts in per_task.items():
        if counts["finite_pairs"] == 0:
            raise ValueError(f"no finite SQL pairs for {dataset_config}")
        rate = counts["harmful_units"] / counts["finite_pairs"]
        counts["harm_rate"] = rate
        task_rates.append(rate)

    if len(task_rates) != 4:
        raise ValueError(f"expected four tasks, found {len(task_rates)}")
    task_macro = sum(task_rates) / len(task_rates)
    total_finite = sum(v["finite_pairs"] for v in per_task.values())
    total_harmful = sum(v["harmful_units"] for v in per_task.values())
    unit_micro = total_harmful / total_finite
    return {
        "harmful_relative_loss": harmful_relative_loss,
        "aggregation_primary": "task_macro_harm_rate",
        "per_task": dict(sorted(per_task.items())),
        "task_macro_harm_rate": task_macro,
        "unit_micro_harm_rate": unit_micro,
        "mean_relative_sql_gain_all_minus_target_only": sum(all_gains) / len(all_gains),
        "finite_pair_count": total_finite,
        "excluded_nonfinite_pair_count": sum(
            v["excluded_nonfinite_pairs"] for v in per_task.values()
        ),
        "minimum_task_macro_harm_rate": minimum_task_macro_harm_rate,
        "h1_gate_passed": task_macro >= minimum_task_macro_harm_rate,
    }


def write_backbone_report(
    output_root: str | Path,
    *,
    config: dict[str, Any],
    backbone: str,
    source_git_commit: str,
    scientific_code_sha256: str,
    model_metadata: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Validate all units, compute H1, and durably write the backbone report."""
    records = read_backbone_records(
        output_root,
        config,
        backbone,
        scientific_code_sha256,
    )
    result = summarize_harm_records(
        records,
        harmful_relative_loss=float(config["harm_definition"]["harmful_relative_loss"]),
        minimum_task_macro_harm_rate=float(
            config["gate"]["minimum_task_macro_harm_rate_per_backbone"]
        ),
    )
    report = {
        "experiment": config["experiment"],
        "schema_version": config["schema_version"],
        "result_status": "screening_only",
        "config_hash": EXPECTED_P0A_CONFIG_HASH,
        "parent_p0_config_hash": config["parent_p0_config_hash"],
        "git_commit": source_git_commit,
        "scientific_code_sha256": scientific_code_sha256,
        "backbone": backbone,
        "model": model_metadata,
        "runtime": runtime,
        "scope": {
            "calibration_origins_only": True,
            "sealed_evaluation_origins_instantiated": False,
            "candidate_subset_search_enabled": False,
            "full_p0_gate_computed": False,
        },
        "h1_screen": result,
    }
    report_path = Path(output_root) / "reports" / f"{backbone}_p0a_harm_screen.json"
    _atomic_json(report_path, report)
    return report


def current_git_commit(repo_root: str | Path) -> str:
    """Small public wrapper used by notebook runners."""
    return git_commit(Path(repo_root).resolve())


def scientific_code_hash(repo_root: str | Path, backbone: str) -> str:
    """Hash only files that can change P0a predictions, units, or H1 aggregation."""
    root = Path(repo_root).resolve()
    common = [
        P0A_CONFIG_PATH,
        PARENT_P0_CONFIG_PATH,
        Path("src/covsafe/config.py"),
        Path("src/covsafe/diagnostics.py"),
        Path("src/covsafe/fev_smoke.py"),
        Path("src/covsafe/p0a.py"),
        Path("src/covsafe/protocol.py"),
    ]
    model_specific = {
        "chronos_2": [Path("src/covsafe/chronos2_p0a.py")],
        "timesfm_3": [
            Path("src/covsafe/timesfm3_p0a.py"),
            Path("src/covsafe/timesfm3_smoke.py"),
        ],
    }
    if backbone not in model_specific:
        raise ValueError(f"unknown P0a backbone: {backbone}")
    digest = hashlib.sha256()
    for relative in sorted(common + model_specific[backbone], key=str):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
