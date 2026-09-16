"""Chronos-2 calibration-origin smoke runner.

Heavy runtime dependencies are imported lazily so the core diagnostic package and
its unit tests remain CPU-only.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from covsafe.config import canonical_config_hash, load_yaml
from covsafe.diagnostics import relative_gain
from covsafe.protocol import temporal_origin_partition

EXPECTED_SMOKE_CONFIG_HASH = "dc37b99f601c"
REQUIRED_METRICS = ("SQL", "WQL", "MASE", "WAPE")


def make_task_fields(
    config: dict[str, Any],
    variant: str,
    cutoff: int | str,
    *,
    num_windows: int = 1,
) -> dict[str, Any]:
    """Build FEV task fields for a declared smoke variant."""
    if variant not in config["variants"]:
        raise ValueError(f"undeclared variant: {variant}")
    dataset = config["dataset"]
    include_covariates = variant == "all_dynamic"
    fields: dict[str, Any] = {
        "dataset_path": dataset["dataset_path"],
        "dataset_config": dataset["dataset_config"],
        "horizon": dataset["horizon"],
        "num_windows": num_windows,
        "initial_cutoff": cutoff,
        "seasonality": dataset["seasonality"],
        "known_dynamic_columns": (
            dataset["known_dynamic_columns"] if include_covariates else []
        ),
        "past_dynamic_columns": (
            dataset["past_dynamic_columns"] if include_covariates else []
        ),
        "static_columns": [],
        "eval_metric": config["metrics"]["primary"],
        "extra_metrics": [
            {"name": "WQL", "epsilon": 1.0},
            "MASE",
            {"name": "WAPE", "epsilon": 1.0},
        ],
        "quantile_levels": config["metrics"]["quantile_levels"],
    }
    if dataset.get("target") is not None:
        fields["target"] = dataset["target"]
    return fields


def select_finite_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """Extract required finite metric values from a FEV evaluation summary."""
    selected: dict[str, float] = {}
    for metric in REQUIRED_METRICS:
        if metric not in summary:
            raise KeyError(f"FEV summary is missing {metric}")
        value = float(summary[metric])
        if not math.isfinite(value):
            raise ValueError(f"{metric} is not finite")
        selected[metric] = value
    return selected


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def run_chronos2_smoke(
    repo_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run target-only and all-dynamic Chronos-2 inference on one calibration origin."""
    import fev
    import torch
    from chronos import BaseChronosPipeline

    root = Path(repo_root).resolve()
    config_path = root / "configs/smoke/chronos2_epf_np.yaml"
    config = load_yaml(config_path)
    config_hash = canonical_config_hash(config)
    if config_hash != EXPECTED_SMOKE_CONFIG_HASH:
        raise RuntimeError(
            f"smoke config hash mismatch: {config_hash} != {EXPECTED_SMOKE_CONFIG_HASH}"
        )
    if not config["guardrails"]["calibration_origins_only"]:
        raise RuntimeError("calibration-only guardrail must remain enabled")
    if config["guardrails"]["sealed_evaluation_origins_instantiated"]:
        raise RuntimeError("sealed evaluation origins must not be instantiated")
    if config["guardrails"]["scientific_gate_computed"]:
        raise RuntimeError("scientific gates are forbidden in a smoke run")
    if config["guardrails"]["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("Select Runtime > Change runtime type > T4 GPU, then restart")

    p0_config = load_yaml(root / "configs/diagnostic/covariate_utility_p0.yaml")
    if canonical_config_hash(p0_config) != config["parent_p0_config_hash"]:
        raise RuntimeError("parent P0 configuration hash mismatch")
    p0_task = next(
        task
        for task in p0_config["data"]["tasks"]
        if task["dataset_config"] == config["dataset"]["dataset_config"]
    )
    partition = temporal_origin_partition(
        p0_task["num_windows"],
        calibration_fraction=p0_config["data"]["calibration_fraction"],
        minimum_per_partition=p0_config["data"]["minimum_origins_per_partition"],
    )
    origin_index = config["dataset"]["calibration_origin_index"]
    if origin_index not in partition.calibration:
        raise RuntimeError("the smoke origin is not in the calibration partition")

    base_fields = make_task_fields(
        config,
        "all_dynamic",
        cutoff=config["dataset"]["expected_cutoff"],
        num_windows=p0_task["num_windows"],
    )
    base_fields.pop("initial_cutoff")
    base_task = fev.Task(**base_fields)
    cutoff = base_task.cutoffs[origin_index]
    if cutoff != config["dataset"]["expected_cutoff"]:
        raise RuntimeError(f"unexpected cutoff: {cutoff}")

    tasks = {
        variant: fev.Task(**make_task_fields(config, variant, cutoff))
        for variant in config["variants"]
    }
    all_dynamic_task = tasks["all_dynamic"]
    all_dynamic_task.load_full_dataset(num_proc=1)
    dataset_fingerprint = all_dynamic_task._dataset_fingerprint
    if dataset_fingerprint != config["dataset"]["expected_fingerprint"]:
        raise RuntimeError(
            "dataset fingerprint mismatch: "
            f"{dataset_fingerprint} != {config['dataset']['expected_fingerprint']}"
        )

    dtype = getattr(torch, config["model"]["torch_dtype"])
    load_start = time.monotonic()
    pipeline = BaseChronosPipeline.from_pretrained(
        config["model"]["checkpoint"],
        device_map=config["model"]["device"],
        torch_dtype=dtype,
    )
    model_load_seconds = time.monotonic() - load_start

    variants: dict[str, dict[str, Any]] = {}
    for variant, task in tasks.items():
        predictions, inference_time = pipeline.predict_fev(
            task,
            batch_size=config["model"]["batch_size"],
            cross_learning=config["model"]["cross_learning"],
            as_univariate=config["model"]["as_univariate"],
        )
        summary = task.evaluation_summary(
            predictions,
            model_name=f"chronos-2-{variant}",
            training_time_s=0.0,
            inference_time_s=inference_time,
            trained_on_this_dataset=False,
        )
        variants[variant] = {
            "metrics": select_finite_metrics(summary),
            "inference_time_seconds": float(inference_time),
            "num_forecasts": int(summary["num_forecasts"]),
            "prediction_window_count": len(predictions),
        }

    target_sql = variants["target_only"]["metrics"]["SQL"]
    all_sql = variants["all_dynamic"]["metrics"]["SQL"]
    diagnostic_gain = float(relative_gain([target_sql], [all_sql])[0])

    model = getattr(pipeline, "model", None)
    model_config = getattr(model, "config", None)
    manifest: dict[str, Any] = {
        "experiment": config["experiment"],
        "schema_version": config["schema_version"],
        "result_status": "smoke_only",
        "config_hash": config_hash,
        "parent_p0_config_hash": config["parent_p0_config_hash"],
        "git_commit": _git_commit(root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "dataset_config": config["dataset"]["dataset_config"],
            "dataset_fingerprint": dataset_fingerprint,
            "calibration_origin_index": origin_index,
            "cutoff": cutoff,
            "horizon": config["dataset"]["horizon"],
        },
        "model": {
            "checkpoint": config["model"]["checkpoint"],
            "checkpoint_revision": getattr(model_config, "_commit_hash", None),
            "model_load_seconds": model_load_seconds,
            "torch_dtype": config["model"]["torch_dtype"],
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "fev": _package_version("fev"),
            "chronos_forecasting": _package_version("chronos-forecasting"),
        },
        "variants": variants,
        "diagnostic_relative_sql_gain_all_minus_target_only": diagnostic_gain,
        "scientific_gate_computed": False,
        "sealed_evaluation_origins_instantiated": False,
    }

    destination = (
        Path(output_path)
        if output_path is not None
        else root / "outputs/smoke/chronos2_epf_np.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return manifest
