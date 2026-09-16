"""TimesFM 3 calibration-origin smoke runner using the pinned official FEV wrapper."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from covsafe.config import canonical_config_hash, load_yaml
from covsafe.diagnostics import relative_gain
from covsafe.fev_smoke import (
    git_commit,
    make_task_fields,
    package_version,
    select_finite_metrics,
)
from covsafe.protocol import temporal_origin_partition

EXPECTED_SMOKE_CONFIG_HASH = "56041b1ae960"
_WRAPPER_MODULE_NAME = "covsafe_pinned_fev_timesfm3_wrapper"


def _load_wrapper_module(wrapper_path: Path) -> ModuleType:
    """Load the official pinned FEV TimesFM-3 wrapper once per Python process."""
    if _WRAPPER_MODULE_NAME in sys.modules:
        return sys.modules[_WRAPPER_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_WRAPPER_MODULE_NAME, wrapper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load FEV wrapper from {wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_WRAPPER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def run_timesfm3_smoke(
    repo_root: str | Path,
    fev_checkout: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run target-only and all-dynamic TimesFM 3 inference on one calibration origin."""
    import fev
    import torch
    from huggingface_hub import HfApi, snapshot_download

    root = Path(repo_root).resolve()
    upstream = Path(fev_checkout).resolve()
    config = load_yaml(root / "configs/smoke/timesfm3_epf_np.yaml")
    config_hash = canonical_config_hash(config)
    if config_hash != EXPECTED_SMOKE_CONFIG_HASH:
        raise RuntimeError(
            f"smoke config hash mismatch: {config_hash} != {EXPECTED_SMOKE_CONFIG_HASH}"
        )
    if git_commit(upstream) != config["model"]["fev_wrapper_commit"]:
        raise RuntimeError("FEV wrapper checkout does not match the pinned commit")
    if not config["guardrails"]["academic_noncommercial_use_only"]:
        raise RuntimeError("TimesFM 3 weights are restricted to non-commercial use")
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

    checkpoint_id = config["model"]["checkpoint"]
    checkpoint_revision = HfApi().model_info(checkpoint_id).sha
    checkpoint_path = snapshot_download(
        repo_id=checkpoint_id,
        revision=checkpoint_revision,
    )

    wrapper_path = upstream / config["model"]["fev_wrapper_path"]
    wrapper_module = _load_wrapper_module(wrapper_path)
    model = wrapper_module.TimesFM3Model(
        checkpoint_path=checkpoint_path,
        min_batch=config["model"]["min_batch"],
        max_batch=config["model"]["max_batch"],
        per_core_batch_size=config["model"]["per_core_batch_size"],
        max_context_length=config["model"]["max_context_length"],
        device=config["model"]["device"],
    )
    load_start = time.monotonic()
    model._get_forecaster()
    model_load_seconds = time.monotonic() - load_start

    variants: dict[str, dict[str, Any]] = {}
    for variant, task in tasks.items():
        predictions = model.fit_predict(task)
        summary = task.evaluation_summary(
            predictions,
            model_name=f"timesfm-3-{variant}",
            training_time_s=0.0,
            inference_time_s=model.inference_time,
            trained_on_this_dataset=False,
        )
        variants[variant] = {
            "metrics": select_finite_metrics(summary),
            "inference_time_seconds": float(model.inference_time),
            "num_forecasts": int(summary["num_forecasts"]),
            "prediction_window_count": len(predictions),
        }

    target_sql = variants["target_only"]["metrics"]["SQL"]
    all_sql = variants["all_dynamic"]["metrics"]["SQL"]
    diagnostic_gain = float(relative_gain([target_sql], [all_sql])[0])

    manifest: dict[str, Any] = {
        "experiment": config["experiment"],
        "schema_version": config["schema_version"],
        "result_status": "smoke_only",
        "config_hash": config_hash,
        "parent_p0_config_hash": config["parent_p0_config_hash"],
        "git_commit": git_commit(root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "dataset_config": config["dataset"]["dataset_config"],
            "dataset_fingerprint": dataset_fingerprint,
            "calibration_origin_index": origin_index,
            "cutoff": cutoff,
            "horizon": config["dataset"]["horizon"],
        },
        "model": {
            "checkpoint": checkpoint_id,
            "checkpoint_revision": checkpoint_revision,
            "model_load_seconds": model_load_seconds,
            "source_commit": config["model"]["source_commit"],
            "fev_wrapper_commit": config["model"]["fev_wrapper_commit"],
            "license": config["model"]["license"],
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "fev": package_version("fev"),
            "timesfm": package_version("timesfm"),
        },
        "variants": variants,
        "diagnostic_relative_sql_gain_all_minus_target_only": diagnostic_gain,
        "scientific_gate_computed": False,
        "sealed_evaluation_origins_instantiated": False,
    }

    destination = (
        Path(output_path)
        if output_path is not None
        else root / "outputs/smoke/timesfm3_epf_np.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return manifest
