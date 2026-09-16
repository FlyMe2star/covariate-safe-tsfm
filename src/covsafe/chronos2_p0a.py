"""Chronos-2 runner for the calibration-only P0a harm screen."""

from __future__ import annotations

import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from covsafe.fev_smoke import package_version, select_finite_metrics
from covsafe.p0a import (
    EXPECTED_P0A_CONFIG_HASH,
    completed_unit,
    current_git_commit,
    load_frozen_p0a,
    make_calibration_task,
    normalize_score_records,
    scientific_code_hash,
    write_backbone_report,
    write_unit_artifact,
)


def run_chronos2_p0a(repo_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Run or resume every Chronos-2 P0a task-variant unit, then compute H1."""
    import torch
    from chronos import BaseChronosPipeline

    root = Path(repo_root).resolve()
    destination = Path(output_root)
    config, parent = load_frozen_p0a(root)
    if config["execution"]["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime, then restart and run all cells")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))

    backbone = "chronos_2"
    model_config = config["backbones"][backbone]
    source_commit = current_git_commit(root)
    code_hash = scientific_code_hash(root, backbone)
    dtype = getattr(torch, model_config["torch_dtype"])
    load_start = time.monotonic()
    pipeline = BaseChronosPipeline.from_pretrained(
        model_config["checkpoint"],
        revision=model_config["checkpoint_revision"],
        device_map=model_config["device"],
        dtype=dtype,
    )
    model_load_seconds = time.monotonic() - load_start

    runtime = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "fev": package_version("fev"),
        "chronos_forecasting": package_version("chronos-forecasting"),
    }
    model_metadata = {
        "checkpoint": model_config["checkpoint"],
        "checkpoint_revision": model_config["checkpoint_revision"],
        "torch_dtype": model_config["torch_dtype"],
        "model_load_seconds": model_load_seconds,
    }

    for dataset_config, task_config in config["tasks"].items():
        for variant in config["scope"]["variants"]:
            existing = completed_unit(
                destination,
                backbone=backbone,
                dataset_config=dataset_config,
                variant=variant,
                config_hash=EXPECTED_P0A_CONFIG_HASH,
                scientific_code_sha256=code_hash,
            )
            if existing is not None:
                print(f"RESUME {backbone}/{dataset_config}/{variant}")
                continue

            print(f"RUN {backbone}/{dataset_config}/{variant}")
            task = make_calibration_task(config, parent, dataset_config, variant)
            full_dataset = task.load_full_dataset(num_proc=1)
            if task._dataset_fingerprint != task_config["expected_fingerprint"]:
                raise RuntimeError(f"dataset fingerprint mismatch for {dataset_config}")
            if len(full_dataset) != task_config["expected_item_count"]:
                raise RuntimeError(f"dataset item-count mismatch for {dataset_config}")

            predictions, inference_time = pipeline.predict_fev(
                task,
                batch_size=model_config["batch_size"],
                cross_learning=model_config["cross_learning"],
                as_univariate=model_config["as_univariate"],
            )
            summary = task.evaluation_summary(
                predictions,
                model_name=f"chronos-2-{variant}",
                training_time_s=0.0,
                inference_time_s=inference_time,
                trained_on_this_dataset=False,
            )
            records = normalize_score_records(
                task.scores_per_item(predictions),
                task=task,
                backbone=backbone,
                dataset_config=dataset_config,
                variant=variant,
                origin_indices=task_config["calibration_origin_indices"],
            )
            write_unit_artifact(
                destination,
                records,
                {
                    "schema_version": 1,
                    "result_status": "screening_only",
                    "config_hash": EXPECTED_P0A_CONFIG_HASH,
                    "parent_p0_config_hash": config["parent_p0_config_hash"],
                    "git_commit": source_commit,
                    "scientific_code_sha256": code_hash,
                    "backbone": backbone,
                    "dataset_config": dataset_config,
                    "dataset_fingerprint": task._dataset_fingerprint,
                    "variant": variant,
                    "origin_indices": task_config["calibration_origin_indices"],
                    "checkpoint_revision": model_config["checkpoint_revision"],
                    "inference_time_seconds": float(inference_time),
                    "aggregate_metrics": select_finite_metrics(summary),
                    "runtime": runtime,
                },
            )
            print(f"SAVED {backbone}/{dataset_config}/{variant}: {len(records)} rows")

    return write_backbone_report(
        destination,
        config=config,
        backbone=backbone,
        source_git_commit=source_commit,
        scientific_code_sha256=code_hash,
        model_metadata=model_metadata,
        runtime=runtime,
    )
