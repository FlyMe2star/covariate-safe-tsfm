"""TimesFM 3 runner for the calibration-only P0b finite-oracle screen."""

from __future__ import annotations

import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from covsafe.fev_smoke import git_commit, package_version
from covsafe.p0b import (
    EXPECTED_P0B_CONFIG_HASH,
    completed_origin,
    current_git_commit,
    historical_policy_set,
    json_safe_metric_summary,
    load_frozen_p0b,
    make_origin_task,
    normalize_policy_score_records,
    prepare_task_context,
    read_p0a_baselines,
    scientific_code_hash,
    write_backbone_report,
    write_origin_artifact,
)
from covsafe.timesfm3_smoke import _load_wrapper_module


def run_timesfm3_p0b(
    repo_root: str | Path,
    fev_checkout: str | Path,
    p0a_output_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run or resume all TimesFM-3 P0b origins, then compute oracle headroom."""
    import torch
    from huggingface_hub import snapshot_download

    root = Path(repo_root).resolve()
    upstream = Path(fev_checkout).resolve()
    destination = Path(output_root)
    config, parent, p0a = load_frozen_p0b(root)
    if config["execution"]["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime, then restart and run all cells")

    backbone = "timesfm_3"
    model_config = config["backbones"][backbone]
    if git_commit(upstream) != model_config["fev_wrapper_commit"]:
        raise RuntimeError("FEV wrapper checkout does not match the pinned commit")
    source_commit = current_git_commit(root)
    code_hash = scientific_code_hash(root, backbone)
    read_p0a_baselines(root, p0a_output_root, p0a, backbone)
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))

    checkpoint_path = snapshot_download(
        repo_id=model_config["checkpoint"],
        revision=model_config["checkpoint_revision"],
    )
    wrapper = _load_wrapper_module(upstream / model_config["fev_wrapper_path"])
    model = wrapper.TimesFM3Model(
        checkpoint_path=checkpoint_path,
        min_batch=model_config["min_batch"],
        max_batch=model_config["max_batch"],
        per_core_batch_size=model_config["per_core_batch_size"],
        max_context_length=model_config["max_context_length"],
        device=model_config["device"],
    )
    load_start = time.monotonic()
    model._get_forecaster()
    model_load_seconds = time.monotonic() - load_start

    runtime = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "fev": package_version("fev"),
        "timesfm": package_version("timesfm"),
    }
    model_metadata = {
        "checkpoint": model_config["checkpoint"],
        "checkpoint_revision": model_config["checkpoint_revision"],
        "source_commit": model_config["source_commit"],
        "fev_wrapper_commit": model_config["fev_wrapper_commit"],
        "license": model_config["license"],
        "model_load_seconds": model_load_seconds,
    }

    for dataset_config, expected in p0a["tasks"].items():
        context = prepare_task_context(parent, p0a, dataset_config)
        for origin_index in expected["calibration_origin_indices"]:
            existing = completed_origin(
                destination,
                backbone=backbone,
                dataset_config=dataset_config,
                origin_index=origin_index,
                config_hash=EXPECTED_P0B_CONFIG_HASH,
                scientific_code_sha256=code_hash,
            )
            if existing is not None:
                print(f"RESUME {backbone}/{dataset_config}/origin_{origin_index:03d}")
                continue

            policies, ranking = historical_policy_set(config, context, origin_index)
            evaluated = policies
            print(
                f"RUN {backbone}/{dataset_config}/origin_{origin_index:03d} "
                f"with {len(evaluated)} same-hardware policies"
            )
            records: list[dict[str, Any]] = []
            policy_results: dict[str, Any] = {}
            for policy in evaluated:
                task = make_origin_task(parent, context, origin_index, policy)
                predictions = model.fit_predict(task)
                inference_time = float(model.inference_time)
                summary = task.evaluation_summary(
                    predictions,
                    model_name=f"timesfm-3-{policy.policy_id}",
                    training_time_s=0.0,
                    inference_time_s=inference_time,
                    trained_on_this_dataset=False,
                )
                records.extend(
                    normalize_policy_score_records(
                        task.scores_per_item(predictions),
                        task=task,
                        backbone=backbone,
                        dataset_config=dataset_config,
                        origin_index=origin_index,
                        policy_id=policy.policy_id,
                    )
                )
                policy_results[policy.policy_id] = {
                    "inference_time_seconds": inference_time,
                    "aggregate_metrics": json_safe_metric_summary(summary),
                }
            write_origin_artifact(
                destination,
                records,
                {
                    "schema_version": 1,
                    "result_status": "screening_only",
                    "config_hash": EXPECTED_P0B_CONFIG_HASH,
                    "parent_p0_config_hash": config["parent_p0_config_hash"],
                    "prerequisite_p0a_config_hash": config[
                        "prerequisite_p0a_config_hash"
                    ],
                    "git_commit": source_commit,
                    "scientific_code_sha256": code_hash,
                    "backbone": backbone,
                    "dataset_config": dataset_config,
                    "dataset_fingerprint": context.reference_task._dataset_fingerprint,
                    "origin_index": origin_index,
                    "cutoff": str(context.reference_task.cutoffs[origin_index]),
                    "checkpoint_revision": model_config["checkpoint_revision"],
                    "historical_correlation_ranking": ranking,
                    "unique_policies": [policy.to_dict() for policy in policies],
                    "evaluated_policy_ids": [policy.policy_id for policy in evaluated],
                    "policy_results": policy_results,
                    "runtime": runtime,
                },
            )
            print(
                f"SAVED {backbone}/{dataset_config}/origin_{origin_index:03d}: "
                f"{len(records)} rows"
            )

    return write_backbone_report(
        root,
        p0a_output_root,
        destination,
        config=config,
        p0a=p0a,
        backbone=backbone,
        source_git_commit=source_commit,
        scientific_code_sha256=code_hash,
        model_metadata=model_metadata,
        runtime=runtime,
    )
