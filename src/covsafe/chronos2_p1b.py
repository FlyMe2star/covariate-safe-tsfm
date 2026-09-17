"""Chronos-2 runner for the frozen P1b sealed-origin evaluation."""

from __future__ import annotations

import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from covsafe.fev_smoke import git_commit, package_version
from covsafe.p0b import (
    historical_policy_set,
    json_safe_metric_summary,
    make_origin_task,
    normalize_policy_score_records,
    prepare_task_context,
)
from covsafe.p1b import (
    EXPECTED_P1B_CONFIG_HASH,
    atomic_json,
    load_frozen_p1b,
    scientific_code_hash,
    sha256_file,
)
from covsafe.p1b_eval import (
    build_origin_decisions,
    completed_sealed_origin,
    ensure_decision_artifact,
    evaluation_origin_indices,
    load_router_state,
    origin_unit_keys,
    write_loss_artifact,
)


def run_chronos2_p1b(
    repo_root: str | Path,
    state_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run or safely resume exhaustive Chronos-2 sealed policy evaluation."""
    import torch
    from chronos import BaseChronosPipeline

    root = Path(repo_root).resolve()
    destination = Path(output_root)
    config, p0b, parent, p0a = load_frozen_p1b(root)
    state, score_records, state_path = load_router_state(root, state_root)
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime, then restart and run all cells")

    backbone = "chronos_2"
    model_config = p0b["backbones"][backbone]
    code_hash = scientific_code_hash(root)
    state_sha256 = sha256_file(state_path)
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
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

    completed_units = 0
    evaluated_policy_runs = 0
    inference_seconds = 0.0
    for dataset_config in config["scope"]["tasks"]:
        context = prepare_task_context(parent, p0a, dataset_config)
        for origin_index in evaluation_origin_indices(
            parent, p0a, dataset_config
        ):
            existing = completed_sealed_origin(
                destination,
                backbone=backbone,
                dataset_config=dataset_config,
                origin_index=origin_index,
                scientific_code_sha256=code_hash,
                router_state_sha256=state_sha256,
            )
            if existing is not None:
                completed_units += 1
                evaluated_policy_runs += len(existing["evaluated_policy_ids"])
                inference_seconds += float(existing["origin_inference_time_seconds"])
                print(f"RESUME {backbone}/{dataset_config}/origin_{origin_index:03d}")
                continue

            policies, ranking = historical_policy_set(p0b, context, origin_index)
            cutoff = str(context.reference_task.cutoffs[origin_index])
            decisions = build_origin_decisions(
                config=config,
                state_report=state,
                score_records=score_records,
                backbone=backbone,
                dataset_config=dataset_config,
                origin_index=origin_index,
                cutoff=cutoff,
                units=origin_unit_keys(context, origin_index),
                policies=policies,
            )
            decision_sidecar = ensure_decision_artifact(
                destination,
                decisions,
                {
                    "git_commit": git_commit(root),
                    "scientific_code_sha256": code_hash,
                    "router_state_sha256": state_sha256,
                    "backbone": backbone,
                    "dataset_config": dataset_config,
                    "origin_index": origin_index,
                    "cutoff": cutoff,
                    "applicable_group": bool(
                        config["applicability_map"][backbone][dataset_config]
                    ),
                },
            )
            print(
                f"RUN {backbone}/{dataset_config}/origin_{origin_index:03d} "
                f"with {len(policies)} policies after frozen decisions"
            )
            records: list[dict[str, Any]] = []
            policy_results: dict[str, Any] = {}
            origin_inference_seconds = 0.0
            for policy in policies:
                task = make_origin_task(parent, context, origin_index, policy)
                predictions, elapsed = pipeline.predict_fev(
                    task,
                    batch_size=model_config["batch_size"],
                    cross_learning=model_config["cross_learning"],
                    as_univariate=model_config["as_univariate"],
                )
                elapsed = float(elapsed)
                origin_inference_seconds += elapsed
                summary = task.evaluation_summary(
                    predictions,
                    model_name=f"chronos-2-p1b-{policy.policy_id}",
                    training_time_s=0.0,
                    inference_time_s=elapsed,
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
                    "inference_time_seconds": elapsed,
                    "aggregate_metrics": json_safe_metric_summary(summary),
                }
            loss_sidecar = write_loss_artifact(
                destination,
                records,
                {
                    "git_commit": git_commit(root),
                    "scientific_code_sha256": code_hash,
                    "router_state_sha256": state_sha256,
                    "decision_csv_sha256": decision_sidecar["csv_sha256"],
                    "backbone": backbone,
                    "dataset_config": dataset_config,
                    "dataset_fingerprint": context.reference_task._dataset_fingerprint,
                    "origin_index": origin_index,
                    "cutoff": cutoff,
                    "checkpoint_revision": model_config["checkpoint_revision"],
                    "historical_correlation_ranking": ranking,
                    "unique_policies": [policy.to_dict() for policy in policies],
                    "evaluated_policy_ids": [policy.policy_id for policy in policies],
                    "policy_results": policy_results,
                    "origin_inference_time_seconds": origin_inference_seconds,
                    "runtime": runtime,
                },
            )
            completed_units += 1
            evaluated_policy_runs += len(policies)
            inference_seconds += origin_inference_seconds
            print(
                f"SAVED {backbone}/{dataset_config}/origin_{origin_index:03d}: "
                f"{loss_sidecar['row_count']} loss rows"
            )

    report = {
        "experiment": config["experiment"],
        "schema_version": 1,
        "result_status": "paper_eligible",
        "artifact_role": "sealed_inference_completion_not_aggregate_result",
        "config_hash": EXPECTED_P1B_CONFIG_HASH,
        "git_commit": git_commit(root),
        "scientific_code_sha256": code_hash,
        "router_state_sha256": state_sha256,
        "backbone": backbone,
        "model": {
            "checkpoint": model_config["checkpoint"],
            "checkpoint_revision": model_config["checkpoint_revision"],
            "torch_dtype": model_config["torch_dtype"],
            "model_load_seconds": model_load_seconds,
        },
        "runtime": runtime,
        "completed_origin_count": completed_units,
        "evaluated_policy_run_count": evaluated_policy_runs,
        "total_inference_time_seconds": inference_seconds,
        "sealed_results_aggregated": False,
    }
    atomic_json(destination / "reports" / f"{backbone}_p1b_completion.json", report)
    return report
