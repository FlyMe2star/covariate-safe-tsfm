"""Shared, model-independent helpers for FEV smoke runners."""

from __future__ import annotations

import math
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

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


def package_version(name: str) -> str:
    """Return an installed package version or ``unknown``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def git_commit(repo_root: Path) -> str:
    """Return the repository HEAD commit."""
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
