"""Configuration loading and stable hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, rejecting non-mapping roots."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("The configuration root must be a mapping.")
    return value


def canonical_config_hash(config: dict[str, Any], length: int = 12) -> str:
    """Return a stable SHA-256 prefix over a canonical JSON representation."""
    if length < 8 or length > 64:
        raise ValueError("length must be between 8 and 64")
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
