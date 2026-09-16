"""Print the canonical hash of a YAML experiment configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from covsafe.config import canonical_config_hash, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--length", type=int, default=12)
    args = parser.parse_args()
    config = load_yaml(args.config)
    print(canonical_config_hash(config, length=args.length))


if __name__ == "__main__":
    main()
