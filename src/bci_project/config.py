"""YAML configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file and initialize output directories."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Create result and figure directories before an experiment writes artifacts.
    output = config.get("output", {})
    for key in ("results_dir", "figures_dir"):
        if key in output:
            Path(output[key]).mkdir(parents=True, exist_ok=True)

    return config
