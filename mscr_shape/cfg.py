"""Single source of truth for parameters: loads config.yaml as a nested dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import json
    print(json.dumps(load_config(), indent=2))
