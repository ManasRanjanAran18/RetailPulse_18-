"""Configuration loading and path resolution."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("RETAILPULSE_HOME", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


@lru_cache(maxsize=4)
def _load_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load YAML config (env RETAILPULSE_CONFIG wins over default) and apply optional overrides."""
    path = str(path or os.environ.get("RETAILPULSE_CONFIG") or DEFAULT_CONFIG)
    cfg = copy.deepcopy(_load_file(path))
    if overrides:
        _deep_update(cfg, overrides)
    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking:
        cfg["mlflow"]["tracking_uri"] = tracking
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Return an absolute, existing directory for cfg['paths'][key]."""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p
