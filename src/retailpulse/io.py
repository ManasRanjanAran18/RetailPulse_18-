"""Loading helpers shared by the pipeline and the dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from retailpulse.config import load_config, resolve_path


def processed_dir(cfg: dict | None = None) -> Path:
    return resolve_path(cfg or load_config(), "processed_dir")


def reports_dir(cfg: dict | None = None) -> Path:
    return resolve_path(cfg or load_config(), "reports_dir")


def artifacts_dir(cfg: dict | None = None) -> Path:
    return resolve_path(cfg or load_config(), "artifacts_dir")


def read_table(name: str, cfg: dict | None = None) -> pd.DataFrame:
    p = processed_dir(cfg) / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run `retailpulse all` first.")
    return pd.read_parquet(p)


def write_table(df: pd.DataFrame, name: str, cfg: dict | None = None) -> Path:
    p = processed_dir(cfg) / f"{name}.parquet"
    df.to_parquet(p, index=False)
    return p


def read_report(name: str, cfg: dict | None = None) -> dict:
    p = reports_dir(cfg) / f"{name}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def write_report(obj: dict, name: str, cfg: dict | None = None) -> Path:
    p = reports_dir(cfg) / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p
