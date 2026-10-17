import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

from retailpulse.config import load_config
from retailpulse.data.etl import run_etl
from retailpulse.data.generate import generate_dataset


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("rp")
    cfg = load_config(overrides={
        "paths": {k: str(root / k) for k in ("raw_dir", "processed_dir", "artifacts_dir", "reports_dir")},
        "data": {"n_customers": 500, "n_products": 14, "n_stores": 3, "start_date": "2023-01-01", "end_date": "2024-04-30"},
        "mlflow": {"tracking_uri": f"sqlite:///{root}/mlflow.db"},
        "forecasting": {"models": ["seasonal_naive"], "test_days": 14, "horizon_days": 14},
        "segmentation": {"k_range": [4]},
        "churn": {"snapshot_date": "2024-01-15", "cv_folds": 3},
        "monitoring": {"demand_recent_days": 14, "demand_reference_days": 60},
    })
    generate_dataset(cfg)
    run_etl(cfg)
    return cfg


@pytest.fixture(scope="session")
def tables(small_cfg):
    from retailpulse.io import read_table
    return {n: read_table(n, small_cfg) for n in ("transactions", "customers", "products", "inventory")}
