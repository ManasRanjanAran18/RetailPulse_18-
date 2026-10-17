import json

from retailpulse.io import read_report, read_table
from retailpulse.pipeline import STAGES, run_all


def test_end_to_end_pipeline(small_cfg):
    run_all(small_cfg)
    for t in ("demand_daily", "forecast", "forecast_metrics", "product_forecast", "segments", "segment_profile",
              "churn_scores", "inventory_plan", "feature_drift", "demand_drift"):
        assert len(read_table(t, small_cfg)) > 0, t
    mon = read_report("monitoring_report", small_cfg)
    assert {"retrain_recommended", "reasons"} <= set(mon)
    from retailpulse.io import artifacts_dir
    a = artifacts_dir(small_cfg)
    assert (a / "churn_model.joblib").exists() and (a / "segmentation.joblib").exists()
    meta = json.loads((a / "churn_meta.json").read_text())
    assert 0 < meta["threshold"] < 1


def test_pipeline_survives_unreachable_mlflow(small_cfg, monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    cfg = {**small_cfg, "mlflow": {**small_cfg["mlflow"], "tracking_uri": "http://127.0.0.1:9"}}
    STAGES["segment"](cfg)  # must not raise even though the tracking server is down


def test_cli_unknown_stage_rejected():
    import pytest

    from retailpulse.pipeline import main
    with pytest.raises(SystemExit):
        main(["bogus"])
