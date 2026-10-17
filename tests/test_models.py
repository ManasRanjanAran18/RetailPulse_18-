import numpy as np
import pandas as pd
import pytest

from retailpulse.features import (
    build_churn_features,
    churn_labels,
    rfm_table,
)
from retailpulse.models import churn, inventory, segmentation
from retailpulse.models.forecasting import SeasonalNaive, forecast_series, regression_metrics, wape


def test_metrics():
    y, p = np.array([10.0, 20.0]), np.array([12.0, 18.0])
    m = regression_metrics(y, p)
    assert m["mae"] == 2 and m["bias"] == 0 and wape(y, p) == pytest.approx(4 / 30)


def _series(n=200):
    idx = pd.date_range("2024-01-01", periods=n)
    return pd.Series(100 + 20 * (idx.dayofweek >= 5) + np.random.default_rng(0).normal(0, 2, n), index=idx)


def test_seasonal_naive_captures_weekday_pattern():
    s = _series()
    pred = SeasonalNaive().fit(s).predict(14)
    assert len(pred) == 14 and pred.index[0] == s.index[-1] + pd.Timedelta(days=1)
    weekend = pred[pred.index.dayofweek >= 5].mean()
    weekday = pred[pred.index.dayofweek < 5].mean()
    assert weekend - weekday > 15


def test_forecast_series_pipeline(small_cfg):
    res = forecast_series("x", _series(), small_cfg)
    assert res.best_model == "seasonal_naive" and len(res.forecast) == small_cfg["forecasting"]["horizon_days"]
    assert (res.forecast["yhat_lower"] <= res.forecast["yhat"]).all() and (res.forecast["yhat"] <= res.forecast["yhat_upper"]).all()
    assert (res.forecast["yhat_lower"] >= 0).all()


def test_prophet_and_lstm_smoke(small_cfg):
    cfg = {**small_cfg, "forecasting": {**small_cfg["forecasting"], "models": ["seasonal_naive", "prophet", "lstm"],
                                        "lstm": {"units": 8, "epochs": 2, "batch_size": 32, "patience": 1}}}
    res = forecast_series("x", _series(240), cfg)
    assert set(res.backtest["model"]) == {"seasonal_naive", "prophet", "lstm"}
    assert np.isfinite(res.forecast["yhat"]).all()


def test_segmentation(tables):
    rfm = rfm_table(tables["transactions"])
    _, segs, diag = segmentation.fit_segments(rfm, [4, 5])
    assert segs["segment"].notna().all() and segs["segment"].nunique() == diag["best_k"]
    prof = segmentation.segment_profile(segs).set_index("segment")
    assert prof["revenue_share"].sum() == pytest.approx(1.0)
    best = prof["avg_spend"].idxmax()
    assert best in ("Champions", "Loyal Customers")


def test_churn_model_beats_chance(small_cfg, tables):
    snap = pd.Timestamp(small_cfg["churn"]["snapshot_date"])
    f = build_churn_features(tables["transactions"], tables["customers"], snap)
    y = churn_labels(tables["transactions"], f["customer_id"], snap, 90)
    res = churn.train_churn(f, y, small_cfg)
    assert res["metrics"]["test_roc_auc"] > 0.7
    scores = churn.score_customers(res["model"], f, res["metrics"]["threshold"])
    assert scores["churn_probability"].between(0, 1).all() and set(scores["risk_band"]) <= {"Low", "Medium", "High"}


def test_inventory_formulas():
    cfg = {"inventory": {"service_level": 0.95, "order_cost": 50, "holding_rate": 0.2, "review_days": 7, "overstock_days_cover": 60, "demand_lookback_days": 30}}
    dates = pd.date_range("2024-01-01", periods=60)
    tx = pd.DataFrame({"order_date": dates, "product_id": "A", "category": "c", "quantity": 10})
    inv = pd.DataFrame({"product_id": ["A"], "on_hand": [50], "lead_time_days": [4], "snapshot_date": dates[-1], "supplier_id": ["S"]})
    prod = pd.DataFrame({"product_id": ["A"], "product_name": ["a"], "category": ["c"], "unit_cost": [5.0]})
    fc = pd.DataFrame({"product_id": ["A"] * 7, "yhat": [10.0] * 7})
    plan = inventory.build_inventory_plan(inv, prod, fc, tx, cfg).iloc[0]
    assert plan["lead_time_demand"] == 40 and plan["safety_stock"] == 0           # zero variance -> no safety stock
    assert plan["reorder_point"] == 40 and plan["stockout_risk"] == "Low"          # 50 on hand >= ROP
    assert plan["eoq"] == np.ceil(np.sqrt(2 * 3650 * 50 / (0.2 * 5)))
    assert plan["recommended_order_qty"] == 0

    inv.loc[0, "on_hand"] = 20
    plan = inventory.build_inventory_plan(inv, prod, fc, tx, cfg).iloc[0]
    assert plan["stockout_risk"] == "High" and plan["units_at_risk"] == 20 and plan["recommended_order_qty"] > 0
