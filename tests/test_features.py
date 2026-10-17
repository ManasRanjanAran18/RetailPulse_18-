import pandas as pd

from retailpulse.features import build_churn_features, churn_labels, daily_demand, rfm_table


def test_daily_demand_is_dense_and_excludes_returns(tables):
    tx = tables["transactions"]
    d = daily_demand(tx, "category")
    days = tx["order_date"].nunique()
    full = (tx["order_date"].max() - tx["order_date"].min()).days + 1
    assert (d.groupby("series_id").size() == full).all() and days <= full
    pos_units = tx.loc[tx["quantity"] > 0, "quantity"].sum()
    assert d["units"].sum() == pos_units


def test_churn_features_have_no_leakage(tables):
    tx, cu = tables["transactions"], tables["customers"]
    snap = pd.Timestamp("2024-01-15")
    full = build_churn_features(tx, cu, snap).set_index("customer_id").sort_index()
    truncated = build_churn_features(tx[tx["order_date"] < snap], cu, snap).set_index("customer_id").sort_index()
    pd.testing.assert_frame_equal(full, truncated)  # future rows can never change features


def test_churn_labels_binary_and_consistent(tables):
    tx, cu = tables["transactions"], tables["customers"]
    snap = pd.Timestamp("2024-01-15")
    f = build_churn_features(tx, cu, snap)
    y = churn_labels(tx, f["customer_id"], snap, 90)
    assert set(y.unique()) <= {0, 1} and 0.05 < y.mean() < 0.95
    buyer = tx[(tx["order_date"] >= snap) & (tx["order_date"] < snap + pd.Timedelta(days=90)) & (tx["quantity"] > 0)]["customer_id"].dropna().iloc[0]
    assert y[f["customer_id"] == buyer].iloc[0] == 0


def test_rfm(tables):
    r = rfm_table(tables["transactions"])
    assert (r["frequency"] >= 1).all() and (r["recency"] >= 1).all() and r["customer_id"].is_unique
