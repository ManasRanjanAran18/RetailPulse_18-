"""Feature engineering for demand forecasting, segmentation and churn."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------ demand
def daily_demand(tx: pd.DataFrame, level: str = "category") -> pd.DataFrame:
    """Dense daily demand per series.

    Returns columns ``date, series_id, units, revenue`` with zero-filled days for every series, so that
    models always see a complete, regular time index. Returns are excluded from demand.
    """
    if level not in {"category", "product"}:
        raise ValueError("level must be 'category' or 'product'")
    key = "category" if level == "category" else "product_id"
    sales = tx.loc[tx["quantity"] > 0]
    g = sales.groupby([key, "order_date"]).agg(units=("quantity", "sum"), revenue=("revenue", "sum")).reset_index()
    g = g.rename(columns={key: "series_id", "order_date": "date"})
    full_idx = pd.date_range(tx["order_date"].min(), tx["order_date"].max(), freq="D")
    series = g["series_id"].unique()
    grid = pd.MultiIndex.from_product([series, full_idx], names=["series_id", "date"])
    out = g.set_index(["series_id", "date"]).reindex(grid, fill_value=0).reset_index()
    return out.sort_values(["series_id", "date"]).reset_index(drop=True)


def calendar_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """Cyclical calendar encodings used by the LSTM (dow + day-of-year)."""
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    return np.column_stack([
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
    ])


# ------------------------------------------------------------------------------------------ RFM
def rfm_table(tx: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Recency / Frequency / Monetary per identified customer (returns excluded)."""
    s = tx.loc[(tx["quantity"] > 0) & tx["customer_id"].notna()]
    as_of = as_of or (s["order_date"].max() + pd.Timedelta(days=1))
    s = s[s["order_date"] < as_of]
    orders = s.groupby(["customer_id", "order_id"]).agg(d=("order_date", "first"), rev=("revenue", "sum")).reset_index()
    rfm = orders.groupby("customer_id").agg(
        recency=("d", lambda x: (as_of - x.max()).days),
        frequency=("d", "count"),
        monetary=("rev", "sum"),
    )
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    return rfm.reset_index()


# ------------------------------------------------------------------------------------------ churn
CHURN_NUMERIC = [
    "recency", "frequency_30", "frequency_90", "frequency_180", "frequency_all", "monetary_90", "monetary_all",
    "avg_order_value", "avg_basket_lines", "tenure_days", "mean_gap_days", "std_gap_days", "recency_to_gap",
    "frequency_trend", "discount_share", "return_rate", "n_categories", "n_stores", "age",
]
CHURN_CATEGORICAL = ["gender", "region", "loyalty_member", "acquisition_channel"]


def build_churn_features(tx: pd.DataFrame, customers: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Customer-level features using only information strictly **before** ``snapshot`` (no leakage)."""
    snapshot = pd.Timestamp(snapshot)
    h = tx[(tx["order_date"] < snapshot) & tx["customer_id"].notna()]
    sales = h[h["quantity"] > 0]
    orders = sales.groupby(["customer_id", "order_id"]).agg(
        d=("order_date", "first"), rev=("revenue", "sum"), lines=("product_id", "count"),
        disc=("discount_pct", "mean"),
    ).reset_index()
    if orders.empty:
        raise ValueError("No transactions before snapshot date")
    orders["age_days"] = (snapshot - orders["d"]).dt.days

    def window_count(days: int) -> pd.Series:
        return orders[orders["age_days"] <= days].groupby("customer_id")["order_id"].count()

    def window_sum(days: int) -> pd.Series:
        return orders[orders["age_days"] <= days].groupby("customer_id")["rev"].sum()

    f = orders.groupby("customer_id").agg(
        recency=("age_days", "min"), frequency_all=("order_id", "count"), monetary_all=("rev", "sum"),
        avg_order_value=("rev", "mean"), avg_basket_lines=("lines", "mean"), first_order=("d", "min"),
        discount_share=("disc", "mean"),
    )
    f["frequency_30"], f["frequency_90"], f["frequency_180"] = window_count(30), window_count(90), window_count(180)
    f["monetary_90"] = window_sum(90)
    f[["frequency_30", "frequency_90", "frequency_180", "monetary_90"]] = f[["frequency_30", "frequency_90", "frequency_180", "monetary_90"]].fillna(0)
    f["tenure_days"] = (snapshot - f["first_order"]).dt.days

    gaps = orders.sort_values("d").groupby("customer_id")["d"].apply(lambda s: s.diff().dt.days.dropna())
    gap_stats = gaps.groupby(level=0).agg(["mean", "std"]) if len(gaps) else pd.DataFrame(columns=["mean", "std"])
    f["mean_gap_days"] = gap_stats["mean"]
    f["std_gap_days"] = gap_stats["std"]
    f["mean_gap_days"] = f["mean_gap_days"].fillna(f["tenure_days"].clip(lower=30))
    f["std_gap_days"] = f["std_gap_days"].fillna(0)
    f["recency_to_gap"] = f["recency"] / f["mean_gap_days"].clip(lower=1)
    f["frequency_trend"] = f["frequency_30"] / (f["frequency_90"] / 3 + 0.5)  # >1 accelerating, <1 slowing

    ret = h.groupby("customer_id")["is_return"].mean().rename("return_rate")
    f = f.join(ret)
    f["return_rate"] = f["return_rate"].fillna(0)
    f["n_categories"] = sales.groupby("customer_id")["category"].nunique()
    f["n_stores"] = sales.groupby("customer_id")["store_id"].nunique()

    f = f.reset_index().merge(customers[["customer_id", "age", *CHURN_CATEGORICAL]], on="customer_id", how="left")
    return f.drop(columns=["first_order"])


def churn_labels(tx: pd.DataFrame, customer_ids: pd.Series, snapshot: pd.Timestamp, horizon_days: int) -> pd.Series:
    """1 if the customer makes **no** purchase in ``[snapshot, snapshot + horizon)``."""
    snapshot = pd.Timestamp(snapshot)
    end = snapshot + pd.Timedelta(days=horizon_days)
    fut = tx[(tx["order_date"] >= snapshot) & (tx["order_date"] < end) & (tx["quantity"] > 0) & tx["customer_id"].notna()]
    buyers = set(fut["customer_id"].unique())
    return (~customer_ids.isin(buyers)).astype(int).rename("churned")
