"""Inventory optimisation: safety stock, reorder point, EOQ and stock-out risk per SKU.

Category-level forecasts are allocated to SKUs top-down using each SKU's recent share of category demand.
Variability (sigma) is estimated from the SKU's recent daily demand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def allocate_forecast_to_products(cat_forecast: pd.DataFrame, tx: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    sales = tx[tx["quantity"] > 0]
    recent = sales[sales["order_date"] > sales["order_date"].max() - pd.Timedelta(days=lookback_days)]
    units = recent.groupby(["category", "product_id"])["quantity"].sum().reset_index()
    units["share"] = units["quantity"] / units.groupby("category")["quantity"].transform("sum")
    fc = cat_forecast.rename(columns={"series_id": "category"}).merge(units[["category", "product_id", "share"]], on="category")
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        fc[col] = fc[col] * fc["share"]
    return fc[["date", "category", "product_id", "yhat", "yhat_lower", "yhat_upper"]]


def build_inventory_plan(inventory: pd.DataFrame, products: pd.DataFrame, product_forecast: pd.DataFrame,
                         tx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["inventory"]
    z = float(norm.ppf(c["service_level"]))
    sales = tx[tx["quantity"] > 0]
    lb = sales["order_date"].max() - pd.Timedelta(days=c["demand_lookback_days"])
    recent = sales[sales["order_date"] > lb]
    full_idx = pd.date_range(lb + pd.Timedelta(days=1), sales["order_date"].max(), freq="D")
    daily = (recent.groupby(["product_id", "order_date"])["quantity"].sum().unstack(fill_value=0)
             .reindex(columns=full_idx, fill_value=0))
    sigma = daily.std(axis=1, ddof=1).rename("demand_std")

    mean_fc = product_forecast.groupby("product_id")["yhat"].mean().rename("forecast_daily_demand")
    plan = (inventory.merge(products[["product_id", "product_name", "category", "unit_cost"]], on="product_id")
            .merge(mean_fc, on="product_id", how="left").merge(sigma, on="product_id", how="left"))
    plan[["forecast_daily_demand", "demand_std"]] = plan[["forecast_daily_demand", "demand_std"]].fillna(0)

    d, L = plan["forecast_daily_demand"], plan["lead_time_days"]
    plan["lead_time_demand"] = d * L
    plan["safety_stock"] = np.ceil(z * plan["demand_std"] * np.sqrt(L))
    plan["reorder_point"] = np.ceil(plan["lead_time_demand"] + plan["safety_stock"])
    annual = d * 365
    hold = (c["holding_rate"] * plan["unit_cost"]).clip(lower=1e-6)
    plan["eoq"] = np.ceil(np.sqrt(2 * annual * c["order_cost"] / hold))
    plan["order_up_to"] = plan["reorder_point"] + d * c["review_days"]
    plan["days_of_cover"] = np.where(d > 0, plan["on_hand"] / d.replace(0, np.nan), np.inf)

    plan["stockout_risk"] = np.select(
        [plan["on_hand"] < plan["lead_time_demand"], plan["on_hand"] < plan["reorder_point"]], ["High", "Medium"], default="Low")
    need = (plan["order_up_to"] - plan["on_hand"]).clip(lower=0)
    plan["recommended_order_qty"] = np.where(plan["on_hand"] <= plan["reorder_point"], np.ceil(np.maximum(need, plan["eoq"])), 0).astype(int)
    plan["overstock"] = plan["days_of_cover"] > c["overstock_days_cover"]
    plan["excess_units"] = np.where(plan["overstock"], np.maximum(0, plan["on_hand"] - d * c["overstock_days_cover"]), 0).round().astype(int)
    plan["excess_capital"] = plan["excess_units"] * plan["unit_cost"]
    # units we expect to be unable to serve during the replenishment lead time if we reorder today
    plan["units_at_risk"] = np.maximum(0, plan["lead_time_demand"] - plan["on_hand"]).round().astype(int)
    plan["days_of_cover"] = plan["days_of_cover"].replace(np.inf, np.nan)
    risk_rank = {"High": 0, "Medium": 1, "Low": 2}
    plan = plan.sort_values(by=["stockout_risk", "days_of_cover"], key=lambda s: s.map(risk_rank) if s.name == "stockout_risk" else s)
    return plan.reset_index(drop=True)
