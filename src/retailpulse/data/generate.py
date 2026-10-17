"""Synthetic retail dataset generator.

No public dataset can be bundled or downloaded in every environment, so this module produces a realistic,
fully reproducible multi-table retail dataset:

* ``transactions.csv`` - order lines (order_id, order_date, customer_id, store_id, product_id, quantity,
  unit_price, discount_pct). Includes weekly + yearly seasonality, a festive peak, category-specific
  seasonality, promotions, guest checkouts, returns and a few duplicate rows (to exercise the ETL).
* ``customers.csv``    - customer master data (no latent labels are exposed -> no leakage).
* ``products.csv``     - SKU master data.
* ``stores.csv``       - store master data.
* ``inventory.csv``    - current stock snapshot per SKU with supplier lead times.

Customer churn is simulated: churners' purchase rate decays in the ~60 days before they leave, giving a
learnable (but noisy) signal for the churn model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retailpulse.config import load_config, resolve_path
from retailpulse.logging_utils import get_logger

log = get_logger(__name__)

# category -> (price_low, price_high, mean extra units/line, popularity weight)
CATEGORIES = {
    "Grocery": (2, 15, 1.2, 0.30),
    "Beverages": (1.5, 8, 1.0, 0.18),
    "Household": (5, 40, 0.5, 0.16),
    "Personal Care": (4, 30, 0.4, 0.16),
    "Electronics": (25, 300, 0.05, 0.08),
    "Apparel": (10, 90, 0.2, 0.12),
}
ARCHETYPES = {  # name: (share, orders per year, lifetime churn probability)
    "loyal": (0.15, 32, 0.08),
    "regular": (0.35, 14, 0.30),
    "occasional": (0.35, 5, 0.50),
    "bargain": (0.15, 9, 0.45),
}
REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["organic", "paid_search", "social", "referral", "email"]
WEEKDAY_FACTOR = np.array([0.88, 0.90, 0.92, 0.97, 1.10, 1.30, 1.18])  # Mon..Sun


def _season_matrix(dates: pd.DatetimeIndex, cats: list[str]) -> np.ndarray:
    """(n_days, n_categories) multiplicative seasonality."""
    doy = dates.dayofyear.to_numpy()
    month = dates.month.to_numpy()
    out = np.ones((len(dates), len(cats)))
    festive = ((month == 11) & (dates.day.to_numpy() >= 15)) | ((month == 12) & (dates.day.to_numpy() <= 26))
    summer = np.isin(month, [4, 5, 6, 7])
    for j, c in enumerate(cats):
        base = 1 + 0.10 * np.sin(2 * np.pi * (doy - 80) / 365)
        if c == "Beverages":
            base = base * np.where(summer, 1.45, 0.9)
        elif c in ("Electronics", "Apparel"):
            base = base * np.where(festive, 1.9 if c == "Electronics" else 1.7, 1.0)
        elif c == "Grocery":
            base = base * np.where(festive, 1.25, 1.0)
        out[:, j] = base
    return out


def generate_dataset(cfg: dict | None = None, out_dir: str | Path | None = None) -> dict[str, Path]:
    cfg = cfg or load_config()
    d = cfg["data"]
    out = Path(out_dir) if out_dir else resolve_path(cfg, "raw_dir")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(d["seed"])

    dates = pd.date_range(d["start_date"], d["end_date"], freq="D")
    n_days = len(dates)

    # ---------------------------------------------------------------- products & stores
    cat_names = list(CATEGORIES)
    n_prod = d["n_products"]
    cat_of_prod = rng.choice(cat_names, size=n_prod, p=[CATEGORIES[c][3] for c in cat_names])
    for c in cat_names:  # guarantee every category has at least one SKU
        if c not in cat_of_prod:
            cat_of_prod[rng.integers(n_prod)] = c
    prices = np.array([np.exp(rng.uniform(np.log(CATEGORIES[c][0]), np.log(CATEGORIES[c][1]))) for c in cat_of_prod]).round(2)
    products = pd.DataFrame({
        "product_id": [f"P{i + 1:03d}" for i in range(n_prod)],
        "product_name": [f"{c} Item {i + 1}" for i, c in enumerate(cat_of_prod)],
        "category": cat_of_prod,
        "list_price": prices,
        "unit_cost": (prices * rng.uniform(0.55, 0.80, n_prod)).round(2),
    })
    popularity = rng.gamma(2.0, 1.0, n_prod)

    n_stores = d["n_stores"]
    stores = pd.DataFrame({
        "store_id": [f"S{i + 1:02d}" for i in range(n_stores)],
        "region": [REGIONS[i % 4] for i in range(n_stores)],
        "size_sqft": rng.integers(4000, 22000, n_stores),
    })
    store_w = stores["size_sqft"].to_numpy() / stores["size_sqft"].sum()

    # ---------------------------------------------------------------- customers
    n_cust = d["n_customers"]
    arche_names = list(ARCHETYPES)
    arche = rng.choice(arche_names, n_cust, p=[ARCHETYPES[a][0] for a in arche_names])
    signup_offset = np.where(rng.random(n_cust) < 0.55, 0, rng.integers(0, n_days - 30, n_cust))
    signup_idx = signup_offset.astype(int)
    rate = np.array([ARCHETYPES[a][1] for a in arche]) * rng.gamma(4, 0.25, n_cust)  # orders / year

    churn_p = np.array([ARCHETYPES[a][2] for a in arche])
    churns = rng.random(n_cust) < churn_p
    active_len = n_days - signup_idx
    churn_idx = np.where(churns, signup_idx + (rng.beta(1.3, 1.6, n_cust) * active_len).astype(int), n_days + 10_000)

    customers = pd.DataFrame({
        "customer_id": [f"C{i + 1:05d}" for i in range(n_cust)],
        "signup_date": dates[signup_idx].strftime("%Y-%m-%d"),
        "age": np.clip(rng.normal(37, 12, n_cust), 18, 80).astype(int),
        "gender": rng.choice(["F", "M", "O"], n_cust, p=[0.49, 0.48, 0.03]),
        "region": rng.choice(REGIONS, n_cust),
        "loyalty_member": np.where(rng.random(n_cust) < np.where(arche == "loyal", 0.85, 0.35), "Y", "N"),
        "acquisition_channel": rng.choice(CHANNELS, n_cust, p=[0.3, 0.22, 0.2, 0.15, 0.13]),
    })

    # ---------------------------------------------------------------- order intensity matrix
    day_idx = np.arange(n_days)
    season_cat = _season_matrix(dates, cat_names)
    season_total = (season_cat * np.array([CATEGORIES[c][3] for c in cat_names])).sum(1)
    season_total = season_total / season_total.mean()
    trend = 1 + 0.12 * day_idx / n_days
    wk = WEEKDAY_FACTOR[dates.dayofweek.to_numpy()]
    promo = np.zeros(n_days)
    for start in rng.choice(np.arange(10, n_days - 10), size=max(2, n_days // 60), replace=False):
        promo[start:start + 7] = 1
    calendar = season_total * trend * wk * (1 + 0.18 * promo)  # (n_days,)

    active = (day_idx[None, :] >= signup_idx[:, None]) & (day_idx[None, :] < churn_idx[:, None])
    fade = np.clip((churn_idx[:, None] - day_idx[None, :]) / 60.0, 0.12, 1.0)  # purchase decay before churn
    lam = (rate[:, None] / 365.0) * calendar[None, :] * active * fade
    n_orders_matrix = rng.poisson(lam)
    ci, di = np.nonzero(n_orders_matrix)
    reps = n_orders_matrix[ci, di]
    ci, di = np.repeat(ci, reps), np.repeat(di, reps)

    anon_counts = rng.poisson(0.10 * lam.sum(0))  # guest checkouts
    a_di = np.repeat(day_idx, anon_counts)
    cust_id_arr = np.concatenate([customers["customer_id"].to_numpy()[ci], np.full(len(a_di), None, dtype=object)])
    order_day = np.concatenate([di, a_di])
    n_orders = len(order_day)

    order_store = rng.choice(n_stores, n_orders, p=store_w)
    # customers have a 75% preference for a "home" store
    home = rng.choice(n_stores, n_cust, p=store_w)
    cust_of_order = np.concatenate([ci, np.zeros(len(a_di), dtype=int)])  # guest orders get a dummy 0 (masked below)
    pref = (np.arange(n_orders) < len(ci)) & (rng.random(n_orders) < 0.75)
    order_store[pref] = home[cust_of_order[pref]]

    # ---------------------------------------------------------------- order lines
    lines_per_order = 1 + rng.poisson(1.3, n_orders)
    line_order = np.repeat(np.arange(n_orders), lines_per_order)
    line_day = order_day[line_order]

    prod_cat_idx = np.array([cat_names.index(c) for c in cat_of_prod])
    weights = popularity[None, :] * season_cat[:, prod_cat_idx] * np.array([CATEGORIES[c][3] for c in cat_of_prod])[None, :]
    cdf = np.cumsum(weights / weights.sum(1, keepdims=True), axis=1)
    u = rng.random(len(line_order))
    prod_idx = np.minimum((cdf[line_day] < u[:, None]).sum(1), n_prod - 1)

    extra = np.array([CATEGORIES[c][2] for c in cat_of_prod])[prod_idx]
    qty = 1 + rng.poisson(extra)
    base_price = prices[prod_idx]
    disc = np.where((promo[line_day] > 0) & (rng.random(len(line_order)) < 0.6), rng.choice([10, 15, 20, 25], len(line_order)), 0)
    disc = np.where((disc == 0) & (rng.random(len(line_order)) < 0.04), 5, disc)
    unit_price = (base_price * rng.uniform(0.98, 1.02, len(line_order))).round(2)

    tx = pd.DataFrame({
        "order_id": np.array([f"O{i + 1:07d}" for i in range(n_orders)])[line_order],
        "order_date": dates[line_day].strftime("%Y-%m-%d"),
        "customer_id": cust_id_arr[line_order],
        "store_id": stores["store_id"].to_numpy()[order_store[line_order]],
        "product_id": products["product_id"].to_numpy()[prod_idx],
        "quantity": qty,
        "unit_price": unit_price,
        "discount_pct": disc,
    })
    # returns (negative qty) + duplicates -> data-quality issues for the ETL step to handle
    ret_idx = rng.choice(len(tx), int(0.005 * len(tx)), replace=False)
    tx.loc[ret_idx, "quantity"] = -tx.loc[ret_idx, "quantity"]
    dup_idx = rng.choice(len(tx), int(0.002 * len(tx)), replace=False)
    tx = pd.concat([tx, tx.iloc[dup_idx]], ignore_index=True).sample(frac=1, random_state=d["seed"]).sort_values(
        ["order_date", "order_id"], kind="stable").reset_index(drop=True)

    # ---------------------------------------------------------------- inventory snapshot
    clean = tx[tx["quantity"] > 0]
    recent = clean[pd.to_datetime(clean["order_date"]) > dates[-1] - pd.Timedelta(days=28)]
    daily_units = (recent.groupby("product_id")["quantity"].sum() / 28).reindex(products["product_id"]).fillna(0.2)
    lead = rng.integers(3, 15, n_prod)
    cover_days = rng.choice([4, 7, 12, 20, 35, 60, 90], n_prod, p=[0.12, 0.2, 0.2, 0.2, 0.14, 0.09, 0.05])
    inventory = pd.DataFrame({
        "product_id": products["product_id"],
        "snapshot_date": dates[-1].strftime("%Y-%m-%d"),
        "on_hand": np.maximum(0, (daily_units.to_numpy() * cover_days * rng.uniform(0.8, 1.2, n_prod))).round().astype(int),
        "lead_time_days": lead,
        "supplier_id": [f"SUP{x:02d}" for x in rng.integers(1, 9, n_prod)],
    })

    paths = {}
    for name, df in {"transactions": tx, "customers": customers, "products": products, "stores": stores, "inventory": inventory}.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
        log.info("wrote %-13s rows=%-8d -> %s", name, len(df), p)
    return paths


if __name__ == "__main__":
    generate_dataset()
