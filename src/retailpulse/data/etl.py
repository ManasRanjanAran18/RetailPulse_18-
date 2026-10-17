"""Automated ETL: extract -> validate -> transform -> load (parquet).

The step is idempotent and fails fast (``DataQualityError``) if critical checks do not pass, so bad data
never reaches model training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from retailpulse.config import load_config, resolve_path
from retailpulse.logging_utils import get_logger

log = get_logger(__name__)

SCHEMAS: dict[str, list[str]] = {
    "transactions": ["order_id", "order_date", "customer_id", "store_id", "product_id", "quantity", "unit_price", "discount_pct"],
    "customers": ["customer_id", "signup_date", "age", "gender", "region", "loyalty_member", "acquisition_channel"],
    "products": ["product_id", "product_name", "category", "list_price", "unit_cost"],
    "stores": ["store_id", "region", "size_sqft"],
    "inventory": ["product_id", "snapshot_date", "on_hand", "lead_time_days", "supplier_id"],
}


class DataQualityError(RuntimeError):
    """Raised when a critical data-quality check fails."""


@dataclass
class QualityReport:
    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", critical: bool = True) -> None:
        self.checks.append({"check": name, "passed": bool(passed), "critical": critical, "detail": detail})
        (log.info if passed else log.warning)("DQ %-34s %s %s", name, "PASS" if passed else "FAIL", detail)

    @property
    def ok(self) -> bool:
        return all(c["passed"] for c in self.checks if c["critical"])

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": self.checks}


# ------------------------------------------------------------------------------------------ extract
def extract(raw_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for name in SCHEMAS:
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file {path}. Run `retailpulse generate-data` or provide your own CSVs.")
        frames[name] = pd.read_csv(path)
        log.info("extracted %-13s rows=%d", name, len(frames[name]))
    return frames


# ------------------------------------------------------------------------------------------ validate
def validate(frames: dict[str, pd.DataFrame]) -> QualityReport:
    rep = QualityReport()
    for name, cols in SCHEMAS.items():
        missing = set(cols) - set(frames[name].columns)
        rep.add(f"schema:{name}", not missing, f"missing={sorted(missing)}" if missing else "")
    if not rep.ok:
        return rep  # cannot continue with structural problems

    tx, cust, prod, stores = frames["transactions"], frames["customers"], frames["products"], frames["stores"]
    dates = pd.to_datetime(tx["order_date"], errors="coerce")
    rep.add("tx:dates_parseable", dates.notna().all(), f"bad={int(dates.isna().sum())}")
    rep.add("tx:no_future_dates", (dates <= pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).all())
    rep.add("tx:key_fields_not_null", tx[["order_id", "product_id", "store_id"]].notna().all().all())
    rep.add("tx:unit_price_positive", (tx["unit_price"] > 0).all())
    rep.add("tx:discount_range", tx["discount_pct"].between(0, 90).all())
    rep.add("tx:zero_quantity", (tx["quantity"] != 0).all(), critical=False)
    rep.add("tx:fk_product", tx["product_id"].isin(prod["product_id"]).all())
    rep.add("tx:fk_store", tx["store_id"].isin(stores["store_id"]).all())
    known = tx["customer_id"].dropna()
    rep.add("tx:fk_customer", known.isin(cust["customer_id"]).all())
    dup = int(tx.duplicated().sum())
    rep.add("tx:duplicate_rows", dup == 0, f"duplicates={dup} (will be removed)", critical=False)
    guest = float(tx["customer_id"].isna().mean())
    rep.add("tx:guest_share<30%", guest < 0.30, f"{guest:.1%}", critical=False)
    rep.add("customers:unique_id", cust["customer_id"].is_unique)
    rep.add("products:unique_id", prod["product_id"].is_unique)
    return rep


# ------------------------------------------------------------------------------------------ transform
def transform(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    tx = frames["transactions"].drop_duplicates().copy()
    tx["order_date"] = pd.to_datetime(tx["order_date"])
    prod = frames["products"]
    tx = tx.merge(prod[["product_id", "category", "unit_cost"]], on="product_id", how="left")
    tx["is_return"] = tx["quantity"] < 0
    tx["is_guest"] = tx["customer_id"].isna()
    tx["net_unit_price"] = tx["unit_price"] * (1 - tx["discount_pct"] / 100.0)
    tx["revenue"] = tx["quantity"] * tx["net_unit_price"]
    tx["cogs"] = tx["quantity"] * tx["unit_cost"]
    tx["margin"] = tx["revenue"] - tx["cogs"]
    tx = tx.sort_values(["order_date", "order_id"]).reset_index(drop=True)

    cust = frames["customers"].copy()
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])
    inv = frames["inventory"].copy()
    inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"])
    return {"transactions": tx, "customers": cust, "products": prod.copy(), "stores": frames["stores"].copy(), "inventory": inv}


# ------------------------------------------------------------------------------------------ load
def load(frames: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Path]:
    paths = {}
    for name, df in frames.items():
        p = out_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p
    log.info("loaded %d parquet tables -> %s", len(paths), out_dir)
    return paths


def run_etl(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    raw, processed, reports = (resolve_path(cfg, k) for k in ("raw_dir", "processed_dir", "reports_dir"))
    frames = extract(raw)
    report = validate(frames)
    (reports / "data_quality.json").write_text(json.dumps(report.to_dict(), indent=2))
    if not report.ok:
        failed = [c["check"] for c in report.checks if c["critical"] and not c["passed"]]
        raise DataQualityError(f"Critical data-quality checks failed: {failed}")
    clean = transform(frames)
    load(clean, processed)
    tx = clean["transactions"]
    summary = {
        "rows": len(tx),
        "date_min": str(tx["order_date"].min().date()),
        "date_max": str(tx["order_date"].max().date()),
        "revenue": float(tx.loc[~tx["is_return"], "revenue"].sum()),
        "orders": int(tx["order_id"].nunique()),
        "customers": int(tx["customer_id"].nunique()),
    }
    log.info("ETL complete: %s", summary)
    return summary
