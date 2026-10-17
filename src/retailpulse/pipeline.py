"""End-to-end pipeline + CLI.

    retailpulse generate-data | etl | forecast | segment | churn | inventory | monitor | all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from retailpulse import __version__
from retailpulse.config import load_config
from retailpulse.data.etl import run_etl
from retailpulse.data.generate import generate_dataset
from retailpulse.features import (
    CHURN_CATEGORICAL,
    CHURN_NUMERIC,
    build_churn_features,
    churn_labels,
    daily_demand,
    rfm_table,
)
from retailpulse.io import artifacts_dir, read_table, reports_dir, write_report, write_table
from retailpulse.logging_utils import get_logger
from retailpulse.models import churn as churn_mod
from retailpulse.models import inventory as inv_mod
from retailpulse.models import segmentation as seg_mod
from retailpulse.models.forecasting import run_forecasting
from retailpulse.monitoring import drift
from retailpulse.tracking import track

log = get_logger(__name__)


# ------------------------------------------------------------------------------------------ stages
def stage_generate(cfg):
    generate_dataset(cfg)


def stage_etl(cfg):
    summary = run_etl(cfg)
    write_report(summary, "etl_summary", cfg)


def stage_forecast(cfg):
    tx, products = read_table("transactions", cfg), read_table("products", cfg)
    level = cfg["forecasting"]["level"]
    demand = daily_demand(tx, level=level)
    write_table(demand, "demand_daily", cfg)
    with track(cfg, "demand_forecasting") as t:
        t.params({"level": level, **{k: v for k, v in cfg["forecasting"].items() if k != "lstm"},
                  **{f"lstm_{k}": v for k, v in cfg["forecasting"]["lstm"].items()}})
        res = run_forecasting(demand, cfg)
        write_table(res["metrics"], "forecast_metrics", cfg)
        write_table(res["backtest"], "forecast_backtest", cfg)
        write_table(res["forecast"], "forecast", cfg)
        sel = res["metrics"][res["metrics"]["selected"]]
        t.metrics({f"wape_{r.series_id}": r.wape for r in sel.itertuples()})
        t.metrics({"wape_mean_selected": sel["wape"].mean(), "rmse_mean_selected": sel["rmse"].mean()})
        for r in sel.itertuples():
            t.tag(f"best_model_{r.series_id}", r.model)
        out = reports_dir(cfg) / "forecast_summary.json"
        write_report({"level": level, "selected": sel[["series_id", "model", "wape", "rmse", "mae"]].to_dict("records"),
                      "wape_mean_selected": float(sel["wape"].mean())}, "forecast_summary", cfg)
        t.artifact(out)

    fc = res["forecast"]
    if level == "category":
        pf = inv_mod.allocate_forecast_to_products(fc, tx, cfg["inventory"]["demand_lookback_days"])
    else:
        pf = fc.rename(columns={"series_id": "product_id"}).merge(products[["product_id", "category"]], on="product_id")[
            ["date", "category", "product_id", "yhat", "yhat_lower", "yhat_upper"]]
    write_table(pf, "product_forecast", cfg)


def stage_segment(cfg):
    tx = read_table("transactions", cfg)
    rfm = rfm_table(tx)
    with track(cfg, "customer_segmentation") as t:
        pipe, segs, diag = seg_mod.fit_segments(rfm, cfg["segmentation"]["k_range"], cfg["data"]["seed"])
        prof = seg_mod.segment_profile(segs)
        write_table(segs, "segments", cfg)
        write_table(prof, "segment_profile", cfg)
        seg_mod.save(pipe, artifacts_dir(cfg) / "segmentation.joblib")
        write_report(diag, "segmentation_summary", cfg)
        t.params({"k_range": cfg["segmentation"]["k_range"], "best_k": diag["best_k"]})
        t.metrics({"silhouette": diag["silhouette_by_k"][diag["best_k"]], "n_customers": len(segs)})
        t.artifact(artifacts_dir(cfg) / "segmentation.joblib")


def stage_churn(cfg):
    tx, cust = read_table("transactions", cfg), read_table("customers", cfg)
    c = cfg["churn"]
    snap = pd.Timestamp(c["snapshot_date"])
    feats = build_churn_features(tx, cust, snap)
    labels = churn_labels(tx, feats["customer_id"], snap, c["horizon_days"])
    with track(cfg, "churn_prediction") as t:
        res = churn_mod.train_churn(feats, labels, cfg)
        m = res["metrics"]
        t.params({"snapshot_date": c["snapshot_date"], "horizon_days": c["horizon_days"], "selected_model": m["model"],
                  "threshold": m["threshold"]})
        t.metrics({k: v for k, v in m.items() if isinstance(v, float)})
        t.metrics({f"cv_auc_{k}": v for k, v in m["cv_roc_auc"].items()})
        art = artifacts_dir(cfg)
        churn_mod.save(res["model"], art / "churn_model.joblib")
        res["train_features"].to_parquet(art / "churn_reference_features.parquet", index=False)
        (art / "churn_meta.json").write_text(json.dumps({"threshold": m["threshold"], "numeric": CHURN_NUMERIC,
                                                         "categorical": CHURN_CATEGORICAL, "model": m["model"],
                                                         "trained_at": datetime.now(timezone.utc).isoformat()}, indent=2))
        write_table(res["importance"], "churn_importance", cfg)
        write_report(m, "churn_metrics", cfg)
        t.artifact(art / "churn_model.joblib")
        t.artifact(art / "churn_meta.json")

    # score everyone as of "today" (day after last transaction) -> forward-looking risk
    today = tx["order_date"].max() + pd.Timedelta(days=1)
    live = build_churn_features(tx, cust, today)
    live.to_parquet(artifacts_dir(cfg) / "churn_scoring_features.parquet", index=False)
    scores = churn_mod.score_customers(res["model"], live, m["threshold"])
    scores = scores.merge(read_table("customers", cfg)[["customer_id", "region", "loyalty_member"]], on="customer_id", how="left")
    write_table(scores, "churn_scores", cfg)
    log.info("scored %d customers; high-risk=%d", len(scores), int((scores["risk_band"] == "High").sum()))


def stage_inventory(cfg):
    tx = read_table("transactions", cfg)
    plan = inv_mod.build_inventory_plan(read_table("inventory", cfg), read_table("products", cfg),
                                        read_table("product_forecast", cfg), tx, cfg)
    write_table(plan, "inventory_plan", cfg)
    summary = {
        "skus": len(plan), "high_risk_skus": int((plan["stockout_risk"] == "High").sum()),
        "medium_risk_skus": int((plan["stockout_risk"] == "Medium").sum()),
        "overstocked_skus": int(plan["overstock"].sum()), "excess_capital": float(plan["excess_capital"].sum()),
        "units_at_risk": int(plan["units_at_risk"].sum()), "units_to_order": int(plan["recommended_order_qty"].sum()),
    }
    write_report(summary, "inventory_summary", cfg)
    with track(cfg, "inventory_optimization") as t:
        t.params(cfg["inventory"])
        t.metrics(summary)
    log.info("inventory: %s", summary)


def stage_monitor(cfg):
    m = cfg["monitoring"]
    art = artifacts_dir(cfg)
    ref_p, cur_p = art / "churn_reference_features.parquet", art / "churn_scoring_features.parquet"
    if not (ref_p.exists() and cur_p.exists()):
        raise FileNotFoundError("Run the churn stage before monitoring.")
    ref, cur = pd.read_parquet(ref_p), pd.read_parquet(cur_p)
    skip = set(m.get("ignore_features", []))  # features that drift mechanically with time (e.g. tenure)
    fd = drift.feature_drift_report(ref, cur, [f for f in CHURN_NUMERIC if f not in skip],
                                    [f for f in CHURN_CATEGORICAL if f not in skip], m["psi_threshold"], m["ks_alpha"])
    dd = drift.demand_drift_report(read_table("demand_daily", cfg), m["demand_recent_days"], m["demand_reference_days"],
                                   m["psi_threshold"], m["ks_alpha"], m["demand_yoy_tolerance"])
    decision = drift.recommend_retraining(fd, dd, read_table("forecast_metrics", cfg))
    write_table(fd, "feature_drift", cfg)
    write_table(dd, "demand_drift", cfg)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "version": __version__, **decision,
              "n_features_checked": len(fd), "n_features_drifted": int(fd["drifted"].sum()),
              "max_feature_psi": float(fd["psi"].max())}
    write_report(report, "monitoring_report", cfg)
    with track(cfg, "monitoring") as t:
        t.metrics({"n_features_drifted": report["n_features_drifted"], "max_feature_psi": report["max_feature_psi"],
                   "retrain_recommended": float(report["retrain_recommended"])})
    log.info("monitoring: retrain_recommended=%s %s", report["retrain_recommended"], report["reasons"])


STAGES = {"generate-data": stage_generate, "etl": stage_etl, "forecast": stage_forecast, "segment": stage_segment,
          "churn": stage_churn, "inventory": stage_inventory, "monitor": stage_monitor}
ORDER = ["etl", "forecast", "segment", "churn", "inventory", "monitor"]


def run_all(cfg, with_data: bool = False):
    order = (["generate-data"] if with_data else []) + ORDER
    for name in order:
        t0 = time.time()
        log.info("=== stage: %s ===", name)
        STAGES[name](cfg)
        log.info("=== stage %s done in %.1fs ===", name, time.time() - t0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="retailpulse", description=__doc__)
    ap.add_argument("stage", choices=[*STAGES, "all"], help="pipeline stage to run")
    ap.add_argument("--config", help="path to YAML config (default config/config.yaml or $RETAILPULSE_CONFIG)")
    ap.add_argument("--with-data", action="store_true", help="with `all`: (re)generate the synthetic dataset first")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    try:
        run_all(cfg, args.with_data) if args.stage == "all" else STAGES[args.stage](cfg)
    except Exception:
        log.exception("pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
