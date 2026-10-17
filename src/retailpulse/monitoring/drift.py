"""Data / prediction drift detection (PSI + Kolmogorov-Smirnov) and a retraining recommendation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-6


def psi_numeric(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with reference-quantile bins. <0.1 stable, 0.1-0.25 moderate, >0.25 major."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature
        return 0.0 if np.allclose(np.unique(ref), np.unique(cur)) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    r = reference.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    c = current.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift_report(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str], categorical: list[str],
                         psi_threshold: float = 0.2, ks_alpha: float = 0.01) -> pd.DataFrame:
    rows = []
    for col in numeric:
        ks = ks_2samp(reference[col].dropna(), current[col].dropna())
        psi = psi_numeric(reference[col].to_numpy(), current[col].to_numpy())
        rows.append({"feature": col, "type": "numeric", "psi": psi, "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                     "drifted": bool(psi > psi_threshold and ks.pvalue < ks_alpha)})
    for col in categorical:
        psi = psi_categorical(reference[col], current[col])
        rows.append({"feature": col, "type": "categorical", "psi": psi, "ks_stat": np.nan, "ks_pvalue": np.nan,
                     "drifted": bool(psi > psi_threshold)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def demand_drift_report(demand: pd.DataFrame, recent_days: int, reference_days: int, psi_threshold: float, ks_alpha: float,
                        yoy_tolerance: float = 0.35) -> pd.DataFrame:
    """Seasonality-aware demand drift per series.

    * With >= 1 year of history the recent window is compared with the **same calendar window one year earlier**
      (so predictable seasonal peaks such as the festive season are *not* flagged). A series drifts when its level
      changed by more than ``yoy_tolerance`` year-over-year, or its day-of-week-normalised distribution shifted
      (PSI and KS both significant).
    * Otherwise it falls back to the immediately preceding ``reference_days`` window.
    The change vs. the immediately preceding window is always reported for information only.
    """
    rows = []
    last = demand["date"].max()
    r_start = last - pd.Timedelta(days=recent_days)
    prev_start = r_start - pd.Timedelta(days=reference_days)
    for sid, g in demand.groupby("series_id"):
        cur = g[g["date"] > r_start]
        prev = g[(g["date"] > prev_start) & (g["date"] <= r_start)]
        yoy = g[(g["date"] > r_start - pd.Timedelta(days=365)) & (g["date"] <= last - pd.Timedelta(days=365))]
        use_yoy = len(yoy) >= recent_days * 0.9
        ref = yoy if use_yoy else prev
        dow_mean = ref.groupby(ref["date"].dt.dayofweek)["units"].mean()
        ref_rel = ref["units"].to_numpy() / ref["date"].dt.dayofweek.map(dow_mean).clip(lower=EPS).to_numpy()
        cur_rel = cur["units"].to_numpy() / cur["date"].dt.dayofweek.map(dow_mean).clip(lower=EPS).to_numpy()
        ks = ks_2samp(ref_rel, cur_rel)
        psi = psi_numeric(ref_rel, cur_rel, bins=5)
        level_change = float(cur["units"].mean() / max(ref["units"].mean(), EPS) - 1)
        shape_shift = bool(psi > psi_threshold and ks.pvalue < ks_alpha)
        rows.append({
            "series_id": sid, "reference": "same window last year" if use_yoy else "preceding window",
            "ref_mean": float(ref["units"].mean()), "recent_mean": float(cur["units"].mean()),
            "level_change_pct": level_change,
            "change_vs_prior_window_pct": float(cur["units"].mean() / max(prev["units"].mean(), EPS) - 1),
            "psi": psi, "ks_pvalue": float(ks.pvalue),
            "drifted": bool(abs(level_change) > yoy_tolerance or shape_shift) if use_yoy else shape_shift,
        })
    return pd.DataFrame(rows)


def recommend_retraining(feature_drift: pd.DataFrame, demand_drift: pd.DataFrame, forecast_metrics: pd.DataFrame,
                         max_drift_share: float = 0.2) -> dict:
    n_feat = int(feature_drift["drifted"].sum())
    share = n_feat / max(len(feature_drift), 1)
    reasons = []
    if share > max_drift_share:
        reasons.append(f"{n_feat}/{len(feature_drift)} churn features drifted ({share:.0%})")
    drifted_series = demand_drift.loc[demand_drift["drifted"], "series_id"].tolist()
    if drifted_series:
        reasons.append(f"demand distribution shifted for: {', '.join(drifted_series)}")
    sel = forecast_metrics[forecast_metrics["selected"]]
    naive = forecast_metrics[forecast_metrics["model"] == "seasonal_naive"].set_index("series_id")["wape"]
    worse = [r.series_id for r in sel.itertuples() if r.series_id in naive.index and r.model != "seasonal_naive" and r.wape > naive[r.series_id]]
    if worse:
        reasons.append(f"selected forecast model underperforms naive baseline for: {', '.join(worse)}")
    return {"retrain_recommended": bool(reasons), "reasons": reasons, "drifted_features": feature_drift.loc[feature_drift["drifted"], "feature"].tolist(),
            "drifted_series": drifted_series}
