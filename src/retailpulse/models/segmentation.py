"""RFM customer segmentation with K-Means (k chosen by silhouette score)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from retailpulse.logging_utils import get_logger

log = get_logger(__name__)

# Clusters whose mean recency is below ACTIVE_RECENCY_DAYS are "active" and named by value rank; the rest are "lapsed".
ACTIVE_RECENCY_DAYS = 90
ACTIVE_NAMES = ["Champions", "Loyal Customers", "Potential Loyalists", "Promising"]
LAPSED_NAMES = ["At Risk", "Hibernating", "Lost"]
FEATURES = ["recency", "frequency", "monetary"]


def _transform(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df[FEATURES].to_numpy(float))


def fit_segments(rfm: pd.DataFrame, k_range: list[int], seed: int = 42) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Return (pipeline, rfm with `cluster` + `segment`, diagnostics)."""
    X = _transform(rfm)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        scores[k] = float(silhouette_score(Z, labels, sample_size=min(len(Z), 3000), random_state=seed))
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=20, random_state=seed).fit(Z)
    pipe = Pipeline([("scaler", scaler), ("kmeans", km)])

    out = rfm.copy()
    out["cluster"] = km.labels_
    # rank clusters: high frequency & monetary, low recency => best
    prof = out.groupby("cluster")[FEATURES].mean()
    zprof = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)
    value = zprof["frequency"] + zprof["monetary"] - zprof["recency"]
    order = value.sort_values(ascending=False).index.tolist()
    active = [c for c in order if prof.loc[c, "recency"] <= ACTIVE_RECENCY_DAYS]
    lapsed = [c for c in order if c not in active]
    mapping = {}
    for i, c in enumerate(active):
        mapping[c] = ACTIVE_NAMES[i] if i < len(ACTIVE_NAMES) else f"Active {i + 1}"
    for i, c in enumerate(lapsed):
        mapping[c] = LAPSED_NAMES[i] if i < len(LAPSED_NAMES) else f"Lapsed {i + 1}"
    out["segment"] = out["cluster"].map(mapping)
    diag = {"silhouette_by_k": scores, "best_k": best_k, "cluster_to_segment": {int(c): n for c, n in mapping.items()}}
    log.info("segmentation: best_k=%d silhouette=%.3f", best_k, scores[best_k])
    return pipe, out, diag


def segment_profile(segments: pd.DataFrame) -> pd.DataFrame:
    p = segments.groupby("segment").agg(
        customers=("customer_id", "count"), avg_recency_days=("recency", "mean"), avg_orders=("frequency", "mean"),
        avg_spend=("monetary", "mean"), total_revenue=("monetary", "sum"),
    ).reset_index()
    p["revenue_share"] = p["total_revenue"] / p["total_revenue"].sum()
    return p.sort_values("avg_spend", ascending=False).reset_index(drop=True)


def save(pipe: Pipeline, path) -> None:
    joblib.dump(pipe, path)
