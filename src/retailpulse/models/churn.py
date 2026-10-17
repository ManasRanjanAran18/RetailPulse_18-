"""Churn prediction: compares logistic regression vs gradient boosting, selects by CV ROC-AUC."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retailpulse.features import CHURN_CATEGORICAL, CHURN_NUMERIC
from retailpulse.logging_utils import get_logger

log = get_logger(__name__)


def _preprocessor(scale: bool) -> ColumnTransformer:
    num = Pipeline([("scale", StandardScaler())]) if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, CHURN_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CHURN_CATEGORICAL),
    ])


def candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([("prep", _preprocessor(True)),
                                         ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))]),
        "hist_gradient_boosting": Pipeline([("prep", _preprocessor(False)),
                                            ("clf", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250,
                                                                                    l2_regularization=1.0, random_state=seed))]),
    }


def train_churn(features: pd.DataFrame, labels: pd.Series, cfg: dict) -> dict:
    c, seed = cfg["churn"], cfg["data"]["seed"]
    X = features[CHURN_NUMERIC + CHURN_CATEGORICAL]
    y = labels.to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=c["test_size"], stratify=y, random_state=seed)

    cv = StratifiedKFold(c["cv_folds"], shuffle=True, random_state=seed)
    cv_scores = {}
    for name, pipe in candidates(seed).items():
        cv_scores[name] = float(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean())
        log.info("churn CV ROC-AUC %-24s %.4f", name, cv_scores[name])
    best = max(cv_scores, key=cv_scores.get)
    model = candidates(seed)[best].fit(X_tr, y_tr)

    # operating threshold: maximise F1 on *out-of-fold* training predictions (the test set is never used for tuning)
    oof = cross_val_predict(candidates(seed)[best], X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    thr = float(thresholds[int(np.argmax([f1_score(y_tr, oof >= t) for t in thresholds]))])
    proba = model.predict_proba(X_te)[:, 1]
    pred = proba >= thr
    metrics = {
        "model": best, "cv_roc_auc": cv_scores, "threshold": thr,
        "test_roc_auc": float(roc_auc_score(y_te, proba)), "test_pr_auc": float(average_precision_score(y_te, proba)),
        "test_precision": float(precision_score(y_te, pred)), "test_recall": float(recall_score(y_te, pred)),
        "test_f1": float(f1_score(y_te, pred)), "test_brier": float(brier_score_loss(y_te, proba)),
        "base_rate": float(y.mean()), "n_train": len(X_tr), "n_test": len(X_te),
    }
    log.info("churn[%s] test AUC=%.3f PR-AUC=%.3f F1=%.3f", best, metrics["test_roc_auc"], metrics["test_pr_auc"], metrics["test_f1"])

    imp = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=seed)
    importance = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean, "std": imp.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
    # refit on all labelled data for deployment
    final = candidates(seed)[best].fit(X, y)
    return {"model": final, "metrics": metrics, "importance": importance, "train_features": X}


def score_customers(model: Pipeline, features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    X = features[CHURN_NUMERIC + CHURN_CATEGORICAL]
    p = model.predict_proba(X)[:, 1]
    out = features[["customer_id", "recency", "frequency_all", "monetary_all"]].copy()
    out["churn_probability"] = p
    out["risk_band"] = pd.cut(p, [-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"]).astype(str)
    out["predicted_churn"] = p >= threshold
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def save(model: Pipeline, path) -> None:
    joblib.dump(model, path)
