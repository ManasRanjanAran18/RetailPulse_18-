import numpy as np
import pandas as pd

from retailpulse.monitoring import drift


def test_psi_identical_vs_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert drift.psi_numeric(a, rng.normal(0, 1, 5000)) < 0.05
    assert drift.psi_numeric(a, rng.normal(1.5, 1, 5000)) > 0.25


def test_psi_constant_feature():
    assert drift.psi_numeric(np.ones(100), np.ones(100)) == 0.0
    assert drift.psi_numeric(np.ones(100), np.full(100, 2.0)) == 1.0


def test_feature_drift_report_flags_only_shifted():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(0, 1, 3000), "c": rng.choice(["x", "y"], 3000)})
    cur = pd.DataFrame({"a": rng.normal(2, 1, 3000), "b": rng.normal(0, 1, 3000), "c": rng.choice(["x", "y"], 3000)})
    rep = drift.feature_drift_report(ref, cur, ["a", "b"], ["c"]).set_index("feature")
    assert rep.loc["a", "drifted"] and not rep.loc["b", "drifted"] and not rep.loc["c", "drifted"]


def test_demand_drift_ignores_seasonal_peak_but_catches_level_shift():
    dates = pd.date_range("2023-01-01", "2024-12-31")
    season = 1 + 0.8 * ((dates.month == 12).astype(float))          # recurring December peak
    base = 100 * season
    rows = []
    for sid, mult in {"stable": 1.0, "broken": 0.4}.items():
        units = base * np.where(dates >= "2024-12-04", mult, 1.0) * (1 + np.random.default_rng(3).normal(0, 0.03, len(dates)))
        rows.append(pd.DataFrame({"series_id": sid, "date": dates, "units": units}))
    rep = drift.demand_drift_report(pd.concat(rows), 28, 90, 0.2, 0.01).set_index("series_id")
    assert not rep.loc["stable", "drifted"] and rep.loc["broken", "drifted"]
    assert rep.loc["stable", "change_vs_prior_window_pct"] > 0.3   # big raw jump, correctly not flagged


def test_recommend_retraining_reasons():
    fd = pd.DataFrame({"feature": list("abcde"), "drifted": [True, True, False, False, False]})
    dd = pd.DataFrame({"series_id": ["s"], "drifted": [False]})
    fm = pd.DataFrame({"series_id": ["s", "s"], "model": ["seasonal_naive", "prophet"], "wape": [0.1, 0.2], "selected": [False, True]})
    out = drift.recommend_retraining(fd, dd, fm)
    assert out["retrain_recommended"] and len(out["reasons"]) == 2
