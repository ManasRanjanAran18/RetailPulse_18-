"""Demand forecasting: seasonal-naive baseline, Prophet and LSTM with automatic model selection.

For every series (e.g. product category) we
1. back-test each candidate on the last ``test_days`` days,
2. pick the best model by ``selection_metric`` (WAPE by default),
3. refit that model on all data and forecast ``horizon_days`` ahead,
4. attach 80 % prediction intervals derived from the back-test residuals.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd

from retailpulse.features import calendar_features
from retailpulse.logging_utils import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------------------------------ metrics
def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.abs(y - yhat).sum() / max(np.abs(y).sum(), 1e-9))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = (np.abs(y) + np.abs(yhat)) / 2
    return float(np.mean(np.where(denom == 0, 0, np.abs(y - yhat) / np.where(denom == 0, 1, denom))))


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {
        "mae": float(np.mean(np.abs(y - yhat))),
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "wape": wape(y, yhat),
        "smape": smape(y, yhat),
        "bias": float(np.mean(yhat - y)),
    }


# ------------------------------------------------------------------------------------------ models
class SeasonalNaive:
    """Forecast = mean of the same weekday over the last ``weeks`` weeks."""

    name = "seasonal_naive"

    def __init__(self, weeks: int = 4):
        self.weeks = weeks

    def fit(self, s: pd.Series) -> SeasonalNaive:
        self.s = s.astype(float)
        return self

    def predict(self, horizon: int) -> pd.Series:
        idx = pd.date_range(self.s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        recent = self.s.iloc[-7 * self.weeks:]
        by_dow = recent.groupby(recent.index.dayofweek).mean()
        return pd.Series(by_dow.reindex(idx.dayofweek).to_numpy(), index=idx)


class ProphetModel:
    name = "prophet"

    def fit(self, s: pd.Series) -> ProphetModel:
        import logging

        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.ERROR)
        df = pd.DataFrame({"ds": s.index, "y": s.to_numpy(float)})
        # multiplicative seasonality suits retail demand whose amplitude scales with the level
        self.model = Prophet(weekly_seasonality=True, yearly_seasonality=len(s) >= 365 * 1.5,
                             daily_seasonality=False, seasonality_mode="multiplicative", changepoint_prior_scale=0.05)
        self.model.fit(df)
        return self

    def predict(self, horizon: int) -> pd.Series:
        fut = self.model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
        fc = self.model.predict(fut)
        return pd.Series(np.clip(fc["yhat"].to_numpy(), 0, None), index=pd.DatetimeIndex(fc["ds"]))


class LSTMModel:
    """Single-layer LSTM on a sliding window of (log-scaled demand + calendar encodings); recursive forecast."""

    name = "lstm"

    def __init__(self, lookback: int = 28, units: int = 32, epochs: int = 60, batch_size: int = 32,
                 patience: int = 8, seed: int = 42):
        self.lookback, self.units, self.epochs = lookback, units, epochs
        self.batch_size, self.patience, self.seed = batch_size, patience, seed

    def _features(self, values_scaled: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        return np.column_stack([values_scaled, calendar_features(dates)])

    def fit(self, s: pd.Series) -> LSTMModel:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        y = np.log1p(s.to_numpy(float))
        self.mu, self.sd = y.mean(), y.std() or 1.0
        z = (y - self.mu) / self.sd
        feats = self._features(z, s.index)
        X, t = [], []
        for i in range(self.lookback, len(z)):
            X.append(feats[i - self.lookback:i])
            # target day's calendar features are known in advance -> append to the window as last step input
            t.append(z[i])
        X, t = np.array(X), np.array(t)
        # give the network the calendar of the day to predict (known at forecast time)
        cal_next = calendar_features(s.index[self.lookback:])
        n_val = max(14, int(0.1 * len(X)))
        Xtr, Xva, ctr, cva, ttr, tva = X[:-n_val], X[-n_val:], cal_next[:-n_val], cal_next[-n_val:], t[:-n_val], t[-n_val:]

        seq_in = tf.keras.Input(shape=(self.lookback, X.shape[2]))
        cal_in = tf.keras.Input(shape=(cal_next.shape[1],))
        h = tf.keras.layers.LSTM(self.units)(seq_in)
        h = tf.keras.layers.Concatenate()([h, cal_in])
        h = tf.keras.layers.Dense(16, activation="relu")(h)
        out = tf.keras.layers.Dense(1)(h)
        self.model = tf.keras.Model([seq_in, cal_in], out)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        self.model.fit(
            [Xtr, ctr], ttr, validation_data=([Xva, cva], tva), epochs=self.epochs, batch_size=self.batch_size, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)],
        )
        self._z, self._dates = z, s.index
        return self

    def predict(self, horizon: int) -> pd.Series:
        z = list(self._z)
        dates = list(self._dates)
        fut_idx = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        preds = []
        for d in fut_idx:
            win_dates = pd.DatetimeIndex(dates[-self.lookback:])
            feats = self._features(np.array(z[-self.lookback:]), win_dates)[None]
            cal = calendar_features(pd.DatetimeIndex([d]))
            zhat = float(self.model.predict([feats, cal], verbose=0)[0, 0])
            z.append(zhat)
            dates.append(d)
            preds.append(zhat)
        yhat = np.expm1(np.array(preds) * self.sd + self.mu)
        return pd.Series(np.clip(yhat, 0, None), index=fut_idx)


def build_model(name: str, cfg: dict):
    f = cfg["forecasting"]
    if name == "seasonal_naive":
        return SeasonalNaive()
    if name == "prophet":
        return ProphetModel()
    if name == "lstm":
        p = f["lstm"]
        return LSTMModel(lookback=f["lookback"], units=p["units"], epochs=p["epochs"],
                         batch_size=p["batch_size"], patience=p["patience"], seed=cfg["data"]["seed"])
    raise ValueError(f"unknown model {name}")


# ------------------------------------------------------------------------------------------ orchestration
@dataclass
class SeriesResult:
    series_id: str
    best_model: str
    backtest: pd.DataFrame      # model comparison (one row per model)
    backtest_pred: pd.DataFrame  # date, model, actual, predicted
    forecast: pd.DataFrame      # date, yhat, yhat_lower, yhat_upper


def forecast_series(series_id: str, s: pd.Series, cfg: dict) -> SeriesResult:
    f = cfg["forecasting"]
    test_days, horizon, metric = f["test_days"], f["horizon_days"], f["selection_metric"]
    train, test = s.iloc[:-test_days], s.iloc[-test_days:]

    rows, bt_preds, residual_sd = [], [], {}
    for name in f["models"]:
        try:
            m = build_model(name, cfg).fit(train)
            pred = m.predict(test_days)
            met = regression_metrics(test.to_numpy(), pred.to_numpy())
            rows.append({"series_id": series_id, "model": name, **met})
            residual_sd[name] = float(np.std(test.to_numpy() - pred.to_numpy()))
            bt_preds.append(pd.DataFrame({"series_id": series_id, "date": test.index, "model": name,
                                          "actual": test.to_numpy(), "predicted": pred.to_numpy()}))
            log.info("[%s] %-15s WAPE=%.3f RMSE=%.2f", series_id, name, met["wape"], met["rmse"])
        except Exception as exc:  # a failing candidate must never break the whole run
            log.exception("[%s] model %s failed: %s", series_id, name, exc)
    if not rows:
        raise RuntimeError(f"all models failed for series {series_id}")

    comp = pd.DataFrame(rows)
    comp["selected"] = False
    best = comp.sort_values(metric).iloc[0]["model"]
    comp.loc[comp["model"] == best, "selected"] = True

    final = build_model(best, cfg).fit(s)
    yhat = final.predict(horizon)
    sd = residual_sd[best]
    z = f["interval_z"]
    fc = pd.DataFrame({"date": yhat.index, "yhat": yhat.to_numpy(),
                       "yhat_lower": np.clip(yhat.to_numpy() - z * sd, 0, None), "yhat_upper": yhat.to_numpy() + z * sd})
    fc.insert(0, "series_id", series_id)
    fc["model"] = best
    return SeriesResult(series_id, best, comp, pd.concat(bt_preds, ignore_index=True), fc)


def run_forecasting(demand: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """demand: output of ``features.daily_demand``. Returns metrics, back-test and forecast frames."""
    metrics, backtests, forecasts = [], [], []
    for sid, g in demand.groupby("series_id"):
        s = g.set_index("date")["units"].astype(float).asfreq("D", fill_value=0)
        res = forecast_series(str(sid), s, cfg)
        metrics.append(res.backtest)
        backtests.append(res.backtest_pred)
        forecasts.append(res.forecast)
    return {"metrics": pd.concat(metrics, ignore_index=True),
            "backtest": pd.concat(backtests, ignore_index=True),
            "forecast": pd.concat(forecasts, ignore_index=True)}
