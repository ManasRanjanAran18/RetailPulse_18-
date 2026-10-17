"""Thin, failure-tolerant MLflow wrapper: experiment tracking must never break a pipeline run."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from retailpulse.logging_utils import get_logger

log = get_logger(__name__)


class Tracker:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _safe(self, fn, *a, **k):
        if not self.enabled:
            return
        try:
            fn(*a, **k)
        except Exception as exc:  # noqa: BLE001
            log.warning("MLflow call failed (%s) - continuing without tracking", exc)

    def params(self, d: dict):
        import mlflow
        self._safe(mlflow.log_params, {k: str(v)[:250] for k, v in d.items()})

    def metrics(self, d: dict, step: int | None = None):
        import mlflow
        clean = {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and v == v}
        self._safe(mlflow.log_metrics, clean, step=step)

    def artifact(self, path: str | Path):
        import mlflow
        if Path(path).exists():
            self._safe(mlflow.log_artifact, str(path))

    def tag(self, k: str, v: str):
        import mlflow
        self._safe(mlflow.set_tag, k, v)


@contextmanager
def track(cfg: dict, run_name: str):
    enabled = True
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        run = mlflow.start_run(run_name=run_name)
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow unavailable (%s) - tracking disabled for run '%s'", exc, run_name)
        enabled = False
        run = None
    try:
        yield Tracker(enabled)
    finally:
        if run is not None:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass
