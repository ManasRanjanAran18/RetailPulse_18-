# Operations runbook

## Routine
| task | command |
|---|---|
| Refresh data & retrain everything | `retailpulse all` (K8s: CronJob `retailpulse-retrain`, manual: `kubectl create job --from=cronjob/retailpulse-retrain manual-1 -n retailpulse`) |
| Re-score customers only | `retailpulse churn` |
| Re-check drift only | `retailpulse monitor` |
| Inspect experiments | MLflow UI (`make mlflow` / service `mlflow:5000`) |

## Reading the monitoring verdict (`reports/monitoring_report.json`)
`retrain_recommended = true` when any of:
1. > 20 % of churn features have PSI > 0.2 **and** KS p < 0.01 (`tenure_days` is excluded — it drifts mechanically);
2. a demand series moved > 35 % year-over-year, or its day-of-week-normalised distribution shifted (PSI & KS);
3. the auto-selected forecast model is worse than the seasonal-naive baseline.

Rule of thumb for PSI: < 0.1 stable · 0.1–0.25 moderate · > 0.25 major.

## Failure modes
| symptom | cause / fix |
|---|---|
| `DataQualityError` | open `reports/data_quality.json`; fix the failing critical check at the source. Nothing downstream is overwritten. |
| Dashboard says "No processed data" | run `retailpulse all --with-data` (or mount your CSVs in `data/raw` and run `retailpulse all`) |
| `No transactions before snapshot date` / empty churn labels | `churn.snapshot_date` must lie inside your data and ≥ `horizon_days` before the last order date |
| MLflow warnings, run continues | tracking server unreachable — by design it never blocks the pipeline |
| One forecast model missing in comparison | that candidate raised an exception; check logs (`log.exception`) — others still run |
| Dashboard pods `Pending` | PVC needs a ReadWriteMany storage class |

## Release checklist
1. `make lint test` green (CI also runs on 3.10–3.12).
2. Build + push image; bump tag in `k8s/`.
3. `kubectl apply -k k8s/`; watch `kubectl -n retailpulse rollout status deploy/retailpulse-dashboard`.
4. Trigger a manual retrain job and confirm the dashboard's *Model monitoring* tab updates.
