# Data dictionary

All raw files are UTF-8 CSV in `data/raw/`. Dates are ISO `YYYY-MM-DD`.

## transactions.csv — one row per order line
| column | type | notes |
|---|---|---|
| order_id | str | groups lines of one basket |
| order_date | date | |
| customer_id | str, nullable | null = guest checkout (kept for demand, excluded from customer analytics) |
| store_id | str | FK → stores |
| product_id | str | FK → products |
| quantity | int | negative = return |
| unit_price | float | list price at sale time |
| discount_pct | int | 0–90 |

Derived in ETL: `category`, `unit_cost`, `is_return`, `is_guest`, `net_unit_price`, `revenue`, `cogs`, `margin`.

## customers.csv
`customer_id, signup_date, age, gender (F/M/O), region, loyalty_member (Y/N), acquisition_channel`

## products.csv
`product_id, product_name, category, list_price, unit_cost`

## stores.csv
`store_id, region, size_sqft`

## inventory.csv — latest stock snapshot
`product_id, snapshot_date, on_hand, lead_time_days, supplier_id`

## Generated outputs (`data/processed/`, `reports/`, `artifacts/`)
| file | content |
|---|---|
| demand_daily.parquet | dense daily units/revenue per forecast series |
| forecast.parquet / forecast_metrics / forecast_backtest | forecasts with intervals; model comparison; back-test predictions |
| product_forecast.parquet | SKU-level forecast used by the inventory optimiser |
| segments / segment_profile | RFM segment per customer; segment KPIs |
| churn_scores.parquet | churn probability + risk band (Low <0.3, Medium <0.6, High) per customer |
| churn_importance.parquet | permutation importance (ΔAUC) |
| inventory_plan.parquet | SS, ROP, EOQ, days of cover, stock-out risk, recommended order qty |
| feature_drift / demand_drift | drift diagnostics |
| reports/*.json | data quality, metrics, monitoring verdict |
| artifacts/churn_model.joblib, segmentation.joblib | fitted sklearn pipelines (+ churn_meta.json with threshold) |
