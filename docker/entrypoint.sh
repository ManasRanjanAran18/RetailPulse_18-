#!/usr/bin/env bash
# Container entrypoint:  dashboard | pipeline [args] | any other command
set -euo pipefail

case "${1:-dashboard}" in
  dashboard)
    # first boot on an empty volume: build everything so the UI is never empty
    if [ ! -f /app/data/processed/forecast.parquet ] && [ "${AUTO_BOOTSTRAP:-true}" = "true" ]; then
      echo "[entrypoint] no processed data found -> running full pipeline (AUTO_BOOTSTRAP=true)"
      [ -f /app/data/raw/transactions.csv ] || python -m retailpulse.pipeline generate-data
      python -m retailpulse.pipeline all
    fi
    exec streamlit run /app/app/dashboard.py \
      --server.port=8501 --server.address=0.0.0.0 --server.headless=true \
      --browser.gatherUsageStats=false
    ;;
  pipeline)
    shift
    exec python -m retailpulse.pipeline "${@:-all}"
    ;;
  *)
    exec "$@"
    ;;
esac
