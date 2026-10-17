# syntax=docker/dockerfile:1.6
# One image, three roles: dashboard (default), pipeline job (`retailpulse all`), tests.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    RETAILPULSE_HOME=/app \
    PYTHONPATH=/app/src \
    PATH="/app/bin:${PATH}"

# libgomp: required by Prophet/cmdstan + scikit-learn; curl: container health checks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config ./config
COPY src ./src
COPY app ./app
COPY data ./data
COPY data/processed ./data/processed
COPY docker/entrypoint.sh /app/bin/entrypoint.sh
COPY docker/retailpulse /app/bin/retailpulse

# run as an unprivileged user; data/artifacts/reports are volumes
RUN useradd --create-home --uid 10001 rp \
    && chmod +x /app/bin/entrypoint.sh /app/bin/retailpulse \
    && mkdir -p data/raw data/processed artifacts reports \
    && chown -R rp:rp /app
USER rp

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/app/bin/entrypoint.sh"]
CMD ["dashboard"]
