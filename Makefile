.PHONY: help install data pipeline etl forecast churn segment inventory monitor dashboard mlflow test lint docker-build docker-up docker-down k8s-apply clean
PY ?= python
export PYTHONPATH := src

help:            ## show this help
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:         ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

data:            ## (re)generate the synthetic raw dataset
	$(PY) -m retailpulse.pipeline generate-data

pipeline:        ## run everything: generate data -> ETL -> models -> monitoring
	$(PY) -m retailpulse.pipeline all --with-data

etl forecast churn segment inventory monitor:  ## run a single stage
	$(PY) -m retailpulse.pipeline $@

dashboard:       ## launch the Streamlit dashboard on :8501
	streamlit run app/dashboard.py

mlflow:          ## launch the MLflow UI on :5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:            ## run the test-suite
	$(PY) -m pytest -q

lint:            ## static analysis
	ruff check src tests app

docker-build:    ## build the image
	docker build -t retailpulse:latest .

docker-up:       ## full stack with docker compose
	docker compose up --build

docker-down:
	docker compose down

k8s-apply:       ## deploy to the current kube-context
	kubectl apply -k k8s/

clean:           ## remove generated artefacts (keeps raw data)
	rm -rf data/processed/*.parquet artifacts/* reports/* mlflow.db mlruns .pytest_cache
