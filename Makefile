PYTHON ?= python3.11

.PHONY: help install real-pipeline synthetic-demo dashboard test clean-outputs

help:
	@echo "Available targets:"
	@echo "  install         Install Python dependencies"
	@echo "  real-pipeline   Run full pipeline on real Kaggle data (requires data/raw/*.csv)"
	@echo "  synthetic-demo  Run pipeline on synthetic fallback data"
	@echo "  dashboard       Launch Streamlit dashboard"
	@echo "  test            Run pytest test suite"
	@echo "  clean-outputs   Remove generated outputs (keep raw + processed data)"

install:
	$(PYTHON) -m pip install -r requirements.txt

real-pipeline:
	$(PYTHON) src/data_ingestion.py
	$(PYTHON) src/provider_feature_engineering.py
	$(PYTHON) src/modeling.py
	$(PYTHON) src/explainability.py
	$(PYTHON) src/rag_claim_review.py
	$(PYTHON) src/monitoring.py

synthetic-demo:
	$(PYTHON) src/synthetic_data_generation.py
	$(PYTHON) src/preprocessing.py
	$(PYTHON) src/feature_engineering.py
	$(PYTHON) src/modeling.py
	$(PYTHON) src/explainability.py
	$(PYTHON) src/rag_claim_review.py
	$(PYTHON) src/monitoring.py

dashboard:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest -q

clean-outputs:
	rm -rf outputs/figures/*.png outputs/reports/*.csv outputs/reports/*.json outputs/sample_reviews/*.txt outputs/models/*.pkl
