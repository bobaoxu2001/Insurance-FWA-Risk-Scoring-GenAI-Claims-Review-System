PYTHON ?= python3.11

.PHONY: help install install-llm real-pipeline temporal-eval synthetic-demo \
        fairness llm-reviews dashboard test clean-outputs

help:
	@echo "Available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install          Install core Python dependencies"
	@echo "    install-llm      Install optional LLM/semantic-retrieval deps"
	@echo ""
	@echo "  Pipelines:"
	@echo "    real-pipeline    Full pipeline on real Kaggle data (random 80/20 split)"
	@echo "    temporal-eval    Re-evaluate models with chronological train/test split"
	@echo "    synthetic-demo   Pipeline on synthetic fallback data"
	@echo ""
	@echo "  Analyses:"
	@echo "    fairness         Demographic disparate-impact audit on patient panels"
	@echo "    llm-reviews      Generate semantic-retrieval + flan-t5 LLM provider reviews"
	@echo ""
	@echo "  Run:"
	@echo "    dashboard        Launch Streamlit dashboard"
	@echo "    test             pytest test suite (17 checks)"
	@echo "    clean-outputs    Remove generated outputs (keep raw + processed data)"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-llm:
	$(PYTHON) -m pip install -r requirements-llm.txt

real-pipeline:
	$(PYTHON) src/data_ingestion.py
	$(PYTHON) src/provider_feature_engineering.py
	$(PYTHON) src/modeling.py --split random
	$(PYTHON) src/explainability.py
	$(PYTHON) src/rag_claim_review.py
	$(PYTHON) src/monitoring.py

temporal-eval:
	$(PYTHON) src/modeling.py --split temporal

fairness:
	$(PYTHON) src/fairness_audit.py

llm-reviews:
	$(PYTHON) -m src.llm_review --tier 2 --mixed

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
