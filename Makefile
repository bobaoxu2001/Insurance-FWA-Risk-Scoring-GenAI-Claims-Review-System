PYTHON ?= python3.11

.PHONY: help install install-llm download-real download-partb real-pipeline \
        graph-features temporal-eval synthetic-demo fairness llm-reviews \
        tune-rf tune-gb psi feedback oig-leie cms-ltc partb-ltc partb-all \
        docker docker-llm dashboard test clean-outputs

help:
	@echo "Available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install          Install core Python dependencies"
	@echo "    install-llm      Install optional LLM/semantic-retrieval deps"
	@echo "    download-real    Download OIG LEIE + CMS Nursing Home data (~25MB total)"
	@echo ""
	@echo "  Real-data pipelines:"
	@echo "    oig-leie         Analyze 83K real federal-fraud exclusion records"
	@echo "    cms-ltc          Train LTC FWA model on 14,699 real US nursing homes"
	@echo "    download-partb   Download Medicare Physician 2023 (~470 MB)"
	@echo "    partb-ltc        Train on 193K LTC providers w/ real NPI LEIE labels"
	@echo "    partb-all        Train on full 1.26M Medicare Part B universe"
	@echo ""
	@echo "  Kaggle pipelines:"
	@echo "    real-pipeline    Full pipeline on real Kaggle data (random 80/20 split)"
	@echo "    graph-features   Add 5 bipartite-graph features to provider table"
	@echo "    temporal-eval    Re-evaluate models with chronological train/test split"
	@echo "    synthetic-demo   Pipeline on synthetic fallback data"
	@echo ""
	@echo "  Production-ML modules:"
	@echo "    tune-rf          RandomizedSearchCV hyperparameter tuning"
	@echo "    psi              PSI drift detection (train vs test under temporal split)"
	@echo "    fairness         Demographic disparate-impact audit on patient panels"
	@echo "    feedback         Analyst-disposition feedback loop + optional retrain"
	@echo "    llm-reviews      Generate semantic-retrieval + flan-t5 LLM provider reviews"
	@echo ""
	@echo "  Containerization:"
	@echo "    docker           Build CPU-only Docker image (core deps)"
	@echo "    docker-llm       Build Docker image with torch + transformers"
	@echo ""
	@echo "  Run:"
	@echo "    dashboard        Launch Streamlit dashboard"
	@echo "    test             pytest test suite (17 checks)"
	@echo "    clean-outputs    Remove generated outputs (keep raw + processed data)"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-llm:
	$(PYTHON) -m pip install -r requirements-llm.txt

download-real:
	bash scripts/download_real_data.sh

oig-leie:
	$(PYTHON) src/oig_leie_analysis.py

cms-ltc:
	$(PYTHON) src/cms_ltc_pipeline.py

download-partb:
	@mkdir -p data/raw/cms_partb
	@if [ ! -s data/raw/cms_partb/medicare_physician_by_provider_2023.csv ]; then \
	    echo "Downloading Medicare Physician & Other Practitioners 2023 (~470 MB)..."; \
	    curl -L -o data/raw/cms_partb/medicare_physician_by_provider_2023.csv \
	        "https://data.cms.gov/sites/default/files/2025-04/22edfd1e-d17a-4478-ad6b-92cac2a5a3c4/MUP_PHY_R25_P05_V20_D23_Prov.csv"; \
	else \
	    echo "Already present: data/raw/cms_partb/medicare_physician_by_provider_2023.csv"; \
	fi

partb-ltc:
	$(PYTHON) src/medicare_partb_pipeline.py --population ltc

partb-all:
	$(PYTHON) src/medicare_partb_pipeline.py --population all

docker:
	docker build -t fwa-portfolio .

docker-llm:
	docker build --build-arg INSTALL_LLM=1 -t fwa-portfolio-llm .

real-pipeline:
	$(PYTHON) src/data_ingestion.py
	$(PYTHON) src/provider_feature_engineering.py
	$(PYTHON) src/graph_features.py
	$(PYTHON) src/modeling.py --split random
	$(PYTHON) src/explainability.py
	$(PYTHON) src/rag_claim_review.py
	$(PYTHON) src/monitoring.py

graph-features:
	$(PYTHON) src/graph_features.py

temporal-eval:
	$(PYTHON) src/modeling.py --split temporal

tune-rf:
	$(PYTHON) src/hyperparameter_tuning.py --model rf --n-iter 30

tune-gb:
	$(PYTHON) src/hyperparameter_tuning.py --model gb --n-iter 30

psi:
	$(PYTHON) src/psi_drift.py

feedback:
	$(PYTHON) src/feedback_loop.py --retrain

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
