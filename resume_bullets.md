# Resume Bullets — Insurance FWA Risk Scoring & GenAI Claims Review System

---

## Short Versions (1 line each)

- Built end-to-end insurance FWA risk scoring pipeline using Random Forest / XGBoost achieving ROC-AUC >0.93; deployed interactive Streamlit dashboard with RAG-style claim review using TF-IDF policy retrieval
- Engineered 8 domain-specific features from 5,000 synthetic insurance claims; trained supervised classifiers and Isolation Forest anomaly detection with explainability via feature importance and permutation analysis
- Designed human-in-the-loop AI audit system generating structured claim review reports with policy evidence retrieval and risk-tiered analyst escalation, following CMS FWA compliance guidelines (42 CFR Part 455)

---

## Long Versions (2–3 sentences each)

### ML Engineering Focus
Designed and implemented a full insurance Fraud, Waste & Abuse (FWA) analytics pipeline: generated 5,000 synthetic claims with sigmoid-derived realistic fraud labels (~8% rate), engineered 8 domain-specific features (claim-to-provider ratio, documentation risk, provider risk score), and trained Logistic Regression, Random Forest, XGBoost, and Isolation Forest models achieving best ROC-AUC of 0.93+. Applied class-imbalance strategies (`class_weight='balanced'`, `scale_pos_weight`) and evaluated models on precision, recall, F1, and confusion matrices. Delivered outputs as serialized model artifacts with feature importance visualizations and a JSON metrics report.

### NLP / RAG Focus
Built a RAG-style claim review system requiring no paid APIs: indexed insurance policy rules and 50 synthetic claim documents with TF-IDF vectorization, then used cosine similarity retrieval to surface the top 3 relevant policy citations for each flagged claim. Generated structured review reports containing risk level, key fraud indicators (e.g., "Claim is 4.2x provider average"), retrieved policy evidence with similarity scores, analyst action recommendation, and audit trail. System is fully deterministic and transparent, designed for auditability in regulated insurance environments.

### Data Science / Analytics Focus
Developed an end-to-end data science portfolio project in insurance FWA detection: from synthetic data generation (NumPy/Pandas) through preprocessing, feature engineering, ML modeling (scikit-learn/XGBoost), SHAP/permutation explainability, and an interactive 5-tab Streamlit dashboard surfacing executive KPIs, FWA pattern analysis, model performance visualizations, per-claim risk profiles, and responsible AI documentation. Followed responsible AI practices including bias-free feature selection, human-in-the-loop escalation thresholds, and explicit model limitation disclosures.

---

## Skills Demonstrated

| Category | Technologies |
|---|---|
| Languages | Python 3 |
| ML Libraries | scikit-learn, XGBoost (optional), NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| NLP / Retrieval | TF-IDF, Cosine Similarity (sklearn) |
| Explainability | Feature Importance, Permutation Importance, SHAP (optional) |
| Dashboard | Streamlit |
| Deployment | Joblib model serialization, JSON metrics export |
| MLOps Concepts | Train/test stratified split, class imbalance handling, model versioning |
| Domain Knowledge | Insurance FWA, CMS 42 CFR Part 455, HIPAA awareness, ICD-10/CPT codes |
| Responsible AI | Human-in-the-loop design, bias documentation, auditability |
