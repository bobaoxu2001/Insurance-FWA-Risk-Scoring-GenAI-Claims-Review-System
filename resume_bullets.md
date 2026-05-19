# Resume Bullets — Healthcare Provider FWA Risk Scoring & GenAI Review System

**Project Name:** Healthcare Provider FWA Risk Scoring & GenAI Review System

---

## Short (3 bullets for resume)

- Built a provider-level healthcare FWA analytics pipeline using the public Kaggle Healthcare
  Provider Fraud Detection dataset (558K+ claims across 5,410 providers), joining inpatient,
  outpatient, and beneficiary records to engineer 27 provider-level risk features; best model
  (Random Forest) achieved ROC-AUC 0.9535 and F1 0.6927 on a held-out test set.

- Trained and evaluated Logistic Regression, Random Forest, Gradient Boosting/XGBoost, and
  Isolation Forest models on imbalanced provider fraud labels, reporting ROC-AUC, PR-AUC,
  precision/recall tradeoffs, and performing full threshold sweep analysis to support
  operational capacity planning.

- Developed a RAG-style provider review assistant using TF-IDF retrieval over synthetic
  healthcare billing audit policy rules to generate structured, auditable provider risk
  summaries with quantified risk indicators, policy citations, documentation gaps, and
  suggested analyst actions — supporting human-in-the-loop FWA workflows.

---

## Long (4 bullets)

- Built an end-to-end healthcare provider FWA analytics pipeline using the Kaggle Healthcare
  Provider Fraud Detection Analysis dataset: ingested and joined inpatient, outpatient, and
  beneficiary CSVs using defensive column-matching logic; engineered 25+ provider-level
  features including reimbursement outlier z-scores, inpatient billing ratio, physician count
  diversity, chronic condition complexity, admission duration, and diagnosis/procedure code
  diversity per provider.

- Trained Logistic Regression, Random Forest, Gradient Boosting/XGBoost, and Isolation Forest
  models with class-imbalance handling (balanced class weights, scale_pos_weight); evaluated
  with ROC-AUC, PR-AUC, precision/recall/F1, confusion matrix, and a threshold sweep table
  (threshold 0.05–0.95) mapping the precision/recall tradeoff to analyst review capacity;
  saved annotated model metrics JSON indicating real vs. synthetic data source.

- Built a RAG-style provider audit review assistant: TF-IDF + cosine similarity retrieval
  over synthetic healthcare billing policy rules; generates 15 structured review packets per
  run (10 HIGH, 3 MEDIUM, 2 LOW risk) — each containing quantified risk indicators, retrieved
  policy citations with similarity scores, data and documentation gaps, suggested analyst
  actions, human review notes, and a system limitations / data disclaimer section.

- Implemented a monitoring module producing data quality reports (column-level missing rates,
  duplicate provider checks, class balance), reimbursement distribution charts, fraud rate by
  provider volume quintile, and feature missingness heatmaps; wrapped all outputs in a 6-tab
  Streamlit dashboard with graceful degradation to synthetic demo mode when real Kaggle data
  is unavailable, and clear "Dataset Setup Required" prompts.

---

## GitHub Pinned Repository Description (<=120 chars)

Healthcare provider FWA: Kaggle claims -> 25+ features -> ML models -> RAG audit reviews -> Streamlit dashboard

---

## LinkedIn Featured Project (~100 words)

Built a healthcare provider fraud, waste & abuse (FWA) detection system using the public
Kaggle Healthcare Provider Fraud Detection dataset. The pipeline joins inpatient, outpatient,
and beneficiary records into a provider-level feature table with 25+ engineered features, then
trains ensemble ML models with class-imbalance handling. A RAG-style audit review assistant
uses TF-IDF policy retrieval to generate structured provider risk summaries — with quantified
risk indicators, policy citations, and suggested analyst actions — designed for human-in-the-loop
workflows. Includes model monitoring, a full threshold sweep analysis, and a 6-tab Streamlit
dashboard with graceful synthetic fallback. Tech: Python, scikit-learn, pandas, SHAP, Streamlit.

---

## 30-Second Pitch

"I built a healthcare provider fraud detection system using a public Kaggle dataset of
inpatient, outpatient, and beneficiary claims. The pipeline joins those three data sources
and engineers about 25 provider-level features -- reimbursement outlier scores, inpatient
billing ratio, physician diversity, chronic-condition complexity. I trained Random Forest,
Gradient Boosting, and Logistic Regression models and did a full threshold sweep to show
the precision/recall tradeoff at each operating point. On top of the model, I built a
RAG-style review system that retrieves relevant billing audit rules via TF-IDF and generates
structured, human-readable audit packets for each high-risk provider. The whole thing runs in
Streamlit with a graceful fallback to synthetic data when the Kaggle download is not present."

---

## Copy-Ready Final Resume Version (Real Results, Paste Directly)

> Use these after running the full Kaggle pipeline. Numbers reflect real test-set results.

- Engineered a provider-level healthcare FWA risk scoring pipeline in Python using the public
  Kaggle Healthcare Provider Fraud Detection dataset; joined 558K+ inpatient/outpatient claims
  with beneficiary demographics to produce a 5,410-provider modeling table with 27 risk
  features (reimbursement outlier scores, inpatient billing ratio, admission duration,
  chronic-condition complexity, physician diversity).

- Trained and evaluated Logistic Regression, Random Forest, and Gradient Boosting classifiers
  with class-imbalance handling; best model (Random Forest) achieved ROC-AUC **0.9535**,
  F1 **0.6927**, and Recall **0.7030** at the default threshold (9.35% fraud base rate);
  generated a full precision/recall threshold sweep table to support analyst capacity planning.

- Developed a RAG-style provider audit review assistant using TF-IDF retrieval over synthetic
  healthcare billing audit rules; generates structured review packets — quantified risk
  indicators, policy citations, documentation gaps, suggested analyst actions — for 15
  providers per run, supporting human-in-the-loop FWA workflows.

- Built a model monitoring module (data quality checks, reimbursement distribution drift,
  fraud rate by provider volume bucket) and a 6-tab Streamlit dashboard exposing risk scores,
  model performance, threshold analysis, and provider review summaries to FWA analysts.

---

## 2-Minute Pitch (Manulife / John Hancock LTC FWA Team Framing)

"The project addresses the same core challenge your LTC FWA team faces: given large volumes
of provider billing, how do you systematically identify anomalous patterns before you pay
claims you cannot recover?

I used the Kaggle Healthcare Provider Fraud Detection Analysis dataset -- a public proxy with
the key structural elements: inpatient and outpatient claims, beneficiary demographics
including chronic conditions, physician identifiers, and binary fraud labels at the provider
level. It is not LTC-specific, but the analytical architecture directly transfers.

The feature engineering step joins three data sources and aggregates to the provider level,
computing things like average reimbursement per claim relative to peers, inpatient billing
ratio, physician diversity, and chronic-condition complexity of the patient panel -- the kind
of signals a human FWA analyst would look for manually.

On the modeling side I trained Random Forest and Gradient Boosting models with class-imbalance
handling, and produced a threshold sweep table. The threshold choice matters operationally:
you can tune precision vs. recall based on how many reviewers you have.

The RAG-style review assistant is the human-in-the-loop layer: when a provider scores HIGH,
the system retrieves the most relevant billing audit rules via TF-IDF and generates a
structured packet -- risk indicators, policy citations, documentation gaps, and suggested
analyst actions. This gives the analyst everything they need to make an informed decision
quickly, without the model making the decision for them.

The pipeline has a clean fallback to synthetic data so it can be demonstrated without
proprietary data, and a Streamlit dashboard wraps everything. The architecture mirrors what
a production FWA system would look like before adding real-time scoring and case management
integration."

---

## 5 Likely Interview Questions & Answer Outlines

### Q1: "Walk me through your approach to engineering provider-level features from claim-level data."
- Started with three Kaggle tables: ~558K inpatient+outpatient claims joined with beneficiary demographics
- Joined claims to beneficiary on BeneID to attach chronic conditions, age, death indicators
- Aggregated to Provider granularity using groupby + multiple agg functions
- Engineered 27 features across 5 categories: volume, financial, physician diversity, patient mix, code diversity
- Key design choice: use *peer-relative* metrics (percentile rank, z-scores) so the model can detect outliers regardless of absolute scale

### Q2: "Your Random Forest got 0.95 AUC. Is that real or is there leakage?"
- Honest answer: the metric is real on the held-out 20% test split, but I want to be transparent about caveats
- Labels are provider-level binary flags from Kaggle — they aggregate claim-level noise, so edge cases are ambiguous
- No target leakage in features: all features are computed before label is touched, and I drop the Provider ID
- The 9.35% fraud rate is somewhat balanced compared to real-world prevalence (typically <2%), which inflates discriminability
- In production I'd expect lower AUC and would calibrate against analyst feedback

### Q3: "How would you decide what threshold to use in production?"
- Threshold is an operational decision, not a modeling one
- Generated a full threshold sweep table (`outputs/reports/threshold_analysis.csv`) showing precision, recall, F1, and number flagged at each threshold
- The right threshold depends on reviewer capacity: if I have 10 reviewers who can each handle 50 cases/week, I can flag ~500 providers/week
- Pick threshold where precision-flagged-volume ≈ analyst capacity
- For LTC FWA, I'd lean toward recall-oriented (lower threshold) since missed fraud is expensive and irreversible

### Q4: "Why TF-IDF instead of LLM embeddings for the RAG component?"
- Deliberate choice for this portfolio: TF-IDF is deterministic, auditable, no API keys, runs offline
- For real audit/compliance work, deterministic retrieval is often preferred — analysts can trace exactly why a policy rule was retrieved
- Embeddings (sentence-transformers) would handle synonyms better and I'd swap in a local model for v2
- Both approaches are fundamentally cosine-similarity-on-vectors; the embedding choice is the difference

### Q5: "How would you monitor this model in production?"
- Already prototyped in `src/monitoring.py`:
  - Data drift: monthly distributions of reimbursement amounts, claim volume, provider feature means
  - Class balance: rolling fraud-flag rate to catch label-pipeline issues
  - Feature stability: per-feature missingness and distribution drift vs training baseline
- For real production I'd add:
  - Prediction drift (PSI on score distribution)
  - Calibration drift (Brier score on labeled dispositions)
  - Reviewer feedback loop: flagged-but-not-fraud rate → retraining signal
  - Alerting on threshold-crossing for any of the above
