# Healthcare Provider FWA Risk Scoring & GenAI Review System

> A healthcare fraud, waste, and abuse analytics project using real public claims data
> combined with ML risk scoring and RAG-style provider review.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![CI](https://github.com/bobaoxu2001/Insurance-FWA-Risk-Scoring-GenAI-Claims-Review-System/actions/workflows/ci.yml/badge.svg)](https://github.com/bobaoxu2001/Insurance-FWA-Risk-Scoring-GenAI-Claims-Review-System/actions/workflows/ci.yml)

## TL;DR

Provider-level healthcare FWA pipeline trained on **three real public datasets**:

| Dataset | Real? | What it gives us |
|---|---|---|
| Kaggle Healthcare Provider Fraud (Train) | Real Medicare claim structure, anonymized IDs | 5,410 providers × 32 features, 506 fraud labels |
| **CMS Nursing Home Provider Information** | **Real LTC providers with real names + CCNs** | 14,699 US nursing homes, 22.55% flagged (abuse / SFF / fines / payment denials) |
| **HHS-OIG List of Excluded Individuals/Entities** | **Real federal fraud exclusions** | 83,256 actual exclusions, 1,818 LTC-specific (1,447 HHA, 256 SNF, 77 Hospice) |
| **Medicare Physician & Other Practitioners 2023** ⋈ **OIG LEIE** | **Real NPI-keyed Medicare billing + real federal-exclusion labels** | 1.26M real provider NPIs; LTC subset 193K with 80 LEIE-matched fraud labels; full universe 207 matches |

Two complete modeling pipelines, real LTC fraud labels, cross-referenced via business-name matching, with semantic + LLM RAG, temporal validation, PSI drift, fairness audit, and reviewer-feedback loop.

### Headline Numbers

| | |
|---|---|
| **Kaggle pipeline (random split, RF)** | ROC-AUC **0.9526** · PR-AUC **0.7343** · F1 0.6731 · Brier 0.042 |
| **Kaggle pipeline (temporal split, GB)** | ROC-AUC **0.8899** · PR-AUC **0.5529** — the realistic deployment number |
| **CMS Nursing Home pipeline (GB)** | ROC-AUC **0.8538** · PR-AUC **0.6668** · CV PR-AUC **0.6816 ± 0.0078** — on REAL US nursing homes with REAL labels |
| **CMS ↔ OIG LEIE cross-reference** | 4 modeled providers whose legal business name matches a federal exclusion record (candidates for verification) |
| **Tuned RF (Kaggle, 5-fold CV, PR-AUC)** | **0.7491** (Δ +0.015 vs default hyperparameters) |
| **Drift (Kaggle temporal)** | 13 / 32 features cross PSI 0.25 → would trigger retraining alert |
| **RAG corpus** | Synthetic policy rules + **real federal exclusion-code taxonomy** from §1128 (15 codes, 83K active cases) |
| **RAG layer** | Three tiers: TF-IDF + template → semantic dense retrieval (sentence-transformers) → semantic + local LLM generation (flan-t5-base, deterministic decoding) |
| **Fairness** | Patient-panel demographic audit with 4/5ths-rule disparate-impact check |
| **Feedback loop** | Analyst-disposition CSV → model-vs-analyst agreement metrics → retrain trigger |
| **Reproducibility** | `make download-real && make real-pipeline && make cms-ltc` · 37 pytest checks · GitHub Actions CI |

> **Disclaimer:** Public educational dataset only — not Manulife/John Hancock data, not LTC-specific, no PHI. RAG policy text is synthetic. See §4 Data Disclaimer.

---

## 1. Project Overview

An end-to-end FWA analytics pipeline that:

- Ingests and joins the **Kaggle Healthcare Provider Fraud Detection Analysis** dataset
  (inpatient claims + outpatient claims + beneficiary demographics)
- Engineers **27 provider-level risk features** (volume, reimbursement, physician patterns,
  chronic-condition complexity, inpatient ratio, admission duration, code diversity)
- Trains **Logistic Regression, Random Forest, Gradient Boosting / XGBoost, and Isolation
  Forest** models with class-imbalance handling
- Evaluates with ROC-AUC, PR-AUC, F1, and a full **threshold sweep** analysis
- Generates **provider-level feature-importance explanations** (model importances by default; SHAP if `shap` is installed)
- Produces **RAG-style provider audit review packets** using TF-IDF retrieval over
  synthetic healthcare billing audit rules
- Runs **model monitoring and data-quality checks** on the provider feature table
- Serves a **6-tab Streamlit dashboard** for executives and FWA analysts

**Fallback mode:** When Kaggle data is not available, the pipeline runs on a synthetic
5,000-claim dataset with a hidden latent fraud-intent variable (leakage-aware) — for
portability only; real headline metrics below come from the Kaggle dataset.

---

## 2. Real Datasets

This project trains on **three real public datasets** — not one. The Kaggle dataset is the legacy pipeline; the OIG LEIE and CMS Nursing Home Compare additions move the project from "anonymized educational data" to "real LTC providers with real names and real federal-fraud exclusion records."

### Dataset 1 — Kaggle Healthcare Provider Fraud Detection (anonymized claims)

- Real Medicare claim structure, anonymized provider/beneficiary IDs
- 558K+ inpatient/outpatient claims, 138K beneficiaries → 5,410 providers
- Binary `PotentialFraud` labels (9.35% fraud rate)
- Used for the primary modeling pipeline (Sections 9–14)

### Dataset 2 — CMS Nursing Home Provider Information (real LTC providers)

- **Real, named US nursing homes** with CMS Certification Numbers (CCNs)
- 14,699 facilities, downloaded from `data.cms.gov` via paginated API
- Real labels:
  - `Abuse Icon == Y` (1,497 cited for resident abuse)
  - `Special Focus Status` ∈ {SFF, SFF Candidate} — CMS's official quality watch list (526 facilities)
  - `Number of Fines ≥ 5` (515 facilities)
  - `Number of Payment Denials ≥ 1` (1,950 facilities)
- Combined risk flag: 3,315 / 14,699 = **22.55% flag rate**
- Features include ownership type, beds, occupancy, 5-star ratings, staffing hours, turnover, deficiency counts, chain affiliation
- This is the closest publicly available approximation to the data an LTC FWA analytics team would actually work with.

**Pipeline (`src/cms_ltc_pipeline.py`):**

| Model | ROC-AUC | PR-AUC | F1 | CV PR-AUC (mean ± std) |
|---|---|---|---|---|
| Logistic Regression | 0.8485 | 0.6475 | 0.6001 | 0.6667 ± 0.0047 |
| Random Forest | 0.8503 | 0.6479 | 0.6067 | 0.6653 ± 0.0074 |
| **Gradient Boosting** ⭐ | **0.8538** | **0.6668** | 0.5296 | **0.6816 ± 0.0078** |

Note that AUC on real LTC data (~0.85) is meaningfully lower than the Kaggle pipeline's 0.95 — and PR-AUC of 0.67 with stable CV (std ≈ 0.008) is the kind of result one would expect on a genuine LTC FWA problem. This is the **more credible number to cite in interviews**.

### Dataset 3 — HHS-OIG List of Excluded Individuals/Entities (LEIE)

- **Real federal healthcare-fraud exclusion records** (not synthetic, not labels-from-rules)
- 83,256 currently-excluded individuals and entities, 8,608 with real NPIs
- Downloaded directly from `oig.hhs.gov/exclusions/downloadables/UPDATED.csv`
- **1,818 LTC-specific exclusions** (1,447 Home Health Agency, 256 Skilled Nursing Facility, 77 Hospice, 38 Nursing Firm)
- Each record cites the **real legal authority** under §1128 of the Social Security Act:
  - `1128a1` — Conviction of program-related crimes (25,760 cases)
  - `1128a2` — Conviction relating to patient abuse (8,073 cases)
  - `1128a3` — Felony health-care-fraud conviction (5,826 cases)
  - `1128b4` — License revocation/suspension (33,136 cases — most common)
  - `1128b7` — Anti-Kickback Statute violations (735 cases)
  - …(15 codes total, see `data/documents/oig_exclusion_codes.txt`)

**Integration:**

- `src/oig_leie_analysis.py` produces real fraud descriptive analytics (top specialties, annual trends, LTC subset)
- The exclusion-code taxonomy is auto-emitted to `data/documents/oig_exclusion_codes.txt` and **indexed by the RAG retriever alongside the synthetic policy rules** — so when the LLM generates an audit summary, it can cite **real federal authority** instead of synthetic rules.
- The CMS LTC pipeline cross-references each modeled provider's legal business name against LEIE-excluded entities → `outputs/reports/cms_ltc_leie_overlap.csv` (4 candidate matches — names like "MEMORIAL MEDICAL CENTER" appear multiple times in LEIE; require human verification).

> **Honesty note:** the direct NPI overlap between LEIE and the Kaggle claims is zero because Kaggle's physician identifiers are anonymized (`PHY412132`-style). This is exposed in `src/oig_leie_analysis.py` and reported in the output rather than hidden.

### Dataset 4 — Medicare Physician & Other Practitioners 2023 ⋈ OIG LEIE (real NPI fraud labels)

The strongest real-data join in the project: cross-references real Medicare provider billing with real federal fraud exclusions to produce real NPI-keyed fraud labels.

- **Source 1:** `data.cms.gov/sites/default/files/.../MUP_PHY_R25_P05_V20_D23_Prov.csv` — Medicare Physician & Other Practitioners by Provider, calendar year 2023, ~472 MB / 1.26M real US providers
- **Source 2:** HHS-OIG LEIE NPI list (8,429 real exclusion NPIs)
- **Join logic:** `Rndrng_NPI` ∈ LEIE NPIs → `excluded_for_fraud = 1`
- **LTC subset:** Nurse Practitioner + Geriatric Medicine + Hospice & Palliative Care + Geriatric Psychiatry = 193,290 providers, **80 with real LEIE-fraud labels**
- **Full universe:** 1,259,343 providers, **207 with real LEIE-fraud labels**
- **Features:** 49 numeric — Medicare billing volume (HCPCS codes, services, charges, payments), beneficiary demographics (age bands, race counts, dual-eligible, chronic-condition prevalences, average risk score)
- **Engineered ratios:** payment-per-beneficiary, services-per-beneficiary, allowed-to-submitted ratio, dual-share

This is the only pipeline in the project where **every positive case is a real US provider with real Medicare billing data who appears on the federal HHS-OIG exclusion list**. No synthetic labels, no aggregated proxies.

The class imbalance is extreme (~0.04% in the LTC subset, ~0.02% in the full universe), so this pipeline is the most-realistic stress test of the entire methodology. Run: `make partb-ltc` (~3 min on CPU) or `make partb-all` (~10 min).

### Reproducibility

```bash
make download-real     # ~25 MB OIG + CMS Nursing Home in 30s
# Medicare Part B 2023 download (~470 MB, 1-2 min on good connection):
make partb-ltc         # trains the NPI-keyed real-fraud pipeline

make oig-leie          # OIG LEIE descriptive analysis + RAG taxonomy
make cms-ltc           # CMS Nursing Home pipeline
```

Raw real-data CSVs are gitignored (downloadable, not committed). Processed modeling tables are committed.

---

## 3. Kaggle Pipeline Results

Full pipeline run on the **Kaggle Healthcare Provider Fraud Detection Analysis** dataset (Train split):

| Source | Providers | Features | Fraud Rate |
|---|---|---|---|
| Train_Beneficiarydata | 138,556 beneficiaries | — | — |
| Train_Inpatientdata | 40,474 claims | — | — |
| Train_Outpatientdata | 517,737 claims | — | — |
| **Provider table (processed)** | **5,410 providers** | **32** (27 aggregations + 5 graph) | **9.35% (506 fraudulent)** |

**Model performance — held-out 80/20 stratified test set (with graph features):**

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier | Log-loss |
|---|---|---|---|---|---|---|---|
| **Random Forest** ⭐ | **0.9526** | **0.7343** | 0.6731 | 0.6731 | 0.6731 | **0.0421** | 0.1958 |
| Gradient Boosting | 0.9448 | 0.6479 | 0.6557 | 0.6053 | 0.7156 | 0.0457 | **0.1660** |
| Logistic Regression | 0.9286 | 0.7098 | 0.5611 | 0.7679 | 0.4416 | 0.1051 | 0.4093 |
| Isolation Forest | anomaly scoring | — | — | — | — | — | — |

**5-fold stratified cross-validation** (full dataset, scoring = ROC-AUC):

| Model | CV-AUC (mean ± std) | Per-fold AUC |
|---|---|---|
| **Random Forest** ⭐ | **0.9499 ± 0.0145** | 0.962, 0.958, 0.960, 0.923, 0.948 |
| Gradient Boosting | 0.9525 ± 0.0138 | 0.968, 0.961, 0.958, 0.928, 0.947 |
| Logistic Regression | 0.9146 ± 0.0287 | 0.950, 0.949, 0.889, 0.887, 0.897 |

Adding graph features lifted LR's PR-AUC from 0.638 → 0.710 (+11%) — the largest jump because graph features encode signal a linear model otherwise cannot get. RF and GB were already saturating on the non-graph features. Random Forest is selected as the production model by **PR-AUC on the held-out set** (imbalance-aware metric).

### Hyperparameter tuning

`src/hyperparameter_tuning.py` runs RandomizedSearchCV (20 iter, 5-fold stratified CV, PR-AUC scoring) on the chosen model class:

| | Baseline RF | Tuned RF (`make tune-rf`) |
|---|---|---|
| PR-AUC | 0.7343 | **0.7491** |
| Best params | n_estimators=200, max_depth=12, class_weight=balanced (defaults) | n_estimators=500, max_depth=15, min_samples_leaf=8, max_features=0.3 |

Tuned model and full search log persisted to `outputs/models/best_fwa_model_tuned_rf.pkl` and `outputs/reports/hp_tuning_results.json`.

**Top 5 features by Random Forest importance:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `max_admission_duration` | 0.144 |
| 2 | `total_reimbursed` | 0.134 |
| 3 | `total_deductible_amount` | 0.098 |
| 4 | `total_inpatient_reimbursed` | 0.075 |
| 5 | `inpatient_claim_count` | 0.062 |

> All metrics serialized to `outputs/reports/model_metrics.json` with `"_data_source": "real_kaggle_provider"` and an `_evaluation` block documenting the test split, CV scheme, and selection metric. Classification report saved to `outputs/reports/classification_report.txt`. Calibration plot at `outputs/figures/calibration_curve.png`.

### Temporal-split evaluation — the realistic number

The random 80/20 split shares timestamps between train and test, which is unrealistic for production. We re-trained and re-evaluated with a **chronological split**: train on providers whose median claim date is in the earliest 80% of the timeline, test on the most recent 20%. The drop is large and honest:

| Model | Random ROC-AUC | **Temporal ROC-AUC** | Random PR-AUC | **Temporal PR-AUC** |
|---|---|---|---|---|
| Logistic Regression | 0.9286 | **0.8424** | 0.7098 | **0.2905** |
| Random Forest | 0.9526 | **0.8828** | 0.7343 | **0.3921** |
| **Gradient Boosting** ⭐ | 0.9448 | **0.8899** | 0.6479 | **0.5529** |

Under temporal evaluation **Gradient Boosting wins decisively on PR-AUC** (0.55 vs 0.39 for RF). GB's lower-variance score distribution generalizes better across the time gap. RF still wins on ROC-AUC but is over-confident on out-of-time data (visible in the calibration curve). This is the kind of finding that only surfaces with the right validation methodology — and it changes the production-model choice. Graph features lifted GB's temporal PR-AUC from 0.41 (pre-graph) to 0.55 (+35% relative), confirming that collusion-ring signals generalize across time better than provider-level aggregations alone.

Reproduce with: `make temporal-eval` (writes `outputs/reports/model_metrics_temporal.json`).

---

## 4. Business Problem & FWA Context

> "How do we systematically identify which healthcare providers are billing anomalously —
> before we pay claims we can't recover — while protecting legitimate providers and
> maintaining audit explainability for regulators?"

Healthcare Fraud, Waste, and Abuse (FWA) is a persistent source of improper payments
across public and private insurance programs in the US. Detection is difficult because
fraudulent providers are a small minority of a large population, the patterns evolve
over time, and most cases require human review for regulatory defensibility. Common
provider-level FWA patterns include:

- **Upcoding:** billing a higher-acuity service (e.g. inpatient) than actually rendered
- **Unbundling:** splitting a single procedure into multiple separately-billed components
- **Phantom billing:** charging for services not rendered
- **Duplicate billing:** submitting the same service claim multiple times
- **Medically unnecessary services:** high reimbursement for treatments without clinical justification
- **Physician ID fraud:** billing under multiple physician NPIs to obscure patterns

---

## 5. Kaggle Dataset Details

**Name:** Healthcare Provider Fraud Detection Analysis
**Source:** Kaggle (public educational dataset)
**URL:** https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis

### Data Disclaimer

> **IMPORTANT:** This dataset is a public educational resource only.
> - NOT Manulife data
> - NOT John Hancock data
> - NOT Long Term Care-specific claims data
> - No real patient records, clinical documents, or insurance company data
> - No PHI (Protected Health Information)
> - Synthetic text used for RAG demo purposes only

The dataset contains:
- **Beneficiary demographics:** DOB, DOD, chronic conditions (14 flags), race, state
- **Inpatient claims:** admission/discharge dates, reimbursement, deductible, attending/operating physicians, diagnosis & procedure codes
- **Outpatient claims:** same structure, without admission dates
- **Provider labels:** binary PotentialFraud flag (Yes/No) per provider

---

## 6. Why this is relevant to LTC FWA

The Kaggle dataset is **not** Long Term Care-specific — it covers Medicare-style
inpatient/outpatient claims. However, the analytical workflow transfers almost
directly to LTC FWA work:

- **Provider-level risk scoring.** LTC FWA programs review providers and facilities,
  not individual claims in isolation. The aggregation pattern here (claim → provider
  feature table → binary risk score) is the same shape an LTC program would use.
- **Reimbursement anomaly detection.** Outlier reimbursement, abnormal admission
  duration, and inpatient-billing ratio are exactly the signals LTC FWA analysts
  watch for in skilled-nursing and home-health billing.
- **Utilization patterns.** Unique beneficiaries served, physician diversity, and
  chronic-condition complexity translate directly into LTC member-mix and care-team
  composition checks.
- **Documentation / audit evidence.** The RAG-style review packet — risk indicators,
  retrieved policy citations, documentation gaps — mirrors the structured case file
  an LTC compliance / SIU reviewer assembles before acting on a flagged provider.
- **Human-in-the-loop review.** No automated payment hold; every HIGH flag is
  surfaced to a licensed analyst. This is the same operating model required for
  LTC fraud findings to hold up under regulator review.

In short: different claim taxonomy, same analytical scaffolding, same audit deliverable.

---

## 7. Why This Dataset

This is the closest freely-available public proxy for healthcare FWA analytics:

| Attribute | This dataset | Real-world FWA |
|---|---|---|
| Unit of analysis | Provider-level | Provider-level (most programs) |
| Label type | Binary fraud flag | Binary or scored |
| Claim types | Inpatient + Outpatient | Full continuum |
| Beneficiary data | Demographics + chronic conditions | Demographics + clinical |
| Physician data | Attending + Operating + Other | Full credentialing records |

---

## 8. Solution Architecture

```mermaid
graph TD
    A[Raw CSV Files<br>Inpatient + Outpatient + Beneficiary] --> B[data_ingestion.py<br>File discovery & validation]
    B --> C[provider_feature_engineering.py<br>27 provider-level features]
    C --> D[modeling.py<br>LR + RF + GB/XGB + IsoForest]
    D --> E[Risk Scores per Provider]
    E --> F[explainability.py<br>Feature importance + provider explanations]
    E --> G[rag_claim_review.py<br>TF-IDF policy retrieval + review templates]
    D --> H[monitoring.py<br>Data quality + model monitoring]
    F --> I[Streamlit Dashboard<br>6-tab analyst interface]
    G --> I
    H --> I
```

---

## 9. Data Pipeline

```
data/raw/
    Train_Beneficiarydata-*.csv
    Train_Inpatientdata-*.csv
    Train_Outpatientdata-*.csv
    Train-*.csv  (provider labels)
         |
         v
src/data_ingestion.py        — find files, validate columns, load DataFrames
         |
         v
src/provider_feature_engineering.py
         — join inpatient + outpatient on BeneID to beneficiary
         — aggregate to provider level (27 features)
         — attach fraud labels
         — save data/processed/provider_modeling_table.csv
         |
         v
src/modeling.py              — train + evaluate + save models
src/explainability.py        — feature importance + provider explanations
src/rag_claim_review.py      — generate 15 provider audit reviews
src/monitoring.py            — data quality + monitoring charts
```

---

## 10. Feature Engineering

Features engineered at provider level from joined inpatient/outpatient/beneficiary data:

**Volume features:**
- `total_claims`, `inpatient_claim_count`, `outpatient_claim_count`
- `inpatient_ratio` = inpatient / total
- `unique_beneficiaries`
- `claim_frequency_per_beneficiary`

**Physician features:**
- `unique_attending_physicians`, `unique_operating_physicians`, `unique_other_physicians`

**Financial features:**
- `total_inpatient_reimbursed`, `total_outpatient_reimbursed`, `total_reimbursed`
- `avg_reimbursed_per_claim`, `reimbursement_per_beneficiary`
- `total_deductible_amount`, `inpatient_reimbursement_share`

**Patient demographic features:**
- `avg_patient_age`, `std_patient_age`
- `death_rate` (fraction of patients with recorded death)
- `avg_chronic_conditions` (mean # chronic conditions per patient)

**Duration features** (inpatient only):
- `avg_admission_duration`, `max_admission_duration` (days)

**Code diversity features:**
- `diagnosis_code_diversity` (mean unique ICD codes per claim)
- `procedure_code_diversity` (mean unique CPT codes per claim)

**Risk / outlier features:**
- `provider_volume_percentile`
- `reimbursement_outlier_score` = z-score of avg reimbursement vs all providers

**Graph features (`src/graph_features.py`):**

FWA schemes are often collusive: small groups of providers share beneficiaries (kickbacks, identity fraud) or share attending physicians (NPI laundering). Provider-level aggregations miss these — a single provider's history can look perfectly normal while the *network* reveals a ring.

Built from two bipartite graphs (Provider ↔ Beneficiary, Provider ↔ AttendingPhysician):

- `beneficiary_sharing_rate` — fraction of the provider's beneficiaries who also see ≥1 other provider
- `avg_co_provider_count` — average number of *other* providers that this provider's beneficiaries also see
- `physician_sharing_rate` — fraction of the provider's attending physicians who also bill under another Provider NPI (classic upcoding / NPI laundering signal)
- `provider_clustering_coefficient` — local clustering coefficient in the projected provider-provider graph (high = part of a tightly-connected ring of providers sharing patients)
- `provider_pagerank` — PageRank on the same projected graph, weighted by shared-beneficiary count

Adding these 5 features lifted LR PR-AUC from 0.638 → 0.710 and GB temporal PR-AUC from 0.41 → 0.55 (the largest single-feature-group gain in this project).

---

## 11. Modeling Approach

| Model | Notes |
|---|---|
| Logistic Regression | Baseline; class_weight=balanced |
| Random Forest | n_estimators=200; class_weight=balanced; max_depth=12 |
| Gradient Boosting / XGBoost | scale_pos_weight for imbalance; 200 trees |
| Isolation Forest | Unsupervised anomaly detection baseline |

Class imbalance is handled with `class_weight="balanced"` for supervised models and
`contamination=fraud_base_rate` for Isolation Forest.

---

## 12. Model Evaluation

Two-stage evaluation guards against split-of-the-day artifacts:

1. **Held-out 80/20 stratified test set** — for the headline operating-point metrics (P/R/F1, calibration).
2. **5-fold stratified cross-validation** on the full dataset — for a stability estimate (mean ± std ROC-AUC).

Per-model metrics persisted to `outputs/reports/model_metrics.json`:

| Metric | What it measures | Why it matters for FWA |
|---|---|---|
| ROC-AUC | Ranking quality across all thresholds | Overall separability |
| **PR-AUC** | Area under precision-recall curve | More informative than ROC under class imbalance — used for model selection |
| Precision @ threshold | True flagged / total flagged | Analyst capacity / false-positive cost |
| Recall @ threshold | True flagged / total fraud | Fraud catch rate / financial exposure |
| F1 @ threshold | Harmonic mean of P and R | Balanced summary |
| **Brier score** | Mean squared probability error | Calibration of the predicted probabilities |
| **Log-loss** | Cross-entropy | Calibration + ranking combined |
| **CV-AUC (mean ± std)** | Stability across 5 folds | Detects overfit to a lucky split |
| Threshold sweep table | P/R/F1/n_flagged across 0.05–0.95 | Operations picks threshold by reviewer capacity |

See **Section 2 — Real Dataset Results** above for headline numbers from the Kaggle pipeline.

### Why Metrics Should Be Interpreted Carefully

- Kaggle labels have ~9.4% fraud rate at the provider level, with some label noise from aggregation
- Provider-level labels aggregate claim-level noise; edge cases are ambiguous
- Test set performance is not a guarantee of production performance
- Real LTC fraud detection would require domain-specific features, clinical context, compliance review, and ongoing model validation
- Isolation Forest AUC not reported (unsupervised; no predict_proba threshold calibration)

---

## 13. Explainability

- **Feature importance:** model `.feature_importances_` by default; uses SHAP TreeExplainer if the optional `shap` package is installed; permutation importance as a final fallback
- **Provider explanations:** for each high-risk provider, 3-5 business-readable bullets
  (e.g. "Avg reimbursement per claim is 3.2x the overall median")
- **Top risk factors:** `outputs/reports/top_risk_factors.csv`
- **Provider explanations:** `outputs/reports/high_risk_provider_explanations.csv`

---

## 14. RAG Provider Review Assistant

Three retrieval+generation tiers, chosen at runtime based on installed packages:

| Tier | Retrieval | Generation | Dependencies | When it runs |
|---|---|---|---|---|
| 0 (always available) | TF-IDF + cosine similarity | Deterministic template | scikit-learn only | `src/rag_claim_review.py` |
| 1 (opt-in) | **Semantic dense embeddings** (`all-MiniLM-L6-v2`) | Deterministic template | + sentence-transformers | `src/llm_review.py --tier 1` |
| 2 (opt-in) | **Semantic dense embeddings** | **Local LLM** (`flan-t5-base`, deterministic decoding) | + transformers + torch | `src/llm_review.py --tier 2` |

Install the optional path: `make install-llm` (≈700 MB of wheels; the flan-t5-base model auto-downloads to `~/.cache/huggingface` on first use, ~250M params, ~1 GB). The LLM call uses `do_sample=False` so every run produces the same output for the same input — auditable, reproducible.

The pipeline scores every provider, then composes a structured review packet:
- Provider ID, Risk Level, Model Risk Score, retrieval+generation backend (printed on every review)
- **Audit Reviewer Summary** — 1-2 sentence LLM-generated (tier 2) or template-rendered (tier 0/1) explanation, grounded in the top risk indicator and top retrieved policy
- Key Risk Indicators (3-5 specific, quantified bullets — deterministic from the feature values)
- Retrieved Policy / Audit Evidence (top 3 chunks with similarity scores)
- Data & Documentation Gaps
- Suggested Analyst Action
- Human Review Notes (analyst checklist)
- System Limitations disclaimer

Reviews are saved with a backend suffix so the three tiers do not collide:
`review_{Provider}.txt`, `review_{Provider}_llm.txt`, etc.

> **Honest note on tier 2 output quality.** flan-t5-base is a 250M-parameter model running on CPU. With few-shot prompting it produces grounded 1-2 sentence summaries, but it is not a 70B production model — output occasionally borrows wording from the in-context examples. The architecture is the deliverable; swap in a larger local model (`flan-t5-large`, `mistral-7b-instruct` via `llama.cpp`) for production-grade prose.

---

## 15. Model Monitoring & Data Quality

`src/monitoring.py` produces:

| Output | Description |
|---|---|
| `data_quality_report.csv` | Column-level missing rates, n_unique, dtypes |
| `model_monitoring_report.csv` | Provider-level statistical summary per feature |
| `provider_risk_distribution.png` | Risk score distribution by fraud label |
| `reimbursement_distribution.png` | Reimbursement per claim with median / P99 |
| `fraud_rate_by_volume_bucket.png` | Fraud prevalence by provider volume quintile |
| `feature_missingness.png` | Missing rate per feature column |

### Population Stability Index (PSI)

`src/psi_drift.py` computes the Population Stability Index between the *earliest 80%* and *most recent 20%* of providers (the same chronological cut used by the temporal-split modeling). PSI is the industry-standard drift metric for credit-scoring and fraud applications.

Thresholds (standard):
- `PSI < 0.10` → stable
- `0.10 ≤ PSI < 0.25` → moderate shift, investigate
- `PSI ≥ 0.25` → significant shift, retraining candidate

**Current run on the Kaggle data:** 13 / 32 features exceed the 0.25 threshold under chronological split. The top drifters are volume features (`total_claims`, `provider_volume_percentile`, `unique_beneficiaries`) and graph features (`provider_pagerank`) — which directly explains the temporal-split AUC drop we documented above. In production this would trigger a retraining alert.

Outputs: `outputs/reports/psi_drift_report.csv`, `outputs/figures/psi_top_features.png`.

### Analyst Feedback Loop

`src/feedback_loop.py` closes the loop a static classifier alone cannot: analyst dispositions on flagged providers become labels for the next training cycle.

**CSV schema** (compatible with most case-management exports):
```
provider_id, model_score, model_flag, analyst_disposition, disposition_date
```
where `analyst_disposition ∈ {confirmed_fraud, cleared, needs_more_info}`.

The module computes the agreement metrics a production FWA team would watch weekly:

| Metric | What it means |
|---|---|
| `precision_of_flag` | Fraction of model-flagged providers the analyst confirmed |
| `false_confirm_rate` | Fraction of model-flagged providers the analyst cleared |
| `miss_rate_on_audit` | Among random-audit (unflagged) cases, fraction the analyst confirmed |

A configurable retraining-trigger rule (`precision_of_flag < 0.55` AND `n_labels ≥ 100`) fires when feedback indicates the model has drifted away from analyst judgement.

When `--retrain` is set, the module merges analyst-confirmed labels with the original training set and persists a `best_fwa_model_feedback_retrained.pkl` — the exact mechanic a production system would invoke on the retrain trigger.

Outputs: `outputs/reports/feedback_log.csv`, `outputs/reports/feedback_loop_metrics.json`, `outputs/figures/feedback_calibration.png`.

---

## 16. Fairness Audit

`src/fairness_audit.py` runs a disparate-impact analysis on the trained model. Because Kaggle providers do not carry protected attributes themselves, the audit operates on the **patient panel each provider serves**: it joins the beneficiary table back in, aggregates demographics per provider (majority race, average age band, dominant state), and compares model behaviour across cohorts.

Outputs (after `make fairness`):

| File | Contents |
|---|---|
| `outputs/reports/fairness_audit_report.csv` | Per-race-cohort: n_providers, fraud_rate, mean_risk_score, model_flag_rate, **disparate_impact_ratio**, **passes_4_5ths_rule** |
| `outputs/reports/fairness_audit_age.csv` | Same metrics bucketed by patient-panel average age |
| `outputs/figures/fairness_score_by_race.png` | Score-distribution boxplot by majority patient race |
| `outputs/figures/fairness_flag_rate_by_cohort.png` | HIGH-flag rate bar chart with 4/5ths-rule threshold line |

**Honest finding from the current Kaggle run:** the model's HIGH-flag rate concentrates almost entirely in providers serving majority-white patient panels (11.5%) versus 0% for majority-Black, majority-Hispanic, and majority-Other panels. The 4/5ths-rule **fails** for the latter three cohorts. This reflects the underlying label distribution — 505 of the 506 fraud labels in this dataset are attached to providers with majority-white panels (which is itself a known characteristic of the Medicare population in this dataset, not a model defect). In a real deployment this finding would trigger a compliance conversation about label sourcing before any model action.

This is intentionally a *descriptive* audit, not a hypothesis test. A production deployment would add intersectional cohorts (Race × Age × State), statistical significance testing, and per-cohort calibration analysis.

---

## 17. Responsible AI & Auditability

- **Human-in-the-loop:** every HIGH risk flag surfaces to a human analyst before action
- **Quantified uncertainty:** model probability scores (not black-box flags)
- **Policy traceability:** each review cites the most relevant audit rule chunks
- **Data lineage:** data source clearly labeled (real Kaggle / synthetic demo)
- **Limitations logged:** every review includes a System Limitations section
- **No real PHI:** Kaggle dataset and synthetic documents only

---

## 18. Repository Structure

```
Insurance-FWA-Risk-Scoring-GenAI-Claims-Review-System/
├── src/
│   ├── data_generation.py              # Original synthetic data generation
│   ├── synthetic_data_generation.py    # Same, with synthetic-mode header
│   ├── data_ingestion.py               # Kaggle file discovery + loaders
│   ├── provider_feature_engineering.py # Provider-level feature table builder
│   ├── preprocessing.py                # Synthetic claim preprocessing
│   ├── feature_engineering.py          # Synthetic claim feature engineering
│   ├── graph_features.py               # 5 bipartite-graph features (sharing rates, clustering, PageRank)
│   ├── modeling.py                     # ML training + evaluation (random / temporal split)
│   ├── hyperparameter_tuning.py        # RandomizedSearchCV on PR-AUC
│   ├── explainability.py               # Feature importance + provider explanations
│   ├── rag_claim_review.py             # Tier-0 TF-IDF + template RAG (no extra deps)
│   ├── llm_review.py                   # Tier-1/2 semantic retrieval + local-LLM generation
│   ├── monitoring.py                   # Data quality + monitoring charts
│   ├── psi_drift.py                    # PSI drift detection (train vs temporal-test)
│   ├── fairness_audit.py               # Patient-panel disparate-impact audit
│   ├── feedback_loop.py                # Analyst-disposition feedback + retraining
│   ├── oig_leie_analysis.py            # Real federal-fraud (LEIE) integration + RAG taxonomy
│   ├── cms_ltc_pipeline.py             # Real LTC pipeline (CMS Nursing Home Provider Information)
│   ├── medicare_partb_pipeline.py      # Real NPI-labeled pipeline (Medicare Part B ⋈ LEIE)
│   └── utils.py
├── data/
│   ├── raw/                            # Place Kaggle CSVs here (see data/README.md)
│   ├── processed/                      # provider_modeling_table.csv / claims_features.csv
│   └── documents/                      # policy_rules.txt + synthetic claim docs
├── sql/
│   └── provider_features.sql           # Portable SQL impl of provider feature aggregation
├── scripts/
│   └── download_real_data.sh           # Idempotent downloader for OIG LEIE + CMS Nursing Home
├── tests/
│   ├── test_project_structure.py       # Structural / output-schema checks
│   └── test_logic.py                   # Unit tests for PSI, graph features, risk thresholds
├── outputs/
│   ├── figures/                        # PNG charts
│   ├── models/                         # best_fwa_model.pkl, isolation_forest.pkl
│   ├── reports/                        # metrics JSON, threshold CSV, explanations, executive_summary.md
│   └── sample_reviews/                 # review_{ID}.txt files
├── notebooks/
├── .github/workflows/ci.yml            # Compile + pytest on push
├── Makefile                            # make install / real-pipeline / dashboard / test
├── app.py                              # Streamlit 11-tab dashboard
├── config.py                           # Path constants
├── requirements.txt                    # Core deps
├── requirements-llm.txt                # Optional: torch + transformers + sentence-transformers
├── Dockerfile                          # CPU image; build --build-arg INSTALL_LLM=1 for LLM tier
├── .dockerignore
└── README.md
```

---

## 19. How to Download the Data

1. Go to: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
2. Click **Download** (free Kaggle account required)
3. Unzip the downloaded archive
4. Place all CSV files in `data/raw/`
5. Verify: `python src/data_ingestion.py` — should print "All files loaded successfully"

---

## 20. How to Run

### Quick start with the Makefile

```bash
# Core pipeline (no torch/transformers required)
make install         # pip install -r requirements.txt
make real-pipeline   # full pipeline on real Kaggle data (random 80/20 split)
make graph-features  # add 5 bipartite-graph features to the provider table
make temporal-eval   # re-evaluate with chronological train/test split
make tune-rf         # RandomizedSearchCV hyperparameter tuning
make psi             # PSI drift report (chronological split)
make fairness        # patient-panel disparate-impact audit
make feedback        # analyst-disposition feedback loop (synthesized) + retrain
make dashboard       # launch Streamlit dashboard
make test            # run pytest (30 checks)

# Optional GenAI layer (~700MB of torch + transformers + flan-t5-base download)
make install-llm     # pip install -r requirements-llm.txt
make llm-reviews     # 10 semantic-retrieval + local-LLM generated provider reviews

# Real public data — automatic downloads
make download-real   # OIG LEIE + CMS Nursing Home (~25 MB total)
make download-partb  # Medicare Physician 2023 (~470 MB)
make oig-leie        # real-data fraud taxonomy + LTC subset
make cms-ltc         # real LTC FWA model on 14,699 US nursing homes
make partb-ltc       # real-NPI-labeled fraud model (193K LTC providers ⋈ LEIE)
```

### Docker

```bash
make docker          # CPU-only image with core deps
make docker-llm      # Same + torch + transformers (for tier-2 RAG)
docker run -p 8501:8501 fwa-portfolio        # launches the dashboard

# Mount real data from the host so the image stays small (recommended):
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
    -p 8501:8501 fwa-portfolio
```

### Run tests

```bash
pytest -q
```

### Full pipeline with real Kaggle data

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place Kaggle CSV files in data/raw/  (see section 15)

# 3. Run pipeline
python src/data_ingestion.py                  # validate files
python src/provider_feature_engineering.py   # build provider feature table
python src/modeling.py                        # train + evaluate models
python src/explainability.py                  # feature importance + explanations
python src/rag_claim_review.py               # generate 15 sample reviews
python src/monitoring.py                      # monitoring charts + reports

# 4. Launch dashboard
streamlit run app.py
```

### Synthetic demo (no Kaggle download required)

```bash
pip install -r requirements.txt

python src/synthetic_data_generation.py      # generates synthetic_claims.csv
python src/preprocessing.py
python src/feature_engineering.py
python src/modeling.py
python src/explainability.py
python src/rag_claim_review.py
python src/monitoring.py

streamlit run app.py
```

---

## 21. Sample Outputs

- `outputs/reports/model_metrics.json` — ROC-AUC, F1, precision, recall per model
- `outputs/reports/threshold_analysis.csv` — full threshold sweep (0.05 → 0.95)
- `outputs/reports/top_risk_factors.csv` — top 20 features by importance
- `outputs/reports/high_risk_provider_explanations.csv` — provider-level risk bullets
- `outputs/sample_reviews/review_*.txt` — formatted RAG audit reviews
- `outputs/figures/roc_curve.png`, `precision_recall_curve.png`, `feature_importance.png`
- `outputs/figures/provider_risk_distribution.png`, `fraud_rate_by_volume_bucket.png`
- `outputs/reports/executive_summary.md` — **leadership-facing 1-pager** (business framing, headline metrics, production roadmap)
- `sql/provider_features.sql` — **portable SQL implementation** of the provider feature aggregation (Snowflake / BigQuery / Redshift compatible)

---

## 22. Dashboard

The Streamlit dashboard (`app.py`) is a 6-tab analyst-facing interface over the trained models, monitoring reports, and RAG review packets. It is the screenshot-friendly surface for the whole pipeline.

### Launching locally

```bash
make dashboard            # equivalent to: streamlit run app.py
```

The app starts at `http://localhost:8501` and reads from `data/processed/` and `outputs/`. If real Kaggle data is present, it shows the real provider table; otherwise it falls back to the synthetic demo and labels every panel accordingly.

### Tabs

| # | Tab | What it shows |
|---|---|---|
| 1 | Executive Overview | Provider count, fraud rate, headline model metrics (AUC, F1), data-source banner |
| 2 | Provider FWA Pattern Explorer | Reimbursement distribution, fraud rate by volume bucket, top high-risk providers, inpatient vs outpatient mix |
| 3 | Model Performance | Per-model metrics, ROC, PR curve, calibration plot, confusion matrix, feature importance, threshold sweep table |
| 4 | High-Risk Provider Review Assistant | Provider-ID selector, model score, top risk indicators, full RAG review packet |
| 5 | Model Monitoring & Data Quality | Missing-value chart, class balance, provider volume distribution, monitoring report |
| 6 | Auditability & Responsible AI | Data disclaimer, model assumptions, RAG limitations, human-in-the-loop framing |

Screenshots are not committed to keep the repo light. Capture from the running app and embed in any external deck or write-up.

---

## 23. Project Deliverables

The pipeline produces a complete set of artifacts spanning data, modeling, monitoring, audit, and presentation layers:

| Layer | Deliverable | Location |
|---|---|---|
| Data | Provider-level modeling table (5,410 providers × 27 features + label) | `data/processed/provider_modeling_table.csv` |
| Data | Portable SQL reference for the provider feature aggregation | `sql/provider_features.sql` |
| Modeling | Trained best model (Random Forest) + Isolation Forest anomaly model | `outputs/models/best_fwa_model.pkl`, `isolation_forest.pkl` |
| Modeling | Model metrics JSON (ROC-AUC, PR-AUC, P/R/F1, Brier, log-loss, 5-fold CV) — random split | `outputs/reports/model_metrics.json` |
| Modeling | **Same schema under chronological train/test split** (the realistic metric) | `outputs/reports/model_metrics_temporal.json` |
| Modeling | sklearn classification report for the best model | `outputs/reports/classification_report.txt` |
| Modeling | Threshold sweep analysis (0.05 → 0.95 with precision/recall/F1/n_flagged) | `outputs/reports/threshold_analysis.csv` |
| Modeling | ROC, precision-recall, **calibration**, confusion matrix, feature importance | `outputs/figures/*.png` |
| Fairness | **Patient-panel disparate-impact audit** (4/5ths-rule check per cohort) | `outputs/reports/fairness_audit_report.csv` + figures |
| GenAI | **Local-LLM-generated audit summaries** (semantic retrieval + flan-t5-base) | `outputs/sample_reviews/review_*_llm.txt` |
| Graph features | **Bipartite-graph features** (sharing rates, clustering coef, PageRank) added to the provider table | `src/graph_features.py` |
| Tuning | **RandomizedSearchCV** tuned model + full search log | `outputs/models/best_fwa_model_tuned_rf.pkl` + `outputs/reports/hp_tuning_*.{json,csv}` |
| Drift | **PSI drift report** with retrain-trigger verdict per feature | `outputs/reports/psi_drift_report.csv` + figure |
| Feedback | **Analyst-disposition feedback loop** with retrain trigger | `outputs/reports/feedback_log.csv`, `feedback_loop_metrics.json` |
| Real LTC pipeline | **CMS Nursing Home Provider Information** — 14,699 real US nursing homes, real labels | `src/cms_ltc_pipeline.py` + `outputs/reports/cms_ltc_*` |
| Real federal fraud | **HHS-OIG LEIE** — 83K real federal exclusions, 1,818 LTC-specific | `src/oig_leie_analysis.py` + `outputs/reports/oig_leie_*` |
| Real NPI-labeled | **Medicare Part B 2023 ⋈ LEIE** — 1.26M real provider NPIs, 207 real fraud labels | `src/medicare_partb_pipeline.py` + `outputs/reports/medicare_partb_*` |
| Container | Dockerfile (CPU base + optional LLM build-arg) | `Dockerfile` / `make docker` |
| Logic tests | Unit tests for PSI math, graph features, risk thresholds | `tests/test_logic.py` |
| Explainability | Top risk factors ranked by importance | `outputs/reports/top_risk_factors.csv` |
| Explainability | Per-provider business-readable risk explanations | `outputs/reports/high_risk_provider_explanations.csv` |
| GenAI / RAG | 15 structured provider review packets (10 High / 3 Medium / 2 Low risk) | `outputs/sample_reviews/review_*.txt` |
| Monitoring | Data quality and feature monitoring reports | `outputs/reports/data_quality_summary.csv`, `model_monitoring_report.csv` |
| Monitoring | Provider risk, reimbursement, and volume-bucket fraud-rate charts | `outputs/figures/provider_risk_distribution.png`, etc. |
| Stakeholder | Executive summary 1-pager (business framing, headline metrics, roadmap) | `outputs/reports/executive_summary.md` |
| Application | 6-tab Streamlit dashboard | `app.py` |
| Engineering | pytest suite (13 tests) | `tests/` |
| Engineering | GitHub Actions CI (compile + tests) | `.github/workflows/ci.yml` |
| Engineering | Makefile (install / real-pipeline / synthetic-demo / dashboard / test) | `Makefile` |

---

## 24. What Would Be Required in a Real Insurance Environment

If this workflow were adapted into a real FWA program rather than run on a public reference dataset, the next investments would be:

- **Real LTC claim notes and care-plan documentation.** Replace synthetic policy text with actual SOC documentation, plan-of-care narratives, and progress notes so the RAG layer can do clinical/contextual evidence retrieval rather than generic billing-rule matching.
- **Provider network / graph features.** Build referral graphs and shared-physician edges across providers; graph-based risk scoring catches collusive billing rings that provider-level features miss.
- **Temporal drift monitoring on rolling windows.** Move from a single train/test split to monthly retrains with PSI/KS drift alerts on feature distributions and score distributions.
- **Case management system integration.** Push HIGH-risk flags directly into the analyst's case queue with a structured payload (risk indicators, retrieved policy, prior dispositions for the same provider).
- **Fairness / compliance / bias review across protected attributes.** Audit score distributions across race, state, and age-band cohorts; document any disparate impact before deployment.
- **Reviewer feedback loop.** Capture analyst dispositions (confirmed fraud / cleared / needs more info) as labels for the next training cycle — this is what turns the model from a static classifier into a learning system.
- **Calibrated thresholds tuned to current analyst capacity.** Auto-adjust the operating threshold so flagged volume matches the rolling-window analyst headcount, instead of a fixed cutoff.

---

## 25. Limitations & Future Improvements

| Current limitation | Potential improvement |
|---|---|
| TF-IDF retrieval only | Add sentence-transformer embeddings for semantic search |
| No temporal modeling | Time-series billing sequences (weekly volume trends) |
| No graph features | Provider-beneficiary referral network analysis |
| No NLP on clinical text | Would require real EHR access |
| Static threshold | Dynamic threshold based on rolling reviewer capacity |
| Binary labels | Ordinal or multi-class fraud severity scoring |
| Train set only (Kaggle) | Cross-validate against held-out Test split |
