"""
Streamlit dashboard — Insurance FWA Risk Scoring & GenAI Claims Review System.

Six tabs:
  1. Executive Overview
  2. FWA Pattern Explorer
  3. Model Performance
  4. Claim Review Assistant
  5. Model Monitoring & Data Quality
  6. Auditability & Responsible AI
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.utils import get_risk_level, format_currency

st.set_page_config(
    page_title="Insurance FWA Risk Scoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


ID_LIKE_COLS = [
    "claim_id", "policyholder_id", "provider_id", "claim_date",
    "service_type", "diagnosis_group", "state",
]
EXCLUDE_FROM_MODEL = ["rule_based_risk_score"]


# ── Data loading ────────────────────────────────────────────────────────────

@st.cache_data
def load_claims():
    for p in [
        os.path.join(config.DATA_PROCESSED, "claims_features.csv"),
        os.path.join(config.DATA_PROCESSED, "claims_encoded.csv"),
        os.path.join(config.DATA_RAW,       "synthetic_claims.csv"),
    ]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "claim_date" in df.columns:
                df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")
            return df
    return None


@st.cache_data
def load_metrics():
    p = os.path.join(config.OUTPUTS_REPORTS, "model_metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_threshold_table():
    p = os.path.join(config.OUTPUTS_REPORTS, "threshold_analysis.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def load_monitoring_report():
    p = os.path.join(config.OUTPUTS_REPORTS, "model_monitoring_report.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def load_data_quality():
    p = os.path.join(config.OUTPUTS_REPORTS, "data_quality_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_resource
def load_model():
    p = os.path.join(config.OUTPUTS_MODELS, "best_fwa_model.pkl")
    return joblib.load(p) if os.path.exists(p) else None


def load_review(claim_id):
    p = os.path.join(config.OUTPUTS_REVIEWS, f"review_{claim_id}.txt")
    if os.path.exists(p):
        with open(p) as f:
            return f.read()
    return None


def get_risk_scores(df, model):
    if model is None:
        return None
    feature_cols = [
        c for c in df.columns
        if c not in ID_LIKE_COLS + ["fraud_label"] + EXCLUDE_FROM_MODEL
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    try:
        X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
        return model.predict_proba(X)[:, 1]
    except Exception:
        return None


# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("🛡️ FWA Risk Scoring")
st.sidebar.markdown("**Insurance Fraud, Waste & Abuse Analytics**")
st.sidebar.caption("Aligned to Long Term Care FWA — synthetic data only.")
st.sidebar.markdown("---")
tab_choice = st.sidebar.radio(
    "Navigate to",
    [
        "Executive Overview",
        "FWA Pattern Explorer",
        "Model Performance",
        "Claim Review Assistant",
        "Model Monitoring & Data Quality",
        "Auditability & Responsible AI",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Portfolio demo — not clinical or production grade.")


# ── Load data ───────────────────────────────────────────────────────────────

df      = load_claims()
metrics = load_metrics()
model   = load_model()

if df is not None and model is not None:
    risk_scores = get_risk_scores(df, model)
    if risk_scores is not None:
        df = df.copy()
        df["model_risk_score"] = risk_scores


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Overview
# ════════════════════════════════════════════════════════════════════════════

if tab_choice == "Executive Overview":
    st.title("📊 Executive Overview")
    st.info("Synthetic claims dataset. Every prediction shown here is meant to be "
            "reviewed by a human FWA analyst before any payment action is taken.")

    if df is None:
        st.warning("No claims data found. Run `python src/data_generation.py` first.")
    else:
        total_claims  = len(df)
        fraud_rate    = df["fraud_label"].mean() if "fraud_label" in df.columns else 0.0
        avg_amount    = df["claim_amount"].mean() if "claim_amount" in df.columns else 0.0
        high_risk_cnt = int((df["model_risk_score"] >= config.HIGH_RISK_THRESHOLD).sum()) \
                        if "model_risk_score" in df.columns else "N/A"

        best_auc = best_f1 = "N/A"
        if metrics:
            best_name = max(metrics, key=lambda n: metrics[n].get("roc_auc", 0))
            best_auc  = f"{metrics[best_name]['roc_auc']:.3f}"
            best_f1   = f"{metrics[best_name]['f1']:.3f}"

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Claims",      f"{total_claims:,}")
        c2.metric("Fraud Rate",        f"{fraud_rate:.1%}")
        c3.metric("Avg Claim",         format_currency(avg_amount))
        c4.metric("High-Risk Claims",  f"{high_risk_cnt:,}" if isinstance(high_risk_cnt, int) else high_risk_cnt)
        c5.metric("Best AUC",          best_auc)
        c6.metric("Best F1",           best_f1)

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Risk Score Distribution")
            if "model_risk_score" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(df["model_risk_score"], bins=40, color="#2196F3", edgecolor="white", alpha=0.8)
                ax.axvline(config.HIGH_RISK_THRESHOLD, color="red", ls="--",
                           label=f"High threshold ({config.HIGH_RISK_THRESHOLD})")
                ax.set_xlabel("Model Risk Score")
                ax.set_ylabel("Count")
                ax.legend()
                st.pyplot(fig); plt.close()
                st.caption("Most claims sit at low risk; the long right tail is what the analyst team reviews first.")

        with col_b:
            st.subheader("Fraud Label Distribution")
            if "fraud_label" in df.columns:
                counts = df["fraud_label"].value_counts().rename({0: "Legitimate", 1: "Fraudulent"})
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
                       colors=["#4CAF50", "#F44336"],
                       startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
                st.pyplot(fig); plt.close()
                st.caption("Class imbalance is realistic — fraud is rare, so we evaluate with PR / recall, "
                           "not just accuracy.")

        st.markdown("---")
        st.subheader("Top 10 Highest-Risk Claims (model score)")
        if "model_risk_score" in df.columns:
            top_risky = df.nlargest(10, "model_risk_score")[
                ["claim_id", "service_type", "claim_amount",
                 "model_risk_score", "fraud_label"]
            ].reset_index(drop=True)
            top_risky["risk_level"] = top_risky["model_risk_score"].apply(get_risk_level)
            st.dataframe(top_risky, use_container_width=True)
            st.caption("These are the claims an FWA analyst would queue first. "
                       "`fraud_label` is shown for evaluation only — in production it would not be available.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — FWA Pattern Explorer
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "FWA Pattern Explorer":
    st.title("🔍 FWA Pattern Explorer")
    st.info("Use these charts to understand WHERE fraud-like patterns concentrate "
            "(service type, provider, documentation quality). This is what an analyst "
            "would look at before drilling into individual claims.")

    if df is None:
        st.warning("No data available.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Claim Amount by Fraud Label")
            fig, ax = plt.subplots(figsize=(6, 4))
            fraud_0 = df[df["fraud_label"] == 0]["claim_amount"] if "fraud_label" in df.columns else df["claim_amount"]
            fraud_1 = df[df["fraud_label"] == 1]["claim_amount"] if "fraud_label" in df.columns else None
            ax.hist(fraud_0.clip(upper=50000), bins=50, alpha=0.6, label="Legitimate", color="#4CAF50")
            if fraud_1 is not None:
                ax.hist(fraud_1.clip(upper=50000), bins=50, alpha=0.6, label="Fraudulent", color="#F44336")
            ax.set_xlabel("Claim Amount ($)")
            ax.legend()
            st.pyplot(fig); plt.close()
            st.caption("Fraudulent claims skew higher but overlap heavily with legit "
                       "high-cost claims (e.g. inpatient, oncology) — amount alone is not enough.")

        with col2:
            st.subheader("Fraud Rate by Service Type")
            if "service_type" in df.columns and "fraud_label" in df.columns:
                st_fraud = (
                    df.groupby("service_type")["fraud_label"]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "fraud_rate", "count": "total"})
                    .sort_values("fraud_rate", ascending=True)
                )
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(st_fraud.index, st_fraud["fraud_rate"] * 100,
                        color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(st_fraud))))
                ax.set_xlabel("Fraud Rate (%)")
                st.pyplot(fig); plt.close()
                st.caption("Service-type fraud-rate ranking informs where to deploy targeted reviewer capacity.")

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Top 15 Risky Providers (by avg model score)")
            if "provider_id" in df.columns and "model_risk_score" in df.columns:
                prov_risk = (
                    df.groupby("provider_id")
                    .agg(
                        avg_risk_score=("model_risk_score", "mean"),
                        claim_count=("claim_id", "count"),
                        fraud_rate=("fraud_label", "mean"),
                    )
                    .sort_values("avg_risk_score", ascending=False)
                    .head(15)
                    .reset_index()
                )
                prov_risk["avg_risk_score"] = prov_risk["avg_risk_score"].round(3)
                prov_risk["fraud_rate"]     = (prov_risk["fraud_rate"] * 100).round(1)
                st.dataframe(prov_risk, use_container_width=True)
                st.caption("Provider-level aggregation is the unit of investigation for most FWA programs.")

        with col4:
            st.subheader("Documentation Score vs Claim Amount")
            if "documentation_score" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sample = df.sample(min(1000, len(df)), random_state=42)
                colors = sample["fraud_label"].map({0: "#4CAF50", 1: "#F44336"})
                ax.scatter(sample["documentation_score"], sample["claim_amount"].clip(upper=30000),
                           c=colors, alpha=0.4, s=12)
                ax.set_xlabel("Documentation Score")
                ax.set_ylabel("Claim Amount ($)")
                st.pyplot(fig); plt.close()
                st.caption("Low documentation + high amount is a classic FWA red zone, but plenty of "
                           "legitimate claims also have sloppy paperwork.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Model Performance":
    st.title("🤖 Model Performance")
    st.info("Because fraud is rare, the precision-recall tradeoff matters more "
            "than raw accuracy. A precision-oriented threshold sends fewer false "
            "positives to analysts; a recall-oriented threshold catches more fraud "
            "at the cost of more reviews.")

    if not metrics:
        st.warning("No model metrics found. Run `python src/modeling.py` first.")
    else:
        st.subheader("Evaluation Metrics (Test Set)")
        metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
        metrics_df = metrics_df.sort_values("roc_auc", ascending=False)
        st.dataframe(metrics_df.style.highlight_max(subset=["roc_auc", "f1", "recall"]),
                     use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    conf_path = os.path.join(config.OUTPUTS_FIGURES, "confusion_matrix.png")
    roc_path  = os.path.join(config.OUTPUTS_FIGURES, "roc_curve.png")
    pr_path   = os.path.join(config.OUTPUTS_FIGURES, "precision_recall_curve.png")
    fi_path   = os.path.join(config.OUTPUTS_FIGURES, "feature_importance.png")

    with col1:
        st.subheader("Confusion Matrix")
        if os.path.exists(conf_path):
            st.image(conf_path, use_container_width=True)
            st.caption("Top-right = missed fraud (worst kind of error for FWA).")
        else:
            st.info("Run `python src/modeling.py`.")

    with col2:
        st.subheader("ROC Curves")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
            st.caption("ROC summarizes ranking quality but can look optimistic on imbalanced data.")
        else:
            st.info("Run `python src/modeling.py`.")

    st.markdown("---")
    col3, col4 = st.columns([2, 3])
    with col3:
        st.subheader("Precision-Recall Curve")
        if os.path.exists(pr_path):
            st.image(pr_path, use_container_width=True)
            st.caption("This is the chart we actually optimize against in FWA — "
                       "it tells you how much precision you lose to catch more fraud.")
        else:
            st.info("Run `python src/modeling.py` to generate PR curve.")

    with col4:
        st.subheader("Threshold Sweep")
        thr = load_threshold_table()
        if thr is not None:
            st.dataframe(thr, use_container_width=True, height=320)
            st.caption("Lowering the threshold flags more claims (higher recall, lower precision). "
                       "Operations teams pick the threshold based on reviewer capacity.")
        else:
            st.info("Threshold analysis not found. Run `python src/modeling.py`.")

    st.markdown("---")
    st.subheader("Feature Importance")
    if os.path.exists(fi_path):
        st.image(fi_path, use_container_width=True)
        st.caption("Importance ranks observable features only — hidden latent drivers "
                   "(provider integrity, policyholder propensity) are intentionally NOT exposed to the model.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Claim Review Assistant
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Claim Review Assistant":
    st.title("🔎 Claim Review Assistant")
    st.info("Pick a claim → the assistant retrieves the most relevant policy "
            "language and renders an analyst-ready review packet.")

    if df is None:
        st.warning("No claims data found.")
    else:
        # Prefer claims that already have pre-generated reviews (richer demo)
        review_dir = config.OUTPUTS_REVIEWS
        precomputed = []
        if os.path.isdir(review_dir):
            precomputed = sorted([
                f.replace("review_", "").replace(".txt", "")
                for f in os.listdir(review_dir) if f.startswith("review_") and f.endswith(".txt")
            ])

        all_ids = df["claim_id"].dropna().tolist() if "claim_id" in df.columns else []
        default_ids = precomputed if precomputed else all_ids
        selected_id = st.selectbox(
            "Select Claim ID (pre-generated reviews appear first)",
            default_ids + [c for c in all_ids if c not in default_ids],
            index=0,
        )

        if selected_id:
            row = df[df["claim_id"] == selected_id].iloc[0]

            risk_score = row.get("model_risk_score", row.get("rule_based_risk_score", 0.0))
            risk_level = get_risk_level(risk_score)
            risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level, "⚪")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Risk Score", f"{risk_score:.3f}")
            col2.metric("Risk Level", f"{risk_color} {risk_level}")
            col3.metric("Claim Amount", format_currency(row.get("claim_amount", 0)))
            col4.metric("Approved", format_currency(row.get("approved_amount", 0)))

            st.markdown("---")
            st.subheader("AI-Generated Review (RAG)")
            review_text = load_review(selected_id)
            if review_text:
                st.code(review_text, language="text")
            else:
                try:
                    from src.rag_claim_review import (
                        load_policy_rules, build_policy_index, generate_review
                    )
                    policy_chunks = load_policy_rules()
                    if policy_chunks:
                        vectorizer, tfidf_matrix = build_policy_index(policy_chunks)
                        on_the_fly = generate_review(
                            selected_id, row, float(risk_score),
                            vectorizer, tfidf_matrix, policy_chunks
                        )
                        st.code(on_the_fly, language="text")
                    else:
                        st.warning("Policy rules document not found.")
                except Exception as e:
                    st.error(f"Could not auto-generate review: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Monitoring & Data Quality
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Model Monitoring & Data Quality":
    st.title("📈 Model Monitoring & Data Quality")
    st.info("A model is only as good as the data feeding it. This tab tracks "
            "volume, label drift, dollar drift, and column-level quality — the "
            "minimum monitoring surface any production FWA system needs.")

    report = load_monitoring_report()
    qa = load_data_quality()

    if report is None:
        st.warning("No monitoring report found. Run `python src/monitoring.py` first.")
    else:
        st.subheader("Monthly Claim Volume & Fraud Rate")
        col1, col2 = st.columns(2)
        with col1:
            p = os.path.join(config.OUTPUTS_FIGURES, "monthly_fraud_rate.png")
            if os.path.exists(p):
                st.image(p, use_container_width=True)
                st.caption("Sudden spikes in fraud rate are a label-drift / data-pipeline alarm.")
        with col2:
            p = os.path.join(config.OUTPUTS_FIGURES, "claim_amount_drift.png")
            if os.path.exists(p):
                st.image(p, use_container_width=True)
                st.caption("Mean vs P95 separation tells you whether drift is broad or tail-only.")

        st.subheader("Monthly Monitoring Report")
        st.dataframe(report, use_container_width=True)
        st.caption("In production this table would be re-computed daily and compared "
                   "against a rolling baseline to fire PSI / KS drift alerts.")

    st.markdown("---")
    st.subheader("Data Quality — Column Summary")
    if qa is not None:
        st.dataframe(qa, use_container_width=True, height=400)
        st.caption("Missing rates, unique counts, and dtypes — the first thing to "
                   "check when a model suddenly degrades.")
    else:
        st.info("Run `python src/monitoring.py` to generate the data-quality summary.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Auditability & Responsible AI
# ════════════════════════════════════════════════════════════════════════════

else:  # "Auditability & Responsible AI"
    st.title("📋 Auditability & Responsible AI")

    st.markdown("""
### Synthetic Data Disclaimer
All data is **synthetic** (NumPy / pandas generators). No PHI, PII, or real
claims data is used. This project is a portfolio demonstration of FWA analytics
patterns relevant to Long Term Care insurance — not a clinical or production tool.

---
### Why the Metrics Are Interpreted Carefully
- Labels come from a *known* data-generating process, so any model that recovers
  enough of that process can score very high. To avoid trivial leakage, the
  fraud label is driven by a **hidden intent variable** plus heavy stochastic
  noise; only **noisy proxies** are exposed to the model.
- Even so, headline AUC/F1 should be read as **upper bounds** on what a real
  model could achieve — real claims data has label noise, regime shifts, and
  adversarial behavior that synthetic data cannot reproduce.
- We report both ROC and Precision-Recall metrics, and we publish a full
  threshold sweep so operations can pick the precision/recall point that
  matches reviewer capacity.

---
### Model Assumptions
- **Class imbalance** (~7-9% fraud) handled via `class_weight='balanced'` for
  LR / RF and `scale_pos_weight` for XGBoost.
- **80/20 stratified split**, fraud rate preserved across splits.
- `rule_based_risk_score` is excluded from model inputs — it is a transparent
  baseline for dashboards, not a feature.

---
### GenAI / RAG Limitations
- The claim-review module uses **TF-IDF cosine similarity** over a policy-rules
  corpus, not an LLM. Output is fully deterministic and requires no external
  API keys.
- Template-based generation means no hallucinated facts — every field is sourced
  from claim data or retrieved policy text.
- A real production stack would use dense embeddings (e.g. sentence-transformers)
  plus an LLM with **citation guardrails** and **prompt-injection defenses**.

---
### Human-in-the-Loop
- Every HIGH-risk recommendation **must** be reviewed by a licensed analyst
  before any payment suspension or denial.
- The model produces probabilities, not verdicts. Final adjudication remains
  with humans who can access complete medical records and provider attestations.

---
### Bias & Fairness Considerations
- No demographic features (race, gender, religion) are used as model inputs.
- Age is included only as a clinical indicator.
- Fraud labels are based on behavioral signals, not identity.
- In production: scheduled fairness audits, segmented PR analysis, and
  reviewer-feedback loops to detect drift in false-positive disparities.

---
### Data Governance
- HIPAA / state privacy rules would apply to all real claims data.
- Audit logs for every prediction and analyst action.
- Model registry + version pinning + rollback procedures.
- Drift monitoring (PSI, KS) on key features and on label rate.
""")
