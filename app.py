"""
Streamlit Dashboard — Insurance FWA Risk Scoring & GenAI Claims Review System
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# ── Config ──────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.utils import get_risk_level, format_currency

st.set_page_config(
    page_title="Insurance FWA Risk Scoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading helpers ─────────────────────────────────────────────────────

@st.cache_data
def load_claims():
    for p in [
        os.path.join(config.DATA_PROCESSED, "claims_features.csv"),
        os.path.join(config.DATA_PROCESSED, "claims_encoded.csv"),
        os.path.join(config.DATA_RAW,       "synthetic_claims.csv"),
    ]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


@st.cache_data
def load_metrics():
    p = os.path.join(config.OUTPUTS_REPORTS, "model_metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_model():
    p = os.path.join(config.OUTPUTS_MODELS, "best_fwa_model.pkl")
    if os.path.exists(p):
        return joblib.load(p)
    return None


def load_review(claim_id):
    p = os.path.join(config.OUTPUTS_REVIEWS, f"review_{claim_id}.txt")
    if os.path.exists(p):
        with open(p) as f:
            return f.read()
    return None


ID_LIKE_COLS = [
    "claim_id", "policyholder_id", "provider_id", "claim_date",
    "service_type", "diagnosis_group", "state",
]


def get_risk_scores(df, model):
    if model is None:
        return None
    feature_cols = [
        c for c in df.columns
        if c not in ID_LIKE_COLS + ["fraud_label"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    try:
        return model.predict_proba(df[feature_cols])[:, 1]
    except Exception:
        return None


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🛡️ FWA Risk Scoring")
st.sidebar.markdown("**Insurance Fraud, Waste & Abuse Analytics**")
st.sidebar.markdown("---")
tab_choice = st.sidebar.radio(
    "Navigate to",
    ["Executive Overview", "FWA Pattern Explorer",
     "Model Performance", "Claim Review Assistant",
     "Auditability Notes"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Synthetic data only — for portfolio demonstration.")

# ── Load data ────────────────────────────────────────────────────────────────

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
    st.markdown("Key performance indicators and high-level FWA analytics.")

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
        c3.metric("Avg Claim Amount",  format_currency(avg_amount))
        c4.metric("High-Risk Claims",  f"{high_risk_cnt:,}" if isinstance(high_risk_cnt, int) else high_risk_cnt)
        c5.metric("Best Model AUC",    best_auc)
        c6.metric("Best Model F1",     best_f1)

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Risk Score Distribution")
            if "model_risk_score" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(df["model_risk_score"], bins=40, color="#2196F3", edgecolor="white", alpha=0.8)
                ax.axvline(config.HIGH_RISK_THRESHOLD, color="red", ls="--", label=f"High-Risk Threshold ({config.HIGH_RISK_THRESHOLD})")
                ax.set_xlabel("Model Risk Score")
                ax.set_ylabel("Count")
                ax.set_title("Risk Score Distribution")
                ax.legend()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Model risk scores not available.")

        with col_b:
            st.subheader("Fraud Label Distribution")
            if "fraud_label" in df.columns:
                counts = df["fraud_label"].value_counts().rename({0: "Legitimate", 1: "Fraudulent"})
                fig, ax = plt.subplots(figsize=(5, 3))
                colors = ["#4CAF50", "#F44336"]
                ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors,
                       startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
                ax.set_title("Claim Fraud Distribution")
                st.pyplot(fig)
                plt.close()

        st.markdown("---")
        st.subheader("Most Recent High-Risk Claims")
        if "model_risk_score" in df.columns:
            top_risky = df.nlargest(10, "model_risk_score")[
                ["claim_id", "service_type", "claim_amount",
                 "model_risk_score", "fraud_label"]
            ].reset_index(drop=True)
            top_risky["risk_level"] = top_risky["model_risk_score"].apply(get_risk_level)
            st.dataframe(top_risky, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — FWA Pattern Explorer
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "FWA Pattern Explorer":
    st.title("🔍 FWA Pattern Explorer")

    if df is None:
        st.warning("No data available.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Claim Amount Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            fraud_0 = df[df["fraud_label"] == 0]["claim_amount"] if "fraud_label" in df.columns else df["claim_amount"]
            fraud_1 = df[df["fraud_label"] == 1]["claim_amount"] if "fraud_label" in df.columns else None
            ax.hist(fraud_0.clip(upper=50000), bins=50, alpha=0.6, label="Legitimate", color="#4CAF50")
            if fraud_1 is not None:
                ax.hist(fraud_1.clip(upper=50000), bins=50, alpha=0.6, label="Fraudulent", color="#F44336")
            ax.set_xlabel("Claim Amount ($)")
            ax.set_ylabel("Count")
            ax.set_title("Claim Amount by Fraud Label")
            ax.legend()
            st.pyplot(fig)
            plt.close()

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
                bars = ax.barh(st_fraud.index, st_fraud["fraud_rate"] * 100,
                               color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(st_fraud))))
                ax.set_xlabel("Fraud Rate (%)")
                ax.set_title("Fraud Rate by Service Type")
                st.pyplot(fig)
                plt.close()

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Provider Risk Ranking (Top 15)")
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

        with col4:
            st.subheader("Documentation Score vs Fraud")
            if "documentation_score" in df.columns and "fraud_label" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sample = df.sample(min(1000, len(df)), random_state=42)
                colors = sample["fraud_label"].map({0: "#4CAF50", 1: "#F44336"})
                ax.scatter(sample["documentation_score"], sample["claim_amount"].clip(upper=30000),
                           c=colors, alpha=0.4, s=12)
                ax.set_xlabel("Documentation Score")
                ax.set_ylabel("Claim Amount ($)")
                ax.set_title("Documentation Score vs Claim Amount")
                st.pyplot(fig)
                plt.close()

        st.markdown("---")
        st.subheader("Top Suspicious Claims (High Keyword Count)")
        if "suspicious_keyword_count" in df.columns:
            suspicious = df.nlargest(20, "suspicious_keyword_count")[
                ["claim_id", "provider_id", "service_type", "claim_amount",
                 "suspicious_keyword_count", "duplicate_claim_flag",
                 "late_submission_flag", "fraud_label"]
            ].reset_index(drop=True)
            st.dataframe(suspicious, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Model Performance":
    st.title("🤖 Model Performance")

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
    fi_path   = os.path.join(config.OUTPUTS_FIGURES, "feature_importance.png")

    with col1:
        st.subheader("Confusion Matrix")
        if os.path.exists(conf_path):
            st.image(conf_path, use_container_width=True)
        else:
            st.info("Confusion matrix not found. Run `python src/modeling.py`.")

    with col2:
        st.subheader("ROC Curves")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info("ROC curve not found. Run `python src/modeling.py`.")

    st.markdown("---")
    st.subheader("Feature Importance")
    if os.path.exists(fi_path):
        st.image(fi_path, use_container_width=True)
    else:
        st.info("Feature importance plot not found. Run `python src/modeling.py`.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Claim Review Assistant
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Claim Review Assistant":
    st.title("🔎 Claim Review Assistant")
    st.markdown("Select a claim to view its risk profile and AI-generated review summary.")

    if df is None:
        st.warning("No claims data found.")
    else:
        claim_ids = df["claim_id"].dropna().tolist() if "claim_id" in df.columns else []
        selected_id = st.selectbox("Select Claim ID", claim_ids, index=0)

        if selected_id:
            row = df[df["claim_id"] == selected_id].iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            risk_score = row.get("model_risk_score", row.get("rule_based_risk_score", 0.0))
            risk_level = get_risk_level(risk_score)

            risk_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(risk_level, "gray")

            col1.metric("Risk Score", f"{risk_score:.3f}")
            col2.metric("Risk Level", risk_level)
            col3.metric("Claim Amount", format_currency(row.get("claim_amount", 0)))
            col4.metric("Fraud Label", "Yes" if row.get("fraud_label", 0) == 1 else "No")

            st.markdown("---")
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Claim Details")
                detail_fields = [
                    "service_type", "diagnosis_group", "state",
                    "claimant_age", "provider_id", "days_since_policy_start",
                    "prior_claim_count", "provider_claim_volume",
                    "provider_avg_claim_amount", "approved_amount",
                ]
                detail_df = pd.DataFrame(
                    [(f, row.get(f, "N/A")) for f in detail_fields if f in row.index],
                    columns=["Field", "Value"]
                )
                st.dataframe(detail_df, use_container_width=True)

            with col_b:
                st.subheader("Top Risk Indicators")
                indicators = []
                ratio = row.get("claim_to_provider_avg_ratio", 1.0)
                if not pd.isna(ratio) and ratio > 1.5:
                    indicators.append(f"Claim amount is **{ratio:.1f}x** provider average")
                if row.get("documentation_score", 1.0) < 0.4:
                    indicators.append(f"Low documentation score: **{row.get('documentation_score', 0):.2f}**")
                if row.get("duplicate_claim_flag", 0) == 1:
                    indicators.append("**Duplicate claim** flag triggered")
                if row.get("late_submission_flag", 0) == 1:
                    indicators.append("**Late submission** flag triggered")
                if row.get("suspicious_keyword_count", 0) >= 3:
                    indicators.append(f"**{int(row['suspicious_keyword_count'])}** suspicious keywords detected")
                if row.get("high_cost_outlier_flag", 0) == 1:
                    indicators.append("**High-cost outlier** (top 10%)")

                if indicators:
                    for ind in indicators:
                        st.markdown(f"- {ind}")
                else:
                    st.markdown("No single dominant indicator; composite score elevated.")

            st.markdown("---")
            st.subheader("AI-Generated Review Summary (RAG)")
            review_text = load_review(selected_id)
            if review_text:
                st.code(review_text, language="text")
            else:
                st.info(
                    "No pre-generated review found for this claim. "
                    "Run `python src/rag_claim_review.py` to generate reviews for high-risk claims, "
                    "or a review will be auto-generated here."
                )
                # Auto-generate on the fly
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
# TAB 5 — Auditability Notes
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Auditability Notes":
    st.title("📋 Auditability & Responsible AI Notes")

    st.markdown("""
## Synthetic Data Disclaimer
All data in this system is **synthetically generated** using NumPy random distributions.
No real patient, provider, or claims data is used. The dataset is designed to simulate realistic
FWA patterns for portfolio and educational purposes only.

---
## Model Assumptions
- **Fraud labels** are derived from a sigmoid-transformed composite risk score based on domain-relevant
  features (claim amount vs provider average, documentation quality, duplicate flags, etc.)
- **Class imbalance** (~8% fraud rate) is addressed via `class_weight='balanced'` in supervised models
  and `contamination` parameter in Isolation Forest.
- **Train/test split**: 80/20 stratified split ensures the fraud rate is preserved across splits.
- Models are trained on engineered features including ratios, flags, and normalized scores.

---
## GenAI / RAG Limitations
- The **RAG claim review** module uses **TF-IDF cosine similarity** — not a large language model.
  This approach is fully deterministic, requires no API keys, and is reproducible.
- Retrieved policy evidence is **approximate** — similarity scores depend on vocabulary overlap,
  not semantic understanding.
- Review summaries are **template-based**, not generatively hallucinated. Every field is
  deterministically derived from claim data and policy document retrieval.
- A real production RAG system would use dense embeddings (e.g., sentence-transformers)
  and an LLM for natural-language generation.

---
## Human-in-the-Loop
- **This system is designed to assist, not replace, human analysts.**
- All HIGH-risk recommendations require human review before payment suspension or denial.
- Model scores are probabilistic; a high score indicates elevated risk, not confirmed fraud.
- Final adjudication must involve a licensed insurance professional with access to
  complete medical records and provider attestation.

---
## Bias & Fairness Considerations
- No demographic features (race, gender, religion) are used as model inputs.
- Age is included as a clinical indicator, not a discriminatory factor.
- Fraud labels are based on behavioral signals, not identity characteristics.
- Regular retraining and fairness audits would be required in production deployment.

---
## Data Governance
- In production, all claims data would be subject to HIPAA privacy protections.
- Audit logs of all model predictions and analyst actions would be maintained.
- Model versioning and drift monitoring would be implemented.
- Explainability reports (feature importance, SHAP values) would accompany each risk score.
""")
