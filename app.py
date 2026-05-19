"""
Streamlit dashboard — Healthcare Provider FWA Risk Scoring & GenAI Review System.

Six tabs:
  1. Executive Overview
  2. Provider FWA Pattern Explorer
  3. Model Performance
  4. High-Risk Provider Review Assistant
  5. Model Monitoring & Data Quality
  6. Auditability & Responsible AI

Data-source awareness:
  - If data/processed/provider_modeling_table.csv exists → real Kaggle mode
  - Otherwise → synthetic demo mode
  - App NEVER crashes if files are missing; shows setup instructions instead.
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
    page_title="Healthcare Provider FWA Risk Scoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Constants ────────────────────────────────────────────────────────────────

ID_LIKE_COLS = [
    "claim_id", "policyholder_id", "provider_id", "claim_date",
    "service_type", "diagnosis_group", "state",
    "Provider",
]
EXCLUDE_FROM_MODEL = ["rule_based_risk_score"]
TARGET_COLS = ["PotentialFraud", "fraud_label"]


def _detect_target(df):
    for t in TARGET_COLS:
        if t in df.columns:
            return t
    return None


# ── Data loading ────────────────────────────────────────────────────────────

@st.cache_data
def detect_data_mode():
    """Returns 'real' or 'synthetic' based on what files exist."""
    if os.path.exists(os.path.join(config.DATA_PROCESSED, "provider_modeling_table.csv")):
        return "real"
    return "synthetic"


@st.cache_data
def load_modeling_data():
    """Load the best available modeling table."""
    provider_path = os.path.join(config.DATA_PROCESSED, "provider_modeling_table.csv")
    if os.path.exists(provider_path):
        return pd.read_csv(provider_path), "real"

    for p in [
        os.path.join(config.DATA_PROCESSED, "claims_features.csv"),
        os.path.join(config.DATA_PROCESSED, "claims_encoded.csv"),
        os.path.join(config.DATA_RAW,       "synthetic_claims.csv"),
    ]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "claim_date" in df.columns:
                df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")
            return df, "synthetic"

    return None, "none"


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
    for name in ["data_quality_report.csv", "data_quality_summary.csv"]:
        p = os.path.join(config.OUTPUTS_REPORTS, name)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


@st.cache_resource
def load_model():
    p = os.path.join(config.OUTPUTS_MODELS, "best_fwa_model.pkl")
    return joblib.load(p) if os.path.exists(p) else None


def load_review(entity_id):
    p = os.path.join(config.OUTPUTS_REVIEWS, f"review_{entity_id}.txt")
    if os.path.exists(p):
        with open(p) as f:
            return f.read()
    return None


def get_risk_scores(df, model):
    if model is None:
        return None
    target = _detect_target(df)
    drop = ID_LIKE_COLS + ([target] if target else []) + EXCLUDE_FROM_MODEL
    feature_cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    try:
        X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
        return model.predict_proba(X)[:, 1]
    except Exception:
        return None


def list_review_ids():
    review_dir = config.OUTPUTS_REVIEWS
    if not os.path.isdir(review_dir):
        return []
    return sorted([
        f.replace("review_", "").replace(".txt", "")
        for f in os.listdir(review_dir)
        if f.startswith("review_") and f.endswith(".txt")
    ])


# ── Load everything ──────────────────────────────────────────────────────────

df, data_mode = load_modeling_data()
metrics       = load_metrics()
model         = load_model()

data_source_label = (
    "Real Kaggle Dataset" if data_mode == "real"
    else "Synthetic Demo Data" if data_mode == "synthetic"
    else "No Data"
)

if df is not None and model is not None:
    risk_scores = get_risk_scores(df, model)
    if risk_scores is not None:
        df = df.copy()
        df["model_risk_score"] = risk_scores

target_col = _detect_target(df) if df is not None else None
id_col     = "Provider" if data_mode == "real" else "claim_id"


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🛡️ Provider FWA Risk Scoring")
st.sidebar.markdown("**Healthcare Fraud, Waste & Abuse Analytics**")
st.sidebar.markdown(f"**Data Source:** `{data_source_label}`")
if data_mode == "real":
    st.sidebar.success("Real Kaggle data loaded.")
elif data_mode == "synthetic":
    st.sidebar.warning("Synthetic demo data — download Kaggle dataset for full pipeline.")
else:
    st.sidebar.error("No data found. Run the pipeline first.")

st.sidebar.markdown("---")
tab_choice = st.sidebar.radio(
    "Navigate to",
    [
        "Executive Overview",
        "Provider FWA Pattern Explorer",
        "Model Performance",
        "High-Risk Provider Review Assistant",
        "Model Monitoring & Data Quality",
        "Auditability & Responsible AI",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Portfolio demo — not clinical or production grade.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Disclaimer:** This project uses the public Kaggle Healthcare Provider "
    "Fraud Detection dataset. It is NOT Manulife/John Hancock data, NOT LTC-specific, "
    "and contains no PHI. RAG policy text is synthetic."
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Overview
# ════════════════════════════════════════════════════════════════════════════

if tab_choice == "Executive Overview":
    st.title("📊 Executive Overview")

    if data_mode == "real":
        st.info(
            "Data source: **Kaggle Healthcare Provider Fraud Detection Analysis** "
            "(public educational dataset). Not Manulife / John Hancock data. "
            "Not Long Term Care-specific. All model outputs require human analyst review."
        )
    elif data_mode == "synthetic":
        st.warning(
            "**Demo/Sample mode** — Running on synthetic data. "
            "Download the Kaggle dataset (see Auditability tab) for real provider analysis."
        )
    else:
        st.error("No modeling data found. Run the data pipeline first.")
        st.stop()

    if df is None:
        st.warning("No data loaded.")
    else:
        if data_mode == "real":
            n_providers      = len(df)
            fraud_rate       = df[target_col].mean() if target_col else 0.0
            total_claims_rep = int(df["total_claims"].sum()) if "total_claims" in df.columns else 0
            avg_reimb        = df["avg_reimbursed_per_claim"].mean() if "avg_reimbursed_per_claim" in df.columns else 0.0
            high_risk_cnt    = int((df["model_risk_score"] >= config.HIGH_RISK_THRESHOLD).sum()) \
                               if "model_risk_score" in df.columns else "N/A"
        else:
            n_providers      = df["provider_id"].nunique() if "provider_id" in df.columns else len(df)
            fraud_rate       = df[target_col].mean() if target_col else 0.0
            total_claims_rep = len(df)
            avg_reimb        = df["claim_amount"].mean() if "claim_amount" in df.columns else 0.0
            high_risk_cnt    = int((df["model_risk_score"] >= config.HIGH_RISK_THRESHOLD).sum()) \
                               if "model_risk_score" in df.columns else "N/A"

        best_auc = best_f1 = "N/A"
        clean_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
        if clean_metrics:
            best_name = max(clean_metrics, key=lambda n: clean_metrics[n].get("roc_auc", 0))
            best_auc  = f"{clean_metrics[best_name]['roc_auc']:.3f}"
            best_f1   = f"{clean_metrics[best_name]['f1']:.3f}"

        label1 = "Providers Analyzed" if data_mode == "real" else "Total Claims"
        label3 = "Total Claims Rep." if data_mode == "real" else "Total Claims"
        label4 = "Avg Reimbursement" if data_mode == "real" else "Avg Claim Amount"

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric(label1,          f"{n_providers:,}")
        c2.metric("Fraud Rate",    f"{fraud_rate:.1%}")
        c3.metric(label3,          f"{total_claims_rep:,}")
        c4.metric(label4,          format_currency(avg_reimb))
        c5.metric("Best AUC",      best_auc)
        c6.metric("Best F1",       best_f1)

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

        with col_b:
            st.subheader("Fraud Label Distribution")
            if target_col and target_col in df.columns:
                label_map = {0: "Legitimate", 1: "Fraudulent"}
                counts = df[target_col].value_counts().rename(label_map)
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
                       colors=["#4CAF50", "#F44336"],
                       startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
                st.pyplot(fig); plt.close()

        st.markdown("---")
        entity_label = "Providers" if data_mode == "real" else "Claims"
        st.subheader(f"Top 10 Highest-Risk {entity_label}")
        if "model_risk_score" in df.columns:
            show_cols = [c for c in [
                id_col, "total_claims", "avg_reimbursed_per_claim",
                "claim_amount", "service_type",
                "model_risk_score", target_col,
            ] if c and c in df.columns]
            top_risky = df.nlargest(10, "model_risk_score")[show_cols].reset_index(drop=True)
            top_risky["risk_level"] = top_risky["model_risk_score"].apply(get_risk_level)
            st.dataframe(top_risky, use_container_width=True)
            if data_mode == "synthetic":
                st.caption("(Demo/Sample) Synthetic data — for illustration only.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Provider FWA Pattern Explorer
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Provider FWA Pattern Explorer":
    st.title("🔍 Provider FWA Pattern Explorer")

    if data_mode == "synthetic":
        st.warning("(Demo/Sample) Showing synthetic claim-level patterns. "
                   "Download Kaggle data for real provider-level analytics.")

    if df is None:
        st.warning("No data available.")
    else:
        if data_mode == "real":
            # Real provider-level charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Provider Risk Score Distribution")
                p = os.path.join(config.OUTPUTS_FIGURES, "provider_risk_distribution.png")
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
                elif "model_risk_score" in df.columns:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(df["model_risk_score"], bins=40, color="#2196F3", edgecolor="white")
                    ax.set_xlabel("Model Risk Score")
                    st.pyplot(fig); plt.close()

            with col2:
                st.subheader("Reimbursement Distribution")
                p = os.path.join(config.OUTPUTS_FIGURES, "reimbursement_distribution.png")
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
                elif "avg_reimbursed_per_claim" in df.columns:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    data = df["avg_reimbursed_per_claim"].dropna()
                    ax.hist(data[data <= data.quantile(0.99)], bins=50, color="#1565C0", alpha=0.8)
                    ax.set_xlabel("Avg Reimbursement / Claim")
                    st.pyplot(fig); plt.close()
                st.caption("Provider-level total reimbursement; outliers above P99 warrant review regardless of model score.")

            st.markdown("---")
            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Fraud Rate by Provider Volume Bucket")
                p = os.path.join(config.OUTPUTS_FIGURES, "fraud_rate_by_volume_bucket.png")
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
                    st.caption("Higher-volume providers don't always have more fraud — "
                               "but extreme outliers are worth investigating.")

            with col4:
                st.subheader("Inpatient vs Outpatient Mix")
                if "inpatient_ratio" in df.columns and target_col in df.columns:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    for label, grp in df.groupby(target_col):
                        ax.hist(
                            grp["inpatient_ratio"].dropna(), bins=30, alpha=0.6,
                            label=f"Fraud={label}",
                            color="#C62828" if label == 1 else "#1565C0"
                        )
                    ax.set_xlabel("Inpatient Ratio")
                    ax.set_ylabel("Provider count")
                    ax.legend()
                    st.pyplot(fig); plt.close()
                    st.caption("Providers with a very high inpatient ratio may be upcoding "
                               "outpatient visits to the more costly inpatient setting.")

            st.markdown("---")
            st.subheader("Top 15 Highest-Risk Providers")
            if "model_risk_score" in df.columns and "Provider" in df.columns:
                show = [c for c in [
                    "Provider", "total_claims", "unique_beneficiaries",
                    "avg_reimbursed_per_claim", "inpatient_ratio",
                    "model_risk_score", target_col,
                ] if c and c in df.columns]
                top15 = df.nlargest(15, "model_risk_score")[show].reset_index(drop=True)
                for col in ["avg_reimbursed_per_claim"]:
                    if col in top15.columns:
                        top15[col] = top15[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                if "inpatient_ratio" in top15.columns:
                    top15["inpatient_ratio"] = top15["inpatient_ratio"].apply(
                        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                    )
                st.dataframe(top15, use_container_width=True)

        else:
            # Synthetic claim-level charts (legacy)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Claim Amount by Fraud Label")
                fig, ax = plt.subplots(figsize=(6, 4))
                if "fraud_label" in df.columns and "claim_amount" in df.columns:
                    for lab, col_str in [(0, "#4CAF50"), (1, "#F44336")]:
                        data = df[df["fraud_label"] == lab]["claim_amount"].clip(upper=50000)
                        ax.hist(data, bins=50, alpha=0.6, color=col_str,
                                label=["Legitimate", "Fraudulent"][lab])
                ax.set_xlabel("Claim Amount ($)")
                ax.legend()
                st.pyplot(fig); plt.close()
                st.caption("(Demo/Sample)")

            with col2:
                st.subheader("Fraud Rate by Service Type")
                if "service_type" in df.columns and target_col in df.columns:
                    st_fraud = (
                        df.groupby("service_type")[target_col].mean()
                          .sort_values(ascending=True)
                    )
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(st_fraud.index, st_fraud.values * 100,
                            color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(st_fraud))))
                    ax.set_xlabel("Fraud Rate (%)")
                    st.pyplot(fig); plt.close()
                    st.caption("(Demo/Sample)")

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Top 15 Risky Providers (by avg score)")
                if "provider_id" in df.columns and "model_risk_score" in df.columns:
                    prov_risk = (
                        df.groupby("provider_id")
                          .agg(
                              avg_risk_score=("model_risk_score", "mean"),
                              claim_count=("claim_id", "count"),
                          )
                          .sort_values("avg_risk_score", ascending=False)
                          .head(15).reset_index()
                    )
                    st.dataframe(prov_risk, use_container_width=True)
                    st.caption("(Demo/Sample)")

            with col4:
                st.subheader("Documentation Score vs Claim Amount")
                if "documentation_score" in df.columns and "claim_amount" in df.columns:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sample = df.sample(min(1000, len(df)), random_state=42)
                    c = sample[target_col].map({0: "#4CAF50", 1: "#F44336"}) \
                        if target_col in sample.columns else "#2196F3"
                    ax.scatter(sample["documentation_score"],
                               sample["claim_amount"].clip(upper=30000),
                               c=c, alpha=0.4, s=12)
                    ax.set_xlabel("Documentation Score")
                    ax.set_ylabel("Claim Amount ($)")
                    st.pyplot(fig); plt.close()
                    st.caption("(Demo/Sample)")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Model Performance":
    st.title("🤖 Model Performance")
    st.info(
        "Because fraud is rare, the precision-recall tradeoff matters more than "
        "raw accuracy. A precision-oriented threshold sends fewer false positives "
        "to analysts; a recall-oriented threshold catches more fraud at the cost "
        "of more reviews."
    )

    if data_mode == "synthetic":
        st.warning("(Demo/Sample) — Metrics are from synthetic training data.")

    if not metrics:
        st.warning("No model metrics found. Run `python src/modeling.py` first.")
    else:
        clean_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
        note = metrics.get("_note", "")
        if note:
            st.caption(f"Data source note: {note}")

        st.subheader("Evaluation Metrics (Test Set)")
        if clean_metrics:
            metrics_df = pd.DataFrame(clean_metrics).T.reset_index().rename(columns={"index": "Model"})
            metrics_df = metrics_df.sort_values("roc_auc", ascending=False)
            st.dataframe(metrics_df, use_container_width=True)

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
            st.caption("Top-right = missed fraud (worst error for FWA).")
        else:
            st.info("Run `python src/modeling.py` to generate.")

    with col2:
        st.subheader("ROC Curves")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
            st.caption("Higher curve = better separation of fraudulent vs. legitimate providers across all thresholds.")
        else:
            st.info("Run `python src/modeling.py`.")

    st.markdown("---")
    col3, col4 = st.columns([2, 3])
    with col3:
        st.subheader("Precision-Recall Curve")
        if os.path.exists(pr_path):
            st.image(pr_path, use_container_width=True)
            st.caption("The chart to optimize against in FWA — "
                       "shows precision/recall tradeoff at each threshold.")
        else:
            st.info("Run `python src/modeling.py`.")

    with col4:
        st.subheader("Threshold Sweep")
        thr = load_threshold_table()
        if thr is not None:
            st.dataframe(thr, use_container_width=True, height=320)
            st.caption("Lowering the threshold flags more entities (higher recall, lower precision). "
                       "Operations teams pick the threshold based on reviewer capacity.")
        else:
            st.info("Run `python src/modeling.py`.")

    st.markdown("---")
    st.subheader("Feature Importance")
    if os.path.exists(fi_path):
        st.image(fi_path, use_container_width=True)
        st.caption("Top features by model importance. Provider-level features "
                   "show which billing patterns the model finds most predictive.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — High-Risk Provider Review Assistant
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "High-Risk Provider Review Assistant":
    entity_label = "Provider" if data_mode == "real" else "Claim"
    st.title(f"🔎 High-Risk {entity_label} Review Assistant")
    st.info(
        f"Select a {entity_label.lower()} ID to view the model risk score, "
        "top risk indicators, and the RAG-generated audit review."
    )

    if data_mode == "synthetic":
        st.warning("(Demo/Sample) — Showing synthetic claim reviews. "
                   "Download Kaggle data for real provider-level reviews.")

    if df is None:
        st.warning(f"No data found.")
    else:
        precomputed = list_review_ids()

        if data_mode == "real" and "Provider" in df.columns:
            all_ids = df["Provider"].dropna().astype(str).tolist()
        elif "claim_id" in df.columns:
            all_ids = df["claim_id"].dropna().astype(str).tolist()
        else:
            all_ids = []

        options = precomputed + [i for i in all_ids if i not in precomputed]
        if not options:
            st.warning(f"No {entity_label.lower()} IDs found. Run the pipeline first.")
        else:
            selected_id = st.selectbox(
                f"Select {entity_label} ID (pre-generated reviews shown first)",
                options, index=0,
            )

            if selected_id:
                if data_mode == "real":
                    row_mask = df["Provider"].astype(str) == selected_id
                elif "claim_id" in df.columns:
                    row_mask = df["claim_id"].astype(str) == selected_id
                else:
                    row_mask = pd.Series([False] * len(df))

                if row_mask.any():
                    row = df[row_mask].iloc[0]

                    risk_score = float(row.get("model_risk_score",
                                               row.get("rule_based_risk_score", 0.0)) or 0.0)
                    risk_level_str = get_risk_level(risk_score)
                    risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level_str, "⚪")

                    if data_mode == "real":
                        cols = st.columns(5)
                        cols[0].metric("Risk Score",           f"{risk_score:.3f}")
                        cols[1].metric("Risk Level",           f"{risk_color} {risk_level_str}")
                        cols[2].metric("Total Claims",         f"{int(row.get('total_claims', 0) or 0):,}")
                        cols[3].metric("Unique Beneficiaries", f"{int(row.get('unique_beneficiaries', 0) or 0):,}")
                        reimb = row.get("avg_reimbursed_per_claim", 0) or 0
                        cols[4].metric("Avg Reimbursement",   format_currency(float(reimb)))

                        st.markdown("---")
                        st.subheader("Provider Feature Summary")
                        disp_cols = [c for c in [
                            "total_claims", "inpatient_claim_count", "outpatient_claim_count",
                            "unique_beneficiaries", "inpatient_ratio",
                            "avg_reimbursed_per_claim", "total_reimbursed",
                            "avg_admission_duration", "avg_chronic_conditions", "death_rate",
                            "unique_attending_physicians", target_col,
                        ] if c and c in df.columns]
                        feat_display = pd.DataFrame(
                            {"Feature": disp_cols,
                             "Value": [row.get(c, "N/A") for c in disp_cols]}
                        )
                        st.dataframe(feat_display, use_container_width=True)
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Risk Score",    f"{risk_score:.3f}")
                        col2.metric("Risk Level",    f"{risk_color} {risk_level_str}")
                        col3.metric("Claim Amount",  format_currency(row.get("claim_amount", 0)))
                        col4.metric("Approved",      format_currency(row.get("approved_amount", 0)))

                    st.markdown("---")
                    st.subheader("AI-Generated Review (RAG)")
                    review_text = load_review(selected_id)
                    if review_text:
                        st.code(review_text, language="text")
                    else:
                        st.info(
                            f"No pre-generated review for {entity_label} `{selected_id}`. "
                            "Run `python src/rag_claim_review.py` to generate reviews."
                        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Monitoring & Data Quality
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Model Monitoring & Data Quality":
    st.title("📈 Model Monitoring & Data Quality")
    st.info(
        "A model is only as good as the data feeding it. This tab tracks "
        "volume, reimbursement distributions, class balance, and column-level "
        "data quality — the minimum monitoring surface any production FWA system needs."
    )

    if data_mode == "synthetic":
        st.warning("(Demo/Sample) — Monitoring on synthetic data.")

    report = load_monitoring_report()
    qa     = load_data_quality()

    # Provider-level charts
    p_prd = os.path.join(config.OUTPUTS_FIGURES, "provider_risk_distribution.png")
    p_re  = os.path.join(config.OUTPUTS_FIGURES, "reimbursement_distribution.png")
    p_fv  = os.path.join(config.OUTPUTS_FIGURES, "fraud_rate_by_volume_bucket.png")
    p_miss= os.path.join(config.OUTPUTS_FIGURES, "feature_missingness.png")
    # Claim-level charts
    p_mfr = os.path.join(config.OUTPUTS_FIGURES, "monthly_fraud_rate.png")
    p_cad = os.path.join(config.OUTPUTS_FIGURES, "claim_amount_drift.png")

    if data_mode == "real":
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(p_prd):
                st.subheader("Provider Risk Distribution")
                st.image(p_prd, use_container_width=True)
        with col2:
            if os.path.exists(p_re):
                st.subheader("Reimbursement Distribution")
                st.image(p_re, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if os.path.exists(p_fv):
                st.subheader("Fraud Rate by Volume Bucket")
                st.image(p_fv, use_container_width=True)
                st.caption("Provider volume alone is not a fraud signal — distribution should be roughly flat across buckets if volume is uninformative.")
        with col4:
            if os.path.exists(p_miss):
                st.subheader("Feature Missingness")
                st.image(p_miss, use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(p_mfr):
                st.subheader("Monthly Fraud Rate")
                st.image(p_mfr, use_container_width=True)
                st.caption("(Demo/Sample)")
        with col2:
            if os.path.exists(p_cad):
                st.subheader("Monthly Claim Amount Drift")
                st.image(p_cad, use_container_width=True)
                st.caption("(Demo/Sample)")
        if os.path.exists(p_miss):
            st.subheader("Feature Missingness")
            st.image(p_miss, use_container_width=True)

    if report is not None:
        st.subheader("Monitoring Report")
        st.dataframe(report, use_container_width=True)
    else:
        st.info("Run `python src/monitoring.py` to generate the monitoring report.")

    st.markdown("---")
    st.subheader("Data Quality — Column Summary")
    if qa is not None:
        st.dataframe(qa, use_container_width=True, height=400)
        st.caption("Missing rates, unique counts, and dtypes per feature column.")
    else:
        st.info("Run `python src/monitoring.py` to generate the data-quality summary.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Auditability & Responsible AI
# ════════════════════════════════════════════════════════════════════════════

else:  # "Auditability & Responsible AI"
    st.title("📋 Auditability & Responsible AI")

    st.markdown("""
### Data Disclaimer

This project uses the **Kaggle Healthcare Provider Fraud Detection Analysis** dataset:
- Public educational dataset only (not proprietary)
- **NOT** Manulife data
- **NOT** John Hancock data
- **NOT** Long Term Care-specific claims data
- No real patient records, clinical documents, or claim notes
- RAG policy text and claim documents are **synthetic** — generated for demo purposes only
- No PHI (Protected Health Information) is present anywhere in this project

Dataset URL: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis

---
### How to Download the Real Dataset

1. Go to: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
2. Click **Download** (requires a free Kaggle account)
3. Unzip the archive
4. Place all CSV files in `data/raw/`
5. Run:
   ```bash
   python src/data_ingestion.py
   python src/provider_feature_engineering.py
   python src/modeling.py
   python src/explainability.py
   python src/rag_claim_review.py
   python src/monitoring.py
   ```

Without the real data, the pipeline runs on synthetic claims (demo mode).

---
### Human-in-the-Loop Design

Every risk score generated by this system is:
- A **recommendation**, not a decision
- Annotated with quantified risk indicators
- Retrieving applicable policy / audit rules via TF-IDF
- Flagged with specific documentation gaps for analyst follow-up
- Assigned a suggested action tied to risk level (LOW / MEDIUM / HIGH)

No payment is suspended or claim denied by the model alone. A licensed FWA
analyst must review every HIGH-risk flag before any action is taken.

---
### Why the Metrics Should Be Interpreted Carefully

- Provider-level labels are binary (Yes/No) with no partial credit for near-fraud patterns
- Class imbalance: ~9.4% fraud rate in Kaggle provider labels (still higher than real-world FWA prevalence, which is typically <2%)
- Provider-level aggregation smooths individual claim noise; granular patterns may be missed
- Model performance on a withheld test set is not a guarantee of production performance
- Feature importance is from training data — distribution shift can degrade results

---
### Model Limitations

| Limitation | Mitigation |
|---|---|
| No semantic RAG (TF-IDF only) | Flag retrieval gaps; note in review |
| No temporal modeling | Future work: time-series billing sequences |
| No graph/network features | Provider-beneficiary network analysis not yet included |
| No NLP on clinical notes | Synthetic text only; would require real EHR access |
| Binary labels (real-world is graded) | Use probability scores, not hard cutoffs |

---
### Repository Architecture

```
Insurance-FWA-Risk-Scoring-GenAI-Claims-Review-System/
├── src/
│   ├── data_generation.py              # Original synthetic data (kept for compatibility)
│   ├── synthetic_data_generation.py    # Renamed copy with synthetic-mode header
│   ├── data_ingestion.py               # NEW: loads & validates Kaggle CSV files
│   ├── provider_feature_engineering.py # NEW: 25+ provider-level features
│   ├── preprocessing.py                # Synthetic claim preprocessing
│   ├── feature_engineering.py          # Synthetic claim feature engineering
│   ├── modeling.py                     # ML models (real or synthetic)
│   ├── explainability.py               # Feature importance + explanations
│   ├── rag_claim_review.py             # TF-IDF RAG reviews (provider or claim level)
│   ├── monitoring.py                   # Data quality + model monitoring
│   └── utils.py
├── data/
│   ├── raw/                            # Place Kaggle CSVs here
│   ├── processed/                      # provider_modeling_table.csv or claims_features.csv
│   └── documents/                      # policy_rules.txt + synthetic claim docs
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── reports/
│   └── sample_reviews/
├── app.py                              # This Streamlit dashboard
├── config.py
├── README.md
└── resume_bullets.md
```
""")
