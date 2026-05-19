"""
Explainability module for FWA risk scoring.
Generates feature importance and human-readable explanations.

When real Kaggle provider data is present (provider_modeling_table.csv):
  - Produces outputs/reports/high_risk_provider_explanations.csv
  - Uses provider-level language ("Provider's avg reimbursement..." etc.)

When using synthetic claim data:
  - Produces outputs/reports/high_risk_claim_explanations.csv  (legacy)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    from sklearn.inspection import permutation_importance


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


def _get_feature_cols(df, target=None):
    if target is None:
        target = _detect_target(df) or "fraud_label"
    return [
        c for c in df.columns
        if c not in ID_LIKE_COLS + [target] + EXCLUDE_FROM_MODEL
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def load_model_and_data():
    model_path = os.path.join(config.OUTPUTS_MODELS, "best_fwa_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run modeling.py first.")
    model = joblib.load(model_path)

    provider_path = os.path.join(config.DATA_PROCESSED, "provider_modeling_table.csv")
    if os.path.exists(provider_path):
        df = pd.read_csv(provider_path)
        print(f"  Loaded real provider data from {provider_path}")
        return model, df, "real"

    for p in [
        os.path.join(config.DATA_PROCESSED, "claims_features.csv"),
        os.path.join(config.DATA_PROCESSED, "claims_encoded.csv"),
        os.path.join(config.DATA_RAW,       "synthetic_claims.csv"),
    ]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"  Loaded synthetic data from {p}")
            return model, df, "synthetic"

    raise FileNotFoundError("No data file found.")


def get_feature_importance(model, df):
    feature_cols = _get_feature_cols(df)
    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    target = _detect_target(df) or "fraud_label"

    if HAS_SHAP and hasattr(model, "estimators_"):
        try:
            print("  Using SHAP for feature importance...")
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X.sample(min(500, len(X)), random_state=42))
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:  # newer SHAP returns (n, p, k)
                sv = sv[..., 1] if sv.shape[-1] == 2 else sv.mean(axis=-1)
            importances = np.abs(sv).mean(axis=0)
            if importances.shape[0] != len(feature_cols):
                raise ValueError("SHAP feature dimension mismatch; falling back")
        except Exception as e:
            print(f"  SHAP failed ({e}); falling back to model importances.")
            importances = (
                model.feature_importances_ if hasattr(model, "feature_importances_")
                else np.abs(model.coef_[0])
            )
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("  Using permutation importance...")
        y = df[target]
        result = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances = result.importances_mean

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi_df


def save_top_risk_factors(fi_df):
    os.makedirs(config.OUTPUTS_REPORTS, exist_ok=True)
    out_path = os.path.join(config.OUTPUTS_REPORTS, "top_risk_factors.csv")
    fi_df.head(20).to_csv(out_path, index=False)
    print(f"  Saved top risk factors to {out_path}")


def generate_claim_explanations(model, df):
    feature_cols = _get_feature_cols(df)
    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y_prob = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["model_risk_score"] = y_prob

    # Select top high-risk claims
    high_risk = df[y_prob >= config.HIGH_RISK_THRESHOLD].copy()
    if len(high_risk) < 10:
        high_risk = df.nlargest(20, "model_risk_score")

    explanations = []
    for _, row in high_risk.head(50).iterrows():
        explanation_parts = []

        # Claim amount vs provider average
        if "claim_to_provider_avg_ratio" in row and not pd.isna(row["claim_to_provider_avg_ratio"]):
            ratio = row["claim_to_provider_avg_ratio"]
            if ratio > 1.5:
                explanation_parts.append(
                    f"Claim amount is {ratio:.1f}x provider average"
                )

        # Documentation quality
        if "documentation_score" in row:
            doc_score = row["documentation_score"]
            if doc_score < 0.4:
                explanation_parts.append(
                    f"Low documentation score ({doc_score:.2f}/1.0)"
                )

        # Duplicate flag
        if row.get("duplicate_claim_flag", 0) == 1:
            explanation_parts.append("Duplicate claim flag triggered")

        # Late submission
        if row.get("late_submission_flag", 0) == 1:
            explanation_parts.append("Late submission (>90 days)")

        # Suspicious keywords
        if row.get("suspicious_keyword_count", 0) >= 3:
            explanation_parts.append(
                f"{int(row['suspicious_keyword_count'])} suspicious keywords detected"
            )

        # High amount vs provider average (replacement for retired high_cost_outlier_flag)
        if row.get("high_amount_risk_flag", 0) == 1:
            explanation_parts.append("Claim amount >2x provider average")

        # Prior claims
        if row.get("prior_claim_count", 0) > 10:
            explanation_parts.append(
                f"High prior claim history ({int(row['prior_claim_count'])} claims)"
            )

        cid = row.get("claim_id", "UNKNOWN")
        explanations.append({
            "claim_id": cid,
            "model_risk_score": round(row["model_risk_score"], 4),
            "fraud_label": int(row.get("fraud_label", -1)),
            "claim_amount": round(row.get("claim_amount", 0), 2),
            "explanation": "; ".join(explanation_parts) if explanation_parts else "Multiple elevated risk signals",
            "explanation_count": len(explanation_parts),
        })

    expl_df = pd.DataFrame(explanations)
    return expl_df


def save_explanations(expl_df, mode="synthetic"):
    os.makedirs(config.OUTPUTS_REPORTS, exist_ok=True)
    if mode == "real":
        out_path = os.path.join(config.OUTPUTS_REPORTS, "high_risk_provider_explanations.csv")
    else:
        out_path = os.path.join(config.OUTPUTS_REPORTS, "high_risk_claim_explanations.csv")
    expl_df.to_csv(out_path, index=False)
    print(f"  Saved {len(expl_df)} explanations to {out_path}")


def generate_provider_explanations(model, df):
    """
    Generate business-readable provider-level explanations for real Kaggle data.
    """
    feature_cols = _get_feature_cols(df)
    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y_prob = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["model_risk_score"] = y_prob

    # Compute comparison stats across all providers
    stats = {}
    for col in ["avg_reimbursed_per_claim", "reimbursement_per_beneficiary",
                "total_claims", "avg_chronic_conditions",
                "avg_admission_duration", "inpatient_ratio"]:
        if col in df.columns:
            stats[col] = {
                "median": df[col].median(),
                "p90":    df[col].quantile(0.90),
                "std":    df[col].std(),
            }

    high_risk = df.nlargest(50, "model_risk_score")

    explanations = []
    for _, row in high_risk.iterrows():
        parts = []
        pid = row.get("Provider", "UNKNOWN")

        # Reimbursement vs median
        if "avg_reimbursed_per_claim" in stats and not pd.isna(row.get("avg_reimbursed_per_claim")):
            med = stats["avg_reimbursed_per_claim"]["median"]
            val = row["avg_reimbursed_per_claim"]
            if med > 0 and val > med * 1.5:
                parts.append(
                    f"Avg reimbursement per claim (${val:,.0f}) is "
                    f"{val/med:.1f}x the overall median (${med:,.0f})"
                )

        # Inpatient ratio vs p90
        if "inpatient_ratio" in stats and not pd.isna(row.get("inpatient_ratio")):
            p90 = stats["inpatient_ratio"]["p90"]
            val = row["inpatient_ratio"]
            if val > p90:
                parts.append(
                    f"Inpatient ratio ({val:.2f}) is above the 90th percentile ({p90:.2f}) — "
                    "unusually high share of costlier inpatient billing"
                )

        # Claim volume
        if "total_claims" in stats and not pd.isna(row.get("total_claims")):
            p90 = stats["total_claims"]["p90"]
            val = row["total_claims"]
            if val > p90:
                parts.append(
                    f"Provider submits {int(val):,} total claims, above the 90th "
                    f"percentile ({int(p90):,}) — elevated billing volume"
                )

        # Beneficiary load
        if "reimbursement_per_beneficiary" in stats and not pd.isna(row.get("reimbursement_per_beneficiary")):
            med = stats["reimbursement_per_beneficiary"]["median"]
            val = row["reimbursement_per_beneficiary"]
            if med > 0 and val > med * 2:
                parts.append(
                    f"Reimbursement per beneficiary (${val:,.0f}) is "
                    f"{val/med:.1f}x median — may indicate unnecessary services"
                )

        # Chronic conditions
        if "avg_chronic_conditions" in stats and not pd.isna(row.get("avg_chronic_conditions")):
            p90 = stats["avg_chronic_conditions"]["p90"]
            val = row["avg_chronic_conditions"]
            if val > p90:
                parts.append(
                    f"Avg chronic conditions per beneficiary ({val:.1f}) is elevated "
                    f"(p90={p90:.1f}) — higher-complexity patient mix"
                )

        if not parts:
            parts.append("Composite model risk score elevated; multiple moderate signals")

        explanations.append({
            "Provider":          pid,
            "model_risk_score":  round(float(row["model_risk_score"]), 4),
            "PotentialFraud":    int(row.get("PotentialFraud", -1)),
            "explanation":       "; ".join(parts[:5]),
            "explanation_count": min(len(parts), 5),
        })

    return pd.DataFrame(explanations)


def main():
    print("Running explainability pipeline...")
    result = load_model_and_data()
    model, df, mode = result

    print("  Computing feature importance...")
    fi_df = get_feature_importance(model, df)
    print(f"  Top 5 features:\n{fi_df.head()}")
    save_top_risk_factors(fi_df)

    if mode == "real":
        print("  Generating provider-level explanations (real Kaggle data)...")
        expl_df = generate_provider_explanations(model, df)
    else:
        print("  Generating claim-level explanations (synthetic data)...")
        expl_df = generate_claim_explanations(model, df)

    save_explanations(expl_df, mode=mode)
    print("Explainability complete.")


if __name__ == "__main__":
    main()
