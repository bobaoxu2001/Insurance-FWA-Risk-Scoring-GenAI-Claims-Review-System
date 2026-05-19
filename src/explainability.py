"""
Explainability module for FWA risk scoring.
Generates feature importance and human-readable claim explanations.
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
]


def _get_feature_cols(df, target="fraud_label"):
    return [
        c for c in df.columns
        if c not in ID_LIKE_COLS + [target]
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def load_model_and_data():
    model_path = os.path.join(config.OUTPUTS_MODELS, "best_fwa_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run modeling.py first.")
    model = joblib.load(model_path)

    for p in [
        os.path.join(config.DATA_PROCESSED, "claims_features.csv"),
        os.path.join(config.DATA_PROCESSED, "claims_encoded.csv"),
        os.path.join(config.DATA_RAW,       "synthetic_claims.csv"),
    ]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"  Loaded data from {p}")
            return model, df

    raise FileNotFoundError("No data file found.")


def get_feature_importance(model, df):
    feature_cols = _get_feature_cols(df)
    X = df[feature_cols]

    if HAS_SHAP:
        print("  Using SHAP for feature importance...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.sample(500, random_state=42))
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        importances = np.abs(shap_values).mean(axis=0)
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("  Using permutation importance...")
        y = df["fraud_label"]
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
    X = df[feature_cols]
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

        # High cost outlier
        if row.get("high_cost_outlier_flag", 0) == 1:
            explanation_parts.append("Claim amount in top 10% (high-cost outlier)")

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


def save_explanations(expl_df):
    os.makedirs(config.OUTPUTS_REPORTS, exist_ok=True)
    out_path = os.path.join(config.OUTPUTS_REPORTS, "high_risk_claim_explanations.csv")
    expl_df.to_csv(out_path, index=False)
    print(f"  Saved {len(expl_df)} high-risk claim explanations to {out_path}")


def main():
    print("Running explainability pipeline...")
    model, df = load_model_and_data()

    print("  Computing feature importance...")
    fi_df = get_feature_importance(model, df)
    print(f"  Top 5 features:\n{fi_df.head()}")
    save_top_risk_factors(fi_df)

    print("  Generating claim-level explanations...")
    expl_df = generate_claim_explanations(model, df)
    save_explanations(expl_df)

    print("Explainability complete.")


if __name__ == "__main__":
    main()
