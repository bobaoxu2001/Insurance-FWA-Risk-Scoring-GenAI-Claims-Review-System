"""
Feature engineering module for FWA risk scoring.
Adds domain-specific engineered features and a rule-based risk score.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data():
    path = os.path.join(config.DATA_PROCESSED, "claims_encoded.csv")
    if not os.path.exists(path):
        # Fall back to raw
        path = os.path.join(config.DATA_RAW, "synthetic_claims.csv")
    df = pd.read_csv(path)
    print(f"  Loaded data from {path}")
    return df


def create_features(df):
    """
    Add engineered features. NOTE: we intentionally avoid features that are
    near-deterministic functions of the (hidden) fraud_label. All engineered
    features below are derived from columns that an analyst could observe at
    claim-intake time.
    """
    # Ratio of claim amount to provider average
    df["claim_to_provider_avg_ratio"] = (
        df["claim_amount"] / (df["provider_avg_claim_amount"] + 1e-6)
    ).round(4)

    # Approval ratio (post-adjudication signal; useful but noisy)
    df["approval_ratio"] = (
        df["approved_amount"] / (df["claim_amount"] + 1e-6)
    ).round(4)

    # Claimant frequency score (normalized prior claims)
    df["claimant_claim_frequency_score"] = (
        df["prior_claim_count"] / (df["prior_claim_count"].max() + 1e-6)
    ).round(4)

    # Provider risk score: high volume + high avg = higher risk
    vol_norm = (df["provider_claim_volume"] - df["provider_claim_volume"].min()) / (
        df["provider_claim_volume"].max() - df["provider_claim_volume"].min() + 1e-6
    )
    avg_norm = (df["provider_avg_claim_amount"] - df["provider_avg_claim_amount"].min()) / (
        df["provider_avg_claim_amount"].max() - df["provider_avg_claim_amount"].min() + 1e-6
    )
    df["provider_risk_score"] = ((vol_norm + avg_norm) / 2).round(4)

    # Documentation risk flag (low documentation score)
    df["documentation_risk_flag"] = (df["documentation_score"] < 0.4).astype(int)

    # Suspicious text risk flag
    df["suspicious_text_risk_flag"] = (df["suspicious_keyword_count"] >= 3).astype(int)

    # High amount risk flag (claim > 2x provider average) — soft signal
    df["high_amount_risk_flag"] = (df["claim_to_provider_avg_ratio"] > 2.0).astype(int)

    # New-policy flag (early policy period historically higher risk)
    df["new_policy_flag"] = (df["days_since_policy_start"] < 180).astype(int)

    return df


def create_rule_based_risk_score(df):
    """Simple weighted rule-based composite risk score (0-1 range)."""
    score = np.zeros(len(df))

    # High amount vs provider avg
    score += 0.25 * np.clip(df["claim_to_provider_avg_ratio"] / 5.0, 0, 1)

    # Low documentation
    score += 0.20 * (1.0 - df["documentation_score"])

    # Duplicate flag
    score += 0.20 * df["duplicate_claim_flag"]

    # Late submission
    score += 0.10 * df["late_submission_flag"]

    # Suspicious keywords
    score += 0.10 * np.clip(df["suspicious_keyword_count"] / 8.0, 0, 1)

    # Prior claims (normalized)
    score += 0.08 * df["claimant_claim_frequency_score"]

    # Provider risk
    score += 0.07 * df["provider_risk_score"]

    df["rule_based_risk_score"] = score.round(4)
    return df


def save_features(df):
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)
    out_path = os.path.join(config.DATA_PROCESSED, "claims_features.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved feature-engineered data ({df.shape}) to {out_path}")


def main():
    print("Running feature engineering...")
    df = load_data()
    df = create_features(df)
    df = create_rule_based_risk_score(df)

    new_features = [
        "claim_to_provider_avg_ratio", "approval_ratio",
        "claimant_claim_frequency_score", "provider_risk_score",
        "documentation_risk_flag", "suspicious_text_risk_flag",
        "high_amount_risk_flag", "new_policy_flag", "rule_based_risk_score"
    ]
    print(f"  Added {len(new_features)} engineered features: {new_features}")

    save_features(df)
    print("Feature engineering complete.")


if __name__ == "__main__":
    main()
