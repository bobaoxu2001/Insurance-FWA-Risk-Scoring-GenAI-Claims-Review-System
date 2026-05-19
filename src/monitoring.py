"""
Model & data monitoring module for the FWA pipeline.

Produces:
  - outputs/reports/model_monitoring_report.csv
        month, n_claims, fraud_rate, mean_claim_amount, p95_claim_amount,
        missing_rate, outlier_rate
  - outputs/reports/data_quality_summary.csv
        column-level missing rates, duplicate rate, outlier rate
  - outputs/figures/monthly_fraud_rate.png
  - outputs/figures/claim_amount_drift.png

These are the building blocks any production FWA system needs day one:
volume, label-rate stability, dollar drift, and data-quality regressions.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _load_claims():
    # Monitor the RAW file so missing-value rates reflect true upstream quality.
    for p in [
        os.path.join(config.DATA_RAW,       "synthetic_claims.csv"),
        os.path.join(config.DATA_PROCESSED, "claims_features.csv"),
        os.path.join(config.DATA_PROCESSED, "claims_encoded.csv"),
    ]:
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["claim_date"])
            print(f"  Loaded {len(df)} rows from {p}")
            return df
    raise FileNotFoundError("No claims data found. Run data_generation first.")


def monthly_report(df):
    df = df.copy()
    df["month"] = df["claim_date"].dt.to_period("M").astype(str)

    p99 = df["claim_amount"].quantile(0.99)

    def agg(g):
        return pd.Series({
            "n_claims": len(g),
            "fraud_rate": round(float(g["fraud_label"].mean()), 4),
            "mean_claim_amount": round(float(g["claim_amount"].mean()), 2),
            "p95_claim_amount": round(float(g["claim_amount"].quantile(0.95)), 2),
            "missing_rate": round(float(g.isna().mean().mean()), 4),
            "outlier_rate": round(float((g["claim_amount"] > p99).mean()), 4),
        })

    report = df.groupby("month").apply(agg).reset_index()
    return report


def data_quality_summary(df):
    """Column-level data quality: missing rate, n unique, dtype."""
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_missing": int(df[col].isna().sum()),
            "missing_rate": round(float(df[col].isna().mean()), 4),
            "n_unique": int(df[col].nunique(dropna=True)),
        })
    qa = pd.DataFrame(rows)

    duplicate_claim_id_rate = (
        round(float(df["claim_id"].duplicated().mean()), 4)
        if "claim_id" in df.columns else None
    )
    p99 = df["claim_amount"].quantile(0.99) if "claim_amount" in df.columns else None
    outlier_rate = (
        round(float((df["claim_amount"] > p99).mean()), 4)
        if p99 is not None else None
    )

    summary = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "duplicate_claim_id_rate": duplicate_claim_id_rate,
        "claim_amount_p99": float(p99) if p99 is not None else None,
        "claim_amount_outlier_rate_above_p99": outlier_rate,
        "overall_missing_rate": round(float(df.isna().mean().mean()), 4),
    }
    return qa, summary


def plot_monthly_fraud_rate(report):
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(report["month"], report["fraud_rate"], marker="o", color="#C62828", lw=2)
    ax.set_ylabel("Fraud rate")
    ax.set_xlabel("Month")
    ax.set_title("Monthly Fraud Rate — Label Drift Monitor")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(config.OUTPUTS_FIGURES, "monthly_fraud_rate.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved {path}")


def plot_claim_amount_drift(df, report):
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(report["month"], report["mean_claim_amount"], marker="o",
            color="#1565C0", lw=2, label="Mean")
    ax.plot(report["month"], report["p95_claim_amount"], marker="s",
            color="#EF6C00", lw=2, label="P95")
    ax.set_ylabel("Claim amount ($)")
    ax.set_xlabel("Month")
    ax.set_title("Monthly Claim-Amount Drift — Mean & P95")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(config.OUTPUTS_FIGURES, "claim_amount_drift.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved {path}")


def main():
    print("Running model & data monitoring...")
    df = _load_claims()

    print("  Computing monthly report...")
    report = monthly_report(df)
    os.makedirs(config.OUTPUTS_REPORTS, exist_ok=True)
    report_path = os.path.join(config.OUTPUTS_REPORTS, "model_monitoring_report.csv")
    report.to_csv(report_path, index=False)
    print(f"  Saved {report_path}  ({len(report)} months)")

    print("  Computing data-quality summary...")
    qa, summary = data_quality_summary(df)
    qa_path = os.path.join(config.OUTPUTS_REPORTS, "data_quality_summary.csv")
    qa.to_csv(qa_path, index=False)
    print(f"  Saved {qa_path}")
    print("  Summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    print("  Generating charts...")
    plot_monthly_fraud_rate(report)
    plot_claim_amount_drift(df, report)

    print("Monitoring complete.")


if __name__ == "__main__":
    main()
