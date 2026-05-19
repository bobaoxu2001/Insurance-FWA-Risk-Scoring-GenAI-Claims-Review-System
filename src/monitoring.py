"""
Model & data monitoring module for the FWA pipeline.

Mode detection:
  - If data/processed/provider_modeling_table.csv exists → PROVIDER-LEVEL monitoring
  - Otherwise → CLAIM-LEVEL monitoring (legacy synthetic mode)

Provider-level outputs:
  - outputs/reports/data_quality_report.csv          (column-level missing/duplicate stats)
  - outputs/reports/model_monitoring_report.csv      (provider-level summary stats)
  - outputs/figures/provider_risk_distribution.png
  - outputs/figures/reimbursement_distribution.png
  - outputs/figures/fraud_rate_by_volume_bucket.png
  - outputs/figures/feature_missingness.png

Claim-level outputs (legacy):
  - outputs/reports/model_monitoring_report.csv      (monthly)
  - outputs/reports/data_quality_summary.csv
  - outputs/figures/monthly_fraud_rate.png
  - outputs/figures/claim_amount_drift.png
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


TARGET_COLS = ["PotentialFraud", "fraud_label"]


def _detect_target(df):
    for t in TARGET_COLS:
        if t in df.columns:
            return t
    return None


def _load_data():
    provider_path = os.path.join(config.DATA_PROCESSED, "provider_modeling_table.csv")
    if os.path.exists(provider_path):
        df = pd.read_csv(provider_path)
        print(f"  Loaded REAL provider data from {provider_path}  shape={df.shape}")
        return df, "real"

    for p in [
        os.path.join(config.DATA_RAW,       "synthetic_claims.csv"),
        os.path.join(config.DATA_PROCESSED, "claims_features.csv"),
        os.path.join(config.DATA_PROCESSED, "claims_encoded.csv"),
    ]:
        if os.path.exists(p):
            parse = {}
            if "claim_date" in pd.read_csv(p, nrows=0).columns:
                parse = {"parse_dates": ["claim_date"]}
            df = pd.read_csv(p, **parse)
            print(f"  Loaded synthetic data from {p}  shape={df.shape}")
            return df, "synthetic"

    raise FileNotFoundError("No data found. Run data generation or provider feature engineering first.")


# ── Provider-level monitoring ──────────────────────────────────────────────────

def data_quality_report(df):
    """Column-level: missing rate, n_unique, duplicate provider IDs."""
    rows = []
    for col in df.columns:
        rows.append({
            "column":       col,
            "dtype":        str(df[col].dtype),
            "n_missing":    int(df[col].isna().sum()),
            "missing_rate": round(float(df[col].isna().mean()), 4),
            "n_unique":     int(df[col].nunique(dropna=True)),
        })
    qa = pd.DataFrame(rows)

    dup_rate = None
    if "Provider" in df.columns:
        dup_rate = round(float(df["Provider"].duplicated().mean()), 4)

    target = _detect_target(df)
    fraud_rate = round(float(df[target].mean()), 4) if target else None

    summary = {
        "n_providers":       int(len(df)),
        "n_features":        int(df.shape[1]),
        "duplicate_provider_rate": dup_rate,
        "fraud_rate":        fraud_rate,
        "overall_missing_rate": round(float(df.isna().mean().mean()), 4),
    }
    return qa, summary


def provider_monitoring_report(df):
    """High-level provider stats for the monitoring report."""
    target = _detect_target(df) or "PotentialFraud"

    rows = []
    for col in ["total_claims", "total_reimbursed", "avg_reimbursed_per_claim",
                "unique_beneficiaries", "inpatient_ratio", "avg_chronic_conditions",
                "reimbursement_outlier_score", "avg_admission_duration"]:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        rows.append({
            "feature":  col,
            "mean":     round(float(s.mean()), 4) if len(s) else None,
            "median":   round(float(s.median()), 4) if len(s) else None,
            "std":      round(float(s.std()), 4) if len(s) else None,
            "p25":      round(float(s.quantile(0.25)), 4) if len(s) else None,
            "p75":      round(float(s.quantile(0.75)), 4) if len(s) else None,
            "p99":      round(float(s.quantile(0.99)), 4) if len(s) else None,
            "n_missing": int(df[col].isna().sum()),
        })

    # Class balance
    fraud_row = {
        "feature":   target,
        "mean":      round(float(df[target].mean()), 4),
        "median":    None,
        "std":       round(float(df[target].std()), 4),
        "p25":       None,
        "p75":       None,
        "p99":       None,
        "n_missing": int(df[target].isna().sum()),
    }
    rows.append(fraud_row)

    return pd.DataFrame(rows)


def plot_provider_risk_distribution(df, target):
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)
    path = os.path.join(config.OUTPUTS_FIGURES, "provider_risk_distribution.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0: "#1565C0", 1: "#C62828"}
    for label, grp in df.groupby(target):
        if "reimbursement_outlier_score" in df.columns:
            col = "reimbursement_outlier_score"
        elif "total_claims" in df.columns:
            col = "total_claims"
        else:
            break
        ax.hist(grp[col].dropna(), bins=40, alpha=0.6,
                color=colors.get(label, "gray"),
                label=f"Fraud={label}")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Provider count")
    ax.set_title("Provider Risk Distribution by Fraud Label")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved {path}")


def plot_reimbursement_distribution(df):
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)
    path = os.path.join(config.OUTPUTS_FIGURES, "reimbursement_distribution.png")

    col = next((c for c in ["avg_reimbursed_per_claim", "total_reimbursed"] if c in df.columns), None)
    if col is None:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    data = df[col].dropna()
    p99  = data.quantile(0.99)
    ax.hist(data[data <= p99], bins=60, color="#1565C0", alpha=0.8, edgecolor="white")
    ax.axvline(data.median(), color="#C62828", lw=2, ls="--",
               label=f"Median=${data.median():,.0f}")
    ax.axvline(p99, color="orange", lw=1.5, ls=":",
               label=f"P99=${p99:,.0f}")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Provider count")
    ax.set_title(f"Distribution of {col.replace('_', ' ').title()} (truncated at P99)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved {path}")


def plot_fraud_rate_by_volume_bucket(df, target):
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)
    path = os.path.join(config.OUTPUTS_FIGURES, "fraud_rate_by_volume_bucket.png")

    if "total_claims" not in df.columns:
        return

    try:
        df = df.copy()
        df["volume_bucket"] = pd.qcut(df["total_claims"], q=5, labels=False, duplicates="drop")
        summary = df.groupby("volume_bucket")[target].mean()
        counts  = df.groupby("volume_bucket")["total_claims"].agg(["min", "max"])
        labels  = [f"Q{int(i)+1}\n[{int(r['min'])}-{int(r['max'])}]"
                   for i, r in counts.iterrows()]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, summary.values, color="#E53935", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Provider Volume Quintile (total claims)")
        ax.set_ylabel("Fraud rate")
        ax.set_title("Fraud Rate by Provider Volume Bucket")
        for bar, val in zip(bars, summary.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Saved {path}")
    except Exception as e:
        print(f"  WARNING: could not plot fraud rate by volume bucket: {e}")


def plot_feature_missingness(df):
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)
    path = os.path.join(config.OUTPUTS_FIGURES, "feature_missingness.png")

    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(20)

    if miss.empty:
        # All columns complete — save a simple "No missing data" chart
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No missing values detected", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.set_title("Feature Missingness")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Saved {path} (no missing values)")
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(miss) * 0.4 + 1)))
    colors = ["#C62828" if v > 0.1 else "#1565C0" for v in miss.values]
    ax.barh(miss.index[::-1], miss.values[::-1] * 100, color=colors[::-1])
    ax.set_xlabel("Missing rate (%)")
    ax.set_title("Feature Missingness (top columns with missing values)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved {path}")


# ── Claim-level monitoring (legacy synthetic) ──────────────────────────────────

def monthly_report(df):
    df = df.copy()
    df["month"] = df["claim_date"].dt.to_period("M").astype(str)
    p99 = df["claim_amount"].quantile(0.99)

    def agg(g):
        return pd.Series({
            "n_claims":         len(g),
            "fraud_rate":       round(float(g["fraud_label"].mean()), 4),
            "mean_claim_amount":round(float(g["claim_amount"].mean()), 2),
            "p95_claim_amount": round(float(g["claim_amount"].quantile(0.95)), 2),
            "missing_rate":     round(float(g.isna().mean().mean()), 4),
            "outlier_rate":     round(float((g["claim_amount"] > p99).mean()), 4),
        })

    return df.groupby("month").apply(agg).reset_index()


def data_quality_summary_claims(df):
    rows = []
    for col in df.columns:
        rows.append({
            "column":       col,
            "dtype":        str(df[col].dtype),
            "n_missing":    int(df[col].isna().sum()),
            "missing_rate": round(float(df[col].isna().mean()), 4),
            "n_unique":     int(df[col].nunique(dropna=True)),
        })
    qa = pd.DataFrame(rows)
    dup_rate = round(float(df["claim_id"].duplicated().mean()), 4) if "claim_id" in df.columns else None
    p99      = df["claim_amount"].quantile(0.99) if "claim_amount" in df.columns else None
    outlier  = round(float((df["claim_amount"] > p99).mean()), 4) if p99 is not None else None
    summary  = {
        "n_rows":  len(df),
        "n_cols":  df.shape[1],
        "duplicate_claim_id_rate":          dup_rate,
        "claim_amount_p99":                 float(p99) if p99 is not None else None,
        "claim_amount_outlier_rate_above_p99": outlier,
        "overall_missing_rate":             round(float(df.isna().mean().mean()), 4),
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


def plot_claim_amount_drift(report):
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Running monitoring pipeline...")
    df, mode = _load_data()
    os.makedirs(config.OUTPUTS_REPORTS, exist_ok=True)
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)

    target = _detect_target(df)

    if mode == "real":
        print("  Mode: PROVIDER-LEVEL monitoring")

        print("  Computing data quality report...")
        qa, summary = data_quality_report(df)
        qa_path = os.path.join(config.OUTPUTS_REPORTS, "data_quality_report.csv")
        qa.to_csv(qa_path, index=False)
        print(f"  Saved {qa_path}")
        for k, v in summary.items():
            print(f"    {k}: {v}")

        print("  Computing provider monitoring report...")
        mon = provider_monitoring_report(df)
        mon_path = os.path.join(config.OUTPUTS_REPORTS, "model_monitoring_report.csv")
        mon.to_csv(mon_path, index=False)
        print(f"  Saved {mon_path}")

        print("  Generating charts...")
        if target:
            plot_provider_risk_distribution(df, target)
            plot_fraud_rate_by_volume_bucket(df, target)
        plot_reimbursement_distribution(df)
        plot_feature_missingness(df)

    else:
        print("  Mode: CLAIM-LEVEL monitoring (synthetic)")

        if "claim_date" not in df.columns or "fraud_label" not in df.columns:
            print("  WARNING: claim_date or fraud_label missing; skipping monthly report.")
        else:
            report = monthly_report(df)
            mon_path = os.path.join(config.OUTPUTS_REPORTS, "model_monitoring_report.csv")
            report.to_csv(mon_path, index=False)
            print(f"  Saved {mon_path}  ({len(report)} months)")
            plot_monthly_fraud_rate(report)
            plot_claim_amount_drift(report)

        qa, summary = data_quality_summary_claims(df)
        qa_path = os.path.join(config.OUTPUTS_REPORTS, "data_quality_summary.csv")
        qa.to_csv(qa_path, index=False)
        print(f"  Saved {qa_path}")
        for k, v in summary.items():
            print(f"    {k}: {v}")

        plot_feature_missingness(df)

    print("Monitoring complete.")


if __name__ == "__main__":
    main()
