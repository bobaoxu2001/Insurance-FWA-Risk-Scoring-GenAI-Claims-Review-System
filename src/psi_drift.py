"""
psi_drift.py
============
Population Stability Index (PSI) drift detection between two slices of the
provider modeling table.

PSI is the standard industry metric for feature distribution drift in credit-
scoring, insurance, and fraud applications. For each feature it bins both
the reference and the target distribution and computes:

    PSI = Σ (target_pct - ref_pct) * ln(target_pct / ref_pct)

Interpretation (industry-standard thresholds):
    PSI < 0.10        → no significant shift
    0.10 ≤ PSI < 0.25 → moderate shift, investigate
    PSI ≥ 0.25        → significant shift, retrain candidate

This module is split-mode aware: by default it computes PSI between the
*chronological train half* and the *chronological test half* of the
provider table — i.e. it measures real temporal drift, not random-split
sampling noise.

Outputs
-------
outputs/reports/psi_drift_report.csv     per-feature PSI + verdict
outputs/figures/psi_top_features.png     bar chart of top-drifting features
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


PSI_THRESHOLDS = {
    "stable":     0.10,
    "moderate":   0.25,
}


def _verdict(psi: float) -> str:
    if psi < PSI_THRESHOLDS["stable"]:
        return "stable"
    elif psi < PSI_THRESHOLDS["moderate"]:
        return "moderate_shift"
    else:
        return "significant_shift"


def _compute_psi(ref: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    """Quantile-bin PSI; epsilon-smooth empty bins so the ln is finite."""
    ref = ref[np.isfinite(ref)]
    target = target[np.isfinite(target)]
    if len(ref) < 50 or len(target) < 50:
        return float("nan")
    # Quantile bins derived from the reference distribution
    bin_edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return float("nan")
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf

    ref_pct    = np.histogram(ref,    bins=bin_edges)[0] / max(len(ref), 1)
    target_pct = np.histogram(target, bins=bin_edges)[0] / max(len(target), 1)
    # Laplace smooth to avoid log(0)
    eps = 1e-4
    ref_pct    = np.clip(ref_pct,    eps, None)
    target_pct = np.clip(target_pct, eps, None)
    return float(np.sum((target_pct - ref_pct) * np.log(target_pct / ref_pct)))


def split_by_date(df: pd.DataFrame, test_fraction: float = 0.2):
    """Same chronological split as src/modeling.py prepare_features_temporal."""
    if "median_claim_date" not in df.columns:
        raise ValueError("median_claim_date column missing — re-run provider_feature_engineering")
    dates = pd.to_datetime(df["median_claim_date"], errors="coerce")
    df = df.loc[dates.notna()].copy()
    dates = dates.loc[df.index]
    cutoff = dates.quantile(1 - test_fraction)
    train = df.loc[dates <= cutoff]
    test  = df.loc[dates >  cutoff]
    return train, test, cutoff


def compute_drift_report(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    drop = {"Provider", "PotentialFraud", "median_claim_date"}
    feature_cols = [c for c in train.columns
                    if c not in drop and pd.api.types.is_numeric_dtype(train[c])]

    rows = []
    for c in feature_cols:
        psi = _compute_psi(train[c].to_numpy(), test[c].to_numpy())
        rows.append({
            "feature":         c,
            "psi":             round(psi, 4),
            "verdict":         _verdict(psi) if not np.isnan(psi) else "insufficient_data",
            "train_mean":      round(float(train[c].mean()), 4),
            "test_mean":       round(float(test[c].mean()),  4),
            "train_std":       round(float(train[c].std()),  4),
            "test_std":        round(float(test[c].std()),   4),
        })
    out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    return out


def plot_top(report: pd.DataFrame, fig_dir: Path, top_k: int = 15):
    fig_dir.mkdir(parents=True, exist_ok=True)
    top = report.head(top_k).iloc[::-1]   # reverse so largest bar at top
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d62728" if v == "significant_shift" else
              "#ff7f0e" if v == "moderate_shift"   else
              "#2ca02c" for v in top["verdict"]]
    ax.barh(top["feature"], top["psi"], color=colors)
    ax.axvline(PSI_THRESHOLDS["stable"],   color="#666", linestyle=":",  lw=1, label="0.10 (stable threshold)")
    ax.axvline(PSI_THRESHOLDS["moderate"], color="#666", linestyle="--", lw=1, label="0.25 (significant threshold)")
    ax.set_xlabel("Population Stability Index (train → test)")
    ax.set_title(f"Top {top_k} drifting features under chronological split")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = fig_dir / "psi_top_features.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved drift bar chart to {path}")


def main():
    table = Path(config.DATA_PROCESSED) / "provider_modeling_table.csv"
    if not table.is_file():
        raise FileNotFoundError("provider_modeling_table.csv missing")
    df = pd.read_csv(table)

    print("Splitting chronologically by median_claim_date...")
    train, test, cutoff = split_by_date(df)
    print(f"  Cutoff   : {cutoff.date()}")
    print(f"  Train    : {len(train)} providers")
    print(f"  Test     : {len(test)} providers")

    print("\nComputing PSI per feature...")
    report = compute_drift_report(train, test)

    out_path = Path(config.OUTPUTS_REPORTS) / "psi_drift_report.csv"
    report.to_csv(out_path, index=False)
    print(f"\nSaved PSI report to {out_path}")

    print("\nDrift summary:")
    n_total       = len(report)
    n_stable      = (report["verdict"] == "stable").sum()
    n_moderate    = (report["verdict"] == "moderate_shift").sum()
    n_significant = (report["verdict"] == "significant_shift").sum()
    print(f"  Stable                 : {n_stable:>3} / {n_total}")
    print(f"  Moderate shift (≥0.10) : {n_moderate:>3} / {n_total}")
    print(f"  Significant shift (≥0.25): {n_significant:>3} / {n_total}")
    print("\nTop 10 drifting features:")
    print(report.head(10)[["feature", "psi", "verdict"]].to_string(index=False))

    plot_top(report, Path(config.OUTPUTS_FIGURES))

    if n_significant > 0:
        print(f"\n  ⚠ {n_significant} feature(s) crossed the 0.25 threshold — would trigger retraining alert in production")
    elif n_moderate > 0:
        print(f"\n  ⚠ {n_moderate} feature(s) crossed the 0.10 threshold — investigate but no retrain trigger")
    else:
        print(f"\n  ✓ All features stable")


if __name__ == "__main__":
    main()
