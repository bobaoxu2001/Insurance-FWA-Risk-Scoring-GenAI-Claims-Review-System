"""
fairness_audit.py
=================
Demographic-fairness audit for the provider FWA risk score.

The Kaggle dataset has beneficiary-level demographics (Race, Gender, State, age)
but no provider-level demographics. The standard insurance-fairness question is:

  "Does the model score providers differently based on the *patient panel*
   they serve?  i.e. is the model penalizing providers who serve more
   minority / older / specific-state patients?"

This is the right framing for healthcare FWA because providers don't have
protected attributes themselves — but their patient panels do, and a model
that systematically scores higher for providers serving certain cohorts
would push enforcement burden disproportionately onto those cohorts.

Method
------
1. For every provider, aggregate beneficiary attributes across all their
   claims: average patient age, percent of patients in each Race code,
   percent in each Gender code, dominant State.
2. Assign each provider to a "predominant cohort" (majority race, etc.).
3. Compute disparate-impact-style metrics:
     - mean model score by cohort
     - HIGH-risk flag rate by cohort
     - 4/5ths rule: flag-rate ratio of any cohort to the highest-flagged cohort
4. Persist a fairness_audit_report.csv and two figures.

Outputs
-------
outputs/reports/fairness_audit_report.csv      summary by cohort
outputs/figures/fairness_score_by_race.png      score distribution by cohort
outputs/figures/fairness_flag_rate_by_cohort.png  HIGH flag rate by cohort

Important
---------
This is an honest *demonstration* of a fairness audit on a small public
dataset.  Real-deployment fairness review would require:
  - statistical-significance testing (currently descriptive only)
  - intersectional cohorts (Race × Age × State, not marginals)
  - feedback from compliance / legal review
  - calibration analysis per cohort (not just score-distribution)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Kaggle Race codes for the Medicare dataset
RACE_LABELS = {
    1: "White",
    2: "Black",
    3: "Other",       # Asian, Pacific-Islander combined in this dataset
    5: "Hispanic",
}


def _find(pattern_substrings):
    """Find a CSV in data/raw whose lowercased filename contains all substrings."""
    raw = Path(config.DATA_RAW)
    for p in sorted(raw.glob("*.csv")):
        name = p.name.lower()
        if all(sub in name for sub in pattern_substrings):
            return p
    return None


def load_inputs():
    """Load the trained model, the provider table, and the raw beneficiary + claims
    files needed to compute per-provider patient-panel composition."""
    # 1. Provider modeling table
    table = Path(config.DATA_PROCESSED) / "provider_modeling_table.csv"
    if not table.is_file():
        raise FileNotFoundError(
            "data/processed/provider_modeling_table.csv missing. "
            "Run src/provider_feature_engineering.py first."
        )
    providers = pd.read_csv(table)

    # 2. Trained model (random-split version, since temporal model has different
    #    selection behaviour and is reported separately)
    model_path = Path(config.OUTPUTS_MODELS) / "best_fwa_model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(
            "outputs/models/best_fwa_model.pkl missing. Run src/modeling.py first."
        )
    model = joblib.load(model_path)

    # 3. Raw beneficiary + claims for demographic aggregation
    bene_path = _find(["beneficiary", "train"])
    ip_path   = _find(["inpatient",   "train"])
    op_path   = _find(["outpatient",  "train"])
    if not (bene_path and ip_path and op_path):
        raise FileNotFoundError("Could not locate train Beneficiary/Inpatient/Outpatient CSVs.")

    bene = pd.read_csv(bene_path, usecols=["BeneID", "DOB", "Gender", "Race", "State"])
    bene["DOB"] = pd.to_datetime(bene["DOB"], errors="coerce")
    # Approximate age as of the dataset midpoint (2009 in the Kaggle data)
    bene["age"] = (pd.Timestamp("2009-12-31") - bene["DOB"]).dt.days / 365.25

    ip = pd.read_csv(ip_path,  usecols=["Provider", "BeneID"])
    op = pd.read_csv(op_path,  usecols=["Provider", "BeneID"])
    claims = pd.concat([ip, op], ignore_index=True)
    return providers, model, bene, claims


def build_patient_panel(claims: pd.DataFrame, bene: pd.DataFrame) -> pd.DataFrame:
    """For every provider, compute the composition of the patient panel they served:
    avg_patient_age, race shares (white/black/other/hispanic), gender share,
    dominant state."""
    merged = claims.merge(bene, on="BeneID", how="left")

    # average age per provider
    avg_age = merged.groupby("Provider")["age"].mean().rename("panel_avg_age")

    # race shares
    merged["_race_label"] = merged["Race"].map(RACE_LABELS).fillna("Unknown")
    race_pct = (merged.groupby(["Provider", "_race_label"]).size()
                       .unstack(fill_value=0)
                       .pipe(lambda d: d.div(d.sum(axis=1), axis=0)))
    race_pct.columns = [f"panel_pct_{c.lower()}" for c in race_pct.columns]

    # majority race
    majority_race = race_pct.idxmax(axis=1).str.replace("panel_pct_", "").rename("panel_majority_race")

    # gender share (1 = Male, 2 = Female in this dataset)
    gender_pct_female = (merged.assign(_is_female=(merged["Gender"] == 2).astype(int))
                                 .groupby("Provider")["_is_female"].mean()
                                 .rename("panel_pct_female"))

    # dominant state
    dominant_state = (merged.groupby(["Provider", "State"]).size()
                              .reset_index(name="n")
                              .sort_values(["Provider", "n"], ascending=[True, False])
                              .drop_duplicates("Provider")
                              .set_index("Provider")["State"]
                              .rename("panel_dominant_state"))

    panel = pd.concat(
        [avg_age, race_pct, majority_race, gender_pct_female, dominant_state],
        axis=1
    ).reset_index()
    return panel


def score_providers(providers: pd.DataFrame, model) -> pd.DataFrame:
    """Run the trained classifier over the provider table and attach probabilities."""
    drop_cols = {"Provider", "PotentialFraud", "median_claim_date"}
    feat_cols = [c for c in providers.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(providers[c])]
    X = providers[feat_cols].fillna(providers[feat_cols].median(numeric_only=True))
    providers = providers.copy()
    providers["risk_score"] = model.predict_proba(X)[:, 1]
    providers["model_flag_high"] = (providers["risk_score"] >= 0.5).astype(int)
    return providers


def fairness_report(scored: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Per-cohort score + flag-rate + 4/5ths-rule metrics."""
    df = scored.merge(panel, on="Provider", how="inner")

    # Group by predominant patient race
    grp = df.groupby("panel_majority_race")
    summary = pd.DataFrame({
        "n_providers":     grp.size(),
        "n_fraud_labels":  grp["PotentialFraud"].sum(),
        "fraud_rate":      grp["PotentialFraud"].mean().round(4),
        "mean_risk_score": grp["risk_score"].mean().round(4),
        "p90_risk_score":  grp["risk_score"].quantile(0.9).round(4),
        "model_flag_rate": grp["model_flag_high"].mean().round(4),
    }).sort_values("n_providers", ascending=False)

    # Four-fifths rule: ratio of each cohort's flag rate to the highest-flagged cohort
    max_flag = summary["model_flag_rate"].max()
    summary["disparate_impact_ratio"] = (summary["model_flag_rate"] / max_flag).round(3)
    summary["passes_4_5ths_rule"] = summary["disparate_impact_ratio"] >= 0.80
    return summary


def by_age_band(scored: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Same metrics, but cohorts are age-bands of the patient panel."""
    df = scored.merge(panel, on="Provider", how="inner").dropna(subset=["panel_avg_age"])
    df["age_band"] = pd.cut(df["panel_avg_age"],
                            bins=[0, 65, 70, 75, 80, 200],
                            labels=["<65", "65-70", "70-75", "75-80", "80+"])
    grp = df.groupby("age_band", observed=True)
    summary = pd.DataFrame({
        "n_providers":     grp.size(),
        "fraud_rate":      grp["PotentialFraud"].mean().round(4),
        "mean_risk_score": grp["risk_score"].mean().round(4),
        "model_flag_rate": grp["model_flag_high"].mean().round(4),
    })
    return summary


def make_plots(scored: pd.DataFrame, panel: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = scored.merge(panel, on="Provider", how="inner")

    # 1. Risk-score distribution by majority race
    fig, ax = plt.subplots(figsize=(8, 5))
    order = (df.groupby("panel_majority_race").size().sort_values(ascending=False)
                .index.tolist())
    data = [df.loc[df["panel_majority_race"] == r, "risk_score"].values for r in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("Model risk score")
    ax.set_title("Risk-score distribution by predominant patient race")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_dir / "fairness_score_by_race.png", dpi=120)
    plt.close()

    # 2. HIGH-flag rate by cohort
    fig, ax = plt.subplots(figsize=(8, 5))
    summary = fairness_report(scored, panel)
    summary["model_flag_rate"].plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_ylabel("Fraction of providers flagged HIGH risk")
    ax.set_title("HIGH-risk flag rate by predominant patient race")
    ax.axhline(0.8 * summary["model_flag_rate"].max(),
               color="red", linestyle="--", lw=1,
               label=f"4/5ths rule threshold ({0.8 * summary['model_flag_rate'].max():.3f})")
    ax.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fig_dir / "fairness_flag_rate_by_cohort.png", dpi=120)
    plt.close()


def main():
    print("Running fairness audit...")
    providers, model, bene, claims = load_inputs()
    print(f"  Loaded {len(providers)} providers, {len(bene)} beneficiaries, "
          f"{len(claims)} claims")

    print("Scoring providers...")
    scored = score_providers(providers, model)
    print(f"  Score range: [{scored['risk_score'].min():.3f}, {scored['risk_score'].max():.3f}]  "
          f"mean={scored['risk_score'].mean():.3f}")

    print("Building per-provider patient panel composition...")
    panel = build_patient_panel(claims, bene)
    print(f"  Panel features: {panel.columns.tolist()}")

    print("Computing fairness metrics by patient majority race...")
    race_summary = fairness_report(scored, panel)
    print(race_summary)

    print("\nComputing fairness metrics by patient-panel age band...")
    age_summary = by_age_band(scored, panel)
    print(age_summary)

    print("\nGenerating fairness plots...")
    make_plots(scored, panel, Path(config.OUTPUTS_FIGURES))

    out = Path(config.OUTPUTS_REPORTS) / "fairness_audit_report.csv"
    race_summary.to_csv(out)
    print(f"  Saved race-cohort report to {out}")

    out_age = Path(config.OUTPUTS_REPORTS) / "fairness_audit_age.csv"
    age_summary.to_csv(out_age)
    print(f"  Saved age-band report to {out_age}")

    print("\nFairness audit complete.")
    # Surface the headline disparate-impact finding for the operator
    worst = race_summary["disparate_impact_ratio"].min()
    if worst < 0.8:
        worst_cohort = race_summary["disparate_impact_ratio"].idxmin()
        print(f"\n  ⚠ Disparate-impact warning: cohort '{worst_cohort}' has flag-rate ratio "
              f"{worst:.2f} — below the 4/5ths rule threshold of 0.80")
    else:
        print(f"\n  ✓ All cohorts within 4/5ths-rule threshold (min ratio={worst:.2f})")


if __name__ == "__main__":
    main()
