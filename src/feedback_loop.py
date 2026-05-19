"""
feedback_loop.py
================
Analyst-disposition feedback loop. This is the architecture piece that turns
a static classifier into a learning system: when analysts review flagged
providers, their dispositions feed back as labels for the next training cycle.

Three things happen in this module:

  1. **Synthesize a realistic feedback CSV** (or read one the user supplies).
     The CSV represents what an analyst case-management system would emit:
       provider_id, model_score, model_flag, analyst_disposition, disposition_date

  2. **Compute model-vs-analyst agreement metrics** — these are the metrics a
     production FWA team would watch every week:
       - Precision-of-flag rate (confirmed_fraud / flagged_for_review)
       - False-confirm rate (cleared / flagged)
       - Analyst override rate (cleared at high-score band)
       - Calibration of model_score vs analyst_disposition

  3. **Produce a retraining-trigger recommendation** based on:
       - Whether the precision-of-flag rate has dropped below a threshold
       - Whether enough new labels have accumulated to justify a retrain

CSV schema
----------
provider_id            string,  matches Provider in the modeling table
model_score            float,   the score that triggered review
model_flag             int,     1 if score ≥ deployment threshold
analyst_disposition    enum,    one of {confirmed_fraud, cleared, needs_more_info}
disposition_date       ISO date, when the analyst closed the case

Outputs
-------
outputs/reports/feedback_log.csv                 the (synthetic or supplied) feedback
outputs/reports/feedback_loop_metrics.json       headline agreement metrics
outputs/figures/feedback_calibration.png         score-vs-confirmed-fraud-rate
outputs/figures/feedback_disposition_mix.png     disposition distribution

CLI
---
    python src/feedback_loop.py                    # synthesize + report
    python src/feedback_loop.py --feedback FILE    # load a real CSV
    python src/feedback_loop.py --retrain          # also retrain on new labels
"""

from __future__ import annotations

import argparse
import json
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


DISPOSITIONS = ["confirmed_fraud", "cleared", "needs_more_info"]
DEPLOYMENT_THRESHOLD = 0.50

# Retrain triggers — thresholds a production FWA team would set
PRECISION_FLOOR = 0.55        # if precision-of-flag drops below this → consider retrain
MIN_LABELS_FOR_RETRAIN = 100   # don't retrain on <100 new labels


def load_scored_providers():
    """Score all providers with the production model."""
    table = Path(config.DATA_PROCESSED) / "provider_modeling_table.csv"
    df = pd.read_csv(table)

    model_path = Path(config.OUTPUTS_MODELS) / "best_fwa_model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError("best_fwa_model.pkl missing — run src/modeling.py first")
    model = joblib.load(model_path)

    drop = {"Provider", "PotentialFraud", "median_claim_date"}
    feat_cols = [c for c in df.columns
                 if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))
    df["model_score"] = model.predict_proba(X)[:, 1]
    df["model_flag"]  = (df["model_score"] >= DEPLOYMENT_THRESHOLD).astype(int)
    return df, model, feat_cols


def synthesize_feedback(scored: pd.DataFrame, n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Generate a realistic feedback CSV: sample ~80% from flagged providers
    (analysts review what the model flagged) and ~20% from the rest (random
    audit / appeals). Disposition is biased by the underlying PotentialFraud
    label but noisy — analysts get it right ~75% of the time."""
    rng = np.random.default_rng(seed)
    flagged_pool = scored[scored["model_flag"] == 1]
    unflagged_pool = scored[scored["model_flag"] == 0]

    n_flagged   = int(0.8 * n)
    n_unflagged = n - n_flagged

    flagged_pick = flagged_pool.sample(
        min(n_flagged, len(flagged_pool)), random_state=seed
    )
    unflagged_pick = unflagged_pool.sample(
        min(n_unflagged, len(unflagged_pool)), random_state=seed + 1
    )
    sample = pd.concat([flagged_pick, unflagged_pick])

    # Analyst noise: 75% match ground truth, 15% disagree, 10% needs_more_info
    n_rows = len(sample)
    u = rng.random(n_rows)
    disp = []
    for is_fraud, draw in zip(sample["PotentialFraud"].values, u):
        if draw < 0.10:
            disp.append("needs_more_info")
        elif draw < 0.85:
            disp.append("confirmed_fraud" if is_fraud == 1 else "cleared")
        else:  # analyst disagrees with ground truth
            disp.append("cleared" if is_fraud == 1 else "confirmed_fraud")

    feedback = pd.DataFrame({
        "provider_id":         sample["Provider"].values,
        "model_score":         sample["model_score"].round(4).values,
        "model_flag":          sample["model_flag"].values,
        "analyst_disposition": disp,
        "disposition_date":    pd.Timestamp("2009-12-31") + pd.to_timedelta(
            rng.integers(0, 120, n_rows), unit="D"
        ),
    })
    return feedback


def compute_agreement_metrics(feedback: pd.DataFrame) -> dict:
    n_flagged = (feedback["model_flag"] == 1).sum()
    n_unflagged = (feedback["model_flag"] == 0).sum()

    confirmed = feedback["analyst_disposition"] == "confirmed_fraud"
    cleared   = feedback["analyst_disposition"] == "cleared"
    pending   = feedback["analyst_disposition"] == "needs_more_info"

    # Among flagged providers, what fraction did the analyst confirm?
    flag_mask = feedback["model_flag"] == 1
    precision_of_flag = (
        (confirmed & flag_mask).sum() / max(flag_mask.sum(), 1)
    )
    false_confirm_rate = (
        (cleared & flag_mask).sum() / max(flag_mask.sum(), 1)
    )

    # Among unflagged (random-audit), what fraction did the analyst still confirm?
    no_flag_mask = feedback["model_flag"] == 0
    miss_rate = (
        (confirmed & no_flag_mask).sum() / max(no_flag_mask.sum(), 1)
    )

    return {
        "n_total":              int(len(feedback)),
        "n_flagged":            int(n_flagged),
        "n_unflagged":          int(n_unflagged),
        "n_confirmed_fraud":    int(confirmed.sum()),
        "n_cleared":            int(cleared.sum()),
        "n_needs_more_info":    int(pending.sum()),
        "precision_of_flag":    round(float(precision_of_flag), 4),
        "false_confirm_rate":   round(float(false_confirm_rate), 4),
        "miss_rate_on_audit":   round(float(miss_rate), 4),
    }


def plot_calibration(feedback: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = feedback[feedback["analyst_disposition"].isin(["confirmed_fraud", "cleared"])].copy()
    df["confirmed"] = (df["analyst_disposition"] == "confirmed_fraud").astype(int)
    df["score_bin"] = pd.cut(df["model_score"],
                             bins=[0, 0.1, 0.25, 0.5, 0.75, 1.001],
                             labels=["0-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"])
    bands = (df.groupby("score_bin", observed=True)["confirmed"]
                  .agg(["mean", "size"]).rename(columns={"mean": "confirm_rate"}))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(bands)), bands["confirm_rate"],
           tick_label=bands.index, color="#4c78a8")
    for i, (rate, n) in enumerate(zip(bands["confirm_rate"], bands["size"])):
        ax.text(i, rate + 0.02, f"n={n}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Analyst-confirmed fraud rate within score band")
    ax.set_xlabel("Model score band")
    ax.set_title("Model score → analyst-confirmation rate (calibration via feedback)")
    plt.tight_layout()
    plt.savefig(fig_dir / "feedback_calibration.png", dpi=120)
    plt.close()
    print(f"  Saved calibration-by-feedback plot")

    fig, ax = plt.subplots(figsize=(6, 4))
    feedback["analyst_disposition"].value_counts().plot(
        kind="bar", ax=ax, color=["#2ca02c", "#d62728", "#ff7f0e"]
    )
    ax.set_title("Analyst disposition mix")
    ax.set_ylabel("count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fig_dir / "feedback_disposition_mix.png", dpi=120)
    plt.close()


def retrain_with_feedback(feedback: pd.DataFrame, feat_cols: list, original_df: pd.DataFrame):
    """Re-train RF using ground-truth labels OR-ed with analyst-confirmed labels.
    In production this is where you would merge analyst dispositions into your
    training set. Here we demonstrate the mechanics — for the Kaggle dataset
    ground truth already matches the analyst signal, so the retrain is mostly
    a no-op, but the wiring is correct."""
    from sklearn.ensemble import RandomForestClassifier
    print("\nRetraining RandomForest with feedback-merged labels...")

    df = original_df.copy()
    # Map analyst dispositions to additional positive/negative labels
    pos = feedback[feedback["analyst_disposition"] == "confirmed_fraud"]["provider_id"]
    neg = feedback[feedback["analyst_disposition"] == "cleared"]["provider_id"]
    df.loc[df["Provider"].isin(pos), "PotentialFraud"] = 1
    df.loc[df["Provider"].isin(neg), "PotentialFraud"] = 0
    n_overrides = (df["PotentialFraud"] != original_df["PotentialFraud"]).sum()
    print(f"  Labels overridden by analyst feedback: {n_overrides}")

    X = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))
    y = df["PotentialFraud"]
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                max_depth=12, random_state=config.RANDOM_SEED, n_jobs=-1)
    rf.fit(X, y)
    out = Path(config.OUTPUTS_MODELS) / "best_fwa_model_feedback_retrained.pkl"
    joblib.dump(rf, out)
    print(f"  Saved feedback-retrained model to {out}")
    return rf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", default=None,
                        help="Path to an analyst-feedback CSV. If omitted, a synthetic one is generated.")
    parser.add_argument("--n", type=int, default=300,
                        help="Number of synthesized feedback rows when --feedback is not given")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain the production model with the feedback-merged labels")
    args = parser.parse_args()

    print("Loading scored providers...")
    scored, model, feat_cols = load_scored_providers()
    print(f"  {len(scored)} providers, {scored['model_flag'].sum()} flagged at θ={DEPLOYMENT_THRESHOLD}")

    if args.feedback:
        feedback = pd.read_csv(args.feedback, parse_dates=["disposition_date"])
        print(f"  Loaded {len(feedback)} analyst dispositions from {args.feedback}")
    else:
        feedback = synthesize_feedback(scored, n=args.n)
        feedback_path = Path(config.OUTPUTS_REPORTS) / "feedback_log.csv"
        feedback.to_csv(feedback_path, index=False)
        print(f"  Synthesized {len(feedback)} analyst dispositions → {feedback_path}")

    metrics = compute_agreement_metrics(feedback)
    print(f"\nModel-vs-analyst agreement:")
    for k, v in metrics.items():
        print(f"  {k:25s} : {v}")

    out_metrics = Path(config.OUTPUTS_REPORTS) / "feedback_loop_metrics.json"
    with out_metrics.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_metrics}")

    plot_calibration(feedback, Path(config.OUTPUTS_FIGURES))

    # Retraining recommendation
    print("\nRetraining-trigger evaluation:")
    if metrics["n_total"] < MIN_LABELS_FOR_RETRAIN:
        print(f"  ⏸  Only {metrics['n_total']} labels (need {MIN_LABELS_FOR_RETRAIN}) — wait")
    elif metrics["precision_of_flag"] < PRECISION_FLOOR:
        print(f"  ⚠  Precision-of-flag {metrics['precision_of_flag']:.3f} < floor "
              f"{PRECISION_FLOOR} — RETRAIN recommended")
    else:
        print(f"  ✓  Precision-of-flag {metrics['precision_of_flag']:.3f} ≥ floor "
              f"{PRECISION_FLOOR} — no retrain trigger")

    if args.retrain:
        original = pd.read_csv(Path(config.DATA_PROCESSED) / "provider_modeling_table.csv")
        retrain_with_feedback(feedback, feat_cols, original)


if __name__ == "__main__":
    main()
