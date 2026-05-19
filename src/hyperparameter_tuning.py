"""
hyperparameter_tuning.py
========================
RandomizedSearchCV tuning of the best-performing model class, optimizing
for the same metric used for production selection (PR-AUC / average_precision).

Why RandomizedSearchCV instead of GridSearchCV?
  - GridSearchCV scales poorly for >2 hyperparameters with reasonable ranges
  - RandomizedSearchCV with 40 iterations covers the space efficiently and
    is the practical choice in production tuning pipelines

Default search: Random Forest (random-split winner). Use --model gb to tune
Gradient Boosting (temporal-split winner) instead.

Outputs
-------
outputs/models/best_fwa_model_tuned.pkl       tuned best estimator
outputs/reports/hp_tuning_results.json        best params, score, top-10 results
outputs/reports/hp_tuning_search.csv          full search log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data():
    table = Path(config.DATA_PROCESSED) / "provider_modeling_table.csv"
    if not table.is_file():
        raise FileNotFoundError("provider_modeling_table.csv missing")
    df = pd.read_csv(table)
    drop = {"Provider", "PotentialFraud", "median_claim_date"}
    feat_cols = [c for c in df.columns
                 if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))
    y = df["PotentialFraud"]
    return X, y, feat_cols


PARAM_SPACES = {
    "rf": {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [6, 8, 10, 12, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf":  [1, 2, 4, 8],
        "max_features":     ["sqrt", "log2", 0.3, 0.5],
        "class_weight":     ["balanced", "balanced_subsample"],
    },
    "gb": {
        "n_estimators":      [100, 200, 300, 400],
        "max_depth":         [3, 4, 5, 6, 7],
        "learning_rate":     [0.01, 0.02, 0.05, 0.1],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf":  [1, 2, 4, 8],
        "subsample":         [0.7, 0.85, 1.0],
    },
}

ESTIMATORS = {
    "rf": RandomForestClassifier(random_state=config.RANDOM_SEED, n_jobs=-1),
    "gb": GradientBoostingClassifier(random_state=config.RANDOM_SEED),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "gb"], default="rf",
                        help="rf = Random Forest (random-split winner), "
                             "gb = Gradient Boosting (temporal-split winner)")
    parser.add_argument("--n-iter", type=int, default=40,
                        help="Number of RandomizedSearchCV iterations")
    args = parser.parse_args()

    print(f"Loading provider modeling table...")
    X, y, feat_cols = load_data()
    print(f"  X: {X.shape}  fraud rate: {y.mean():.2%}")

    print(f"\nTuning {args.model.upper()} with RandomizedSearchCV "
          f"(n_iter={args.n_iter}, scoring=average_precision, cv=5-fold stratified)")
    print(f"  Param space: {PARAM_SPACES[args.model]}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator=ESTIMATORS[args.model],
        param_distributions=PARAM_SPACES[args.model],
        n_iter=args.n_iter,
        scoring="average_precision",   # PR-AUC — same as production selection
        cv=skf,
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
        verbose=1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y)

    print(f"\nBest CV PR-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # Also report ROC-AUC of the refit estimator on the full data (sanity)
    proba = search.best_estimator_.predict_proba(X)[:, 1]
    print(f"In-sample ROC-AUC: {roc_auc_score(y, proba):.4f}  (sanity check)")

    # Persist artifacts
    os.makedirs(config.OUTPUTS_MODELS, exist_ok=True)
    os.makedirs(config.OUTPUTS_REPORTS, exist_ok=True)
    model_path = Path(config.OUTPUTS_MODELS) / f"best_fwa_model_tuned_{args.model}.pkl"
    joblib.dump(search.best_estimator_, model_path)
    print(f"\nSaved tuned model to {model_path}")

    results = {
        "_model_class": args.model,
        "_n_iter": args.n_iter,
        "_scoring": "average_precision (PR-AUC)",
        "_cv": "5-fold StratifiedKFold",
        "_n_features": X.shape[1],
        "best_score_pr_auc": round(float(search.best_score_), 4),
        "best_params": search.best_params_,
    }
    results_path = Path(config.OUTPUTS_REPORTS) / "hp_tuning_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved best-params summary to {results_path}")

    full = pd.DataFrame(search.cv_results_)
    full = full[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    full = full.sort_values("rank_test_score")
    full_path = Path(config.OUTPUTS_REPORTS) / f"hp_tuning_search_{args.model}.csv"
    full.to_csv(full_path, index=False)
    print(f"Saved full search log to {full_path}")

    # Headline comparison vs baseline (read from model_metrics.json if present)
    metrics_path = Path(config.OUTPUTS_REPORTS) / "model_metrics.json"
    if metrics_path.is_file():
        baseline = json.loads(metrics_path.read_text())
        baseline_pr = baseline.get(
            {"rf": "RandomForest", "gb": "GradientBoosting"}[args.model], {}
        ).get("pr_auc")
        if baseline_pr is not None:
            delta = search.best_score_ - baseline_pr
            print(f"\nBaseline PR-AUC (held-out test): {baseline_pr:.4f}")
            print(f"Tuned     PR-AUC (5-fold CV):    {search.best_score_:.4f}  "
                  f"(Δ = {delta:+.4f})")
            if delta > 0.01:
                print("→ Tuning produced a meaningful improvement.")
            elif delta < -0.01:
                print("→ Default hyperparameters were better than tuned (regularization sweet spot).")
            else:
                print("→ Tuning made no meaningful difference; defaults were near-optimal.")


if __name__ == "__main__":
    main()
