"""
Modeling module for FWA risk scoring.
Trains supervised classifiers and anomaly detection, evaluates, and saves outputs.

Data priority:
  1. data/processed/provider_modeling_table.csv  (real Kaggle provider-level data)
  2. data/processed/claims_features.csv          (synthetic claim-level features)
  3. data/processed/claims_encoded.csv           (synthetic encoded)
  4. data/raw/synthetic_claims.csv               (raw synthetic fallback)

The saved metrics JSON includes a 'data_source' key indicating which was used.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

ID_LIKE_COLS = [
    "claim_id", "policyholder_id", "provider_id", "claim_date",
    "service_type", "diagnosis_group", "state",
    # Provider-level id columns (Kaggle)
    "Provider",
]

# rule_based_risk_score is a hand-coded composite of other features and would
# inflate apparent importance. It is kept in the dataset for the dashboard as a
# transparent baseline, but excluded from supervised model inputs.
EXCLUDE_FROM_MODEL = ["rule_based_risk_score"]

# Possible target column names
TARGET_COLS = ["PotentialFraud", "fraud_label"]

# Track which data source was used (set by load_data)
_DATA_SOURCE = "unknown"


def _detect_target(df):
    """Return the target column name found in df, or raise."""
    for t in TARGET_COLS:
        if t in df.columns:
            return t
    raise ValueError(f"No target column found. Tried: {TARGET_COLS}. Got: {list(df.columns)}")


def _get_model_features(df, target=None):
    """Return only numeric, non-id feature columns."""
    if target is None:
        target = _detect_target(df)
    drop = ID_LIKE_COLS + [target] + EXCLUDE_FROM_MODEL
    feature_cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    return feature_cols


def load_data():
    global _DATA_SOURCE
    provider_path = os.path.join(config.DATA_PROCESSED, "provider_modeling_table.csv")
    feat_path     = os.path.join(config.DATA_PROCESSED, "claims_features.csv")
    enc_path      = os.path.join(config.DATA_PROCESSED, "claims_encoded.csv")
    raw_path      = os.path.join(config.DATA_RAW,       "synthetic_claims.csv")

    if os.path.exists(provider_path):
        df = pd.read_csv(provider_path)
        print(f"  Loaded REAL provider data from {provider_path}  shape={df.shape}")
        _DATA_SOURCE = "real_kaggle_provider"
        return df

    for p, src in [(feat_path, "synthetic_claims_features"),
                   (enc_path,  "synthetic_claims_encoded"),
                   (raw_path,  "synthetic_claims_raw")]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"  Loaded SYNTHETIC data from {p}  shape={df.shape}")
            _DATA_SOURCE = src
            return df

    raise FileNotFoundError(
        "No processed data found.\n"
        "  Real data:    run src/provider_feature_engineering.py first.\n"
        "  Synthetic:    run src/synthetic_data_generation.py + src/preprocessing.py first."
    )


def prepare_features(df):
    from sklearn.model_selection import train_test_split

    target = _detect_target(df)
    feature_cols = _get_model_features(df, target)
    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True)).copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_models(X_train, y_train):
    models = {}

    print("  Training Logistic Regression...")
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=config.RANDOM_SEED
    )
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr

    print("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_depth=12, random_state=config.RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    if HAS_XGB:
        print("  Training XGBoost...")
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos, use_label_encoder=False,
            eval_metric="logloss", random_state=config.RANDOM_SEED,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        models["XGBoost"] = xgb_model
    else:
        print("  XGBoost not available; training GradientBoosting...")
        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=config.RANDOM_SEED
        )
        gb.fit(X_train, y_train)
        models["GradientBoosting"] = gb

    return models


def train_anomaly_detector(X_train):
    print("  Training Isolation Forest (anomaly detection)...")
    iso = IsolationForest(
        n_estimators=200, contamination=config.FRAUD_BASE_RATE,
        random_state=config.RANDOM_SEED, n_jobs=-1
    )
    iso.fit(X_train)
    return iso


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_models(models, X_test, y_test):
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics[name] = {
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
        print(f"  {name}: AUC={metrics[name]['roc_auc']:.4f}  "
              f"F1={metrics[name]['f1']:.4f}  "
              f"Recall={metrics[name]['recall']:.4f}")
    return metrics


def select_best_model(models, metrics):
    best_name = max(metrics, key=lambda n: metrics[n]["roc_auc"])
    print(f"  Best model: {best_name} (AUC={metrics[best_name]['roc_auc']:.4f})")
    return best_name, models[best_name]


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, model_name):
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(config.OUTPUTS_FIGURES, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved confusion matrix to {path}")


def plot_roc_curves(models, X_test, y_test):
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — FWA Risk Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(config.OUTPUTS_FIGURES, "roc_curve.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved ROC curves to {path}")


def plot_precision_recall(models, X_test, y_test):
    """Save PR curves for all models and a per-threshold sweep CSV for the best."""
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)
    os.makedirs(config.OUTPUTS_REPORTS, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    best_name, best_ap, best_probs = None, -1, None
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")
        if ap > best_ap:
            best_ap, best_name, best_probs = ap, name, y_prob

    # Baseline = prevalence
    base = float(np.mean(y_test))
    ax.axhline(base, ls="--", color="gray", lw=1, label=f"Prevalence={base:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — FWA Risk Models")
    ax.legend(loc="lower left")
    plt.tight_layout()
    path = os.path.join(config.OUTPUTS_FIGURES, "precision_recall_curve.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved PR curves to {path}")

    # Threshold sweep for the best model by AP
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        y_hat = (best_probs >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": round(precision_score(y_test, y_hat, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_hat, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_hat, zero_division=0), 4),
            "n_flagged": int(y_hat.sum()),
        })
    thr_df = pd.DataFrame(rows)
    thr_path = os.path.join(config.OUTPUTS_REPORTS, "threshold_analysis.csv")
    thr_df.to_csv(thr_path, index=False)
    print(f"  Saved threshold analysis (best={best_name}) to {thr_path}")


def plot_feature_importance(model, feature_cols, model_name):
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("  Model has no feature_importances_; skipping plot.")
        return

    top_n = 20
    idx = np.argsort(importances)[-top_n:][::-1]
    feat_names = [feature_cols[i] for i in idx]
    feat_vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    ax.barh(range(len(feat_names)), feat_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()
    path = os.path.join(config.OUTPUTS_FIGURES, "feature_importance.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved feature importance to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

def save_outputs(best_model, metrics, iso_forest):
    os.makedirs(config.OUTPUTS_MODELS, exist_ok=True)
    os.makedirs(config.OUTPUTS_REPORTS, exist_ok=True)

    model_path = os.path.join(config.OUTPUTS_MODELS, "best_fwa_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"  Saved best model to {model_path}")

    iso_path = os.path.join(config.OUTPUTS_MODELS, "isolation_forest.pkl")
    joblib.dump(iso_forest, iso_path)
    print(f"  Saved Isolation Forest to {iso_path}")

    # Annotate metrics with the data source used
    annotated = {
        "_data_source": _DATA_SOURCE,
        "_note": (
            "Metrics trained on real Kaggle Healthcare Provider Fraud Detection dataset"
            if _DATA_SOURCE == "real_kaggle_provider"
            else "Metrics trained on SYNTHETIC demo data — not real insurance data"
        ),
    }
    annotated.update(metrics)

    metrics_path = os.path.join(config.OUTPUTS_REPORTS, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(annotated, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Running modeling pipeline...")
    df = load_data()
    target = _detect_target(df)
    print(f"  Target column: {target}  |  Data source: {_DATA_SOURCE}")
    X_train, X_test, y_train, y_test, feature_cols = prepare_features(df)
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}  "
          f"Fraud rate (test): {y_test.mean():.2%}")

    models = train_models(X_train, y_train)
    iso_forest = train_anomaly_detector(X_train)

    print("\nEvaluating models on test set...")
    metrics = evaluate_models(models, X_test, y_test)

    best_name, best_model = select_best_model(models, metrics)

    print("\nGenerating visualizations...")
    plot_confusion_matrix(best_model, X_test, y_test, best_name)
    plot_roc_curves(models, X_test, y_test)
    plot_precision_recall(models, X_test, y_test)
    plot_feature_importance(best_model, feature_cols, best_name)

    save_outputs(best_model, metrics, iso_forest)
    print("\nModeling complete.")


if __name__ == "__main__":
    main()
