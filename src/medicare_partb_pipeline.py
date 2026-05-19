"""
medicare_partb_pipeline.py
==========================
THIRD real-data pipeline — and the only one with **real NPI-keyed fraud labels**.

Joins two real public datasets:
  - Medicare Physician & Other Practitioners by Provider, 2023
    (data.cms.gov; 1.26M real provider NPIs with real Medicare billing data)
  - HHS-OIG LEIE
    (8,429 real federal-exclusion NPIs)

Labelling: a provider is positive (excluded_for_fraud = 1) iff its real NPI
appears in the LEIE. No synthetic logic, no aggregated proxy — actual
federal exclusion as the label.

Configurable scope:
  --population ltc    LTC-relevant provider types only (~193K, ~80 LEIE matches)
  --population all    Full Part B universe (1.26M providers, 207 LEIE matches)

Outputs
-------
data/processed/medicare_partb_modeling_table.csv
outputs/reports/medicare_partb_metrics.json
outputs/reports/medicare_partb_classification_report.txt
outputs/figures/medicare_partb_roc.png
outputs/figures/medicare_partb_pr.png
outputs/figures/medicare_partb_feature_importance.png
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, classification_report, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.base import clone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


PARTB_CSV = Path(config.DATA_RAW) / "cms_partb" / "medicare_physician_by_provider_2023.csv"
LEIE_CSV  = Path(config.DATA_RAW) / "oig" / "oig_leie_updated.csv"


# Numeric feature columns from Medicare Part B 2023 schema
NUMERIC_FEATURES = [
    # Service-volume features
    "Tot_HCPCS_Cds", "Tot_Benes", "Tot_Srvcs",
    "Tot_Sbmtd_Chrg", "Tot_Mdcr_Alowd_Amt", "Tot_Mdcr_Pymt_Amt", "Tot_Mdcr_Stdzd_Amt",
    "Drug_Tot_HCPCS_Cds", "Drug_Tot_Benes", "Drug_Tot_Srvcs",
    "Drug_Sbmtd_Chrg", "Drug_Mdcr_Pymt_Amt",
    "Med_Tot_HCPCS_Cds", "Med_Tot_Benes", "Med_Tot_Srvcs",
    "Med_Sbmtd_Chrg", "Med_Mdcr_Pymt_Amt",
    # Beneficiary demographics
    "Bene_Avg_Age",
    "Bene_Age_LT_65_Cnt", "Bene_Age_65_74_Cnt", "Bene_Age_75_84_Cnt", "Bene_Age_GT_84_Cnt",
    "Bene_Feml_Cnt", "Bene_Male_Cnt",
    "Bene_Race_Wht_Cnt", "Bene_Race_Black_Cnt", "Bene_Race_API_Cnt",
    "Bene_Race_Hspnc_Cnt", "Bene_Race_NatInd_Cnt", "Bene_Race_Othr_Cnt",
    "Bene_Dual_Cnt", "Bene_Ndual_Cnt",
    # Chronic-condition prevalence on the panel
    "Bene_CC_BH_Alz_NonAlzdem_V2_Pct", "Bene_CC_BH_Depress_V1_Pct",
    "Bene_CC_PH_Cancer6_V2_Pct", "Bene_CC_PH_CKD_V2_Pct", "Bene_CC_PH_COPD_V2_Pct",
    "Bene_CC_PH_Diabetes_V2_Pct", "Bene_CC_PH_HF_NonIHD_V2_Pct",
    "Bene_CC_PH_Hypertension_V2_Pct", "Bene_CC_PH_IschemicHeart_V2_Pct",
    "Bene_CC_PH_Stroke_TIA_V2_Pct", "Bene_Avg_Risk_Scre",
]

LTC_PROVIDER_TYPES_REGEX = (
    r"Nurse Practitioner|Geriatric Medicine|Hospice and Palliative Care|"
    r"Geriatric Psychiatry|Skilled Nursing|Home Health"
)


def load_leie_npis() -> set[str]:
    leie = pd.read_csv(LEIE_CSV, usecols=["NPI"], dtype=str)
    leie["NPI"] = leie["NPI"].fillna("0000000000")
    return set(leie.loc[leie["NPI"].str.len() == 10, "NPI"]) - {"0000000000"}


def load_partb(population: str) -> pd.DataFrame:
    if not PARTB_CSV.exists():
        raise FileNotFoundError(
            f"{PARTB_CSV} missing. Download via scripts/download_real_data.sh "
            "or run `make download-real`."
        )
    cols_to_read = ["Rndrng_NPI", "Rndrng_Prvdr_Type", "Rndrng_Prvdr_Crdntls",
                    "Rndrng_Prvdr_State_Abrvtn", "Rndrng_Prvdr_Ent_Cd"] + NUMERIC_FEATURES
    df = pd.read_csv(PARTB_CSV, usecols=lambda c: c in cols_to_read,
                     dtype={"Rndrng_NPI": str, "Rndrng_Prvdr_Type": str,
                            "Rndrng_Prvdr_Crdntls": str,
                            "Rndrng_Prvdr_State_Abrvtn": str,
                            "Rndrng_Prvdr_Ent_Cd": str},
                     low_memory=False)
    print(f"  Loaded {len(df):,} real Medicare Part B providers (2023)")

    if population == "ltc":
        mask = df["Rndrng_Prvdr_Type"].fillna("").str.contains(
            LTC_PROVIDER_TYPES_REGEX, case=False, regex=True
        )
        df = df[mask].copy()
        print(f"  Filtered to LTC-relevant provider types: {len(df):,}")
    return df


def attach_label(df: pd.DataFrame, leie_npis: set[str]) -> pd.DataFrame:
    df = df.copy()
    df["excluded_for_fraud"] = df["Rndrng_NPI"].isin(leie_npis).astype(int)
    n_pos = df["excluded_for_fraud"].sum()
    print(f"  REAL fraud labels (NPI ∈ LEIE): {n_pos:,} / {len(df):,} "
          f"({n_pos/len(df):.3%})")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in NUMERIC_FEATURES:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    # Provider-type one-hot for the LTC-relevant categories (and a catch-all)
    pt = df["Rndrng_Prvdr_Type"].fillna("Unknown")
    for tag, regex in [
        ("nurse_practitioner",  r"Nurse Practitioner"),
        ("geriatric_medicine",  r"Geriatric Medicine"),
        ("hospice_palliative",  r"Hospice and Palliative"),
        ("home_health",         r"Home Health"),
    ]:
        out[f"ptype_{tag}"] = pt.str.contains(regex, case=False, regex=True).astype(int)

    # Entity code: I (individual) vs O (organization)
    out["entity_individual"]   = (df["Rndrng_Prvdr_Ent_Cd"] == "I").astype(int)
    out["entity_organization"] = (df["Rndrng_Prvdr_Ent_Cd"] == "O").astype(int)

    # Engineered ratios — peer-relative billing signals
    out["pymt_per_bene"]   = out.get("Tot_Mdcr_Pymt_Amt", 0) / (out.get("Tot_Benes", 1) + 1)
    out["svcs_per_bene"]   = out.get("Tot_Srvcs",         0) / (out.get("Tot_Benes", 1) + 1)
    out["hcpcs_per_bene"]  = out.get("Tot_HCPCS_Cds",     0) / (out.get("Tot_Benes", 1) + 1)
    out["alwd_to_sbmtd"]   = out.get("Tot_Mdcr_Alowd_Amt", 0) / (out.get("Tot_Sbmtd_Chrg", 1) + 1)
    out["dual_share"]      = out.get("Bene_Dual_Cnt", 0) / (
        (out.get("Bene_Dual_Cnt", 0) + out.get("Bene_Ndual_Cnt", 1)) + 1
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True)).fillna(0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", choices=["ltc", "all"], default="ltc",
                        help="ltc = LTC-relevant provider types (~193K); "
                             "all = full Part B (1.26M, slow)")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip 3-fold cross-validation (faster; cuts ~70%% of runtime)")
    args = parser.parse_args()

    print("Real-NPI fraud-label pipeline (Medicare Part B 2023 ⋈ OIG LEIE)")
    print(f"  Population scope: {args.population}")
    leie_npis = load_leie_npis()
    print(f"  LEIE real NPIs: {len(leie_npis):,}")

    df = load_partb(args.population)
    df = attach_label(df, leie_npis)

    print("\nBuilding feature matrix...")
    X = build_features(df)
    y = df["excluded_for_fraud"].values
    print(f"  Features: {X.shape[1]}  (n={X.shape[0]:,})")

    # Save modeling table (only for the LTC scope; the all scope is too big)
    if args.population == "ltc":
        Path(config.DATA_PROCESSED).mkdir(parents=True, exist_ok=True)
        out_tbl = X.copy()
        out_tbl.insert(0, "NPI",          df["Rndrng_NPI"].values)
        out_tbl.insert(1, "Provider_Type", df["Rndrng_Prvdr_Type"].values)
        out_tbl.insert(2, "State",         df["Rndrng_Prvdr_State_Abrvtn"].values)
        out_tbl["excluded_for_fraud"] = y
        table_path = Path(config.DATA_PROCESSED) / "medicare_partb_modeling_table.csv"
        out_tbl.to_csv(table_path, index=False)
        print(f"  Saved modeling table to {table_path}")

    # 80/20 stratified split. Class is extreme imbalance (~0.04-0.07%) so we use
    # class_weight=balanced and focus on PR-AUC.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
    )
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}  "
          f"Test positive rate: {y_test.mean():.4%}")

    # Hyperparameters tuned for the larger 193K-row population: lower
    # n_estimators and depth keep wall-time tractable. GB is the main bottleneck.
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, solver="liblinear",
            random_state=config.RANDOM_SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced", max_depth=10,
            min_samples_leaf=10, random_state=config.RANDOM_SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.08,
            random_state=config.RANDOM_SEED),
    }

    print("\nTraining models on REAL Medicare provider data with REAL LEIE labels...")
    metrics = {}
    fitted = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        fitted[name] = m
        y_prob = m.predict_proba(X_test)[:, 1]
        y_pred = m.predict(X_test)
        metrics[name] = {
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "pr_auc":    round(average_precision_score(y_test, y_prob), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
        print(f"  {name}: AUC={metrics[name]['roc_auc']:.4f}  "
              f"PR-AUC={metrics[name]['pr_auc']:.4f}  "
              f"F1={metrics[name]['f1']:.4f}  "
              f"Recall={metrics[name]['recall']:.4f}")

    # 3-fold CV on PR-AUC (3 folds because the dataset is large and positives
    # are extremely sparse; std stays meaningful). Skipped if --no-cv passed.
    if args.no_cv:
        print("\nSkipping CV (--no-cv).")
    else:
        print("\n3-fold CV (PR-AUC)...")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_SEED)
        for name, m in models.items():
            scores = cross_val_score(clone(m), X, y, cv=skf,
                                     scoring="average_precision", n_jobs=-1)
            metrics[name]["cv_pr_auc_mean"] = round(float(np.mean(scores)), 4)
            metrics[name]["cv_pr_auc_std"]  = round(float(np.std(scores)),  4)
            metrics[name]["cv_folds"] = len(scores)
            print(f"  {name}: CV PR-AUC = {metrics[name]['cv_pr_auc_mean']:.4f} ± "
                  f"{metrics[name]['cv_pr_auc_std']:.4f}  (n_folds={len(scores)})")

    best_name = max(metrics, key=lambda k: metrics[k]["pr_auc"])
    best = fitted[best_name]
    print(f"\nBest model: {best_name} (PR-AUC={metrics[best_name]['pr_auc']:.4f})")

    # Persist outputs
    Path(config.OUTPUTS_REPORTS).mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUTS_FIGURES).mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUTS_MODELS).mkdir(parents=True, exist_ok=True)

    annotated = {
        "_data_source": "real_medicare_partb_2023_xref_oig_leie",
        "_population":  args.population,
        "_n_providers": int(len(df)),
        "_n_positive":  int(y.sum()),
        "_positive_rate_pct": round(float(y.mean() * 100), 4),
        "_label":      "excluded_for_fraud = NPI ∈ HHS-OIG LEIE",
        "_best_model": best_name,
        "_n_features": int(X.shape[1]),
        **metrics,
    }
    metrics_path = Path(config.OUTPUTS_REPORTS) / "medicare_partb_metrics.json"
    metrics_path.write_text(json.dumps(annotated, indent=2))
    print(f"  Saved metrics to {metrics_path}")

    # Classification report (best model)
    y_pred = best.predict(X_test)
    report = classification_report(y_test, y_pred,
                                    target_names=["OK", "Excluded"], digits=4,
                                    zero_division=0)
    rep_path = Path(config.OUTPUTS_REPORTS) / "medicare_partb_classification_report.txt"
    rep_path.write_text(
        f"Medicare Part B 2023 ⋈ OIG LEIE — best model: {best_name}\n"
        f"Population: {args.population}  n_test={len(y_test)}  "
        f"positive rate={y_test.mean():.4%}\n"
        f"{'='*60}\n{report}"
    )

    # ROC and PR curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={metrics[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Medicare Part B ⋈ LEIE — ROC")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(Path(config.OUTPUTS_FIGURES) / "medicare_partb_roc.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in fitted.items():
        prec, rec, _ = precision_recall_curve(y_test, m.predict_proba(X_test)[:, 1])
        ax.plot(rec, prec, lw=2,
                label=f"{name} (PR-AUC={metrics[name]['pr_auc']:.3f})")
    ax.axhline(y=y_test.mean(), color="gray", lw=1, linestyle="--",
               label=f"Prevalence = {y_test.mean():.4%}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Medicare Part B ⋈ LEIE — Precision-Recall")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(Path(config.OUTPUTS_FIGURES) / "medicare_partb_pr.png", dpi=120)
    plt.close()

    # Feature importance
    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=X.columns)
        top = imp.sort_values(ascending=False).head(20)[::-1]
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top.index, top.values, color="#4c78a8")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top features — {best_name} on Medicare Part B (real LEIE labels)")
        plt.tight_layout()
        plt.savefig(Path(config.OUTPUTS_FIGURES) / "medicare_partb_feature_importance.png",
                    dpi=120)
        plt.close()

    joblib.dump(best, Path(config.OUTPUTS_MODELS) / "medicare_partb_best_model.pkl")
    print(f"\nMedicare Part B ⋈ LEIE pipeline complete.")
    print(f"\nThis is the project's only pipeline with REAL NPI-keyed fraud labels —")
    print(f"every positive case is a real US provider with real Medicare billing data")
    print(f"who appears on the federal HHS-OIG exclusion list.")


if __name__ == "__main__":
    main()
