"""
cms_ltc_pipeline.py
===================
SECOND real-data pipeline: trains LTC FWA-risk models on the **CMS Nursing
Home Provider Information** dataset — real US nursing-home data with real
quality and enforcement labels.

This is the parallel pipeline that addresses the most-cited weakness of the
Kaggle pipeline (anonymized provider IDs, generic Medicare claims). Here:

    - Real provider names and CMS Certification Numbers (CCNs)
    - Real LTC-specific population (14,699 US nursing homes)
    - Real quality-of-care / enforcement labels:
        * Abuse Icon (CMS-cited for resident abuse)
        * Special Focus Facility (SFF / SFF Candidate watch list)
        * Number of Fines (5+ in past 3 years)
        * Number of Payment Denials (any)
    - Real ownership, staffing, deficiency, turnover features
    - Real geography (state, county, lat/lon)

Combined flag rate: ~22% of providers. This is the closest publicly
available approximation to the data Manulife/JH LTC FWA Advanced Analytics
would actually work with.

Cross-reference with OIG LEIE
-----------------------------
After training, we look up each modeled provider's legal business name in
the LEIE — providers whose name matches an LEIE entry have been excluded
from federal programs for actual fraud, kickbacks, or felony conviction.
This gives us a true fraud-ground-truth signal on top of the quality flags.

Outputs
-------
data/processed/cms_ltc_modeling_table.csv       cleaned numeric feature table
outputs/reports/cms_ltc_metrics.json            held-out test metrics
outputs/reports/cms_ltc_classification_report.txt
outputs/reports/cms_ltc_leie_overlap.csv        providers matched to LEIE exclusions
outputs/figures/cms_ltc_roc.png
outputs/figures/cms_ltc_feature_importance.png
outputs/figures/cms_ltc_label_components.png    breakdown of how the combined label was constructed
"""

from __future__ import annotations

import json
import os
import re
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
    f1_score, classification_report, roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


RAW_CSV = Path(config.DATA_RAW) / "cms" / "cms_nursing_home_provider_info.csv"


def load_cms() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} missing. Download with src/download_real_data.sh "
            "or see README §X."
        )
    df = pd.read_csv(RAW_CSV, dtype=str, low_memory=False)
    print(f"  Loaded {len(df):,} real US nursing homes from CMS")
    return df


def build_label(df: pd.DataFrame) -> pd.Series:
    """Combined LTC-FWA risk label:
       Abuse Icon == 'Y'   OR   SFF / SFF Candidate
       OR   Number of Fines >= 5   OR   Number of Payment Denials >= 1
    """
    abuse = df["Abuse Icon"] == "Y"
    sff   = df["Special Focus Status"].fillna("").str.contains(
        "SFF|Special Focus", case=False, regex=True, na=False
    )
    fines = pd.to_numeric(df["Number of Fines"], errors="coerce").fillna(0) >= 5
    denials = pd.to_numeric(df["Number of Payment Denials"], errors="coerce").fillna(0) >= 1
    label = (abuse | sff | fines | denials).astype(int)

    print(f"  Label component counts:")
    print(f"    abuse_icon=Y                : {abuse.sum():>5,}")
    print(f"    SFF or SFF Candidate        : {sff.sum():>5,}")
    print(f"    Fines >= 5                  : {fines.sum():>5,}")
    print(f"    Payment Denials >= 1        : {denials.sum():>5,}")
    print(f"    Combined flagged (any)      : {label.sum():>5,} ({label.mean():.2%})")
    return label, dict(abuse=abuse, sff=sff, fines=fines, denials=denials)


# ── Feature engineering ───────────────────────────────────────────────────────

OWNERSHIP_BUCKETS = {
    "For profit - Limited Liability company": "for_profit_llc",
    "For profit - Corporation":                "for_profit_corp",
    "For profit - Individual":                 "for_profit_indiv",
    "For profit - Partnership":                "for_profit_partner",
    "Non profit - Corporation":                "non_profit_corp",
    "Non profit - Other":                      "non_profit_other",
    "Non profit - Church related":             "non_profit_church",
    "Government - Federal":                    "govt",
    "Government - State":                      "govt",
    "Government - County":                     "govt",
    "Government - City":                       "govt",
    "Government - Hospital district":          "govt",
    "Government - City/county":                "govt",
}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _yes_no(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.upper() == "Y").astype(int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Size and capacity
    out["num_certified_beds"]            = _num(df["Number of Certified Beds"])
    out["avg_residents_per_day"]         = _num(df["Average Number of Residents per Day"])
    out["occupancy_pct"]                 = out["avg_residents_per_day"] / out["num_certified_beds"]

    # Ownership (one-hot via buckets)
    bucket = df["Ownership Type"].map(OWNERSHIP_BUCKETS).fillna("other")
    for b in ["for_profit_llc", "for_profit_corp", "for_profit_indiv",
              "for_profit_partner", "non_profit_corp", "non_profit_other",
              "non_profit_church", "govt"]:
        out[f"own_{b}"] = (bucket == b).astype(int)

    # Provider type
    out["is_medicare_and_medicaid"] = (df["Provider Type"] == "Medicare and Medicaid").astype(int)
    out["resides_in_hospital"]      = _yes_no(df["Provider Resides in Hospital"])
    out["in_chain"]                 = df["Chain Name"].fillna("").astype(str).str.len().gt(0).astype(int)
    out["n_facilities_in_chain"]    = _num(df["Number of Facilities in Chain"]).fillna(0)
    out["is_ccrc"]                  = _yes_no(df["Continuing Care Retirement Community"])

    # CMS 5-star ratings (1-5)
    for col, new in [
        ("Overall Rating",            "rating_overall"),
        ("Health Inspection Rating",  "rating_health"),
        ("Staffing Rating",           "rating_staffing"),
        ("QM Rating",                 "rating_qm"),
        ("Long-Stay QM Rating",       "rating_qm_long"),
        ("Short-Stay QM Rating",      "rating_qm_short"),
    ]:
        if col in df.columns:
            out[new] = _num(df[col])

    # Staffing (hours per resident per day)
    staffing_cols = {
        "Reported Total Nurse Staffing Hours per Resident per Day": "staff_total_hours",
        "Reported RN Staffing Hours per Resident per Day":          "staff_rn_hours",
        "Reported LPN Staffing Hours per Resident per Day":         "staff_lpn_hours",
        "Reported Nurse Aide Staffing Hours per Resident per Day":  "staff_aide_hours",
        "Total Nursing Staff Turnover":                              "turnover_total_nursing",
        "Registered Nurse Turnover":                                 "turnover_rn",
        "Number of Administrators Who Have Left the Nursing Home":   "n_admins_left",
    }
    for src, dst in staffing_cols.items():
        if src in df.columns:
            out[dst] = _num(df[src])

    # Casemix
    out["nursing_casemix_index"] = _num(df["Nursing Case-Mix Index"]) if "Nursing Case-Mix Index" in df.columns else np.nan

    # Inspection / deficiency history
    if "Rating Cycle 1 Total Number of Health Deficiencies" in df.columns:
        out["cycle1_deficiencies"]    = _num(df["Rating Cycle 1 Total Number of Health Deficiencies"]).fillna(0)
        out["cycle1_health_score"]    = _num(df["Rating Cycle 1 Health Deficiency Score"]).fillna(0)
        out["cycle1_total_score"]     = _num(df["Rating Cycle 1 Total Health Score"]).fillna(0)
    if "Total Weighted Health Survey Score" in df.columns:
        out["total_weighted_health"]  = _num(df["Total Weighted Health Survey Score"]).fillna(0)
    if "Number of Citations from Infection Control Inspections" in df.columns:
        out["infection_citations"]    = _num(df["Number of Citations from Infection Control Inspections"]).fillna(0)

    # Compliance / ownership-change risk
    out["health_inspection_overdue"]    = _yes_no(df["Most Recent Health Inspection More Than 2 Years Ago"])
    out["ownership_changed_12m"]        = _yes_no(df["Provider Changed Ownership in Last 12 Months"])
    out["sprinklers_full"]              = _yes_no(df["Automatic Sprinkler Systems in All Required Areas"])

    # Numeric features only; drop columns that are all NaN
    out = out.select_dtypes(include=[np.number])
    out = out.loc[:, out.notna().any()]
    return out


# ── Cross-reference with OIG LEIE ─────────────────────────────────────────────

def cross_reference_leie(cms_df: pd.DataFrame) -> pd.DataFrame:
    """Look up each CMS provider's legal business name in the LEIE."""
    leie_path = Path(config.DATA_RAW) / "oig" / "oig_leie_updated.csv"
    if not leie_path.exists():
        print("  LEIE not available; skipping cross-reference")
        return pd.DataFrame()
    leie = pd.read_csv(leie_path, dtype=str, low_memory=False)
    leie_names = leie["BUSNAME"].dropna().astype(str).str.upper().str.strip()
    leie_set = set(leie_names)
    print(f"  LEIE has {len(leie_set):,} unique business names")

    def _normalize(s):
        if not isinstance(s, str):
            return ""
        s = re.sub(r"[^A-Z0-9 ]", " ", s.upper())
        s = re.sub(r"\s+", " ", s).strip()
        return s

    cms_names = cms_df["Legal Business Name"].fillna("").apply(_normalize)
    leie_norm = pd.Series(list(leie_set)).apply(_normalize)
    leie_norm_set = set(leie_norm)

    matched = cms_names.isin(leie_norm_set)
    print(f"  CMS-LEIE business-name overlap: {matched.sum():,} providers")

    overlap = cms_df.loc[matched, [
        "CMS Certification Number (CCN)", "Provider Name", "State",
        "Legal Business Name", "Ownership Type",
    ]].copy()
    overlap["matched_leie"] = True
    return overlap


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    print("Real-LTC pipeline: CMS Nursing Home Provider Information")
    df = load_cms()

    # Filter to LTC-relevant columns and rows
    label, components = build_label(df)

    print("\nBuilding feature table...")
    X = build_features(df)
    # Replace ±inf with NaN, then median-impute
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    # Anything still NaN (all-NaN columns) → 0
    X = X.fillna(0)
    print(f"  Feature table: {X.shape}")

    # Persist the modeling table
    persist = X.copy()
    persist.insert(0, "CCN", df["CMS Certification Number (CCN)"].values)
    persist.insert(1, "Provider_Name", df["Provider Name"].values)
    persist.insert(2, "State", df["State"].values)
    persist["flagged"] = label.values
    persist_path = Path(config.DATA_PROCESSED) / "cms_ltc_modeling_table.csv"
    persist.to_csv(persist_path, index=False)
    print(f"  Saved modeling table to {persist_path}")

    # 80/20 stratified test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, label, test_size=0.2, stratify=label, random_state=config.RANDOM_SEED
    )
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}  "
          f"Test flag rate: {y_test.mean():.2%}")

    # Train + evaluate three models
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, solver="liblinear",
            random_state=config.RANDOM_SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", max_depth=12,
            random_state=config.RANDOM_SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=config.RANDOM_SEED),
    }

    print("\nTraining and evaluating models...")
    metrics = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
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
              f"F1={metrics[name]['f1']:.4f}")

    # 5-fold CV on the full dataset
    print("\n5-fold stratified CV (PR-AUC)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    for name, m in models.items():
        from sklearn.base import clone
        scores = cross_val_score(clone(m), X, label, cv=skf,
                                  scoring="average_precision", n_jobs=-1)
        metrics[name]["cv_pr_auc_mean"] = round(float(np.mean(scores)), 4)
        metrics[name]["cv_pr_auc_std"]  = round(float(np.std(scores)), 4)
        print(f"  {name}: CV PR-AUC = {metrics[name]['cv_pr_auc_mean']:.4f} ± "
              f"{metrics[name]['cv_pr_auc_std']:.4f}")

    best_name = max(metrics, key=lambda n: metrics[n]["pr_auc"])
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (PR-AUC={metrics[best_name]['pr_auc']:.4f})")

    # Cross-reference with LEIE
    print("\nCross-referencing CMS providers with OIG LEIE exclusions...")
    overlap = cross_reference_leie(df)
    if not overlap.empty:
        overlap_path = Path(config.OUTPUTS_REPORTS) / "cms_ltc_leie_overlap.csv"
        overlap.to_csv(overlap_path, index=False)
        print(f"  Saved {len(overlap):,} matched providers to {overlap_path}")

    # Persist artifacts
    os.makedirs(config.OUTPUTS_REPORTS, exist_ok=True)
    os.makedirs(config.OUTPUTS_FIGURES, exist_ok=True)
    os.makedirs(config.OUTPUTS_MODELS, exist_ok=True)

    annotated = {
        "_data_source":    "real_cms_nursing_home_compare",
        "_n_providers":    int(len(df)),
        "_label_rate":     float(label.mean()),
        "_label_logic":    "Abuse Icon Y OR SFF/SFF Candidate OR Fines>=5 OR Denials>=1",
        "_best_model":     best_name,
        "_n_features":     int(X.shape[1]),
        **metrics,
    }
    metrics_path = Path(config.OUTPUTS_REPORTS) / "cms_ltc_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(annotated, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # Classification report
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["OK", "Flagged"], digits=4)
    rep_path = Path(config.OUTPUTS_REPORTS) / "cms_ltc_classification_report.txt"
    rep_path.write_text(
        f"CMS Nursing Home — best model: {best_name}\n"
        f"Test n={len(y_test)}  flag rate={y_test.mean():.4f}\n"
        f"{'='*60}\n{report}"
    )
    print(f"  Saved classification report to {rep_path}")

    # ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in models.items():
        fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={metrics[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("CMS Nursing Home FWA Risk — ROC")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(Path(config.OUTPUTS_FIGURES) / "cms_ltc_roc.png", dpi=120)
    plt.close()

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=X.columns)
        top = imp.sort_values(ascending=False).head(20)[::-1]
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top.index, top.values, color="#4c78a8")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top features — {best_name} on CMS Nursing Home data")
        plt.tight_layout()
        plt.savefig(Path(config.OUTPUTS_FIGURES) / "cms_ltc_feature_importance.png", dpi=120)
        plt.close()

    # Label-component breakdown
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = {k: int(v.sum()) for k, v in components.items()}
    counts["any_flag"] = int(label.sum())
    ax.bar(list(counts.keys()), list(counts.values()), color="#4c78a8")
    for i, v in enumerate(counts.values()):
        ax.text(i, v + 30, str(v), ha="center", fontsize=10)
    ax.set_ylabel("Providers")
    ax.set_title("Label-component breakdown (CMS Nursing Home, n=14,699)")
    plt.tight_layout()
    plt.savefig(Path(config.OUTPUTS_FIGURES) / "cms_ltc_label_components.png", dpi=120)
    plt.close()

    # Persist model
    model_path = Path(config.OUTPUTS_MODELS) / f"cms_ltc_best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"  Saved best model to {model_path}")

    print("\nCMS LTC pipeline complete.")


if __name__ == "__main__":
    main()
