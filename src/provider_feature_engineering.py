"""
provider_feature_engineering.py
================================
Builds provider-level fraud risk features from joined inpatient / outpatient /
beneficiary data.  Uses the Kaggle Healthcare Provider Fraud Detection Analysis
dataset.

Output: data/processed/provider_modeling_table.csv

All column checks are defensive: if a column doesn't exist in the actual Kaggle
CSV, that feature is silently skipped.
"""

import os
import sys
import re

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_ingestion import (
    check_data_available,
    find_kaggle_files,
    load_beneficiary,
    load_inpatient,
    load_outpatient,
    load_provider_labels,
    print_download_instructions,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, *candidates: str):
    """Return the first matching column name (case-insensitive), or None."""
    col_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in col_lower:
            return col_lower[cand.lower()]
    return None


def _has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _safe_sum(df: pd.DataFrame, col: str):
    if not _has(df, col):
        return pd.Series(0, index=df.index)
    return df[col].fillna(0)


# ── Loading ─────────────────────────────────────────────────────────────────────

def load_raw_data():
    """Find and load all four Kaggle files.  Raises FileNotFoundError if missing."""
    if not check_data_available(config.DATA_RAW):
        print_download_instructions()
        raise FileNotFoundError(
            "Kaggle data not found in data/raw/. See instructions above."
        )

    found = find_kaggle_files(config.DATA_RAW)
    beneficiary = load_beneficiary(found["beneficiary"])
    inpatient   = load_inpatient(found["inpatient"])
    outpatient  = load_outpatient(found["outpatient"])
    labels      = load_provider_labels(found["labels"])
    return beneficiary, inpatient, outpatient, labels


# ── Joining ────────────────────────────────────────────────────────────────────

def join_claims_with_beneficiary(claims: pd.DataFrame, beneficiary: pd.DataFrame,
                                  suffix: str) -> pd.DataFrame:
    """Left-join claims onto beneficiary on BeneID (case-insensitive)."""
    bene_id_claim = _col(claims, "BeneID")
    bene_id_bene  = _col(beneficiary, "BeneID")
    if bene_id_claim is None or bene_id_bene is None:
        print(f"  WARNING [{suffix}] BeneID column not found; skipping beneficiary join.")
        return claims
    merged = claims.merge(
        beneficiary, left_on=bene_id_claim, right_on=bene_id_bene,
        how="left", suffixes=("", f"_bene_{suffix}")
    )
    return merged


# ── Feature builders ───────────────────────────────────────────────────────────

def _volume_features(ip: pd.DataFrame, op: pd.DataFrame, provider_col_ip: str,
                     provider_col_op: str) -> pd.DataFrame:
    """Volume-based provider-level aggregations."""
    ip_counts = ip.groupby(provider_col_ip).size().rename("inpatient_claim_count")
    op_counts = op.groupby(provider_col_op).size().rename("outpatient_claim_count")

    vol = pd.concat([ip_counts, op_counts], axis=1).fillna(0)
    vol["total_claims"] = vol["inpatient_claim_count"] + vol["outpatient_claim_count"]
    vol["inpatient_ratio"] = vol["inpatient_claim_count"] / vol["total_claims"].clip(lower=1)

    # Unique beneficiaries
    bene_col_ip = _col(ip, "BeneID")
    bene_col_op = _col(op, "BeneID")

    ip_bene = (ip.groupby(provider_col_ip)[bene_col_ip].nunique()
               .rename("ip_unique_bene") if bene_col_ip else pd.Series(dtype=int))
    op_bene = (op.groupby(provider_col_op)[bene_col_op].nunique()
               .rename("op_unique_bene") if bene_col_op else pd.Series(dtype=int))

    vol = vol.join(ip_bene, how="left").join(op_bene, how="left")
    vol["unique_beneficiaries"] = (
        vol.get("ip_unique_bene", 0).fillna(0) + vol.get("op_unique_bene", 0).fillna(0)
    )
    vol["claim_frequency_per_beneficiary"] = (
        vol["total_claims"] / vol["unique_beneficiaries"].clip(lower=1)
    )
    vol = vol.drop(columns=["ip_unique_bene", "op_unique_bene"], errors="ignore")
    return vol


def _physician_features(ip: pd.DataFrame, op: pd.DataFrame, provider_col_ip: str,
                         provider_col_op: str) -> pd.DataFrame:
    """Unique physician counts per provider."""
    rows = {}
    for role, col_name in [
        ("attending",  "AttendingPhysician"),
        ("operating",  "OperatingPhysician"),
        ("other",      "OtherPhysician"),
    ]:
        ip_col = _col(ip, col_name)
        op_col = _col(op, col_name)

        ip_agg = (ip.groupby(provider_col_ip)[ip_col].nunique()
                  if ip_col else pd.Series(dtype=int))
        op_agg = (op.groupby(provider_col_op)[op_col].nunique()
                  if op_col else pd.Series(dtype=int))

        combined = pd.concat([ip_agg.rename("ip"), op_agg.rename("op")], axis=1).fillna(0)
        rows[f"unique_{role}_physicians"] = combined.max(axis=1)

    return pd.DataFrame(rows)


def _financial_features(ip: pd.DataFrame, op: pd.DataFrame, provider_col_ip: str,
                         provider_col_op: str) -> pd.DataFrame:
    """Reimbursement / deductible aggregations."""
    amt_ip = _col(ip, "InscClaimAmtReimbursed")
    amt_op = _col(op, "InscClaimAmtReimbursed")
    ded_ip = _col(ip, "DeductibleAmtPaid")
    ded_op = _col(op, "DeductibleAmtPaid")

    feats = {}

    # Reimbursement sums
    if amt_ip:
        feats["total_inpatient_reimbursed"] = ip.groupby(provider_col_ip)[amt_ip].sum()
        feats["avg_ip_reimbursed_per_claim"]= ip.groupby(provider_col_ip)[amt_ip].mean()
    if amt_op:
        feats["total_outpatient_reimbursed"] = op.groupby(provider_col_op)[amt_op].sum()
        feats["avg_op_reimbursed_per_claim"] = op.groupby(provider_col_op)[amt_op].mean()

    # Deductibles
    if ded_ip:
        feats["total_ip_deductible"] = ip.groupby(provider_col_ip)[ded_ip].sum()
    if ded_op:
        feats["total_op_deductible"] = op.groupby(provider_col_op)[ded_op].sum()

    df_fin = pd.DataFrame(feats).fillna(0)

    ip_col = feats.get("total_inpatient_reimbursed", pd.Series(dtype=float))
    op_col = feats.get("total_outpatient_reimbursed", pd.Series(dtype=float))

    if "total_inpatient_reimbursed" in df_fin.columns and "total_outpatient_reimbursed" in df_fin.columns:
        df_fin["total_reimbursed"] = (
            df_fin["total_inpatient_reimbursed"] + df_fin["total_outpatient_reimbursed"]
        )
    elif "total_inpatient_reimbursed" in df_fin.columns:
        df_fin["total_reimbursed"] = df_fin["total_inpatient_reimbursed"]
    elif "total_outpatient_reimbursed" in df_fin.columns:
        df_fin["total_reimbursed"] = df_fin["total_outpatient_reimbursed"]

    if "total_ip_deductible" in df_fin.columns and "total_op_deductible" in df_fin.columns:
        df_fin["total_deductible_amount"] = (
            df_fin["total_ip_deductible"] + df_fin["total_op_deductible"]
        )
    elif "total_ip_deductible" in df_fin.columns:
        df_fin["total_deductible_amount"] = df_fin["total_ip_deductible"]
    elif "total_op_deductible" in df_fin.columns:
        df_fin["total_deductible_amount"] = df_fin["total_op_deductible"]

    # Inpatient reimbursement share
    if "total_inpatient_reimbursed" in df_fin.columns and "total_reimbursed" in df_fin.columns:
        df_fin["inpatient_reimbursement_share"] = (
            df_fin["total_inpatient_reimbursed"] / df_fin["total_reimbursed"].clip(lower=1)
        )

    return df_fin


def _demographic_features(ip_merged: pd.DataFrame, op_merged: pd.DataFrame,
                            provider_col_ip: str, provider_col_op: str) -> pd.DataFrame:
    """Patient demographic aggregations from beneficiary-joined claim tables."""
    feats = {}

    # Patient age: try DOB column; else look for 'Age' directly
    for df, prov_col, tag in [(ip_merged, provider_col_ip, "ip"),
                               (op_merged, provider_col_op, "op")]:
        dob_col = _col(df, "DOB", "BeneDOB", "DateOfBirth")
        age_col = _col(df, "Age")

        if dob_col:
            try:
                df = df.copy()
                df[dob_col] = pd.to_datetime(df[dob_col], errors="coerce")
                dod_col = _col(df, "DOD", "BeneDOD", "DateOfDeath")
                ref_date = (
                    df[dod_col].where(df[dod_col].notna(), pd.Timestamp("2010-12-01"))
                    if dod_col else pd.Timestamp("2010-12-01")
                )
                df["_age_derived"] = (ref_date - df[dob_col]).dt.days / 365.25
                feats[f"{tag}_avg_age"]  = df.groupby(prov_col)["_age_derived"].mean()
                feats[f"{tag}_std_age"]  = df.groupby(prov_col)["_age_derived"].std()
            except Exception:
                pass
        elif age_col:
            feats[f"{tag}_avg_age"]  = df.groupby(prov_col)[age_col].mean()
            feats[f"{tag}_std_age"]  = df.groupby(prov_col)[age_col].std()

    # Derive avg_patient_age from whichever tags were built
    demo = pd.DataFrame(feats).fillna(np.nan)
    age_cols = [c for c in demo.columns if "avg_age" in c]
    if age_cols:
        demo["avg_patient_age"] = demo[age_cols].mean(axis=1)
    std_cols = [c for c in demo.columns if "std_age" in c]
    if std_cols:
        demo["std_patient_age"] = demo[std_cols].mean(axis=1)

    # Death rate
    for df, prov_col, tag in [(ip_merged, provider_col_ip, "ip"),
                               (op_merged, provider_col_op, "op")]:
        dod_col = _col(df, "DOD", "BeneDOD", "DateOfDeath")
        bene_col = _col(df, "BeneID")
        if dod_col and bene_col:
            df = df.copy()
            df[dod_col] = pd.to_datetime(df[dod_col], errors="coerce")
            df["_has_dod"] = df[dod_col].notna().astype(int)
            feats[f"{tag}_death_rate"] = df.groupby(prov_col)["_has_dod"].mean()

    # Chronic conditions
    chronic_agg_frames = []
    for df, prov_col, tag in [(ip_merged, provider_col_ip, "ip"),
                               (op_merged, provider_col_op, "op")]:
        chronic_cols = [c for c in df.columns if re.search(r"chroniccond", c, re.I)]
        if chronic_cols:
            df = df.copy()
            # Kaggle uses 2 = No, 1 = Yes; convert to 0/1
            for cc in chronic_cols:
                df[cc] = df[cc].map({1: 1, 2: 0}).fillna(0)
            df["_n_chronic"] = df[chronic_cols].sum(axis=1)
            chronic_agg_frames.append(
                df.groupby(prov_col)["_n_chronic"].mean().rename(f"{tag}_avg_chronic")
            )

    if chronic_agg_frames:
        chr_df = pd.concat(chronic_agg_frames, axis=1).fillna(np.nan)
        demo = demo.join(chr_df, how="outer")
        avg_chronic_cols = [c for c in chr_df.columns]
        demo["avg_chronic_conditions"] = demo[avg_chronic_cols].mean(axis=1)

    # Death rate combined
    dr_cols = [c for c in demo.columns if "death_rate" in c]
    if dr_cols:
        demo["death_rate"] = demo[dr_cols].mean(axis=1)

    # Drop intermediate per-split columns; keep summary columns
    keep = [c for c in demo.columns if not c.startswith("ip_") and not c.startswith("op_")]
    return demo[keep] if keep else demo


def _duration_features(ip: pd.DataFrame, provider_col_ip: str) -> pd.DataFrame:
    """Inpatient admission duration (days)."""
    admit_col  = _col(ip, "AdmissionDt", "ClmAdmitDate", "AdmitDate")
    discharge_col = _col(ip, "DischargeDt", "ClmDischargeDate", "DischargeDate")

    if admit_col is None or discharge_col is None:
        return pd.DataFrame()

    ip = ip.copy()
    ip[admit_col]    = pd.to_datetime(ip[admit_col],    errors="coerce")
    ip[discharge_col]= pd.to_datetime(ip[discharge_col],errors="coerce")
    ip["_dur"] = (ip[discharge_col] - ip[admit_col]).dt.days.clip(lower=0)

    dur = ip.groupby(provider_col_ip)["_dur"].agg(
        avg_admission_duration="mean",
        max_admission_duration="max",
    )
    return dur


def _code_diversity_features(ip: pd.DataFrame, op: pd.DataFrame,
                               provider_col_ip: str, provider_col_op: str) -> pd.DataFrame:
    """Diagnosis and procedure code diversity per provider."""
    diag_pattern = re.compile(r"clmdiagnosiscode", re.I)
    proc_pattern = re.compile(r"clmprocedurecode", re.I)

    frames = []
    for df, prov_col, tag in [(ip, provider_col_ip, "ip"), (op, provider_col_op, "op")]:
        diag_cols = [c for c in df.columns if diag_pattern.search(c)]
        proc_cols = [c for c in df.columns if proc_pattern.search(c)]

        if diag_cols:
            df = df.copy()
            df["_n_diag"] = df[diag_cols].apply(
                lambda row: row.dropna().nunique(), axis=1
            )
            frames.append(
                df.groupby(prov_col)["_n_diag"].mean().rename(f"{tag}_diag_diversity")
            )
        if proc_cols:
            df = df.copy()
            df["_n_proc"] = df[proc_cols].apply(
                lambda row: row.dropna().nunique(), axis=1
            )
            frames.append(
                df.groupby(prov_col)["_n_proc"].mean().rename(f"{tag}_proc_diversity")
            )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).fillna(0)
    diag_frames = [c for c in combined.columns if "diag_diversity" in c]
    proc_frames = [c for c in combined.columns if "proc_diversity" in c]

    out = pd.DataFrame(index=combined.index)
    if diag_frames:
        out["diagnosis_code_diversity"] = combined[diag_frames].mean(axis=1)
    if proc_frames:
        out["procedure_code_diversity"] = combined[proc_frames].mean(axis=1)
    return out


def _risk_outlier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Provider-level risk / outlier features derived from the full provider table."""
    out = df.copy()

    if "total_claims" in out.columns:
        out["provider_volume_percentile"] = out["total_claims"].rank(pct=True)

    # We need per-claim avg reimbursement for the outlier score.
    if "total_reimbursed" in out.columns and "total_claims" in out.columns:
        out["avg_reimbursed_per_claim"] = (
            out["total_reimbursed"] / out["total_claims"].clip(lower=1)
        )
        overall_median = out["avg_reimbursed_per_claim"].median()
        overall_std    = out["avg_reimbursed_per_claim"].std()
        if overall_std and overall_std > 0:
            out["reimbursement_outlier_score"] = (
                (out["avg_reimbursed_per_claim"] - overall_median) / overall_std
            )

    if "total_reimbursed" in out.columns and "unique_beneficiaries" in out.columns:
        out["reimbursement_per_beneficiary"] = (
            out["total_reimbursed"] / out["unique_beneficiaries"].clip(lower=1)
        )

    return out


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_provider_table() -> pd.DataFrame:
    """
    Full pipeline: load, join, aggregate, compute features, attach labels.

    Returns provider-level DataFrame ready for modeling.
    """
    print("Loading raw Kaggle files...")
    beneficiary, inpatient, outpatient, labels = load_raw_data()

    # Identify provider column (may differ slightly in column name)
    prov_col_ip = _col(inpatient,  "Provider")
    prov_col_op = _col(outpatient, "Provider")
    if prov_col_ip is None or prov_col_op is None:
        raise ValueError("'Provider' column not found in inpatient or outpatient data.")

    print("Joining claims with beneficiary data...")
    ip_merged = join_claims_with_beneficiary(inpatient,  beneficiary, "ip")
    op_merged = join_claims_with_beneficiary(outpatient, beneficiary, "op")

    print("Computing volume features...")
    vol = _volume_features(inpatient, outpatient, prov_col_ip, prov_col_op)

    print("Computing physician features...")
    phy = _physician_features(inpatient, outpatient, prov_col_ip, prov_col_op)

    print("Computing financial features...")
    fin = _financial_features(inpatient, outpatient, prov_col_ip, prov_col_op)

    print("Computing patient demographic features...")
    dem = _demographic_features(ip_merged, op_merged, prov_col_ip, prov_col_op)

    print("Computing duration features...")
    dur = _duration_features(inpatient, prov_col_ip)

    print("Computing code diversity features...")
    div = _code_diversity_features(inpatient, outpatient, prov_col_ip, prov_col_op)

    # Combine all feature blocks
    print("Combining feature blocks...")
    provider_df = vol.copy()
    for block in [phy, fin, dem, dur, div]:
        if not block.empty:
            provider_df = provider_df.join(block, how="left")

    provider_df.index.name = "Provider"
    provider_df = provider_df.reset_index()

    # Add risk/outlier features
    print("Adding risk / outlier features...")
    provider_df = _risk_outlier_features(provider_df)

    # Attach labels
    print("Attaching fraud labels...")
    prov_label_col = _col(labels, "Provider")
    if prov_label_col is None:
        raise ValueError("'Provider' column missing from label file.")

    provider_df = provider_df.merge(
        labels[[prov_label_col, "PotentialFraud"]],
        left_on="Provider", right_on=prov_label_col, how="inner"
    )
    if prov_label_col != "Provider":
        provider_df = provider_df.drop(columns=[prov_label_col])

    print(f"\nProvider table shape : {provider_df.shape}")
    print(f"Fraud rate           : {provider_df['PotentialFraud'].mean():.2%}")
    print(f"Features             : {[c for c in provider_df.columns if c not in ['Provider','PotentialFraud']]}")

    return provider_df


def save_provider_table(df: pd.DataFrame):
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)
    out_path = os.path.join(config.DATA_PROCESSED, "provider_modeling_table.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved provider modeling table to {out_path}")
    return out_path


def main():
    print("Running provider feature engineering pipeline...")
    df = build_provider_table()
    save_provider_table(df)
    print("Provider feature engineering complete.")


if __name__ == "__main__":
    main()
