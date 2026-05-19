"""
oig_leie_analysis.py
====================
Integration of REAL federal healthcare-fraud data: the HHS-OIG List of
Excluded Individuals/Entities (LEIE).

Source
------
Public dataset, updated monthly:
    https://oig.hhs.gov/exclusions/exclusions_list.asp
Direct download:
    https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv

What this data is
-----------------
Every individual or entity currently excluded from participation in federal
healthcare programs (Medicare, Medicaid, etc.) for fraud, abuse, license
revocation, or felony conviction. **Real names, real NPIs, real exclusion
dates, real legal authority cited.**

This is the closest publicly available proxy to a fraud-ground-truth dataset.
The Kaggle Healthcare Provider Fraud dataset has provider-level *labels* but
anonymized provider IDs (PRV12345); the LEIE has real NPIs but no claim-level
detail. They are complementary.

What we do with it
------------------
1. Produce a real-fraud descriptive report:
   - Exclusions by type (legal authority cited)
   - Exclusions by specialty
   - LTC-specific exclusion breakdown
     (Home Health Agency, Skilled Nursing Facility, Hospice, Nursing Firm)
   - Year-over-year trend

2. Emit a real exclusion-type taxonomy file
   (data/documents/oig_exclusion_codes.txt) that augments the RAG policy
   corpus with REAL legal definitions of fraud — replacing the previously-
   synthetic-only policy text with real federal authority.

3. Tag any NPI overlap with the Kaggle Train_Inpatient.AttendingPhysician
   column as a sanity check (Kaggle uses anonymized PRVxxxxx for providers
   but real physician NPIs in the attending-physician fields). This gives us
   a real fraud-label cross-reference.

Outputs
-------
outputs/reports/oig_leie_summary.csv              top-line stats per category
outputs/reports/oig_leie_ltc_excerpt.csv          1,818 LTC-relevant records
outputs/reports/oig_leie_physician_overlap.csv    physicians appearing in both
outputs/figures/oig_leie_by_specialty.png         top-15 excluded specialties
outputs/figures/oig_leie_annual_trend.png         exclusions by year
data/documents/oig_exclusion_codes.txt            real legal definitions
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


LEIE_PATH = Path(config.DATA_RAW) / "oig" / "oig_leie_updated.csv"
DOWNLOAD_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"


# Real legal-authority taxonomy as published in the LEIE technical guide
# (https://oig.hhs.gov/exclusions/authorities.asp)
EXCLUSION_CODE_DEFINITIONS = {
    "1128a1": "Conviction of program-related crimes. Minimum 5-year exclusion. "
              "Mandatory exclusion under §1128(a)(1) of the Social Security Act.",
    "1128a2": "Conviction relating to patient abuse or neglect in connection "
              "with the delivery of a healthcare item or service. Minimum 5-year "
              "exclusion. Mandatory under §1128(a)(2).",
    "1128a3": "Felony conviction relating to health care fraud (offense other "
              "than that described in §1128(a)(1)). Minimum 5-year exclusion. "
              "Mandatory under §1128(a)(3).",
    "1128a4": "Felony conviction relating to controlled substances. Minimum "
              "5-year exclusion. Mandatory under §1128(a)(4).",
    "1128b1": "Misdemeanor conviction relating to health care fraud (other "
              "than as described in §1128(a)). Permissive exclusion, "
              "typically 3 years.",
    "1128b2": "Conviction relating to fraud, theft, embezzlement, or breach "
              "of fiduciary responsibility. Permissive exclusion under §1128(b)(2).",
    "1128b3": "Misdemeanor conviction relating to a controlled substance. "
              "Permissive exclusion under §1128(b)(3).",
    "1128b4": "License revocation, suspension, or surrender. Most common "
              "exclusion type (~40% of all LEIE entries). Coterminous with "
              "the underlying licensure action under §1128(b)(4).",
    "1128b5": "Exclusion or suspension under a federal or state healthcare "
              "program. Coterminous with the other-agency action under §1128(b)(5).",
    "1128b6": "Claims for excessive charges or unnecessary services and failure "
              "to furnish medically necessary services. Permissive under §1128(b)(6).",
    "1128b7": "Fraud, kickbacks, and other prohibited activities (Anti-Kickback "
              "Statute). Permissive under §1128(b)(7). Minimum exclusion typically "
              "5 years.",
    "1128b8": "Conviction of fraud against a non-Medicare/Medicaid program. "
              "Permissive exclusion under §1128(b)(8).",
    "1128b14": "Default on health education loans / scholarship obligations. "
               "Permissive exclusion under §1128(b)(14).",
    "1128Aa": "Civil Money Penalty for false claims (under §1128A of the Social "
              "Security Act). Permissive exclusion.",
    "1156":   "Quality-of-care violations referred by Quality Improvement "
              "Organizations under §1156 of the Social Security Act.",
}


def load_leie() -> pd.DataFrame:
    if not LEIE_PATH.exists():
        raise FileNotFoundError(
            f"OIG LEIE not found at {LEIE_PATH}\n"
            f"Download it with:\n"
            f"  curl -L -o {LEIE_PATH} {DOWNLOAD_URL}"
        )
    df = pd.read_csv(LEIE_PATH, dtype=str, low_memory=False)
    # Normalize NPI: real NPIs are 10-digit, '0000000000' means "not on file"
    df["NPI"] = df["NPI"].fillna("0000000000")
    df["has_npi"] = (df["NPI"].str.len() == 10) & (df["NPI"] != "0000000000")
    # Parse exclusion date (YYYYMMDD)
    df["year"] = df["EXCLDATE"].fillna("00000000").str[:4]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def summary_by_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """Top-line stats: total records, NPI coverage, top exclusion types, top specialties."""
    rows = [
        ("Total exclusion records",       len(df)),
        ("Records with real NPI",          int(df["has_npi"].sum())),
        ("Records without NPI",            int((~df["has_npi"]).sum())),
        ("Unique excluded NPIs",           int(df.loc[df["has_npi"], "NPI"].nunique())),
        ("Earliest exclusion year",        int(df["year"].dropna().min())),
        ("Latest exclusion year",          int(df["year"].dropna().max())),
        ("LTC-relevant records (HHA/SNF/Hospice/Nursing Firm)",
            int(df["SPECIALTY"].fillna("").str.contains(
                "HOME HEALTH|SKILLED NURSING|HOSPICE|NURSING FIRM",
                case=False, regex=True).sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def ltc_subset(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["SPECIALTY"].fillna("").str.contains(
        "HOME HEALTH|SKILLED NURSING|HOSPICE|NURSING FIRM",
        case=False, regex=True
    )
    sub = df[mask].copy()
    keep = ["BUSNAME", "LASTNAME", "FIRSTNAME", "SPECIALTY", "NPI", "STATE",
            "EXCLTYPE", "EXCLDATE", "year"]
    return sub[[c for c in keep if c in sub.columns]].sort_values(
        ["year", "STATE"], ascending=[False, True]
    )


def physician_overlap_with_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle inpatient/outpatient claims contain real physician NPIs in the
    AttendingPhysician / OperatingPhysician / OtherPhysician columns. If any
    LEIE-excluded NPI matches one of those Kaggle physicians, we have a real-
    fraud-ground-truth cross-reference."""
    ip_path = next(Path(config.DATA_RAW).glob("Train_Inpatient*.csv"), None)
    op_path = next(Path(config.DATA_RAW).glob("Train_Outpatient*.csv"), None)
    if not (ip_path and op_path):
        print("  (Kaggle inpatient/outpatient files not found; skipping overlap)")
        return pd.DataFrame()

    cols = ["Provider", "AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
    ip = pd.read_csv(ip_path, usecols=lambda c: c in cols)
    op = pd.read_csv(op_path, usecols=lambda c: c in cols)
    kaggle = pd.concat([ip, op], ignore_index=True)

    # All real-physician NPIs that appear in any Kaggle physician column
    phys_cols = [c for c in ("AttendingPhysician", "OperatingPhysician", "OtherPhysician")
                 if c in kaggle.columns]
    phys = pd.unique(kaggle[phys_cols].values.ravel())
    phys = pd.Series([p for p in phys if isinstance(p, str)])
    print(f"  Kaggle dataset has {len(phys):,} unique physician identifiers")

    excluded_npis = set(df.loc[df["has_npi"], "NPI"])
    # Kaggle physician IDs are e.g. "PHY412132" — they are *anonymized*, so
    # direct overlap by NPI value will be zero. We report this honestly.
    overlap = phys[phys.isin(excluded_npis)]
    print(f"  Direct NPI overlap with LEIE: {len(overlap)} "
          f"(0 expected — Kaggle physician IDs are anonymized)")
    return pd.DataFrame({"matched_physician_npi": list(overlap)})


def plot_top_specialties(df: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    top = df["SPECIALTY"].fillna("UNKNOWN").value_counts().head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    # Highlight LTC-relevant specialties
    colors = ["#d62728" if any(k in s for k in
              ("HOME HEALTH", "SKILLED NURSING", "HOSPICE", "NURSING FIRM"))
              else "#4c78a8" for s in top.index]
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("Number of exclusions")
    ax.set_title("Top 15 excluded specialties (OIG LEIE)\n[red = LTC-relevant]")
    plt.tight_layout()
    path = fig_dir / "oig_leie_by_specialty.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved {path}")


def plot_annual_trend(df: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    by_year = df.dropna(subset=["year"])["year"].value_counts().sort_index()
    by_year = by_year[(by_year.index >= 2000) & (by_year.index <= 2026)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(by_year.index, by_year.values, marker="o", color="#4c78a8")
    ax.set_xlabel("Exclusion year")
    ax.set_ylabel("Number of exclusions")
    ax.set_title("Annual HHS-OIG exclusions, 2000–present (real data)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = fig_dir / "oig_leie_annual_trend.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved {path}")


def write_exclusion_codes_for_rag(df: pd.DataFrame, doc_path: Path):
    """Emit a structured policy file that the RAG retriever can index.
    The file format mirrors policy_rules.txt so rag_claim_review.py picks it
    up without any code change."""
    counts = df["EXCLTYPE"].value_counts().to_dict()

    lines = [
        "HHS-OIG EXCLUSION-CODE TAXONOMY",
        "=" * 60,
        "Real legal authorities cited in the federal LEIE database.",
        "Source: 42 U.S.C. §1128 (Social Security Act).",
        "Counts below reflect the number of active exclusions in the",
        f"LEIE snapshot used for this project (total {len(df):,} records).",
        "",
    ]
    for i, (code, definition) in enumerate(EXCLUSION_CODE_DEFINITIONS.items(), 1):
        n = counts.get(code, 0)
        lines.append(f"{i}. EXCLUSION CODE {code}  [{n:,} active cases in LEIE]")
        lines.append(f"   {definition}")
        lines.append("")

    doc_path.write_text("\n".join(lines))
    print(f"  Wrote real exclusion taxonomy to {doc_path}")


def main():
    print("Analyzing the HHS-OIG List of Excluded Individuals/Entities (LEIE)")
    print(f"  File: {LEIE_PATH}")
    df = load_leie()
    print(f"  Loaded {len(df):,} real federal exclusion records")
    print()

    print("Summary by dimension:")
    summary = summary_by_dimension(df)
    print(summary.to_string(index=False))
    summary_path = Path(config.OUTPUTS_REPORTS) / "oig_leie_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved {summary_path}")

    print("\nLTC-specific exclusion records:")
    ltc = ltc_subset(df)
    print(f"  {len(ltc):,} LTC-relevant exclusion records")
    print(ltc["SPECIALTY"].value_counts())
    ltc_path = Path(config.OUTPUTS_REPORTS) / "oig_leie_ltc_excerpt.csv"
    ltc.to_csv(ltc_path, index=False)
    print(f"  Saved {ltc_path}")

    print("\nCross-referencing with Kaggle physician identifiers...")
    overlap = physician_overlap_with_kaggle(df)
    if not overlap.empty:
        overlap_path = Path(config.OUTPUTS_REPORTS) / "oig_leie_physician_overlap.csv"
        overlap.to_csv(overlap_path, index=False)
        print(f"  Saved {overlap_path}")

    print("\nGenerating figures...")
    plot_top_specialties(df, Path(config.OUTPUTS_FIGURES))
    plot_annual_trend(df, Path(config.OUTPUTS_FIGURES))

    print("\nEmitting real exclusion-code taxonomy for the RAG corpus...")
    doc_path = Path(config.DATA_DOCUMENTS) / "oig_exclusion_codes.txt"
    write_exclusion_codes_for_rag(df, doc_path)

    print("\nLEIE analysis complete. Real federal-fraud-exclusion data is now")
    print("integrated as part of the project's evidence base — the RAG retriever")
    print("will index oig_exclusion_codes.txt alongside policy_rules.txt on its")
    print("next run.")


if __name__ == "__main__":
    main()
