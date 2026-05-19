"""
data_ingestion.py
=================
Loads and joins the Kaggle Healthcare Provider Fraud Detection Analysis dataset.

Dataset:
    https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis

If Kaggle files are missing, prints clear download instructions and raises
FileNotFoundError so the pipeline can fall back to synthetic demo mode.

Expected files in data/raw/ (exact names may vary — timestamp suffix):
    Train_Beneficiarydata-*.csv      beneficiary demographics
    Train_Inpatientdata-*.csv        inpatient claims
    Train_Outpatientdata-*.csv       outpatient claims
    Train-*.csv                      provider fraud labels (Provider, PotentialFraud)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── File discovery ─────────────────────────────────────────────────────────────

def find_kaggle_files(raw_dir: str) -> dict:
    """
    Search raw_dir for Kaggle Healthcare Provider Fraud Detection files.

    Returns a dict with keys: 'beneficiary', 'inpatient', 'outpatient', 'labels'.
    Missing roles are omitted from the dict.  File-name matching is
    case-insensitive and tolerates the timestamp suffix Kaggle adds.
    """
    if not os.path.isdir(raw_dir):
        return {}

    files = os.listdir(raw_dir)
    found = {}

    for fname in files:
        lower = fname.lower()
        if not lower.endswith(".csv"):
            continue
        fpath = os.path.join(raw_dir, fname)

        if "beneficiar" in lower:
            # Keep the one that looks like Training data (prefer 'train' prefix)
            if "beneficiary" not in found or "train" in lower:
                found["beneficiary"] = fpath

        elif "inpatient" in lower:
            if "inpatient" not in found or "train" in lower:
                found["inpatient"] = fpath

        elif "outpatient" in lower:
            if "outpatient" not in found or "train" in lower:
                found["outpatient"] = fpath

        elif "train" in lower and "data" not in lower:
            # Provider label file: contains 'train' but NOT 'data' (e.g. Train-1542865627584.csv)
            # Also skip beneficiary/inpatient/outpatient already matched above
            if "labels" not in found:
                # Verify it looks like a label file by peeking at the header
                try:
                    peek = pd.read_csv(fpath, nrows=2)
                    cols = [c.strip().lower() for c in peek.columns]
                    if "provider" in cols and "potentialfraud" in cols:
                        found["labels"] = fpath
                except Exception:
                    pass

    return found


def check_data_available(raw_dir: str) -> bool:
    """Return True if all four key Kaggle files are found, False otherwise."""
    found = find_kaggle_files(raw_dir)
    required = {"beneficiary", "inpatient", "outpatient", "labels"}
    return required.issubset(found.keys())


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_beneficiary(filepath: str) -> pd.DataFrame:
    """Load beneficiary CSV and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"  [beneficiary] {df.shape}  {filepath}")
    return df


def load_inpatient(filepath: str) -> pd.DataFrame:
    """Load inpatient claims CSV and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"  [inpatient]   {df.shape}  {filepath}")
    return df


def load_outpatient(filepath: str) -> pd.DataFrame:
    """Load outpatient claims CSV and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"  [outpatient]  {df.shape}  {filepath}")
    return df


def load_provider_labels(filepath: str) -> pd.DataFrame:
    """
    Load provider label CSV (Provider, PotentialFraud).
    Converts PotentialFraud 'Yes'/'No' to 1/0.
    Returns DataFrame with columns: Provider, PotentialFraud (int).
    """
    df = pd.read_csv(filepath)
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    fraud_col = None
    for c in df.columns:
        if "fraud" in c.lower():
            fraud_col = c
            break

    if fraud_col is None:
        raise ValueError(f"Could not find a fraud column in {filepath}. Columns: {list(df.columns)}")

    if df[fraud_col].dtype == object:
        df[fraud_col] = df[fraud_col].str.strip().str.lower().map({"yes": 1, "no": 0})

    df = df.rename(columns={fraud_col: "PotentialFraud"})
    df["PotentialFraud"] = df["PotentialFraud"].fillna(0).astype(int)

    print(f"  [labels]      {df.shape}  {filepath}")
    print(f"  Fraud rate in labels: {df['PotentialFraud'].mean():.2%}")
    return df


# ── Instructions ───────────────────────────────────────────────────────────────

def print_download_instructions():
    print("""
=============================================================================
  Kaggle Dataset Not Found — Manual Download Required
=============================================================================

Dataset:
  Healthcare Provider Fraud Detection Analysis
  https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis

Steps:
  1. Go to: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
  2. Click the blue "Download" button (requires a free Kaggle account).
  3. Unzip the downloaded archive (e.g. archive.zip or healthcare-provider-fraud-detection-analysis.zip).
  4. Place ALL extracted CSV files inside:
       {raw_dir}/

  Expected files (Kaggle adds a timestamp suffix):
       Train_Beneficiarydata-<timestamp>.csv
       Train_Inpatientdata-<timestamp>.csv
       Train_Outpatientdata-<timestamp>.csv
       Train-<timestamp>.csv                  <-- provider fraud labels
       Test_Beneficiarydata-<timestamp>.csv   (optional)
       Test_Inpatientdata-<timestamp>.csv     (optional)
       Test_Outpatientdata-<timestamp>.csv    (optional)
       Test-<timestamp>.csv                   (optional)

  After placing the files, re-run:
       python src/data_ingestion.py
       python src/provider_feature_engineering.py

Alternatively, run the synthetic demo pipeline (no download needed):
       python src/synthetic_data_generation.py   # generates synthetic_claims.csv
       python src/preprocessing.py
       python src/feature_engineering.py
       python src/modeling.py
       python src/explainability.py
       python src/rag_claim_review.py
       python src/monitoring.py

=============================================================================
""".format(raw_dir=config.DATA_RAW))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Checking for Kaggle Healthcare Provider Fraud Detection dataset...")
    found = find_kaggle_files(config.DATA_RAW)
    print(f"  Files found: {list(found.keys())}")

    required = {"beneficiary", "inpatient", "outpatient", "labels"}
    missing = required - set(found.keys())

    if missing:
        print(f"\n  Missing files for roles: {missing}")
        print_download_instructions()
        return False

    print("\nLoading all Kaggle files...")
    beneficiary = load_beneficiary(found["beneficiary"])
    inpatient   = load_inpatient(found["inpatient"])
    outpatient  = load_outpatient(found["outpatient"])
    labels      = load_provider_labels(found["labels"])

    print("\nAll files loaded successfully.")
    print(f"  Beneficiary columns : {list(beneficiary.columns[:8])}...")
    print(f"  Inpatient columns   : {list(inpatient.columns[:8])}...")
    print(f"  Outpatient columns  : {list(outpatient.columns[:8])}...")
    print(f"  Label columns       : {list(labels.columns)}")
    return True


if __name__ == "__main__":
    main()
