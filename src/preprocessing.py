"""
Preprocessing module for insurance claims FWA pipeline.
Handles loading, cleaning, encoding, and train/test splitting.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_raw_data():
    path = os.path.join(config.DATA_RAW, "synthetic_claims.csv")
    df = pd.read_csv(path, parse_dates=["claim_date"])
    print(f"  Loaded {len(df)} rows from {path}")
    return df


def handle_missing_values(df):
    """Fill or drop missing values."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categoricals(df):
    """Encode categorical columns using LabelEncoder."""
    cat_cols = ["service_type", "diagnosis_group", "state"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    return df, le_dict


def split_train_test(df, target_col="fraud_label", test_size=0.2, seed=42):
    """Drop non-feature columns, split into train/test."""
    # Preserve claim_id for reference but drop from features
    id_cols = ["claim_id", "policyholder_id", "provider_id", "claim_date",
               "service_type", "diagnosis_group", "state"]
    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols


def save_processed(df, X_train, X_test, y_train, y_test):
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)

    df.to_csv(os.path.join(config.DATA_PROCESSED, "claims_encoded.csv"), index=False)
    X_train.to_csv(os.path.join(config.DATA_PROCESSED, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(config.DATA_PROCESSED, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(config.DATA_PROCESSED, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(config.DATA_PROCESSED, "y_test.csv"), index=False)
    print(f"  Saved processed data to {config.DATA_PROCESSED}/")


def main():
    print("Running preprocessing pipeline...")
    df = load_raw_data()
    df = handle_missing_values(df)
    df, le_dict = encode_categoricals(df)
    X_train, X_test, y_train, y_test, feature_cols = split_train_test(df)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Features: {feature_cols}")
    print(f"  Train fraud rate: {y_train.mean():.2%}")

    save_processed(df, X_train, X_test, y_train, y_test)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
