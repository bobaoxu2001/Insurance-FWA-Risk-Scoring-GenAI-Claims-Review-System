"""Logic-level unit tests for the core algorithmic pieces.
Structural existence is already covered in test_project_structure.py — these
tests verify correctness of the math."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


# ── PSI calculation ──────────────────────────────────────────────────────────


def test_psi_zero_for_identical_distributions():
    from psi_drift import _compute_psi
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    psi = _compute_psi(a, b, n_bins=10)
    assert 0 <= psi < 0.05, f"PSI for identical distributions should be ~0, got {psi}"


def test_psi_large_for_shifted_distributions():
    from psi_drift import _compute_psi
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1,   5000)
    b = rng.normal(2, 1,   5000)   # 2-sigma mean shift
    psi = _compute_psi(a, b, n_bins=10)
    assert psi > 0.25, f"PSI for 2σ-shifted distribution should be >0.25, got {psi}"


def test_psi_verdict_thresholds():
    from psi_drift import _verdict
    assert _verdict(0.05)  == "stable"
    assert _verdict(0.15)  == "moderate_shift"
    assert _verdict(0.30)  == "significant_shift"
    assert _verdict(0.099) == "stable"
    assert _verdict(0.10)  == "moderate_shift"


# ── Graph features sanity ────────────────────────────────────────────────────


def test_beneficiary_sharing_features_zero_overlap():
    """Two providers with disjoint beneficiary sets → sharing rate = 0 for both."""
    from graph_features import beneficiary_sharing_features
    claims = pd.DataFrame({
        "Provider": ["P1", "P1", "P2", "P2"],
        "BeneID":   ["B1", "B2", "B3", "B4"],
    })
    out = beneficiary_sharing_features(claims)
    assert out.loc["P1", "beneficiary_sharing_rate"] == 0.0
    assert out.loc["P2", "beneficiary_sharing_rate"] == 0.0


def test_beneficiary_sharing_features_full_overlap():
    """All providers share all beneficiaries → sharing rate = 1.0."""
    from graph_features import beneficiary_sharing_features
    claims = pd.DataFrame({
        "Provider": ["P1", "P1", "P2", "P2"],
        "BeneID":   ["B1", "B2", "B1", "B2"],
    })
    out = beneficiary_sharing_features(claims)
    assert out.loc["P1", "beneficiary_sharing_rate"] == 1.0
    assert out.loc["P2", "beneficiary_sharing_rate"] == 1.0
    assert out.loc["P1", "avg_co_provider_count"] == 1.0  # exactly 1 other provider


# ── Risk-level thresholding ──────────────────────────────────────────────────


def test_risk_level_thresholds():
    """The risk-level mapping must be monotone and use the configured threshold."""
    from rag_claim_review import _risk_level
    import config
    high_thr = config.HIGH_RISK_THRESHOLD  # default 0.6 per config.py

    # Scores above the high threshold → HIGH
    assert _risk_level(high_thr) == "HIGH"
    assert _risk_level(high_thr + 0.01) == "HIGH"
    assert _risk_level(0.95) == "HIGH"
    # Scores in [0.3, high_thr) → MEDIUM
    assert _risk_level(0.30) == "MEDIUM"
    assert _risk_level(high_thr - 0.01) == "MEDIUM"
    # Below 0.3 → LOW
    assert _risk_level(0.0) == "LOW"
    assert _risk_level(0.29) == "LOW"


# ── Provider feature engineering: ID-like columns dropped ────────────────────


def test_modeling_excludes_split_only_columns():
    """median_claim_date is for splitting, not modeling — must be excluded."""
    from modeling import _get_model_features
    df = pd.DataFrame({
        "Provider":          ["P1"],
        "PotentialFraud":    [1],
        "median_claim_date": ["2009-01-01"],
        "total_claims":      [42],
        "avg_reimbursed":    [1000.0],
    })
    feats = _get_model_features(df, target="PotentialFraud")
    assert "Provider" not in feats
    assert "median_claim_date" not in feats
    assert "PotentialFraud" not in feats
    assert "total_claims"  in feats
    assert "avg_reimbursed" in feats