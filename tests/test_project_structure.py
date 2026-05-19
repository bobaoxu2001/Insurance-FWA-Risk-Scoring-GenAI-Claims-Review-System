"""Structural and sanity tests for the Healthcare Provider FWA project.

These tests are deliberately lightweight — they verify that core source files
and pipeline outputs exist and are well-formed, without re-running the heavy
training pipeline. Metric-content tests skip gracefully if outputs are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


# ── Required source files ──────────────────────────────────────────────────────

REQUIRED_SRC_FILES = [
    "src/data_ingestion.py",
    "src/provider_feature_engineering.py",
    "src/modeling.py",
    "src/explainability.py",
    "src/rag_claim_review.py",
    "src/monitoring.py",
]

REQUIRED_TOP_LEVEL = [
    "README.md",
    "app.py",
    "requirements.txt",
]


@pytest.mark.parametrize("rel_path", REQUIRED_SRC_FILES + REQUIRED_TOP_LEVEL)
def test_required_file_exists(rel_path: str) -> None:
    assert (REPO_ROOT / rel_path).is_file(), f"Missing required file: {rel_path}"


# ── Model metrics JSON ─────────────────────────────────────────────────────────


def _metrics_path() -> Path:
    return REPO_ROOT / "outputs" / "reports" / "model_metrics.json"


def test_model_metrics_json_exists_and_parses() -> None:
    p = _metrics_path()
    assert p.is_file(), "outputs/reports/model_metrics.json is missing"
    with p.open() as f:
        data = json.load(f)
    assert isinstance(data, dict) and data, "model_metrics.json should be a non-empty dict"


def test_real_kaggle_metrics_sanity() -> None:
    """If the metrics were trained on the real Kaggle provider data, the best
    model should have AUC > 0.85. Skips if file is missing or synthetic-mode."""
    p = _metrics_path()
    if not p.is_file():
        pytest.skip("model_metrics.json absent")
    with p.open() as f:
        data = json.load(f)
    if data.get("_data_source") != "real_kaggle_provider":
        pytest.skip("metrics not from real Kaggle pipeline")
    model_aucs = [
        v["roc_auc"] for k, v in data.items()
        if not k.startswith("_") and isinstance(v, dict) and "roc_auc" in v
    ]
    assert model_aucs, "No models with roc_auc found in metrics JSON"
    assert max(model_aucs) > 0.85, (
        f"Real-data run should have at least one model with AUC > 0.85, "
        f"got max={max(model_aucs):.3f}"
    )


# ── Top risk factors ───────────────────────────────────────────────────────────


def test_top_risk_factors_csv() -> None:
    p = REPO_ROOT / "outputs" / "reports" / "top_risk_factors.csv"
    if not p.is_file():
        pytest.skip("top_risk_factors.csv not generated yet")
    # File must have a header and at least one data row
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 2, "top_risk_factors.csv should have header + ≥1 row"


# ── Sample reviews ─────────────────────────────────────────────────────────────


def test_sample_reviews_present() -> None:
    review_dir = REPO_ROOT / "outputs" / "sample_reviews"
    if not review_dir.is_dir():
        pytest.skip("outputs/sample_reviews/ does not exist yet")
    files = list(review_dir.glob("review_*.txt"))
    assert len(files) >= 10, (
        f"Expected at least 10 review files, found {len(files)}"
    )
