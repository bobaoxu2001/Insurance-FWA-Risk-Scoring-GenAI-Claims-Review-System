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
    "src/graph_features.py",        # bipartite-graph derived features
    "src/modeling.py",
    "src/hyperparameter_tuning.py", # RandomizedSearchCV
    "src/explainability.py",
    "src/rag_claim_review.py",
    "src/llm_review.py",            # tier-1/2 semantic + LLM RAG
    "src/monitoring.py",
    "src/psi_drift.py",             # PSI drift detection
    "src/fairness_audit.py",        # demographic disparate-impact check
    "src/feedback_loop.py",         # analyst-disposition retraining loop
    "src/oig_leie_analysis.py",     # real federal exclusion data
    "src/cms_ltc_pipeline.py",      # real CMS Nursing Home pipeline
    "src/medicare_partb_pipeline.py",  # real Medicare Part B ⋈ LEIE NPI labels
    "scripts/download_real_data.sh",
    "Dockerfile",
    ".dockerignore",
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


def test_metrics_schema_completeness() -> None:
    """Every model entry should carry the full new metric schema:
    test-set ranking (roc_auc, pr_auc), classification (precision, recall, f1),
    calibration (brier, log_loss), and 5-fold CV (cv_auc_mean, cv_auc_std)."""
    p = _metrics_path()
    if not p.is_file():
        pytest.skip("model_metrics.json absent")
    with p.open() as f:
        data = json.load(f)

    required = {
        "roc_auc", "pr_auc", "precision", "recall", "f1",
        "brier", "log_loss", "cv_auc_mean", "cv_auc_std",
    }
    model_entries = {k: v for k, v in data.items()
                     if not k.startswith("_") and isinstance(v, dict)}
    assert model_entries, "Expected at least one model entry"
    for name, m in model_entries.items():
        missing = required - set(m.keys())
        assert not missing, f"{name} missing metric fields: {sorted(missing)}"
        # CV std should be small relative to the mean (sanity: not exploding)
        assert m["cv_auc_std"] < 0.10, (
            f"{name} CV-AUC std ({m['cv_auc_std']}) suggests unstable model"
        )


def test_evaluation_block_present() -> None:
    """Metrics JSON should carry the _evaluation provenance block."""
    p = _metrics_path()
    if not p.is_file():
        pytest.skip("model_metrics.json absent")
    with p.open() as f:
        data = json.load(f)
    assert "_evaluation" in data, "Missing _evaluation block"
    ev = data["_evaluation"]
    for key in ("test_split", "cv_scheme", "selection_metric", "best_model"):
        assert key in ev, f"_evaluation missing field: {key}"


def test_classification_report_exists() -> None:
    """classification_report.txt should be saved alongside metrics."""
    p = REPO_ROOT / "outputs" / "reports" / "classification_report.txt"
    if not p.is_file():
        pytest.skip("classification_report.txt not generated yet")
    body = p.read_text()
    # sklearn classification_report always emits these tokens
    for token in ("precision", "recall", "f1-score", "Fraud"):
        assert token in body, f"classification_report.txt missing token: {token}"


def test_calibration_figure_exists() -> None:
    """Reliability diagram should exist after a full modeling run."""
    p = REPO_ROOT / "outputs" / "figures" / "calibration_curve.png"
    if not p.is_file():
        pytest.skip("calibration_curve.png not generated yet")
    # PNG magic bytes
    assert p.read_bytes()[:4] == b"\x89PNG", "calibration_curve.png is not a valid PNG"


# ── Temporal-split metrics (separate artifact alongside random-split) ──────────


def test_temporal_metrics_if_present() -> None:
    """If a temporal-split run has been done, validate its metrics JSON has the
    same schema and that the test_split label correctly indicates chronological."""
    p = REPO_ROOT / "outputs" / "reports" / "model_metrics_temporal.json"
    if not p.is_file():
        pytest.skip("model_metrics_temporal.json not generated yet")
    with p.open() as f:
        data = json.load(f)
    ev = data.get("_evaluation", {})
    assert "chronological" in ev.get("test_split", "").lower(), \
        "temporal metrics _evaluation.test_split should mention 'chronological'"
    # AUC should drop relative to random split (the realistic finding)
    aucs = [v["roc_auc"] for k, v in data.items()
            if not k.startswith("_") and isinstance(v, dict)]
    assert aucs and max(aucs) < 0.95, \
        f"Temporal AUC suspiciously high (max={max(aucs):.3f}); expected ~0.85-0.92"


# ── Fairness audit outputs ────────────────────────────────────────────────────


def test_fairness_audit_report_if_present() -> None:
    p = REPO_ROOT / "outputs" / "reports" / "fairness_audit_report.csv"
    if not p.is_file():
        pytest.skip("fairness_audit_report.csv not generated yet")
    import csv
    with p.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "fairness_audit_report.csv has no rows"
    required = {"panel_majority_race", "n_providers", "fraud_rate",
                "mean_risk_score", "model_flag_rate",
                "disparate_impact_ratio", "passes_4_5ths_rule"}
    assert required <= set(rows[0].keys()), \
        f"fairness_audit_report.csv missing columns: {required - set(rows[0].keys())}"


# ── LLM-augmented reviews ─────────────────────────────────────────────────────


def test_llm_reviews_if_present() -> None:
    """If tier-2 LLM reviews have been generated, they should mention the LLM
    and semantic-retrieval provenance — protects against accidental fallback."""
    review_dir = REPO_ROOT / "outputs" / "sample_reviews"
    if not review_dir.is_dir():
        pytest.skip("outputs/sample_reviews/ missing")
    llm_reviews = list(review_dir.glob("review_*_llm.txt"))
    if not llm_reviews:
        pytest.skip("No *_llm.txt reviews generated yet")
    sample = llm_reviews[0].read_text()
    assert "flan-t5" in sample.lower() or "llm" in sample.lower(), \
        "LLM review should declare its generation backend"
    assert "semantic" in sample.lower() or "sentence-transformers" in sample.lower(), \
        "LLM review should declare its semantic retrieval backend"


# ── Graph features (added to provider table) ──────────────────────────────────


def test_graph_features_in_table_if_present() -> None:
    """If graph_features.py has been run, the provider table should carry
    all five graph-derived columns."""
    p = REPO_ROOT / "data" / "processed" / "provider_modeling_table.csv"
    if not p.is_file():
        pytest.skip("provider table missing")
    # Read just the header to avoid loading the full CSV
    with p.open() as f:
        header = next(f).strip().split(",")
    expected = {
        "beneficiary_sharing_rate", "avg_co_provider_count",
        "physician_sharing_rate", "provider_clustering_coefficient",
        "provider_pagerank",
    }
    present = expected & set(header)
    if not present:
        pytest.skip("graph features not added yet; run src/graph_features.py")
    assert expected <= set(header), (
        f"Provider table missing graph features: {expected - set(header)}"
    )


# ── PSI drift report ──────────────────────────────────────────────────────────


def test_psi_drift_report_if_present() -> None:
    p = REPO_ROOT / "outputs" / "reports" / "psi_drift_report.csv"
    if not p.is_file():
        pytest.skip("psi_drift_report.csv not generated yet")
    import csv
    with p.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "psi_drift_report.csv has no rows"
    required = {"feature", "psi", "verdict", "train_mean", "test_mean"}
    assert required <= set(rows[0].keys()), \
        f"PSI report missing columns: {required - set(rows[0].keys())}"
    # At least one feature should have a non-NaN PSI
    psis = [float(r["psi"]) for r in rows if r["psi"] not in ("", "nan")]
    assert psis, "No valid PSI values in report"


# ── Hyperparameter tuning ─────────────────────────────────────────────────────


def test_hp_tuning_result_if_present() -> None:
    p = REPO_ROOT / "outputs" / "reports" / "hp_tuning_results.json"
    if not p.is_file():
        pytest.skip("hp_tuning_results.json not generated yet")
    with p.open() as f:
        data = json.load(f)
    for key in ("_model_class", "_scoring", "_cv", "best_score_pr_auc", "best_params"):
        assert key in data, f"hp_tuning_results.json missing key: {key}"
    assert data["best_score_pr_auc"] > 0.5, \
        f"Tuned PR-AUC suspiciously low: {data['best_score_pr_auc']}"


# ── Feedback loop ────────────────────────────────────────────────────────────


def test_feedback_loop_metrics_if_present() -> None:
    p = REPO_ROOT / "outputs" / "reports" / "feedback_loop_metrics.json"
    if not p.is_file():
        pytest.skip("feedback_loop_metrics.json not generated yet")
    with p.open() as f:
        data = json.load(f)
    for key in ("n_total", "n_flagged", "precision_of_flag",
                "false_confirm_rate", "miss_rate_on_audit"):
        assert key in data, f"feedback metrics missing key: {key}"
    assert 0 <= data["precision_of_flag"] <= 1, \
        f"precision_of_flag out of [0,1]: {data['precision_of_flag']}"


# ── Real-data artifacts (OIG LEIE + CMS Nursing Home) ─────────────────────────


def test_oig_leie_summary_if_present() -> None:
    p = REPO_ROOT / "outputs" / "reports" / "oig_leie_summary.csv"
    if not p.is_file():
        pytest.skip("OIG LEIE summary not generated yet")
    import csv
    with p.open() as f:
        rows = list(csv.DictReader(f))
    metrics = {r["metric"]: r["value"] for r in rows}
    assert any("LTC-relevant" in m for m in metrics), \
        "OIG LEIE summary should contain an LTC-relevant count"


def test_oig_exclusion_codes_in_rag_corpus() -> None:
    """The OIG LEIE analysis should have written real federal exclusion-code
    definitions to data/documents/, where the RAG retriever picks them up."""
    p = REPO_ROOT / "data" / "documents" / "oig_exclusion_codes.txt"
    if not p.is_file():
        pytest.skip("oig_exclusion_codes.txt not generated yet")
    body = p.read_text()
    # Should mention multiple real exclusion codes
    assert "1128a1" in body and "1128b4" in body, \
        "Exclusion-code taxonomy should cite §1128 authorities"
    assert "Social Security Act" in body, \
        "Should cite legal authority"


def test_cms_ltc_metrics_if_present() -> None:
    p = REPO_ROOT / "outputs" / "reports" / "cms_ltc_metrics.json"
    if not p.is_file():
        pytest.skip("CMS LTC metrics not generated yet")
    with p.open() as f:
        data = json.load(f)
    assert data.get("_data_source") == "real_cms_nursing_home_compare", \
        "CMS LTC metrics must declare real-CMS data source"
    assert data["_n_providers"] >= 10000, \
        f"Expected ~14,699 CMS providers, got {data.get('_n_providers')}"
    # PR-AUC on real LTC data should be in a plausible band
    model_entries = {k: v for k, v in data.items()
                     if not k.startswith("_") and isinstance(v, dict)}
    pr_aucs = [v["pr_auc"] for v in model_entries.values() if "pr_auc" in v]
    assert pr_aucs, "no PR-AUC values in CMS LTC metrics"
    assert 0.5 < max(pr_aucs) < 0.95, \
        f"CMS LTC PR-AUC outside realistic band 0.5-0.95: {max(pr_aucs)}"


def test_cms_leie_overlap_if_present() -> None:
    p = REPO_ROOT / "outputs" / "reports" / "cms_ltc_leie_overlap.csv"
    if not p.is_file():
        pytest.skip("CMS-LEIE overlap not generated yet")
    import csv
    with p.open() as f:
        rows = list(csv.DictReader(f))
    # File should at least have the right schema even if 0 matches
    if rows:
        required = {"CMS Certification Number (CCN)", "Provider Name", "State",
                    "Legal Business Name", "matched_leie"}
        assert required <= set(rows[0].keys()), \
            f"CMS-LEIE overlap missing columns: {required - set(rows[0].keys())}"


# ── Medicare Part B ⋈ LEIE pipeline ───────────────────────────────────────────


def test_partb_metrics_if_present() -> None:
    p = REPO_ROOT / "outputs" / "reports" / "medicare_partb_metrics.json"
    if not p.is_file():
        pytest.skip("medicare_partb_metrics.json not generated yet")
    with p.open() as f:
        data = json.load(f)
    assert data.get("_data_source") == "real_medicare_partb_2023_xref_oig_leie", \
        "Part B metrics must declare real-NPI-join provenance"
    assert data.get("_n_providers", 0) >= 10_000, \
        f"Part B population suspiciously small: {data.get('_n_providers')}"
    assert "LEIE" in data.get("_label", ""), \
        "Part B label must reference LEIE"
    # PR-AUC under extreme imbalance should still beat the prevalence baseline by a lot
    model_entries = {k: v for k, v in data.items()
                     if not k.startswith("_") and isinstance(v, dict)}
    pr_aucs = [v.get("pr_auc", 0) for v in model_entries.values()]
    prevalence = data.get("_positive_rate_pct", 0) / 100
    assert max(pr_aucs) > prevalence * 3, \
        (f"Best PR-AUC ({max(pr_aucs)}) should exceed 3× prevalence "
         f"({prevalence}) — got only {max(pr_aucs) / max(prevalence, 1e-6):.1f}x")


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
