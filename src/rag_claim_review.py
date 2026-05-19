"""
RAG-style claim review module (no paid API required).
Uses TF-IDF + cosine similarity for policy retrieval, then fills a structured template.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


ID_LIKE_COLS = [
    "claim_id", "policyholder_id", "provider_id", "claim_date",
    "service_type", "diagnosis_group", "state",
]


# ──────────────────────────────────────────────────────────────────────────────
# Document loading & indexing
# ──────────────────────────────────────────────────────────────────────────────

def load_policy_rules():
    path = os.path.join(config.DATA_DOCUMENTS, "policy_rules.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        text = f.read()
    # Split into chunks by numbered section
    chunks = re.split(r"\n(?=\d+\.)", text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]
    return chunks


def load_claim_document(claim_id):
    path = os.path.join(config.DATA_DOCUMENTS, f"claim_{claim_id}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def build_policy_index(policy_chunks):
    """Build TF-IDF index over policy chunks."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(policy_chunks)
    return vectorizer, tfidf_matrix


def retrieve_policy_evidence(query, vectorizer, tfidf_matrix, policy_chunks, top_k=3):
    """Retrieve top-k relevant policy chunks for a query."""
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    results = [(policy_chunks[i], float(scores[i])) for i in top_idx]
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Claim review generation
# ──────────────────────────────────────────────────────────────────────────────

def _get_risk_level(score):
    if score >= config.HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif score >= 0.3:
        return "MEDIUM"
    return "LOW"


def _build_query(row):
    """Build a natural language query for policy retrieval."""
    parts = []
    if row.get("duplicate_claim_flag", 0):
        parts.append("duplicate claim billing")
    if row.get("late_submission_flag", 0):
        parts.append("late claim submission over 90 days")
    if row.get("suspicious_keyword_count", 0) >= 3:
        parts.append("suspicious keywords documentation fraud")
    if row.get("documentation_score", 1.0) < 0.4:
        parts.append("incomplete documentation medical records")
    if row.get("claim_to_provider_avg_ratio", 1.0) > 2.0:
        parts.append("claim amount exceeds provider average upcoding")
    if row.get("high_cost_outlier_flag", 0):
        parts.append("high cost claim outlier")
    if not parts:
        parts.append("claim review medical necessity billing guidelines")
    return " ".join(parts)


def _build_risk_indicators(row):
    indicators = []
    ratio = row.get("claim_to_provider_avg_ratio", 1.0)
    if ratio > 1.5:
        indicators.append(f"  - Claim amount is {ratio:.1f}x provider average")
    if row.get("documentation_score", 1.0) < 0.4:
        indicators.append(f"  - Low documentation score ({row['documentation_score']:.2f})")
    if row.get("duplicate_claim_flag", 0):
        indicators.append("  - Duplicate claim flag: POSITIVE")
    if row.get("late_submission_flag", 0):
        indicators.append("  - Late submission flag: POSITIVE")
    if row.get("suspicious_keyword_count", 0) >= 3:
        indicators.append(f"  - Suspicious keywords detected: {int(row['suspicious_keyword_count'])}")
    if row.get("high_cost_outlier_flag", 0):
        indicators.append("  - High-cost outlier (top 10% of all claims)")
    if row.get("prior_claim_count", 0) > 10:
        indicators.append(f"  - Prior claims: {int(row['prior_claim_count'])} (elevated)")
    if not indicators:
        indicators.append("  - No single dominant risk factor; composite score elevated")
    return "\n".join(indicators)


def _suggest_action(risk_level, indicators_text):
    if risk_level == "HIGH":
        return (
            "SUSPEND PAYMENT pending analyst review. Assign to senior investigator. "
            "Request complete medical records and provider attestation within 10 business days."
        )
    elif risk_level == "MEDIUM":
        return (
            "ENHANCED REVIEW required within 5 business days. "
            "Request supplemental documentation before processing payment."
        )
    return (
        "STANDARD PROCESSING. Include in next routine audit sample (10% sampling rate)."
    )


def _audit_notes(row, risk_level):
    notes = []
    notes.append(f"Review generated: automated FWA risk scoring system v1.0")
    notes.append(f"Data source: synthetic claims model (for portfolio demonstration)")
    if risk_level == "HIGH":
        notes.append("Escalation: Human analyst review REQUIRED before any payment action.")
    notes.append(
        "Policy retrieval: TF-IDF cosine similarity over policy_rules.txt "
        "(no external API; deterministic output)."
    )
    notes.append(
        "Model: Best supervised classifier (Random Forest / GradientBoosting / XGBoost). "
        "Anomaly detection: Isolation Forest."
    )
    notes.append(
        "DISCLAIMER: This review is AI-assisted. Final adjudication must involve a licensed "
        "insurance professional. Scores are probabilistic, not definitive."
    )
    return "\n  ".join(notes)


def generate_review(claim_id, row, risk_score, vectorizer, tfidf_matrix, policy_chunks):
    """Generate a structured RAG claim review for one claim."""
    risk_level = _get_risk_level(risk_score)
    query = _build_query(row)
    evidence = retrieve_policy_evidence(query, vectorizer, tfidf_matrix, policy_chunks, top_k=3)
    risk_indicators = _build_risk_indicators(row)
    action = _suggest_action(risk_level, risk_indicators)
    audit = _audit_notes(row, risk_level)

    # Claim doc excerpt
    claim_doc = load_claim_document(claim_id)
    doc_excerpt = ""
    if claim_doc:
        lines = [l for l in claim_doc.split("\n") if l.strip()]
        doc_excerpt = "\n    ".join(lines[:8])
    else:
        doc_excerpt = "(No claim document on file for this claim ID)"

    evidence_text = ""
    for i, (chunk, score) in enumerate(evidence, 1):
        short = chunk[:300].replace("\n", " ")
        evidence_text += f"\n  [{i}] (similarity={score:.3f}) {short}..."

    review = f"""
══════════════════════════════════════════════════════════════════
INSURANCE FWA CLAIM REVIEW REPORT
══════════════════════════════════════════════════════════════════
Claim ID        : {claim_id}
Risk Level      : {risk_level}
Model Risk Score: {risk_score:.4f}
Service Type    : {row.get('service_type', 'N/A')}
Diagnosis Group : {row.get('diagnosis_group', 'N/A')}
Claim Amount    : ${row.get('claim_amount', 0):,.2f}
Approved Amount : ${row.get('approved_amount', 0):,.2f}
Provider ID     : {row.get('provider_id', 'N/A')}
State           : {row.get('state', 'N/A')}

──────────────────────────────────────────────────────────────────
KEY RISK INDICATORS
──────────────────────────────────────────────────────────────────
{risk_indicators}

──────────────────────────────────────────────────────────────────
RETRIEVED POLICY EVIDENCE (TF-IDF Retrieval)
──────────────────────────────────────────────────────────────────
Query used: "{query}"
{evidence_text}

──────────────────────────────────────────────────────────────────
CLAIM DOCUMENT EXCERPT
──────────────────────────────────────────────────────────────────
    {doc_excerpt}

──────────────────────────────────────────────────────────────────
SUGGESTED ANALYST ACTION
──────────────────────────────────────────────────────────────────
{action}

──────────────────────────────────────────────────────────────────
AUDIT NOTES
──────────────────────────────────────────────────────────────────
  {audit}
══════════════════════════════════════════════════════════════════
"""
    return review


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Running RAG claim review pipeline...")

    # Load data
    for p in [
        os.path.join(config.DATA_PROCESSED, "claims_features.csv"),
        os.path.join(config.DATA_PROCESSED, "claims_encoded.csv"),
        os.path.join(config.DATA_RAW,       "synthetic_claims.csv"),
    ]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"  Loaded data from {p}")
            break
    else:
        raise FileNotFoundError("No claims data found.")

    # Load model
    model_path = os.path.join(config.OUTPUTS_MODELS, "best_fwa_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}.")
    model = joblib.load(model_path)

    # Compute risk scores
    id_and_target = ID_LIKE_COLS + ["fraud_label"]
    feature_cols = [
        c for c in df.columns
        if c not in id_and_target and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols]
    risk_scores = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["model_risk_score"] = risk_scores

    # Build policy index
    print("  Building TF-IDF policy index...")
    policy_chunks = load_policy_rules()
    if not policy_chunks:
        print("  WARNING: No policy rules found. Using placeholder.")
        policy_chunks = ["General insurance policy guidelines apply to all claims."]
    vectorizer, tfidf_matrix = build_policy_index(policy_chunks)
    print(f"  Indexed {len(policy_chunks)} policy chunks.")

    # Select top 10 high-risk claims for sample reviews
    os.makedirs(config.OUTPUTS_REVIEWS, exist_ok=True)
    high_risk = df.nlargest(10, "model_risk_score")

    print(f"  Generating reviews for top {len(high_risk)} high-risk claims...")
    for _, row in high_risk.iterrows():
        cid = row.get("claim_id", "UNKNOWN")
        score = row["model_risk_score"]
        review_text = generate_review(
            cid, row, score, vectorizer, tfidf_matrix, policy_chunks
        )
        out_path = os.path.join(config.OUTPUTS_REVIEWS, f"review_{cid}.txt")
        with open(out_path, "w") as f:
            f.write(review_text)

    print(f"  Saved {len(high_risk)} sample reviews to {config.OUTPUTS_REVIEWS}/")
    print("RAG claim review complete.")


if __name__ == "__main__":
    main()
