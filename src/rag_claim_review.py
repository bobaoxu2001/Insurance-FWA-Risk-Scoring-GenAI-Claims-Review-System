"""
RAG-style claim review module (no paid API required).

TF-IDF + cosine similarity over the policy_rules.txt corpus, plus a structured
template populated from model output + observed features. Each review now
covers all of the fields a Long Term Care FWA analyst would expect to see:

  - Claim ID, Risk Level, Model Risk Score
  - Key Risk Indicators (3-5 quantified bullets)
  - Retrieved Policy Evidence (top 2-3 chunks)
  - Documentation Gaps
  - Suggested Analyst Action
  - Human Review Notes (what the analyst should verify)
  - Limitations (synthetic data + model uncertainty)
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
EXCLUDE_FROM_MODEL = ["rule_based_risk_score"]


# ──────────────────────────────────────────────────────────────────────────────
# Document loading & indexing
# ──────────────────────────────────────────────────────────────────────────────

def load_policy_rules():
    path = os.path.join(config.DATA_DOCUMENTS, "policy_rules.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        text = f.read()
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
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(policy_chunks)
    return vectorizer, tfidf_matrix


def retrieve_policy_evidence(query, vectorizer, tfidf_matrix, policy_chunks, top_k=3):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [(policy_chunks[i], float(scores[i])) for i in top_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Review building blocks
# ──────────────────────────────────────────────────────────────────────────────

def _risk_level(score):
    if score >= config.HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif score >= 0.3:
        return "MEDIUM"
    return "LOW"


def _build_query(row):
    parts = []
    if row.get("duplicate_claim_flag", 0):
        parts.append("duplicate claim billing same date")
    if row.get("late_submission_flag", 0):
        parts.append("late claim submission over 90 days")
    if row.get("suspicious_keyword_count", 0) >= 3:
        parts.append("suspicious keywords documentation fraud")
    if row.get("documentation_score", 1.0) < 0.4:
        parts.append("incomplete medical records documentation standards")
    if row.get("claim_to_provider_avg_ratio", 1.0) > 2.0:
        parts.append("claim amount exceeds provider average upcoding")
    if row.get("service_type") == "Pharmacy":
        parts.append("pharmacy claim quantity dosage protocol")
    if not parts:
        parts.append("long term care claim review medical necessity")
    return " ".join(parts)


def _risk_indicators(row):
    out = []
    ratio = row.get("claim_to_provider_avg_ratio", np.nan)
    if pd.notna(ratio) and ratio > 1.5:
        out.append(f"Claim amount is {ratio:.1f}x the provider's average billing")
    ds = row.get("documentation_score", np.nan)
    if pd.notna(ds) and ds < 0.45:
        out.append(f"Low documentation completeness score ({ds:.2f} / 1.00)")
    if row.get("duplicate_claim_flag", 0) == 1:
        out.append("Duplicate-billing flag triggered (same provider/beneficiary window)")
    if row.get("late_submission_flag", 0) == 1:
        out.append("Submitted outside the 90-day window")
    skw = int(row.get("suspicious_keyword_count", 0) or 0)
    if skw >= 3:
        out.append(f"{skw} suspicious keywords detected in claim narrative")
    pc = row.get("prior_claim_count", np.nan)
    if pd.notna(pc) and pc > 10:
        out.append(f"Elevated prior claim history ({int(pc)} claims)")
    if row.get("approval_ratio", 1.0) < 0.5:
        out.append(f"Historical approval ratio low ({row.get('approval_ratio'):.2f})")
    if not out:
        out.append("No single dominant signal; composite model score elevated")
    return out[:5]


def _documentation_gaps(row):
    gaps = []
    ds = row.get("documentation_score", np.nan)
    if pd.isna(ds):
        gaps.append("Documentation score is missing in source data")
    elif ds < 0.45:
        gaps.append("Clinical notes appear incomplete / inconsistent")
    if row.get("late_submission_flag", 0) == 1:
        gaps.append("No clinical justification for >90-day submission delay")
    if row.get("duplicate_claim_flag", 0) == 1:
        gaps.append("Possible duplicate billing — confirm distinct date of service")
    if row.get("suspicious_keyword_count", 0) >= 3:
        gaps.append("Narrative contains language patterns flagged by NLP screen")
    if not gaps:
        gaps.append("No documentation gaps detected by automated screen")
    return gaps


def _suggested_action(level):
    if level == "HIGH":
        return ("SUSPEND PAYMENT pending senior-analyst review. Request complete medical "
                "records, provider attestation, and any supporting plan-of-care within "
                "10 business days. Escalate to SIU if pattern repeats.")
    if level == "MEDIUM":
        return ("ENHANCED REVIEW within 5 business days. Request supplemental "
                "documentation; do not pay until missing fields are reconciled.")
    return ("STANDARD PROCESSING. Include in next routine audit sample (10% sampling).")


def _human_review_notes(row, level):
    notes = [
        "Verify provider NPI is active and not on the OIG exclusion list",
        "Cross-check date of service against beneficiary's other recent claims",
        "Confirm diagnosis codes are consistent with rendered procedures",
    ]
    if level == "HIGH":
        notes.append("Request signed plan-of-care and (for LTC) caregiver visit logs")
        notes.append("Compare billed units against standard utilization for this CPT")
    if row.get("duplicate_claim_flag", 0):
        notes.append("Pull all claims from this provider+beneficiary in the last 30 days")
    if row.get("documentation_score", 1.0) < 0.45:
        notes.append("Request complete EHR audit trail covering the date of service")
    return notes


def _limitations():
    return [
        "Data is fully synthetic; do NOT use any specific value as evidence of real fraud.",
        "Model probabilities are calibrated only against the synthetic generating process.",
        "Retrieval is TF-IDF, not semantic — relevant policy text may be missed if vocabulary differs.",
        "All HIGH-risk recommendations require a licensed analyst before any payment action.",
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Review generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_review(claim_id, row, risk_score, vectorizer, tfidf_matrix, policy_chunks):
    level = _risk_level(risk_score)
    query = _build_query(row)
    evidence = retrieve_policy_evidence(query, vectorizer, tfidf_matrix, policy_chunks, top_k=3)
    indicators = _risk_indicators(row)
    gaps = _documentation_gaps(row)
    action = _suggested_action(level)
    human_notes = _human_review_notes(row, level)
    limits = _limitations()

    evidence_text = ""
    for i, (chunk, score) in enumerate(evidence, 1):
        short = chunk[:280].replace("\n", " ")
        evidence_text += f"\n  [{i}] (similarity={score:.3f}) {short}..."

    ind_text = "\n".join(f"  - {x}" for x in indicators)
    gap_text = "\n".join(f"  - {x}" for x in gaps)
    hum_text = "\n".join(f"  - {x}" for x in human_notes)
    lim_text = "\n".join(f"  - {x}" for x in limits)

    review = f"""
==================================================================
INSURANCE FWA CLAIM REVIEW REPORT  (AI-assisted, human-in-the-loop)
==================================================================
Claim ID         : {claim_id}
Risk Level       : {level}
Model Risk Score : {risk_score:.4f}
Service Type     : {row.get('service_type', 'N/A')}
Diagnosis Group  : {row.get('diagnosis_group', 'N/A')}
Claim Amount     : ${float(row.get('claim_amount', 0) or 0):,.2f}
Approved Amount  : ${float(row.get('approved_amount', 0) or 0):,.2f}
Provider ID      : {row.get('provider_id', 'N/A')}
State            : {row.get('state', 'N/A')}

------------------------------------------------------------------
KEY RISK INDICATORS
------------------------------------------------------------------
{ind_text}

------------------------------------------------------------------
RETRIEVED POLICY EVIDENCE  (TF-IDF cosine similarity)
------------------------------------------------------------------
Query: "{query}"
{evidence_text}

------------------------------------------------------------------
DOCUMENTATION GAPS
------------------------------------------------------------------
{gap_text}

------------------------------------------------------------------
SUGGESTED ANALYST ACTION
------------------------------------------------------------------
{action}

------------------------------------------------------------------
HUMAN REVIEW NOTES  (what an analyst should verify)
------------------------------------------------------------------
{hum_text}

------------------------------------------------------------------
LIMITATIONS
------------------------------------------------------------------
{lim_text}
==================================================================
"""
    return review


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Running RAG claim review pipeline...")

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

    model_path = os.path.join(config.OUTPUTS_MODELS, "best_fwa_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}.")
    model = joblib.load(model_path)

    drop = ID_LIKE_COLS + ["fraud_label"] + EXCLUDE_FROM_MODEL
    feature_cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    risk_scores = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["model_risk_score"] = risk_scores

    print("  Building TF-IDF policy index...")
    policy_chunks = load_policy_rules()
    if not policy_chunks:
        policy_chunks = ["General insurance policy guidelines apply to all claims."]
    vectorizer, tfidf_matrix = build_policy_index(policy_chunks)
    print(f"  Indexed {len(policy_chunks)} policy chunks.")

    os.makedirs(config.OUTPUTS_REVIEWS, exist_ok=True)

    # Sample mix: 8 HIGH, 3 MEDIUM, 2 LOW for contrast (12+ reviews)
    high = df.nlargest(8, "model_risk_score")
    med  = df[(df["model_risk_score"] >= 0.3) &
              (df["model_risk_score"] < config.HIGH_RISK_THRESHOLD)].head(3)
    low  = df[df["model_risk_score"] < 0.3].sample(min(2, max(0, (df["model_risk_score"] < 0.3).sum())),
                                                   random_state=config.RANDOM_SEED)
    sample = pd.concat([high, med, low]).drop_duplicates(subset=["claim_id"])

    print(f"  Generating reviews for {len(sample)} sample claims "
          f"({(sample['model_risk_score'] >= config.HIGH_RISK_THRESHOLD).sum()} HIGH, "
          f"{((sample['model_risk_score']>=0.3)&(sample['model_risk_score']<config.HIGH_RISK_THRESHOLD)).sum()} MED, "
          f"{(sample['model_risk_score']<0.3).sum()} LOW)...")

    for _, row in sample.iterrows():
        cid = row["claim_id"]
        review_text = generate_review(
            cid, row, float(row["model_risk_score"]),
            vectorizer, tfidf_matrix, policy_chunks
        )
        out_path = os.path.join(config.OUTPUTS_REVIEWS, f"review_{cid}.txt")
        with open(out_path, "w") as f:
            f.write(review_text)

    print(f"  Saved {len(sample)} sample reviews to {config.OUTPUTS_REVIEWS}/")
    print("RAG claim review complete.")


if __name__ == "__main__":
    main()
