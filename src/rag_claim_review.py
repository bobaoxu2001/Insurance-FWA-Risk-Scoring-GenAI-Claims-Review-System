"""
RAG-style review module (no paid API required).

TF-IDF + cosine similarity over the policy_rules.txt corpus, plus a structured
template populated from model output + observed features.

Mode detection:
  - If data/processed/provider_modeling_table.csv exists → PROVIDER-LEVEL reviews
    (uses "Provider ID", "billing pattern", etc.)
  - Otherwise → CLAIM-LEVEL reviews (legacy synthetic mode)

Provider reviews include:
  - Provider ID, Risk Level, Model Risk Score
  - Key Risk Indicators (3-5 quantified bullets)
  - Retrieved Policy / Audit Evidence (top 2-3 TF-IDF policy rule chunks)
  - Data & Documentation Gaps
  - Suggested Analyst Action
  - Human Review Notes
  - System Limitations disclaimer

Generates 15 sample reviews: 10 HIGH, 3 MEDIUM, 2 LOW.
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


# ── Shared constants ───────────────────────────────────────────────────────────

ID_LIKE_COLS = [
    "claim_id", "policyholder_id", "provider_id", "claim_date",
    "service_type", "diagnosis_group", "state",
    "Provider",
]
EXCLUDE_FROM_MODEL = ["rule_based_risk_score"]
TARGET_COLS = ["PotentialFraud", "fraud_label"]


def _detect_target(df):
    for t in TARGET_COLS:
        if t in df.columns:
            return t
    return "fraud_label"


# ── Document loading & indexing ────────────────────────────────────────────────

def load_policy_rules():
    """Load all policy-style documents from data/documents/.
    Currently indexes:
      - policy_rules.txt           (synthetic LTC / FWA audit rules)
      - oig_exclusion_codes.txt    (REAL federal exclusion-code taxonomy
                                    emitted by src/oig_leie_analysis.py)
    The latter is grounded in 42 U.S.C. §1128 and counts reflect 83K+
    actual exclusions in the LEIE — so the RAG retriever cites real
    federal authority where applicable."""
    chunks = []
    for filename in ("policy_rules.txt", "oig_exclusion_codes.txt"):
        path = os.path.join(config.DATA_DOCUMENTS, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            text = f.read()
        file_chunks = re.split(r"\n(?=\d+\.)", text)
        file_chunks = [c.strip() for c in file_chunks if len(c.strip()) > 50]
        chunks.extend(file_chunks)
    return chunks


def build_policy_index(policy_chunks):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(policy_chunks)
    return vectorizer, tfidf_matrix


def retrieve_policy_evidence(query, vectorizer, tfidf_matrix, policy_chunks, top_k=3):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [(policy_chunks[i], float(scores[i])) for i in top_idx]


# ── Risk helpers ───────────────────────────────────────────────────────────────

def _risk_level(score):
    if score >= config.HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif score >= 0.3:
        return "MEDIUM"
    return "LOW"


def _limitations(mode):
    if mode == "real":
        return [
            "Data source: Kaggle Healthcare Provider Fraud Detection Analysis (public educational dataset).",
            "NOT Manulife / John Hancock data.  NOT Long Term Care-specific data.",
            "No real patient records, clinical documents, or claim notes are used.",
            "RAG policy text is synthetic.  TF-IDF retrieval may miss semantically relevant rules.",
            "All HIGH-risk flags require a licensed analyst review before any payment action.",
        ]
    return [
        "Data is fully synthetic; do NOT use any specific value as evidence of real fraud.",
        "Model probabilities are calibrated only against the synthetic generating process.",
        "Retrieval is TF-IDF, not semantic — relevant policy text may be missed.",
        "All HIGH-risk recommendations require a licensed analyst before any payment action.",
    ]


# ── Provider-level review ──────────────────────────────────────────────────────

def _provider_query(row):
    parts = []
    if row.get("inpatient_ratio", 0) > 0.9:
        parts.append("high inpatient billing ratio upcoding unnecessary services")
    if row.get("reimbursement_outlier_score", 0) > 2:
        parts.append("claim reimbursement amount exceeds provider average upcoding")
    if row.get("unique_attending_physicians", 0) > 50:
        parts.append("multiple physicians billing under same provider identity")
    if row.get("avg_admission_duration", 0) > 10:
        parts.append("unusually long inpatient admission medically unnecessary")
    if not parts:
        parts.append("healthcare provider billing fraud waste abuse audit review")
    return " ".join(parts)


def _provider_risk_indicators(row, stats):
    out = []

    # Reimbursement vs overall median
    if "avg_reimbursed_per_claim" in row and not pd.isna(row.get("avg_reimbursed_per_claim")):
        med = stats.get("avg_reimbursed_per_claim_median", 0)
        val = float(row["avg_reimbursed_per_claim"])
        if med and val > med * 1.5:
            out.append(
                f"Avg reimbursement per claim (${val:,.0f}) is {val/med:.1f}x "
                f"the overall provider median (${med:,.0f})"
            )

    if "inpatient_ratio" in row and not pd.isna(row.get("inpatient_ratio")):
        p90 = stats.get("inpatient_ratio_p90", 1)
        val = float(row["inpatient_ratio"])
        if val > p90:
            out.append(
                f"Inpatient billing ratio ({val:.2%}) exceeds the 90th percentile "
                f"({p90:.2%}) — potential upcoding to higher-cost inpatient setting"
            )

    if "total_claims" in row and not pd.isna(row.get("total_claims")):
        p90 = stats.get("total_claims_p90", 9999)
        val = int(row["total_claims"])
        if val > p90:
            out.append(
                f"Provider submitted {val:,} total claims — above the 90th-percentile "
                f"volume threshold ({int(p90):,} claims)"
            )

    if "reimbursement_per_beneficiary" in row and not pd.isna(row.get("reimbursement_per_beneficiary")):
        med = stats.get("reimbursement_per_beneficiary_median", 0)
        val = float(row["reimbursement_per_beneficiary"])
        if med and val > med * 2:
            out.append(
                f"Reimbursement per beneficiary (${val:,.0f}) is {val/med:.1f}x median — "
                "unusually high revenue per patient"
            )

    if "avg_chronic_conditions" in row and not pd.isna(row.get("avg_chronic_conditions")):
        p90 = stats.get("avg_chronic_conditions_p90", 9)
        val = float(row["avg_chronic_conditions"])
        if val > p90:
            out.append(
                f"Patient panel has elevated chronic condition complexity "
                f"(avg {val:.1f} conditions, p90={p90:.1f}) — may indicate upcoding comorbidities"
            )

    if not out:
        out.append(
            "Composite provider risk score elevated; no single dominant signal — "
            "review full billing pattern"
        )
    return out[:5]


def _provider_doc_gaps(row):
    gaps = []
    if row.get("unique_attending_physicians", 0) > 30:
        gaps.append(
            "High number of attending physicians billed under this provider NPI — "
            "verify all physicians are credentialed and enrolled"
        )
    if row.get("inpatient_ratio", 0) > 0.85:
        gaps.append(
            "Predominantly inpatient billing — confirm medical necessity documentation "
            "for inpatient admissions vs. outpatient or observation status"
        )
    if not gaps:
        gaps.append("No automated documentation gaps identified; standard audit sampling applies")
    return gaps


def _provider_suggested_action(level):
    if level == "HIGH":
        return (
            "SUSPEND PAYMENT for this provider pending senior-analyst review.  "
            "Request complete billing records, physician attestations, and patient-level "
            "claim detail for the last 12 months.  Escalate to SIU if pattern is confirmed."
        )
    if level == "MEDIUM":
        return (
            "ENHANCED REVIEW within 5 business days.  Pull provider-level claim detail "
            "and cross-check with peer-provider benchmarks.  Do not suspend unless "
            "additional red flags are found."
        )
    return "STANDARD PROCESSING.  Include provider in next routine audit sample (10% sampling)."


def _provider_human_review_notes(row, level):
    notes = [
        "Verify provider NPI is active and not on the OIG exclusion list",
        "Cross-check billed CPT codes against CMS peer-group norms for this specialty",
        "Confirm all attending physicians billed under this NPI are independently enrolled",
    ]
    if level == "HIGH":
        notes.append(
            "Pull individual claim lines for the top 5 highest-reimbursement claims "
            "and request supporting clinical documentation"
        )
        notes.append(
            "Check for overlapping inpatient/outpatient claims on the same beneficiary "
            "on the same date"
        )
    if row.get("inpatient_ratio", 0) > 0.8:
        notes.append(
            "For inpatient claims: verify admission/discharge summaries and confirm "
            "appropriate level-of-care documentation"
        )
    return notes


def generate_provider_review(provider_id, row, risk_score, vectorizer,
                               tfidf_matrix, policy_chunks, stats):
    level    = _risk_level(risk_score)
    query    = _provider_query(row)
    evidence = retrieve_policy_evidence(query, vectorizer, tfidf_matrix, policy_chunks, top_k=3)
    indicators = _provider_risk_indicators(row, stats)
    gaps       = _provider_doc_gaps(row)
    action     = _provider_suggested_action(level)
    notes      = _provider_human_review_notes(row, level)
    limits     = _limitations("real")

    evidence_text = ""
    for i, (chunk, score) in enumerate(evidence, 1):
        short = chunk[:280].replace("\n", " ")
        evidence_text += f"\n  [{i}] (similarity={score:.3f}) {short}..."

    ind_text = "\n".join(f"  - {x}" for x in indicators)
    gap_text = "\n".join(f"  - {x}" for x in gaps)
    hum_text = "\n".join(f"  - {x}" for x in notes)
    lim_text = "\n".join(f"  - {x}" for x in limits)

    # Friendly display for some numeric fields
    def _fmt(col, fmt=".2f", prefix=""):
        v = row.get(col, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{prefix}{float(v):{fmt}}"

    review = f"""
==================================================================
HEALTHCARE PROVIDER FWA REVIEW REPORT  (AI-assisted, human-in-the-loop)
==================================================================
Provider ID              : {provider_id}
Risk Level               : {level}
Model Risk Score         : {risk_score:.4f}
Total Claims             : {_fmt('total_claims', '.0f')}
Inpatient Claims         : {_fmt('inpatient_claim_count', '.0f')}
Outpatient Claims        : {_fmt('outpatient_claim_count', '.0f')}
Inpatient Ratio          : {_fmt('inpatient_ratio', '.2%')}
Unique Beneficiaries     : {_fmt('unique_beneficiaries', '.0f')}
Avg Reimbursement/Claim  : {_fmt('avg_reimbursed_per_claim', ',.2f', '$')}
Total Reimbursed         : {_fmt('total_reimbursed', ',.2f', '$')}
Avg Chronic Conditions   : {_fmt('avg_chronic_conditions', '.2f')}
Avg Admission Duration   : {_fmt('avg_admission_duration', '.1f')} days

------------------------------------------------------------------
KEY RISK INDICATORS
------------------------------------------------------------------
{ind_text}

------------------------------------------------------------------
RETRIEVED POLICY / AUDIT EVIDENCE  (TF-IDF cosine similarity)
------------------------------------------------------------------
Query: "{query}"
{evidence_text}

------------------------------------------------------------------
DATA & DOCUMENTATION GAPS
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
SYSTEM LIMITATIONS & DATA DISCLAIMER
------------------------------------------------------------------
{lim_text}
==================================================================
"""
    return review


# ── Claim-level review (legacy synthetic mode) ─────────────────────────────────

def _risk_level_claim(score):
    return _risk_level(score)


def _build_claim_query(row):
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


def _claim_risk_indicators(row):
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
    if not out:
        out.append("No single dominant signal; composite model score elevated")
    return out[:5]


def _claim_doc_gaps(row):
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


def _claim_suggested_action(level):
    if level == "HIGH":
        return (
            "SUSPEND PAYMENT pending senior-analyst review. Request complete medical "
            "records, provider attestation, and any supporting plan-of-care within "
            "10 business days. Escalate to SIU if pattern repeats."
        )
    if level == "MEDIUM":
        return (
            "ENHANCED REVIEW within 5 business days. Request supplemental "
            "documentation; do not pay until missing fields are reconciled."
        )
    return "STANDARD PROCESSING. Include in next routine audit sample (10% sampling)."


def _claim_human_review_notes(row, level):
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


def generate_claim_review(claim_id, row, risk_score, vectorizer, tfidf_matrix, policy_chunks):
    level      = _risk_level_claim(risk_score)
    query      = _build_claim_query(row)
    evidence   = retrieve_policy_evidence(query, vectorizer, tfidf_matrix, policy_chunks, top_k=3)
    indicators = _claim_risk_indicators(row)
    gaps       = _claim_doc_gaps(row)
    action     = _claim_suggested_action(level)
    notes      = _claim_human_review_notes(row, level)
    limits     = _limitations("synthetic")

    evidence_text = ""
    for i, (chunk, score) in enumerate(evidence, 1):
        short = chunk[:280].replace("\n", " ")
        evidence_text += f"\n  [{i}] (similarity={score:.3f}) {short}..."

    ind_text = "\n".join(f"  - {x}" for x in indicators)
    gap_text = "\n".join(f"  - {x}" for x in gaps)
    hum_text = "\n".join(f"  - {x}" for x in notes)
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


# ── Shared: score data using loaded model ──────────────────────────────────────

def _score_df(df, model):
    target = _detect_target(df)
    drop = ID_LIKE_COLS + [target] + EXCLUDE_FROM_MODEL
    feature_cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    risk_scores = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["model_risk_score"] = risk_scores
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Running RAG review pipeline...")

    # Determine mode
    provider_path = os.path.join(config.DATA_PROCESSED, "provider_modeling_table.csv")
    mode = "real" if os.path.exists(provider_path) else "synthetic"

    if mode == "real":
        df = pd.read_csv(provider_path)
        print(f"  Mode: PROVIDER-LEVEL (real Kaggle data)  shape={df.shape}")
        id_col = "Provider"
    else:
        df = None
        for p in [
            os.path.join(config.DATA_PROCESSED, "claims_features.csv"),
            os.path.join(config.DATA_PROCESSED, "claims_encoded.csv"),
            os.path.join(config.DATA_RAW,       "synthetic_claims.csv"),
        ]:
            if os.path.exists(p):
                df = pd.read_csv(p)
                print(f"  Mode: CLAIM-LEVEL (synthetic data)  from {p}")
                break
        if df is None:
            raise FileNotFoundError("No claims data found. Run data generation first.")
        id_col = "claim_id"

    # Load model
    model_path = os.path.join(config.OUTPUTS_MODELS, "best_fwa_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run modeling.py first.")
    model = joblib.load(model_path)

    df = _score_df(df, model)

    # Build TF-IDF index
    print("  Building TF-IDF policy index...")
    policy_chunks = load_policy_rules()
    if not policy_chunks:
        policy_chunks = ["General insurance policy guidelines apply to all claims."]
    vectorizer, tfidf_matrix = build_policy_index(policy_chunks)
    print(f"  Indexed {len(policy_chunks)} policy chunks.")

    os.makedirs(config.OUTPUTS_REVIEWS, exist_ok=True)

    # Sample: 10 HIGH, 3 MEDIUM, 2 LOW
    high = df.nlargest(10, "model_risk_score")
    med  = df[
        (df["model_risk_score"] >= 0.3) &
        (df["model_risk_score"] < config.HIGH_RISK_THRESHOLD)
    ].head(3)
    low  = df[df["model_risk_score"] < 0.3].sample(
        min(2, max(0, (df["model_risk_score"] < 0.3).sum())),
        random_state=config.RANDOM_SEED
    )
    sample = pd.concat([high, med, low]).drop_duplicates(subset=[id_col])

    print(f"  Generating {len(sample)} reviews "
          f"({(sample['model_risk_score'] >= config.HIGH_RISK_THRESHOLD).sum()} HIGH, "
          f"{((sample['model_risk_score']>=0.3)&(sample['model_risk_score']<config.HIGH_RISK_THRESHOLD)).sum()} MED, "
          f"{(sample['model_risk_score']<0.3).sum()} LOW)...")

    if mode == "real":
        # Precompute population stats for relative comparisons
        stats = {}
        for col, stat in [
            ("avg_reimbursed_per_claim",     "median"),
            ("avg_reimbursed_per_claim",     "p90"),
            ("inpatient_ratio",              "p90"),
            ("total_claims",                 "p90"),
            ("reimbursement_per_beneficiary","median"),
            ("avg_chronic_conditions",       "p90"),
        ]:
            if col in df.columns:
                key = f"{col}_{stat}"
                if stat == "median":
                    stats[key] = float(df[col].median())
                elif stat == "p90":
                    stats[key] = float(df[col].quantile(0.90))

        for _, row in sample.iterrows():
            pid = row[id_col]
            review_text = generate_provider_review(
                pid, row, float(row["model_risk_score"]),
                vectorizer, tfidf_matrix, policy_chunks, stats
            )
            out_path = os.path.join(config.OUTPUTS_REVIEWS, f"review_{pid}.txt")
            with open(out_path, "w") as f:
                f.write(review_text)
    else:
        for _, row in sample.iterrows():
            cid = row[id_col]
            review_text = generate_claim_review(
                cid, row, float(row["model_risk_score"]),
                vectorizer, tfidf_matrix, policy_chunks
            )
            out_path = os.path.join(config.OUTPUTS_REVIEWS, f"review_{cid}.txt")
            with open(out_path, "w") as f:
                f.write(review_text)

    print(f"  Saved {len(sample)} reviews to {config.OUTPUTS_REVIEWS}/")
    print("RAG review complete.")


if __name__ == "__main__":
    main()
