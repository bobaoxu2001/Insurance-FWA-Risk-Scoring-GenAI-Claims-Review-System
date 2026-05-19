"""
synthetic_data_generation.py
============================
SYNTHETIC DEMO MODE — This module generates a fully synthetic insurance claims
dataset for demonstration purposes.  It does NOT use any real patient, provider,
or insurance company data.

Use this when the real Kaggle dataset is not available.  The full production
pipeline uses src/data_ingestion.py + src/provider_feature_engineering.py with
the Kaggle Healthcare Provider Fraud Detection Analysis dataset.

Data generation module for synthetic insurance claims dataset.

Design notes (v2 — leakage-aware)
---------------------------------
The earlier version drove `fraud_label` from a sigmoid of the exact same
features later exposed to the model, which produced unrealistically perfect
metrics (ROC-AUC ~0.99). A senior reviewer would immediately flag this as
target leakage.

This rewrite uses a **hidden intent** variable that is *correlated with* but
not equal to the observable features the model sees:

    intent_to_defraud ~ Bernoulli with prior shaped by latent provider integrity
                        and policyholder propensity (NEITHER exposed to model).

    fraud_label = Bernoulli( sigmoid( w_intent * intent
                                      + w_amount * noisy_amount_signal
                                      + w_doc    * noisy_doc_signal
                                      + w_dup    * noisy_dup_signal
                                      + base_rate_noise + gaussian_noise ) )

Several observed features (documentation_score, suspicious_keyword_count,
duplicate_claim_flag) are *noisy proxies* of the hidden intent — informative,
but far from deterministic. Some legitimate claims look suspicious (high-cost
specialty procedures, sloppy paperwork), and some fraud claims look subtle.

Target test-set performance: ROC-AUC ~0.82-0.92, F1 ~0.45-0.70.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_claims(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    # ── Identifiers ────────────────────────────────────────────────────────
    claim_ids = [f"CLM{str(i).zfill(6)}" for i in range(1, n + 1)]
    policyholder_ids = [f"PH{str(rng.integers(1, 2001)).zfill(5)}" for _ in range(n)]
    provider_ids = [f"PR{str(rng.integers(1, 301)).zfill(4)}" for _ in range(n)]

    # ── Dates ──────────────────────────────────────────────────────────────
    base_date = pd.Timestamp("2022-01-01")
    days_offset = rng.integers(0, 730, size=n)
    claim_dates = [base_date + pd.Timedelta(days=int(d)) for d in days_offset]

    # ── Service & Diagnosis ────────────────────────────────────────────────
    service_types = rng.choice(
        ["Inpatient", "Outpatient", "Pharmacy", "Mental Health", "Dental", "Vision", "Emergency"],
        size=n, p=[0.15, 0.30, 0.20, 0.10, 0.10, 0.08, 0.07]
    )
    diagnosis_groups = rng.choice(
        ["Musculoskeletal", "Cardiovascular", "Respiratory", "Mental_Health",
         "Diabetes", "Oncology", "Gastrointestinal", "Neurological", "Other"],
        size=n
    )

    # ── HIDDEN LATENT DRIVERS (NOT exposed to the model) ───────────────────
    # provider_integrity: lower = more inclined to abusive billing
    provider_list = sorted(set(provider_ids))
    prov_integrity = {p: float(rng.beta(5, 2)) for p in provider_list}  # most ~ honest
    # ~ 8% of providers are "bad actors" with low integrity
    bad_providers = rng.choice(provider_list, size=max(1, int(0.08 * len(provider_list))), replace=False)
    for p in bad_providers:
        prov_integrity[p] = float(rng.beta(2, 6))  # skewed low
    provider_integrity_hidden = np.array([prov_integrity[p] for p in provider_ids])

    # policyholder_propensity: hidden tendency to file inflated/fraudulent claims
    ph_list = sorted(set(policyholder_ids))
    ph_prop = {p: float(rng.beta(2, 8)) for p in ph_list}
    policyholder_propensity_hidden = np.array([ph_prop[p] for p in policyholder_ids])

    # intent_to_defraud: hidden, partly driven by integrity+propensity, partly random
    intent_logit = (
        -2.5
        - 2.2 * provider_integrity_hidden
        + 1.8 * policyholder_propensity_hidden
        + rng.normal(0, 1.0, size=n)
    )
    intent_to_defraud = (rng.random(n) < _sigmoid(intent_logit)).astype(int)

    # ── Provider stats (consistent per provider) — these ARE exposed ───────
    prov_vol = {p: int(rng.integers(10, 500)) for p in provider_list}
    prov_avg = {p: float(rng.uniform(500, 8000)) for p in provider_list}
    provider_claim_volume = np.array([prov_vol[p] for p in provider_ids])
    provider_avg_claim_amount = np.array([prov_avg[p] for p in provider_ids])

    # ── Claim financials ───────────────────────────────────────────────────
    # Amount is mostly driven by service type / diagnosis / provider baseline.
    # Intent nudges it up only modestly, so amount alone is NOT a giveaway.
    base_claim_amount = rng.lognormal(mean=7.5, sigma=1.0, size=n).clip(100, 150000)
    # Legitimate high-cost outliers exist (specialty procedures, oncology, inpatient surgery)
    high_cost_legit = (
        (service_types == "Inpatient") | (diagnosis_groups == "Oncology")
    )
    base_claim_amount = np.where(
        high_cost_legit & (rng.random(n) < 0.25),
        base_claim_amount * rng.uniform(1.5, 4.0, size=n),
        base_claim_amount,
    )

    # Mild inflation from intent (NOT deterministic, NOT large)
    intent_inflation = np.where(
        intent_to_defraud == 1,
        rng.uniform(1.0, 1.6, size=n),  # subtle
        rng.uniform(0.9, 1.1, size=n),  # legit noise
    )
    claim_amount = (base_claim_amount * intent_inflation).clip(100, 250000).round(2)

    claimant_age = rng.integers(18, 85, size=n)
    states = rng.choice(
        ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"],
        size=n
    )

    # ── Policy & history ──────────────────────────────────────────────────
    days_since_policy_start = rng.integers(1, 3650, size=n)
    prior_claim_count = rng.integers(0, 20, size=n)

    # ── Documentation & flags as NOISY PROXIES of intent ───────────────────
    # Documentation score: noisy proxy of intent. Plenty of legit claims have low
    # doc scores; plenty of fraud claims have OK docs.
    doc_noise = rng.normal(0, 0.14, size=n)
    documentation_score = np.clip(
        0.80 - 0.38 * intent_to_defraud + doc_noise,
        0.0, 1.0,
    )

    # Suspicious keyword count: poisson with intent-shifted mean (noisy)
    kw_lambda = 0.7 + 3.0 * intent_to_defraud
    suspicious_keyword_count = rng.poisson(kw_lambda, size=n).clip(0, 10)

    # Duplicate flag: noisy proxy; fraud raises odds but most fraud claims aren't dups.
    dup_prob = 0.025 + 0.25 * intent_to_defraud
    duplicate_claim_flag = (rng.random(n) < dup_prob).astype(int)

    # Late submission flag: weak proxy
    late_prob = 0.05 + 0.15 * intent_to_defraud
    late_submission_flag = (rng.random(n) < late_prob).astype(int)

    # ── Build fraud label from intent + noisy observed signals + noise ─────
    # amount_signal is intentionally weak so the model can't memorize on amount
    amount_ratio = claim_amount / (provider_avg_claim_amount + 1e-6)
    noisy_amount_signal = np.clip(amount_ratio - 1.0, -1, 6) + rng.normal(0, 0.6, size=n)
    noisy_doc_signal = (1.0 - documentation_score) + rng.normal(0, 0.25, size=n)
    noisy_dup_signal = duplicate_claim_flag + 0.6 * late_submission_flag

    latent = (
        4.5 * intent_to_defraud
        + 0.75 * noisy_amount_signal
        + 1.20 * noisy_doc_signal
        + 1.00 * noisy_dup_signal
        + 0.25 * np.clip(suspicious_keyword_count - 1, 0, 6)
        + 0.15 * np.clip(prior_claim_count - 8, 0, 12)
        + 0.30 * (days_since_policy_start < 180).astype(float)
        + rng.normal(0, 0.45, size=n)       # mild aleatoric noise
        - 4.0                                # shift to ~7-10% base rate
    )
    prob_fraud = _sigmoid(latent)

    # Base-rate noise: small fraction of low-risk claims still become fraud,
    # and a slice of high-risk fraud claims look subtle (model misses some).
    base_noise = rng.random(n) < 0.025
    skip_noise = rng.random(n) < 0.10
    fraud_label = (rng.random(n) < prob_fraud).astype(int)
    fraud_label = np.where(base_noise, 1, fraud_label)
    fraud_label = np.where(skip_noise & (fraud_label == 1), 0, fraud_label)

    # ── Approved amount: noisy, NOT directly conditioned on fraud_label ────
    # In reality the adjuster doesn't know the true label; approval ratio
    # depends on doc quality and amount-vs-norm with noise.
    approval_logit = (
        2.0
        - 1.5 * (1.0 - documentation_score)
        - 0.3 * np.clip(amount_ratio - 1.0, 0, 5)
        - 1.0 * duplicate_claim_flag
        + rng.normal(0, 0.6, size=n)
    )
    approval_ratio = np.clip(_sigmoid(approval_logit) + rng.normal(0, 0.08, size=n), 0.05, 1.0)
    approved_amount = (claim_amount * approval_ratio).round(2)

    df = pd.DataFrame({
        "claim_id": claim_ids,
        "policyholder_id": policyholder_ids,
        "provider_id": provider_ids,
        "claim_date": claim_dates,
        "service_type": service_types,
        "diagnosis_group": diagnosis_groups,
        "claim_amount": claim_amount,
        "approved_amount": approved_amount,
        "days_since_policy_start": days_since_policy_start,
        "prior_claim_count": prior_claim_count,
        "provider_claim_volume": provider_claim_volume,
        "provider_avg_claim_amount": provider_avg_claim_amount.round(2),
        "claimant_age": claimant_age,
        "state": states,
        "documentation_score": documentation_score.round(4),
        "suspicious_keyword_count": suspicious_keyword_count,
        "duplicate_claim_flag": duplicate_claim_flag,
        "late_submission_flag": late_submission_flag,
        # NOTE: high_cost_outlier_flag REMOVED — it was a near-deterministic
        # leakage signal in v1. Outlier-ness is implicit in claim_amount.
        "fraud_label": fraud_label,
    })

    # Inject realistic missingness (~1.5% per column) on a few fields
    miss_cols = ["documentation_score", "prior_claim_count", "approved_amount"]
    for c in miss_cols:
        mask = rng.random(n) < 0.015
        df.loc[mask, c] = np.nan

    # Inject ~0.3% exact-duplicate claim_ids? No — keep IDs unique; instead
    # duplicate a small number of claim rows to simulate dup detection load.
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic document generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_claim_document(claim_id, row, rng):
    """Generate a synthetic claim text document. Tone reflects flags, NOT label."""
    service = row["service_type"]
    diagnosis = row["diagnosis_group"]
    amount = row["claim_amount"]
    age = row["claimant_age"]
    doc_score = row.get("documentation_score", 0.7)
    if pd.isna(doc_score):
        doc_score = 0.7

    procedures = {
        "Inpatient": ["hospital admission", "surgical procedure", "post-operative care"],
        "Outpatient": ["office visit", "diagnostic imaging", "lab work"],
        "Pharmacy": ["prescription medication", "compounded medication", "specialty drug"],
        "Mental Health": ["psychotherapy session", "psychiatric evaluation", "counseling"],
        "Dental": ["dental extraction", "root canal treatment", "crown placement"],
        "Vision": ["eye examination", "contact lens fitting", "LASIK consultation"],
        "Emergency": ["emergency room visit", "trauma care", "urgent intervention"],
    }
    proc = rng.choice(procedures.get(service, ["general service"]))

    quality_note = (
        "Documentation is complete and consistent with clinical notes."
        if doc_score > 0.7
        else "Documentation appears incomplete; some fields missing or inconsistent."
    )

    # Caution note keyed to OBSERVED flags only (no leakage of fraud_label)
    caution_lines = []
    if row.get("duplicate_claim_flag", 0):
        caution_lines.append("Duplicate billing codes detected within recent submission window.")
    if row.get("suspicious_keyword_count", 0) >= 3:
        caution_lines.append("Multiple suspicious keywords flagged by NLP screen.")
    if row.get("late_submission_flag", 0):
        caution_lines.append("Submission received outside the 90-day window.")
    caution_block = ("\nFLAGS: " + " ".join(caution_lines)) if caution_lines else ""

    date_str = row['claim_date']
    if hasattr(date_str, 'strftime'):
        date_str = date_str.strftime('%Y-%m-%d')

    doc = f"""INSURANCE CLAIM DOCUMENT
========================
Claim ID      : {claim_id}
Service Type  : {service}
Diagnosis     : {diagnosis.replace('_', ' ')}
Claimant Age  : {age}
Claim Amount  : ${amount:,.2f}

--- CLINICAL NOTES ---
Patient presented for {proc} related to {diagnosis.replace('_', ' ')} condition.
Treating physician documented standard protocol adherence.
{quality_note}

--- PROVIDER NOTES ---
Provider certifies that all services were medically necessary and rendered as described.
Provider ID: {row['provider_id']}
Claim submitted: {date_str}

--- CARE PLAN EXCERPT ---
Continued monitoring and follow-up recommended for {diagnosis.replace('_', ' ')}.
Patient instructed on medication compliance and lifestyle modifications.
Next appointment scheduled within 30-90 days depending on clinical response.
{caution_block}

--- SUBMISSION METADATA ---
Late Submission: {'Yes' if row.get('late_submission_flag', 0) else 'No'}
Duplicate Flag : {'Yes' if row.get('duplicate_claim_flag', 0) else 'No'}
Suspicious Keywords Found: {int(row.get('suspicious_keyword_count', 0))}
"""
    return doc


def generate_policy_rules():
    """Insurance policy rules document used as the RAG retrieval corpus."""
    return """INSURANCE POLICY RULES AND COMPLIANCE GUIDELINES
==================================================

1. FRAUD, WASTE, AND ABUSE (FWA) DEFINITIONS
   1.1 Fraud: Intentional deception or misrepresentation to obtain unauthorized benefit.
   1.2 Waste: Overutilization of services not caused by criminally negligent actions.
   1.3 Abuse: Actions inconsistent with sound fiscal, business, or medical practices.

2. CLAIM SUBMISSION REQUIREMENTS
   2.1 Claims must be submitted within 90 days of service date.
   2.2 All claims require valid diagnosis codes (ICD-10) and procedure codes (CPT/HCPCS).
   2.3 Duplicate claims for the same service, date, and beneficiary are prohibited.
   2.4 Claims must be supported by complete medical documentation.

3. BILLING GUIDELINES
   3.1 Unbundling: Billing separately for services that should be billed together is prohibited.
   3.2 Upcoding: Billing for a higher-intensity service than provided is prohibited.
   3.3 Phantom billing: Billing for services not rendered is prohibited.
   3.4 Kickbacks: Receiving remuneration for referrals is a federal offense.

4. DOCUMENTATION STANDARDS
   4.1 Medical records must support the level of service billed.
   4.2 Physician signatures required on all orders and certifications.
   4.3 Electronic health records must include audit trails.
   4.4 Incomplete or altered documentation triggers automatic claim review.

5. PROVIDER ELIGIBILITY
   5.1 Providers must maintain active licensure in the state of service.
   5.2 Excluded providers (OIG exclusion list) may not submit claims.
   5.3 Provider credentialing must be current and verified annually.

6. HIGH-RISK INDICATORS REQUIRING ENHANCED REVIEW
   6.1 Claim amount exceeds 200% of provider's average claim amount.
   6.2 More than 3 claims from same provider-beneficiary pair within 30 days.
   6.3 Services billed on weekends or holidays for non-emergency care.
   6.4 Specialty services billed by non-specialist providers.
   6.5 Pharmacy claims for quantities exceeding standard dosage protocols.
   6.6 Claims with documentation score below 0.4 (system-calculated).

7. ANOMALY THRESHOLDS
   7.1 Provider billing volume >2 standard deviations above peer group: requires review.
   7.2 Beneficiary with >15 claims per year: requires utilization management review.
   7.3 Late submissions (>90 days): require clinical justification.
   7.4 Approval rate <50%: triggers retrospective audit for provider.

8. ESCALATION PROCEDURES
   8.1 Risk score >= 0.6 (High): Immediate analyst review, potential payment suspension.
   8.2 Risk score 0.3-0.59 (Medium): Enhanced review within 5 business days.
   8.3 Risk score < 0.3 (Low): Standard processing, routine audit sampling.

9. APPEAL RIGHTS
   9.1 Providers may appeal denied or suspended claims within 60 days.
   9.2 Appeals must include additional supporting documentation.
   9.3 First-level appeal decided within 30 days; second-level within 60 days.

10. REGULATORY COMPLIANCE
    10.1 Claims processing subject to CMS guidelines (42 CFR Part 455).
    10.2 HIPAA privacy and security rules apply to all claim records.
    10.3 False Claims Act penalties apply to knowingly fraudulent submissions.
    10.4 State insurance department regulations supersede when more restrictive.

11. LONG TERM CARE (LTC) SPECIFIC GUIDELINES
    11.1 LTC claims must include a current plan of care signed by the attending physician.
    11.2 Activities of Daily Living (ADL) deficits must be documented by a licensed assessor.
    11.3 Home health agency claims require visit logs with caregiver signature.
    11.4 Cognitive impairment determinations require standardized assessment (e.g., MMSE/MoCA).
    11.5 Duplicate caregiver billing across overlapping shifts is a high-priority FWA pattern.
"""


def main():
    os.makedirs(config.DATA_RAW, exist_ok=True)
    os.makedirs(config.DATA_DOCUMENTS, exist_ok=True)

    print("Generating synthetic claims data (leakage-aware v2)...")
    df = generate_claims(n=config.N_CLAIMS, seed=config.RANDOM_SEED)
    out_path = os.path.join(config.DATA_RAW, "synthetic_claims.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} claims to {out_path}")
    print(f"  Fraud rate: {df['fraud_label'].mean():.2%}")

    print("Generating claim documents...")
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample_ids = df.sample(n=50, random_state=config.RANDOM_SEED)["claim_id"].tolist()

    for cid in sample_ids:
        row = df[df["claim_id"] == cid].iloc[0]
        doc_text = generate_claim_document(cid, row, rng)
        doc_path = os.path.join(config.DATA_DOCUMENTS, f"claim_{cid}.txt")
        with open(doc_path, "w") as f:
            f.write(doc_text)

    print(f"  Saved {len(sample_ids)} claim documents to {config.DATA_DOCUMENTS}/")

    rules_path = os.path.join(config.DATA_DOCUMENTS, "policy_rules.txt")
    with open(rules_path, "w") as f:
        f.write(generate_policy_rules())
    print(f"  Saved policy rules to {rules_path}")
    print("Data generation complete.")


if __name__ == "__main__":
    main()
