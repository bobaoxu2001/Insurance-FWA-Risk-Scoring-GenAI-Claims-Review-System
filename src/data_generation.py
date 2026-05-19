"""
Data generation module for synthetic insurance claims dataset.
Generates 5000 claims and ~50 claim documents for FWA risk scoring.
"""

import os
import sys
import numpy as np
import pandas as pd

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_claims(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    # --- Identifiers ---
    claim_ids = [f"CLM{str(i).zfill(6)}" for i in range(1, n + 1)]
    policyholder_ids = [f"PH{str(rng.integers(1, 2001)).zfill(5)}" for _ in range(n)]
    provider_ids = [f"PR{str(rng.integers(1, 301)).zfill(4)}" for _ in range(n)]

    # --- Dates ---
    base_date = pd.Timestamp("2022-01-01")
    days_offset = rng.integers(0, 730, size=n)
    claim_dates = [base_date + pd.Timedelta(days=int(d)) for d in days_offset]

    # --- Service & Diagnosis ---
    service_types = rng.choice(
        ["Inpatient", "Outpatient", "Pharmacy", "Mental Health", "Dental", "Vision", "Emergency"],
        size=n, p=[0.15, 0.30, 0.20, 0.10, 0.10, 0.08, 0.07]
    )
    diagnosis_groups = rng.choice(
        ["Musculoskeletal", "Cardiovascular", "Respiratory", "Mental_Health",
         "Diabetes", "Oncology", "Gastrointestinal", "Neurological", "Other"],
        size=n
    )

    # --- Provider stats (consistent per provider) ---
    provider_list = list(set(provider_ids))
    prov_vol = {p: int(rng.integers(10, 500)) for p in provider_list}
    prov_avg = {p: float(rng.uniform(500, 8000)) for p in provider_list}
    provider_claim_volume = np.array([prov_vol[p] for p in provider_ids])
    provider_avg_claim_amount = np.array([prov_avg[p] for p in provider_ids])

    # --- Claim financials ---
    # Base amount drawn from lognormal; fraudulent claims tend to be inflated later
    base_claim_amount = rng.lognormal(mean=7.5, sigma=1.0, size=n).clip(100, 150000)
    claimant_age = rng.integers(18, 85, size=n)

    states = rng.choice(
        ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"],
        size=n
    )

    # --- Policy & history ---
    days_since_policy_start = rng.integers(1, 3650, size=n)
    prior_claim_count = rng.integers(0, 20, size=n)

    # --- Document quality ---
    documentation_score = rng.uniform(0.0, 1.0, size=n)
    suspicious_keyword_count = rng.integers(0, 8, size=n)

    # Flags (binary)
    duplicate_claim_flag = (rng.random(n) < 0.04).astype(int)
    late_submission_flag = (rng.random(n) < 0.07).astype(int)

    # --- Build risk score (sigmoid-based) ---
    # Each factor contributes to a latent fraud risk score
    risk_score = np.zeros(n)

    # High claim vs provider average
    amount_ratio = base_claim_amount / (provider_avg_claim_amount + 1e-6)
    risk_score += 1.5 * np.clip(amount_ratio - 1.0, 0, 5)

    # Low documentation quality
    risk_score += 2.0 * (1.0 - documentation_score)

    # Duplicate claim
    risk_score += 3.0 * duplicate_claim_flag

    # Late submission
    risk_score += 1.0 * late_submission_flag

    # Suspicious keywords
    risk_score += 0.5 * suspicious_keyword_count

    # High prior claims
    risk_score += 0.1 * np.clip(prior_claim_count - 5, 0, 15)

    # New policy (more fraud in early policy period)
    risk_score += 0.5 * (days_since_policy_start < 180).astype(float)

    # High provider volume (mills)
    risk_score += 0.3 * np.clip((provider_claim_volume - 200) / 100, 0, 3)

    # Apply sigmoid to get probability
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Shift so base fraud rate ~ 8%
    fraud_prob = sigmoid(risk_score - 8.2)
    fraud_label = (rng.random(n) < fraud_prob).astype(int)

    # --- Inflate claim amount for fraudulent claims ---
    fraud_inflation = np.where(fraud_label == 1, rng.uniform(1.2, 3.5, size=n), 1.0)
    claim_amount = (base_claim_amount * fraud_inflation).round(2)

    # Approved amount: typically lower than claimed; fraud claims often get more challenged
    approval_ratio = np.where(
        fraud_label == 1,
        rng.uniform(0.3, 0.8, size=n),
        rng.uniform(0.7, 1.0, size=n)
    )
    approved_amount = (claim_amount * approval_ratio).round(2)

    # High-cost outlier flag
    high_cost_threshold = np.percentile(claim_amount, 90)
    high_cost_outlier_flag = (claim_amount > high_cost_threshold).astype(int)

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
        "high_cost_outlier_flag": high_cost_outlier_flag,
        "fraud_label": fraud_label,
    })

    return df


def generate_claim_document(claim_id, row, rng):
    """Generate a realistic synthetic claim text document."""
    service = row["service_type"]
    diagnosis = row["diagnosis_group"]
    amount = row["claim_amount"]
    age = row["claimant_age"]
    doc_score = row["documentation_score"]
    fraud = row["fraud_label"]

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

    fraud_note = ""
    if fraud == 1:
        fraud_note = (
            "\nCAUTION: Claim flagged for review. Duplicate billing codes detected. "
            "Provider has submitted similar claims within 30 days. "
            "Billed amount significantly exceeds provider average for this service."
        )

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
Claim submitted: {row['claim_date'].strftime('%Y-%m-%d') if hasattr(row['claim_date'], 'strftime') else row['claim_date']}

--- CARE PLAN EXCERPT ---
Continued monitoring and follow-up recommended for {diagnosis.replace('_', ' ')}.
Patient instructed on medication compliance and lifestyle modifications.
Next appointment scheduled within 30-90 days depending on clinical response.
{fraud_note}

--- SUBMISSION METADATA ---
Late Submission: {'Yes' if row['late_submission_flag'] else 'No'}
Duplicate Flag : {'Yes' if row['duplicate_claim_flag'] else 'No'}
Suspicious Keywords Found: {row['suspicious_keyword_count']}
"""
    return doc


def generate_policy_rules():
    """Generate a realistic insurance policy rules document."""
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
   8.2 Risk score 0.3–0.59 (Medium): Enhanced review within 5 business days.
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
"""


def main():
    os.makedirs(config.DATA_RAW, exist_ok=True)
    os.makedirs(config.DATA_DOCUMENTS, exist_ok=True)

    print("Generating synthetic claims data...")
    df = generate_claims(n=config.N_CLAIMS, seed=config.RANDOM_SEED)
    out_path = os.path.join(config.DATA_RAW, "synthetic_claims.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} claims to {out_path}")
    print(f"  Fraud rate: {df['fraud_label'].mean():.2%}")

    # Generate ~50 claim documents (sample from all claims)
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

    # Save policy rules
    rules_path = os.path.join(config.DATA_DOCUMENTS, "policy_rules.txt")
    with open(rules_path, "w") as f:
        f.write(generate_policy_rules())
    print(f"  Saved policy rules to {rules_path}")
    print("Data generation complete.")


if __name__ == "__main__":
    main()
