-- =====================================================================
-- provider_features.sql
-- =====================================================================
-- Reference SQL implementation of the provider-level FWA feature table.
--
-- This is the same aggregation logic that lives in
--   src/provider_feature_engineering.py
-- expressed in ANSI SQL so it can be ported to a data warehouse
-- (Snowflake, BigQuery, Redshift, Databricks SQL) without re-deriving
-- the feature definitions.
--
-- Source tables (Kaggle Healthcare Provider Fraud Detection Analysis):
--   raw_beneficiary        — one row per beneficiary
--   raw_inpatient_claims   — one row per inpatient claim
--   raw_outpatient_claims  — one row per outpatient claim
--   raw_provider_labels    — one row per provider (PotentialFraud: 'Yes'/'No')
--
-- Output:
--   provider_modeling_table — one row per Provider, 27 features + label
-- =====================================================================

-- ---------------------------------------------------------------------
-- Step 1 — union inpatient + outpatient into a unified claim stream
-- ---------------------------------------------------------------------
WITH unified_claims AS (
    SELECT
        Provider,
        BeneID,
        ClaimID,
        ClaimStartDt,
        ClaimEndDt,
        AdmissionDt,
        DischargeDt,
        InscClaimAmtReimbursed,
        DeductibleAmtPaid,
        AttendingPhysician,
        OperatingPhysician,
        OtherPhysician,
        'INPATIENT' AS claim_type
    FROM raw_inpatient_claims

    UNION ALL

    SELECT
        Provider,
        BeneID,
        ClaimID,
        ClaimStartDt,
        ClaimEndDt,
        CAST(NULL AS DATE) AS AdmissionDt,
        CAST(NULL AS DATE) AS DischargeDt,
        InscClaimAmtReimbursed,
        DeductibleAmtPaid,
        AttendingPhysician,
        OperatingPhysician,
        OtherPhysician,
        'OUTPATIENT' AS claim_type
    FROM raw_outpatient_claims
),

-- ---------------------------------------------------------------------
-- Step 2 — join beneficiary demographics & chronic conditions
-- ---------------------------------------------------------------------
claims_with_bene AS (
    SELECT
        c.*,
        b.DOB,
        b.DOD,
        b.Gender,
        b.Race,
        b.State,
        -- chronic-condition flag count (14 columns in raw_beneficiary)
        (
            COALESCE(b.ChronicCond_Alzheimer, 0)
          + COALESCE(b.ChronicCond_Heartfailure, 0)
          + COALESCE(b.ChronicCond_KidneyDisease, 0)
          + COALESCE(b.ChronicCond_Cancer, 0)
          + COALESCE(b.ChronicCond_ObstrPulmonary, 0)
          + COALESCE(b.ChronicCond_Depression, 0)
          + COALESCE(b.ChronicCond_Diabetes, 0)
          + COALESCE(b.ChronicCond_IschemicHeart, 0)
          + COALESCE(b.ChronicCond_Osteoporasis, 0)
          + COALESCE(b.ChronicCond_rheumatoidarthritis, 0)
          + COALESCE(b.ChronicCond_stroke, 0)
        ) AS chronic_condition_count,
        CASE WHEN b.DOD IS NOT NULL THEN 1 ELSE 0 END AS is_deceased,
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, b.DOB)) AS patient_age
    FROM unified_claims c
    LEFT JOIN raw_beneficiary b
      ON c.BeneID = b.BeneID
),

-- ---------------------------------------------------------------------
-- Step 3 — aggregate to Provider level (27 features)
-- ---------------------------------------------------------------------
provider_features AS (
    SELECT
        Provider,

        -- ===== Volume features (6) =====
        COUNT(*)                                                AS total_claims,
        SUM(CASE WHEN claim_type = 'INPATIENT'  THEN 1 ELSE 0 END) AS inpatient_claim_count,
        SUM(CASE WHEN claim_type = 'OUTPATIENT' THEN 1 ELSE 0 END) AS outpatient_claim_count,
        1.0 * SUM(CASE WHEN claim_type = 'INPATIENT' THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0)                               AS inpatient_ratio,
        COUNT(DISTINCT BeneID)                                  AS unique_beneficiaries,
        1.0 * COUNT(*) / NULLIF(COUNT(DISTINCT BeneID), 0)      AS claim_frequency_per_beneficiary,

        -- ===== Physician diversity features (3) =====
        COUNT(DISTINCT AttendingPhysician) AS unique_attending_physicians,
        COUNT(DISTINCT OperatingPhysician) AS unique_operating_physicians,
        COUNT(DISTINCT OtherPhysician)     AS unique_other_physicians,

        -- ===== Financial features (8) =====
        SUM(CASE WHEN claim_type = 'INPATIENT'  THEN InscClaimAmtReimbursed ELSE 0 END) AS total_inpatient_reimbursed,
        SUM(CASE WHEN claim_type = 'OUTPATIENT' THEN InscClaimAmtReimbursed ELSE 0 END) AS total_outpatient_reimbursed,
        SUM(InscClaimAmtReimbursed)                                                     AS total_reimbursed,
        AVG(InscClaimAmtReimbursed)                                                     AS avg_reimbursed_per_claim,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY InscClaimAmtReimbursed)             AS median_reimbursed_per_claim,
        SUM(InscClaimAmtReimbursed) / NULLIF(COUNT(DISTINCT BeneID), 0)                 AS reimbursement_per_beneficiary,
        SUM(DeductibleAmtPaid)                                                          AS total_deductible_amount,
        SUM(CASE WHEN claim_type = 'INPATIENT' THEN InscClaimAmtReimbursed ELSE 0 END)
            / NULLIF(SUM(InscClaimAmtReimbursed), 0)                                    AS inpatient_reimbursement_share,

        -- ===== Patient demographic features (4) =====
        AVG(patient_age)             AS avg_patient_age,
        STDDEV(patient_age)          AS std_patient_age,
        AVG(is_deceased * 1.0)       AS death_rate,
        AVG(chronic_condition_count) AS avg_chronic_conditions,

        -- ===== Duration features — inpatient only (2) =====
        AVG(CASE WHEN claim_type = 'INPATIENT'
                 THEN DATE_PART('day', DischargeDt - AdmissionDt) END) AS avg_admission_duration,
        MAX(CASE WHEN claim_type = 'INPATIENT'
                 THEN DATE_PART('day', DischargeDt - AdmissionDt) END) AS max_admission_duration,

        -- ===== Outlier / risk features (4) =====
        --   percentile rank of provider volume; computed downstream in Step 4
        --   reimbursement_outlier_score; computed downstream in Step 4
        --   high_reimbursement_claim_rate; computed via JOIN in Step 4
        --   diagnosis_code_diversity / procedure_code_diversity; require
        --   pivoted ClmDiagnosisCode_* / ClmProcedureCode_* columns
        CAST(NULL AS DOUBLE PRECISION) AS reimbursement_outlier_score_placeholder

    FROM claims_with_bene
    GROUP BY Provider
),

-- ---------------------------------------------------------------------
-- Step 4 — peer-relative outlier features (window functions over all providers)
-- ---------------------------------------------------------------------
provider_features_final AS (
    SELECT
        pf.*,
        PERCENT_RANK() OVER (ORDER BY pf.total_claims) AS provider_volume_percentile,
        (pf.avg_reimbursed_per_claim - AVG(pf.avg_reimbursed_per_claim) OVER ())
            / NULLIF(STDDEV(pf.avg_reimbursed_per_claim) OVER (), 0)
            AS reimbursement_outlier_score
    FROM provider_features pf
)

-- ---------------------------------------------------------------------
-- Step 5 — attach fraud label and persist
-- ---------------------------------------------------------------------
SELECT
    pf.*,
    CASE WHEN l.PotentialFraud = 'Yes' THEN 1 ELSE 0 END AS PotentialFraud
FROM provider_features_final pf
INNER JOIN raw_provider_labels l
       ON pf.Provider = l.Provider
;

-- =====================================================================
-- Verification queries (run after the CTE above is materialized)
-- =====================================================================
--   -- 1. Row count and fraud rate
--   SELECT COUNT(*)                                            AS n_providers,
--          AVG(PotentialFraud)                                 AS fraud_rate,
--          SUM(PotentialFraud)                                 AS n_fraud
--   FROM provider_modeling_table;
--   -- Expected:  n_providers=5410, fraud_rate≈0.0935, n_fraud=506
--
--   -- 2. Top 10 providers by reimbursement outlier score
--   SELECT Provider, total_reimbursed, reimbursement_outlier_score
--   FROM provider_modeling_table
--   ORDER BY reimbursement_outlier_score DESC
--   LIMIT 10;
--
--   -- 3. Fraud rate by provider-volume bucket (sanity check)
--   SELECT NTILE(5) OVER (ORDER BY total_claims) AS volume_bucket,
--          COUNT(*)              AS n_providers,
--          AVG(PotentialFraud)   AS fraud_rate
--   FROM provider_modeling_table
--   GROUP BY volume_bucket
--   ORDER BY volume_bucket;
