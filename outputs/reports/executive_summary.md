# Executive Summary — Provider FWA Risk Scoring Pilot

**Audience:** FWA program leadership, SIU manager, compliance partner
**Prepared by:** Data Science (portfolio project)
**Status:** Pilot complete on public reference dataset; ready for adaptation to internal claims

---

## 1. What we built

A provider-level fraud-waste-abuse (FWA) risk scoring workflow that:

1. Aggregates inpatient + outpatient claims with beneficiary demographics to produce one risk-ready row per billing provider.
2. Scores every provider on a 0–1 fraud-risk scale using an ensemble of supervised models plus an unsupervised anomaly baseline.
3. Surfaces the top-risk providers to a human analyst inside a structured review packet that cites the relevant audit-policy language.

## 2. Headline numbers (public reference dataset)

| Metric | Value | What it means in business terms |
|---|---|---|
| Providers analyzed | 5,410 | Full Kaggle Train split |
| Underlying claims | 558,211 | Inpatient + outpatient combined |
| Base fraud rate | 9.35% | 506 fraudulent providers in labels |
| Best model | Random Forest | Outperforms LR and GB on F1 |
| ROC-AUC | **0.9535** | Strong ranking ability across thresholds |
| F1 at default threshold | **0.6927** | Balanced precision/recall summary |
| Recall at default threshold | **0.7030** | Catches ~70% of labeled fraud |
| Precision at default threshold | **0.6828** | ~68% of flagged providers are true positives |

> **Important caveat for leadership:** these metrics come from a public educational dataset, not real Manulife / John Hancock LTC claims. They demonstrate the analytical workflow, not production performance.

## 3. Why this matters for LTC FWA

- **Reviewer leverage.** A risk-ranked queue concentrates analyst time on the providers most likely to warrant action. At ~70% recall and ~68% precision, every 10 reviewers manually queued returns ~7 true positives instead of the base-rate ~1.
- **Auditability.** Every HIGH flag ships with quantified risk indicators, retrieved policy citations, and a documented limitations section. This supports regulator review and human-in-the-loop sign-off.
- **Recall-vs-precision is an operational dial.** The threshold sweep table lets the FWA program tune the queue to current reviewer capacity instead of accepting a single fixed threshold.

## 4. Top fraud signals (Random Forest importance)

| Rank | Signal | Why it's actionable |
|---|---|---|
| 1 | `max_admission_duration` | Outlier inpatient stays — possible upcoding or unnecessary admission |
| 2 | `total_reimbursed` | Total $ at risk; primary triage axis |
| 3 | `total_deductible_amount` | Cost-sharing pattern outliers |
| 4 | `total_inpatient_reimbursed` | Inpatient $ concentration |
| 5 | `inpatient_claim_count` | Inpatient billing volume relative to peers |

## 5. What we'd need to move this into production

1. **Real LTC claims access** — current model is trained on Medicare-style data; LTC has distinct service codes, care plans, and reimbursement structures.
2. **Care-plan & clinical-note NLP** — the RAG layer here uses synthetic policy text; production deployment would index real audit policies and clinical documentation.
3. **Provider-network graph features** — shared beneficiaries and physician referral patterns to catch coordinated schemes.
4. **Temporal monitoring** — rolling windows on score distribution, feature drift (PSI), and labeled-disposition feedback.
5. **Fairness & compliance review** — disparate-impact testing across protected attributes before any live deployment.
6. **Case-management integration** — push HIGH flags directly into the SIU case queue with chain-of-custody.

## 6. Risks & limitations (transparent for leadership)

- Public dataset has a higher base fraud rate (9.35%) than typical real-world prevalence; reported AUC likely overstates production performance.
- Labels are provider-level binary flags; real LTC fraud is often a graded continuum and frequently provider+claim-set joint.
- Model is supervised — it can only catch patterns similar to historically labeled fraud. The Isolation Forest baseline partially mitigates this.
- No automated payment hold is recommended at any score level; every action remains analyst-driven.

## 7. Next steps

| Owner | Action | Timeframe |
|---|---|---|
| Data Eng | Provision a sample of internal LTC provider claims for re-training | 2–4 weeks |
| Data Science | Re-train + recalibrate on internal data; produce updated threshold sweep | 2 weeks after data |
| SIU manager | Pilot the HIGH-flag review queue with 2 analysts for one cycle | 1 month |
| Compliance | Review fairness audit and document retention requirements | Parallel to pilot |
| Data Science | Add PSI-based drift monitoring + reviewer-feedback loop | 1 month after pilot |
