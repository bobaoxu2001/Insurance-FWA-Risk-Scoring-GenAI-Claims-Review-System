# Resume & Pitch Material — Insurance FWA Risk Scoring & GenAI Claims Review

Tailored to the **Associate Data Scientist, Long Term Care FWA Advanced
Analytics** role at Manulife / John Hancock.

---

## Short version (3 bullets — resume)

- Built an end-to-end **Long Term Care FWA risk-scoring pipeline** on 5,000
  synthetic claims: leakage-aware data generation (hidden intent driver +
  noisy proxies), feature engineering, and three supervised models
  (LR / RF / GB) plus an Isolation Forest anomaly detector — held-out
  **ROC-AUC 0.82, F1 0.49** with a published precision/recall threshold sweep.
- Shipped a **TF-IDF RAG claim-review module** that retrieves policy-rule
  evidence and renders analyst-ready packets covering risk indicators,
  documentation gaps, suggested action, human-review notes, and limitations
  — fully deterministic, no external API.
- Added a **model-monitoring module** (monthly fraud rate, claim-amount
  drift, column-level data quality) and a **6-tab Streamlit dashboard**
  (executive, pattern explorer, model performance, claim review,
  monitoring, auditability) — human-in-the-loop framing throughout.

---

## Long version (4 bullets — detailed resume / portfolio)

- Designed and built a complete **Fraud, Waste & Abuse analytics pipeline**
  on a synthetic Long Term Care-style claims dataset (5,000 claims):
  data generation with a **hidden 'intent' latent driver** (provider
  integrity + policyholder propensity) so the label is not trivially
  recoverable from observable features, plus engineered features
  (claim-vs-provider ratios, documentation flags, new-policy flag) and a
  transparent rule-based baseline score.
- Trained and evaluated **Logistic Regression, Random Forest, and Gradient
  Boosting** classifiers with `class_weight='balanced'` and an
  **Isolation Forest** anomaly detector; reported ROC, **precision-recall
  curve**, and a full threshold sweep (precision / recall / F1 / n_flagged
  for thresholds 0.05-0.95) so operations can pick the operating point that
  matches reviewer capacity. Held-out **ROC-AUC 0.82, F1 0.49**, recall up
  to 0.70 at lower thresholds.
- Built a **TF-IDF RAG claim-review module** over an insurance
  policy-rules corpus that retrieves the most relevant policy chunks for
  each flagged claim and renders an **analyst-ready review packet**
  (claim ID, risk level, model score, 3-5 quantified risk indicators,
  retrieved policy evidence, documentation gaps, suggested analyst action,
  human-review notes, and synthetic-data limitations) — deterministic,
  reproducible, with an upgrade path to dense embeddings + LLM with
  citation guardrails.
- Productionized the pipeline with a **monitoring module** (monthly fraud
  rate, mean / P95 claim-amount drift, column-level data quality) and a
  **6-tab Streamlit dashboard** covering executive overview, FWA pattern
  explorer, model performance (with PR / threshold tables), claim-review
  assistant, monitoring, and auditability — human-in-the-loop framing,
  synthetic-data banners, and a written Responsible-AI section.

---

## LinkedIn project description (~100 words)

> Built an end-to-end Insurance FWA Risk Scoring & GenAI Claims Review
> System aligned to the Long Term Care FWA Advanced Analytics role at
> Manulife / John Hancock. The pipeline runs on 5,000 synthetic claims
> generated with a deliberately hidden intent variable (so the model can't
> trivially memorize the label), and combines supervised classifiers
> (LR / RF / GB; ROC-AUC ~0.82, F1 ~0.49), an Isolation Forest, and a
> TF-IDF RAG layer over a policy-rules corpus that produces analyst-ready
> review packets. Adds monthly model & data-quality monitoring, a full
> precision/recall threshold sweep, and a six-tab Streamlit dashboard
> covering executives, analysts, ML, monitoring, and auditability.

---

## GitHub pinned-repo description (<120 chars)

> End-to-end LTC FWA pipeline: leakage-aware synthetic claims, ML + Isolation Forest, TF-IDF RAG review, monitoring, Streamlit.

---

## 30-second interview pitch

> "I built an LTC-flavored FWA project to look like the kind of work the
> John Hancock team actually does — claims data, supervised models,
> anomaly detection, a RAG claim-review layer, and monitoring. The
> interesting part is the data: I hide the dominant fraud driver behind a
> latent intent variable, so the model has to learn from *noisy proxies*
> the way it would in production. That brings the metrics into a realistic
> ROC-AUC ~0.82 / F1 ~0.49 range, and the dashboard exposes the full
> threshold sweep so operations can pick the precision/recall point that
> matches reviewer capacity."

---

## 2-minute interview pitch

> "The project is an end-to-end FWA pipeline framed against the Long Term
> Care FWA Advanced Analytics role at John Hancock. I generate ~5,000
> synthetic claims, but the data-generating process intentionally hides
> the dominant fraud driver — a latent 'intent' variable shaped by
> provider integrity and policyholder propensity. The observable features
> are noisy proxies of that hidden intent: documentation completeness,
> suspicious keyword counts, duplicate-billing flags. That design avoids
> the target-leakage trap that gives synthetic FWA projects unrealistic
> 0.99 AUCs — an earlier version of this project landed at AUC 0.99 / F1
> 0.97, and I rewrote the data generator after diagnosing exactly which
> features were leaking.
>
> On top of the data, I train Logistic Regression, Random Forest, and
> Gradient Boosting, evaluate them with ROC, PR, and a full threshold
> sweep, and add an Isolation Forest for unsupervised anomalies. For
> explainability I use SHAP-or-fallback feature importance plus per-claim
> rule-style explanations. The GenAI piece is a TF-IDF RAG layer that
> retrieves policy-rule chunks and renders an analyst-ready review packet
> covering risk indicators, retrieved evidence, documentation gaps,
> suggested action, what a human should verify, and limitations.
>
> Beyond modeling, I built a monitoring module — monthly fraud rate,
> dollar drift, column-level data quality — and a six-tab Streamlit
> dashboard with executive, pattern-explorer, model-performance,
> claim-review, monitoring, and auditability views.
>
> The headline numbers are deliberately realistic: AUC around 0.82, F1
> around 0.49. The bigger story is the engineering discipline: leakage
> diagnosis, human-in-the-loop framing, monitoring from day one, and
> auditable documentation throughout — exactly the muscles an LTC FWA
> team needs."
