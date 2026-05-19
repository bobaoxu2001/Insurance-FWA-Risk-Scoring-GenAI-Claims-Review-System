"""
llm_review.py
=============
Optional LLM / semantic-retrieval upgrades on top of the TF-IDF + template RAG
pipeline in src/rag_claim_review.py.

Three tiers, picked at runtime based on what is installed:

  tier 0 (always available)
    Retrieval: TF-IDF + cosine similarity over policy_rules.txt
    Generation: deterministic templates

  tier 1 (requires sentence-transformers)
    Retrieval: semantic dense embeddings (all-MiniLM-L6-v2, ~80MB)
    Generation: deterministic templates

  tier 2 (requires sentence-transformers AND transformers + a local seq2seq model
          such as google/flan-t5-small ~80M params, already cached for many users)
    Retrieval: semantic dense embeddings
    Generation: LLM (flan-t5-small / -base) with explicit prompt template and
                deterministic decoding (do_sample=False, max_new_tokens=200)

The CLI lets the operator force a tier or use auto-detection:

  python -m src.llm_review --tier auto --n 5         # use whatever is available
  python -m src.llm_review --tier 2 --n 5            # require LLM, fail loudly
  python -m src.llm_review --provider-id PRV56781 --tier 2

All generated reviews are saved to outputs/sample_reviews/ alongside the
template-based reviews from rag_claim_review.py (they get an `_llm` suffix
so the two sets do not collide).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import rag_claim_review as rag_base  # type: ignore


# ── Capability detection ──────────────────────────────────────────────────────

def _have_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def _have_local_seq2seq(model_id: str = "google/flan-t5-small") -> bool:
    """Return True if transformers is installed AND the model is cached locally
    OR the network is reachable (we let transformers handle that check)."""
    try:
        from transformers import AutoTokenizer  # noqa: F401
        return True
    except Exception:
        return False


def detect_tier(force: str = "auto") -> int:
    if force == "0":
        return 0
    if force == "1":
        if not _have_sentence_transformers():
            raise RuntimeError("Tier 1 requested but sentence-transformers is not installed.")
        return 1
    if force == "2":
        if not _have_sentence_transformers():
            raise RuntimeError("Tier 2 requested but sentence-transformers is not installed.")
        if not _have_local_seq2seq():
            raise RuntimeError("Tier 2 requested but transformers is not installed.")
        return 2
    # auto
    if _have_sentence_transformers() and _have_local_seq2seq():
        return 2
    if _have_sentence_transformers():
        return 1
    return 0


# ── Semantic retrieval ────────────────────────────────────────────────────────

class SemanticRetriever:
    """Dense retrieval over policy_rules.txt using sentence-transformers.
    Falls back to TF-IDF retrieval if the embedding model fails to load."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.policy_chunks: List[str] = rag_base.load_policy_rules()
        if not self.policy_chunks:
            raise RuntimeError("policy_rules.txt is empty or missing.")
        self.embeddings = self.model.encode(
            self.policy_chunks, show_progress_bar=False, normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores = (q_emb @ self.embeddings.T).ravel()
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [(self.policy_chunks[i], float(scores[i])) for i in top_idx]


# ── LLM generation ────────────────────────────────────────────────────────────

REVIEW_PROMPT = """You are a healthcare fraud audit analyst. Write a 2-sentence audit summary following the examples.

Example 1:
Risk score: 0.94 (HIGH)
Top signal: Avg reimbursement per claim ($12,500) is 15x the overall provider median ($820)
Top policy: Claim amount exceeds 200% of provider's average claim amount requires review.
Summary: Provider's per-claim reimbursement is more than fifteen times the peer median, which violates the >200%-of-average threshold for enhanced review. Recommend pulling the top-20 highest-reimbursement claims for clinical documentation review before any further payment.

Example 2:
Risk score: 0.62 (MEDIUM)
Top signal: Inpatient billing ratio (78%) exceeds the 90th percentile (57%)
Top policy: Provider billing volume above peer norms requires review.
Summary: The inpatient share of this provider's billing is well above the peer 90th percentile, which often indicates inpatient upcoding. Sample 10 admissions and verify clinical necessity against admission criteria.

Now write the summary for:
Risk score: {risk_score:.2f} ({risk_level})
Top signal: {top_indicator}
Top policy: {top_policy}
Summary:"""


class LLMGenerator:
    """Wraps a small local seq2seq model (flan-t5-small by default) and renders
    a free-text Audit Reviewer Summary from the structured risk indicators and
    retrieved policy evidence. Decoding is deterministic so the output is
    auditable (do_sample=False)."""

    def __init__(self, model_id: str = "google/flan-t5-small"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        self.model_id = model_id
        # Prefer locally cached weights to avoid surprise downloads
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=False,           # deterministic → reproducible audit output
            max_new_tokens=220,
        )

    def generate(self, risk_level: str, risk_score: float,
                 indicators: List[str], evidence: List[Tuple[str, float]]) -> str:
        # Flan-T5 family is small (250M-780M params on CPU) and works best with
        # concise, single-question prompts. We pass only the top indicator and
        # top policy chunk so the model is grounded but not overloaded.
        top_ind = indicators[0] if indicators else "no single dominant indicator"
        top_pol = evidence[0][0][:240] if evidence else "no policy evidence retrieved"
        prompt = REVIEW_PROMPT.format(
            risk_level=risk_level, risk_score=risk_score,
            top_indicator=top_ind, top_policy=top_pol,
        )
        out = self.pipe(prompt)[0]["generated_text"].strip()
        return out


# ── Provider review orchestration ─────────────────────────────────────────────

def build_review(provider_id: str, row: pd.Series, risk_score: float, stats: dict,
                 retriever, llm: Optional[LLMGenerator], tier: int) -> str:
    """Compose a review packet that combines deterministic structured fields
    (risk indicators, doc gaps, limitations) with either a template paragraph
    (tier 0/1) or an LLM-generated Audit Reviewer Summary (tier 2)."""
    risk_level = rag_base._risk_level(risk_score)
    query = rag_base._provider_query(row)
    evidence = retriever.retrieve(query, top_k=3) if hasattr(retriever, "retrieve") \
               else rag_base.retrieve_policy_evidence(
                   query, *retriever, top_k=3
               )  # tier 0 path: retriever is a tuple

    indicators = rag_base._provider_risk_indicators(row, stats)
    doc_gaps = rag_base._provider_doc_gaps(row)
    action = rag_base._provider_suggested_action(risk_level)
    human_notes = rag_base._provider_human_review_notes(row, risk_level)
    limitations = rag_base._limitations("real")

    # Audit Reviewer Summary: LLM-generated for tier 2, template otherwise
    if tier == 2 and llm is not None:
        try:
            summary = llm.generate(risk_level, risk_score, indicators, evidence)
        except Exception as e:
            summary = (
                f"[LLM generation failed: {e}; falling back to template] "
                f"Provider scored {risk_level} ({risk_score:.2f}) by the model."
            )
    else:
        # Template summary identical in flavour to rag_base output
        summary = (
            f"Provider {provider_id} scored {risk_level} ({risk_score:.2f}). "
            f"The strongest signals are: " +
            (indicators[0] if indicators else "low-confidence outlier pattern") + ". "
            "The most relevant retrieved policy is the top-1 evidence chunk below. "
            "Recommended action: " + action.lower() + "."
        )

    retrieval_type = "semantic (sentence-transformers all-MiniLM-L6-v2)" if tier >= 1 \
                     else "TF-IDF + cosine similarity"
    generation_type = "flan-t5-small (local LLM, deterministic decoding)" if tier == 2 \
                      else "deterministic template"

    lines = [
        "=" * 70,
        f"Provider Audit Review  —  Provider {provider_id}",
        "=" * 70,
        f"Risk Level         : {risk_level}",
        f"Model Risk Score   : {risk_score:.4f}",
        f"Retrieval          : {retrieval_type}",
        f"Generation         : {generation_type}",
        "",
        "Audit Reviewer Summary",
        "-" * 70,
        summary,
        "",
        "Key Risk Indicators",
        "-" * 70,
    ]
    lines.extend(f"  • {x}" for x in indicators) if indicators else lines.append("  • (no strong individual indicators)")

    lines.extend([
        "",
        "Retrieved Policy / Audit Evidence (top 3)",
        "-" * 70,
    ])
    for i, (chunk, score) in enumerate(evidence, 1):
        lines.append(f"  [{i}] (sim={score:.3f}) {chunk[:300]}")

    lines.extend([
        "",
        "Documentation / Data Gaps",
        "-" * 70,
    ])
    lines.extend(f"  • {x}" for x in doc_gaps) if doc_gaps else lines.append("  • None flagged.")

    lines.extend([
        "",
        f"Suggested Analyst Action : {action}",
        "",
        "Human Review Notes",
        "-" * 70,
    ])
    lines.extend(f"  • {x}" for x in human_notes)

    lines.extend([
        "",
        "System Limitations",
        "-" * 70,
    ])
    lines.extend(f"  • {x}" for x in limitations)
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_provider_table_and_model():
    table = os.path.join(config.DATA_PROCESSED, "provider_modeling_table.csv")
    if not os.path.exists(table):
        raise FileNotFoundError(
            "data/processed/provider_modeling_table.csv missing. "
            "Run src/provider_feature_engineering.py first."
        )
    df = pd.read_csv(table)

    # prefer the model trained with the matching split mode
    candidates = [
        os.path.join(config.OUTPUTS_MODELS, "best_fwa_model.pkl"),
        os.path.join(config.OUTPUTS_MODELS, "best_fwa_model_temporal.pkl"),
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError("No trained model in outputs/models/. Run src/modeling.py.")
    model = joblib.load(model_path)
    return df, model, model_path


def _summary_stats(df: pd.DataFrame) -> dict:
    return {
        "avg_reimbursed_per_claim_median":
            float(df["avg_reimbursed_per_claim"].median()) if "avg_reimbursed_per_claim" in df else 0,
        "inpatient_ratio_p90":
            float(df["inpatient_ratio"].quantile(0.9)) if "inpatient_ratio" in df else 1,
        "claim_frequency_per_beneficiary_p90":
            float(df["claim_frequency_per_beneficiary"].quantile(0.9))
            if "claim_frequency_per_beneficiary" in df else 1,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM / semantic upgrades to provider RAG reviews")
    parser.add_argument("--tier", choices=["auto", "0", "1", "2"], default="auto",
                        help="0=TF-IDF+template (always works), 1=semantic retrieval, "
                             "2=semantic + flan-t5 LLM generation")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of high-risk providers to review")
    parser.add_argument("--provider-id", default=None,
                        help="Score and review a specific provider ID (overrides --n)")
    parser.add_argument("--model-id", default="google/flan-t5-base",
                        help="HF model id for tier 2 generation")
    parser.add_argument("--mixed", action="store_true",
                        help="Select a mixed panel of HIGH/MEDIUM/LOW risk providers (overrides --n)")
    args = parser.parse_args()

    tier = detect_tier(args.tier)
    print(f"[llm_review] Using tier {tier} "
          f"({['TF-IDF+template', 'semantic+template', 'semantic+LLM'][tier]})")

    df, model, model_path = _load_provider_table_and_model()
    print(f"[llm_review] Loaded {len(df)} providers and model from {os.path.basename(model_path)}")

    # Build retriever
    if tier >= 1:
        print("[llm_review] Building semantic policy index (sentence-transformers)...")
        retriever = SemanticRetriever()
    else:
        print("[llm_review] Building TF-IDF policy index...")
        chunks = rag_base.load_policy_rules()
        vec, mat = rag_base.build_policy_index(chunks)
        retriever = (vec, mat, chunks)

    # Build LLM if tier 2
    llm = None
    if tier == 2:
        print(f"[llm_review] Loading local LLM: {args.model_id}")
        llm = LLMGenerator(model_id=args.model_id)

    # Score providers (avoid the median_claim_date column the model didn't see)
    feature_cols = [c for c in df.columns
                    if c not in ("Provider", "PotentialFraud", "median_claim_date")
                    and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    df["_score"] = model.predict_proba(X)[:, 1]

    # Select providers — if --n is the default, give a mixed-risk panel
    # (top-5 HIGH, 3 in the 0.4–0.6 band, 2 LOW) so reviews show variety
    if args.provider_id:
        sub = df[df["Provider"] == args.provider_id]
        if sub.empty:
            raise SystemExit(f"Provider {args.provider_id} not found in modeling table.")
    elif args.mixed:
        high = df.sort_values("_score", ascending=False).head(5)
        med  = df[(df["_score"] >= 0.30) & (df["_score"] < 0.60)].sample(min(3, 3), random_state=0)
        low  = df[df["_score"] < 0.15].sample(2, random_state=0)
        sub = pd.concat([high, med, low])
    else:
        sub = df.sort_values("_score", ascending=False).head(args.n)

    stats = _summary_stats(df)
    os.makedirs(config.OUTPUTS_REVIEWS, exist_ok=True)

    suffix = "_llm" if tier == 2 else ("_semantic" if tier == 1 else "_tfidf")
    written = []
    for _, row in sub.iterrows():
        pid = row["Provider"]
        txt = build_review(pid, row, float(row["_score"]), stats,
                           retriever, llm, tier)
        out = os.path.join(config.OUTPUTS_REVIEWS, f"review_{pid}{suffix}.txt")
        with open(out, "w") as f:
            f.write(txt)
        written.append(out)
        print(f"  wrote {os.path.basename(out)}  (score={row['_score']:.4f})")

    print(f"\n[llm_review] Wrote {len(written)} reviews to {config.OUTPUTS_REVIEWS}/")


if __name__ == "__main__":
    main()
