"""
graph_features.py
=================
Provider-beneficiary and provider-physician network features.

Why graph features for FWA?
---------------------------
FWA schemes are often collusive: a small group of providers share the same
beneficiaries (kickbacks, identity fraud) or share the same attending physicians
(billing under multiple NPIs to obscure pattern). Pure provider-level features
miss these — a single provider's claim history can look perfectly normal while
the *network* shows a collusion ring.

We build two bipartite graphs:
  (1) Provider ↔ Beneficiary  (edge if any claim links them)
  (2) Provider ↔ AttendingPhysician

And compute per-provider features that summarize the provider's position in
those graphs.

Features added to the provider table
------------------------------------
beneficiary_sharing_rate
    Fraction of this provider's beneficiaries who also see at least one other
    provider in the dataset. Low (<0.3) is unusual for primary care (loyal
    patient panel ≈ normal); very high (>0.95) may indicate an unusually
    shared patient pool (e.g. a clinic colluding with a network).

avg_co_provider_count
    Average number of *other* providers that this provider's beneficiaries also
    see. High values mean the panel is highly cross-referred; very high can
    indicate referral mills or beneficiary identity reuse.

physician_sharing_rate
    Fraction of this provider's attending physicians who also bill under at
    least one other provider. Physicians who appear under multiple Provider
    NPIs are a classic upcoding / NPI-laundering signal.

provider_clustering_coefficient
    Local clustering coefficient of the provider in the projected provider-
    provider graph (two providers are connected if they share ≥1 beneficiary).
    High clustering means this provider is part of a tightly-connected cluster
    of other providers — a collusion-ring signal.

provider_pagerank
    PageRank on the projected provider-provider graph weighted by shared-
    beneficiary count. Centrality in the network of providers who share
    patients.

Output
------
The original provider modeling table is enriched with these 5 columns and
re-saved (in place), so downstream modeling / monitoring / fairness reads the
graph-augmented table automatically.

CLI
---
    python src/graph_features.py            # add features in-place
    python src/graph_features.py --report   # just print the new feature summary
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _find(pattern_substrings):
    raw = Path(config.DATA_RAW)
    for p in sorted(raw.glob("*.csv")):
        name = p.name.lower()
        if all(sub in name for sub in pattern_substrings):
            return p
    return None


def load_claims_for_graph():
    """Concatenate IP + OP claims keeping the four columns we need:
    Provider, BeneID, AttendingPhysician, ClaimID."""
    ip_path = _find(["inpatient", "train"])
    op_path = _find(["outpatient", "train"])
    if not (ip_path and op_path):
        raise FileNotFoundError("Could not locate Train_Inpatient/Outpatient CSVs in data/raw/")

    keep = ["Provider", "BeneID", "AttendingPhysician", "ClaimID"]
    ip = pd.read_csv(ip_path, usecols=keep)
    op = pd.read_csv(op_path, usecols=keep)
    claims = pd.concat([ip, op], ignore_index=True)
    print(f"  Loaded {len(claims):,} claims for graph construction")
    return claims


# ── Feature 1+2: provider ↔ beneficiary sharing ───────────────────────────────

def beneficiary_sharing_features(claims: pd.DataFrame) -> pd.DataFrame:
    """For every (provider, beneficiary) pair, count how many distinct other
    providers also served that beneficiary. Then aggregate per provider:
    - beneficiary_sharing_rate = mean(other_provider_count > 0)
    - avg_co_provider_count    = mean(other_provider_count)"""
    print("  Computing beneficiary sharing rates...")

    # For each beneficiary, number of distinct providers
    bene_provider_count = (claims.groupby("BeneID")["Provider"].nunique()
                                  .rename("n_providers_for_bene"))

    # Join back to each (provider, beneficiary) link; subtract 1 to exclude self
    provider_bene = claims[["Provider", "BeneID"]].drop_duplicates()
    provider_bene = provider_bene.merge(bene_provider_count, on="BeneID", how="left")
    provider_bene["co_providers"] = provider_bene["n_providers_for_bene"] - 1
    provider_bene["is_shared"]    = (provider_bene["co_providers"] > 0).astype(int)

    out = provider_bene.groupby("Provider").agg(
        beneficiary_sharing_rate=("is_shared",   "mean"),
        avg_co_provider_count   =("co_providers", "mean"),
    )
    return out.round(4)


# ── Feature 3: provider ↔ physician sharing ───────────────────────────────────

def physician_sharing_features(claims: pd.DataFrame) -> pd.DataFrame:
    """Fraction of a provider's attending physicians who also bill under at
    least one *other* provider."""
    print("  Computing attending-physician sharing rates...")

    df = claims[["Provider", "AttendingPhysician"]].dropna().drop_duplicates()

    # Number of distinct providers per physician
    phys_provider_count = (df.groupby("AttendingPhysician")["Provider"].nunique()
                              .rename("n_providers_for_phys"))

    df = df.merge(phys_provider_count, on="AttendingPhysician", how="left")
    df["phys_is_shared"] = (df["n_providers_for_phys"] > 1).astype(int)

    out = df.groupby("Provider").agg(
        physician_sharing_rate=("phys_is_shared", "mean")
    )
    return out.round(4)


# ── Features 4+5: projected provider-provider graph ───────────────────────────

def projected_graph_features(claims: pd.DataFrame) -> pd.DataFrame:
    """Project the bipartite provider-beneficiary graph onto provider-provider:
    edge between two providers if they share ≥1 beneficiary, weighted by count.
    Compute local clustering coefficient and PageRank per provider."""
    import networkx as nx

    print("  Building bipartite provider-beneficiary graph...")
    # Provider-provider co-occurrence on beneficiaries (sparse-friendly approach)
    pb = claims[["Provider", "BeneID"]].drop_duplicates()
    # All provider pairs that share at least one beneficiary:
    #   self-join on BeneID, drop trivial pairs (p==q, deduplicate)
    pairs = pb.merge(pb, on="BeneID")
    pairs = pairs[pairs["Provider_x"] < pairs["Provider_y"]]
    edge_weights = (pairs.groupby(["Provider_x", "Provider_y"]).size()
                          .rename("w").reset_index())
    print(f"  Provider-provider edges: {len(edge_weights):,}")

    G = nx.Graph()
    # Add all providers (even isolated ones) so the index is complete
    for p in pb["Provider"].unique():
        G.add_node(p)
    G.add_weighted_edges_from(edge_weights.itertuples(index=False, name=None))

    print(f"  Graph: {G.number_of_nodes():,} providers, "
          f"{G.number_of_edges():,} co-beneficiary edges")

    print("  Computing local clustering coefficient...")
    clustering = nx.clustering(G, weight="w")

    print("  Computing PageRank (may take ~30s)...")
    pagerank = nx.pagerank(G, weight="w", max_iter=100, tol=1e-4)

    out = pd.DataFrame({
        "provider_clustering_coefficient": clustering,
        "provider_pagerank":               pagerank,
    }).rename_axis("Provider")
    return out.round(6)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_graph_features() -> pd.DataFrame:
    claims = load_claims_for_graph()
    bene_share = beneficiary_sharing_features(claims)
    phys_share = physician_sharing_features(claims)
    graph_topo = projected_graph_features(claims)
    feats = bene_share.join(phys_share, how="outer").join(graph_topo, how="outer")
    feats = feats.reset_index().rename(columns={"index": "Provider"})
    return feats


def merge_into_provider_table(graph_feats: pd.DataFrame) -> Path:
    table_path = Path(config.DATA_PROCESSED) / "provider_modeling_table.csv"
    if not table_path.is_file():
        raise FileNotFoundError("provider_modeling_table.csv missing — run provider_feature_engineering first")
    df = pd.read_csv(table_path)
    n_before = df.shape[1]

    # Drop already-merged versions of graph cols so this script is idempotent
    drop_cols = [c for c in graph_feats.columns if c != "Provider" and c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  Dropped existing graph cols before re-merge: {drop_cols}")

    df = df.merge(graph_feats, on="Provider", how="left")
    df.to_csv(table_path, index=False)
    print(f"  Saved enriched table ({n_before} → {df.shape[1]} cols) to {table_path}")
    return table_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true",
                        help="Print summary stats of new features without saving")
    args = parser.parse_args()

    print("Building graph-derived provider features...")
    feats = build_graph_features()
    print(f"\nGraph features shape: {feats.shape}")
    print("\nFeature statistics:")
    print(feats.drop(columns=["Provider"]).describe().round(4))

    if args.report:
        print("\n--report set; not modifying the provider table.")
        return

    merge_into_provider_table(feats)
    print("\nDone. Re-run src/modeling.py to train with the enriched feature set.")


if __name__ == "__main__":
    main()
