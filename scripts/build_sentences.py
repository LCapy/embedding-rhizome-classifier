#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_sentences_rhizome.py — Build a multi-label rhizomatic sentence dataset.

Reads:
  - data/sentences.json       (existing single-label corpus)
  - data/embeddings.npz       (existing LaBSE embeddings)
  - coredrill (v1 or v3)      (for scoring sentences against territories)

Writes:
  - data/sentences_rhizome.json

Format of sentences_rhizome.json:
  Each record keeps all original fields PLUS a "categories" list:
  {
    "text": "...",
    "source": "wikipedia",
    "lang": "en",
    "original_category": "Natural World",   <- original single label preserved
    "categories": [
      {"node": "Natural World",    "z": 0.41, "cos": 0.612, "weight": 0.664},
      {"node": "Cosmology",        "z": 0.52, "cos": 0.589, "weight": 0.595},
      {"node": "Astrophysics",     "z": 0.78, "cos": 0.541, "weight": 0.459},
      {"node": "Earth & Cosmos",   "z": 0.94, "cos": 0.498, "weight": 0.391},
      {"node": "Physical Reality", "z": 1.12, "cos": 0.461, "weight": 0.326}
    ]
  }

This is the proper rhizomatic dataset. Every sentence contributes to every
territory it genuinely inhabits, weighted by geometric proximity.
The KME built from this dataset will correctly represent territory distributions
without arborescent single-label bias.

USAGE:
  python build_sentences_rhizome.py \\
    --sentences  data/sentences.json \\
    --embeddings data/embeddings.npz \\
    --coredrill  coredrill/coredrill_hierarchical.json \\
    --out        data/sentences_rhizome.json \\
    --z-cutoff   1.5 \\
    --min-weight 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("build_sentences_rhizome")

# ── Constants — must match full_build_rhizome_final.py ───────────────────────
SIGMA_CAP = 0.45
SIGMA_FLOOR = 0.25   # minimum sigma for z computation — prevents collapse
MIN_WEIGHT  = 0.01

SKIP_NODES: set = {
    # Original — void centroids
    "Dialogue Scene",
    "Request\u2013Response Dialogue",
    # Collapsed (sigma > 0.70)
    "Instruction & Procedure",
    "Home Repair Request",
    "Household Maintenance",
    "Empathy & Support",
    "Emotion & Social Bonding",
    "Transactions",
    # SGD bulk — task dialogues, not semantic territories
    "Domestic Planning",
    "Task Assignment",
    "Move",
    "Visual Arts",
    "Trip Planning",
    "Restaurant & Ordering",
    "Ordering at a Restaurant",
    "Home Life",
    "Repair",
    "Jobs & Professions",
    "Commuting",
    # Non-wiki bulk — Circa/Dolly single-action nodes
    "Report",
    "Agree",
    "Clarify",
    "Meal Preparation",
    # Anchor-only with high sigma
    "Meaning Relations",
    "Agreement Formation",
    "Project Agreement",
}


# ══════════════════════════════════════════════════════════════════════════════
# § 1  Load
# ══════════════════════════════════════════════════════════════════════════════

def load_sentences(path: str) -> List[dict]:
    log.info(f"Loading sentences: {path}")
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    log.info(f"  {len(records):,} records")
    return records


def load_embeddings(path: str) -> np.ndarray:
    log.info(f"Loading embeddings: {path}")
    data = np.load(path)
    key  = "embeddings" if "embeddings" in data else list(data.keys())[0]
    emb  = data[key].astype(np.float32)
    nrm  = np.linalg.norm(emb, axis=1, keepdims=True)
    emb  = emb / (nrm + 1e-12)
    log.info(f"  shape: {emb.shape}")
    return emb


def load_coredrill(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load centroids and sigmas from any coredrill version.
    Returns (C, S, node_names) where:
      C: (N_nodes, 768) centroid matrix
      S: (N_nodes,) sigma_cap array
      node_names: list of node names
    """
    log.info(f"Loading coredrill: {path}")
    cd   = json.loads(Path(path).read_text(encoding="utf-8"))
    tree = cd.get("tree", {})
    ver  = cd.get("version", "1.0")
    log.info(f"  version: {ver}  nodes: {len(tree)}")

    node_names = []
    centroids  = []
    sigmas     = []

    for node, stats in tree.items():
        if node in SKIP_NODES:
            continue
        c = stats.get("centroid")
        if c is None:
            continue
        v = np.array(c, dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-12
        sig = float(stats.get("sigma") or 0.35)
        sig = max(sig, SIGMA_FLOOR)   # floor prevents z-collapse
        node_names.append(node)
        centroids.append(v)
        sigmas.append(min(sig, SIGMA_CAP))

    C = np.array(centroids, dtype=np.float32)   # (N, 768)
    S = np.array(sigmas,    dtype=np.float32)   # (N,)
    log.info(f"  {len(node_names)} nodes with centroids (excluded {len(SKIP_NODES)} skip)")
    return C, S, node_names


# ══════════════════════════════════════════════════════════════════════════════
# § 2  Score all sentences against all centroids
# ══════════════════════════════════════════════════════════════════════════════

def score_all(embeddings: np.ndarray,
              C:          np.ndarray,
              S:          np.ndarray,
              node_names: List[str],
              cos_threshold: float,
              batch_size: int = 1024,
              ) -> List[List[dict]]:
    """
    Score every sentence against every centroid using a FIXED cosine
    distance threshold — not a fixed z_cutoff.

    A sentence belongs to a territory if:
        (1 - cosine) < cos_threshold   (fixed radius in cosine space)

    This equalizes territory sizes across nodes with different sigmas.
    Broad nodes (high sigma) don't vacuum up the entire corpus.
    Compact nodes (low sigma) don't become too restrictive.

    Weight = exp(-z) where z = (1-cos)/min(sigma, SIGMA_CAP).
    """
    N_sent  = len(embeddings)
    N_nodes = len(node_names)
    log.info(f"Scoring {N_sent:,} sentences × {N_nodes} nodes")
    log.info(f"  Threshold: cosine_dist < {cos_threshold} (fixed radius)")

    all_categories: List[List[dict]] = []
    t0 = time.time()

    for start in range(0, N_sent, batch_size):
        end   = min(start + batch_size, N_sent)
        X     = embeddings[start:end].astype(np.float32)   # (B, 768)
        cos   = X @ C.T                                     # (B, N)
        dist  = 1.0 - cos                                   # (B, N) cosine distances
        z     = dist / (S[None, :] + 1e-12)               # (B, N) z-scores

        for i in range(end - start):
            cats = []
            for j in range(N_nodes):
                dij = float(dist[i, j])
                if dij >= cos_threshold:
                    continue
                cosij = float(cos[i, j])
                zij   = float(z[i, j])
                wij   = float(np.exp(-zij))
                if wij < MIN_WEIGHT:
                    continue
                cats.append({
                    "node":   node_names[j],
                    "z":      round(zij, 4),
                    "cos":    round(cosij, 4),
                    "weight": round(wij, 4),
                })
            cats.sort(key=lambda x: -x["weight"])
            all_categories.append(cats)

        if (start // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            pct     = (end / N_sent) * 100
            log.info(f"  {end:>7,}/{N_sent:,}  ({pct:5.1f}%)  {elapsed:.1f}s")

    counts  = [len(c) for c in all_categories]
    n_empty = sum(1 for c in counts if c == 0)
    log.info(f"  Done in {time.time()-t0:.1f}s")
    log.info(f"  Mean categories/sentence: {np.mean(counts):.1f}")
    log.info(f"  Max: {max(counts)}  Min: {min(counts)}")
    log.info(f"  Sentences with 0 categories: {n_empty:,} ({100*n_empty/N_sent:.1f}%)")
    if n_empty > N_sent * 0.40:
        log.warning(f"  HIGH EMPTY RATE — consider increasing --cos-threshold")
    return all_categories


# ══════════════════════════════════════════════════════════════════════════════
# § 3  Build and write sentences_rhizome.json
# ══════════════════════════════════════════════════════════════════════════════

def build_rhizome_sentences(records:        List[dict],
                             all_categories: List[List[dict]],
                             out_path:       str) -> None:
    """
    Merge original sentence records with multi-label category assignments.
    Write sentences_rhizome.json.
    """
    log.info(f"Building rhizomatic sentence records...")

    out = []
    n_multilabel = 0
    n_singleton  = 0
    n_empty      = 0
    total_cats   = 0

    for rec, cats in zip(records, all_categories):
        # Keep all original fields, rename category → original_category
        new_rec = {
            # Provenance
            "text":              rec.get("text", ""),
            "source":            rec.get("source", ""),
            "lang":              rec.get("lang", "en"),
            "original_category": rec.get("category", ""),
            # Optional metadata — keep if present
        }
        # Preserve any extra metadata fields
        for k in ["lang_name", "lang_family", "lang_tier",
                  "article_title", "source_article", "used_fallback",
                  "level", "parent"]:
            if k in rec:
                new_rec[k] = rec[k]

        # Multi-label categories from geometry
        new_rec["categories"] = cats

        out.append(new_rec)
        total_cats += len(cats)

        if len(cats) == 0:
            n_empty += 1
        elif len(cats) == 1:
            n_singleton += 1
        else:
            n_multilabel += 1

    # Write
    log.info(f"Writing {len(out):,} records → {out_path}")
    Path(out_path).write_text(
        json.dumps(out, ensure_ascii=False, indent=None),
        encoding="utf-8"
    )
    size_mb = Path(out_path).stat().st_size / 1e6

    log.info(f"  Written: {size_mb:.1f} MB")
    log.info(f"  Multi-label: {n_multilabel:,} ({100*n_multilabel/len(out):.1f}%)")
    log.info(f"  Singleton:   {n_singleton:,} ({100*n_singleton/len(out):.1f}%)")
    log.info(f"  Empty:       {n_empty:,} ({100*n_empty/len(out):.1f}%)")
    log.info(f"  Mean cats/sentence: {total_cats/len(out):.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# § 4  Stats report
# ══════════════════════════════════════════════════════════════════════════════

def print_stats(out_path: str, node_names: List[str]) -> None:
    """Show which nodes got the most sentence assignments."""
    log.info("Computing node coverage stats...")
    records = json.loads(Path(out_path).read_text(encoding="utf-8"))

    from collections import Counter
    node_counts   = Counter()
    weight_totals = {}

    for rec in records:
        for cat in rec.get("categories", []):
            n = cat["node"]
            node_counts[n] += 1
            weight_totals[n] = weight_totals.get(n, 0.0) + cat["weight"]

    print()
    print(f"{'='*70}")
    print(f"NODE COVERAGE in sentences_rhizome.json")
    print(f"{'='*70}")
    print(f"  Total sentences: {len(records):,}")
    print(f"  Nodes covered:   {len(node_counts)}/{len(node_names)}")
    print()
    print(f"  {'NODE':<38} {'sentences':>10}  {'total_weight':>13}")
    print(f"  {'-'*38} {'─'*10}  {'─'*13}")

    # Top 30 by sentence count
    for node, cnt in node_counts.most_common(30):
        tw = weight_totals.get(node, 0)
        print(f"  {node:<38} {cnt:>10,}  {tw:>13.1f}")

    # Empty nodes
    empty = [n for n in node_names if n not in node_counts]
    if empty:
        print(f"\n  EMPTY NODES ({len(empty)}) — no sentence assigned:")
        for n in empty:
            print(f"    {n}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════════════
# § 5  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Build multi-label rhizomatic sentence dataset"
    )
    ap.add_argument("--sentences",   required=True,
                    help="data/sentences.json")
    ap.add_argument("--embeddings",  required=True,
                    help="data/embeddings.npz")
    ap.add_argument("--coredrill",   required=True,
                    help="coredrill for scoring (use v1: coredrill_hierarchical.json)")
    ap.add_argument("--out",         required=True,
                    help="output path for sentences_rhizome.json")
    ap.add_argument("--cos-threshold", type=float, default=0.40,
                    help="max cosine distance for category inclusion (default: 0.40). "
                         "Equalizes territory sizes across nodes with different sigmas. "
                         "0.30 = tight, 0.40 = moderate, 0.50 = loose.")
    ap.add_argument("--min-weight",  type=float, default=0.01,
                    help="minimum exp(-z) weight to include (default: 0.01)")
    ap.add_argument("--stats",       action="store_true",
                    help="print node coverage stats after writing")
    args = ap.parse_args()

    print()
    print("=" * 70)
    print("BUILD SENTENCES RHIZOME — Multi-label dataset")
    print("=" * 70)
    print(f"  sentences:  {args.sentences}")
    print(f"  embeddings: {args.embeddings}")
    print(f"  coredrill:  {args.coredrill}")
    print(f"  output:     {args.out}")
    print(f"  cos_thresh: {args.cos_threshold}  (fixed cosine distance radius)")
    print(f"  min_weight: {args.min_weight}")
    print(f"  skip_nodes: {len(SKIP_NODES)}")
    print()
    print("  Every sentence will carry ALL territories it inhabits.")
    print("  No single-label assignment. Overlap is the point.")
    print()

    # Load
    records    = load_sentences(args.sentences)
    embeddings = load_embeddings(args.embeddings)

    if len(records) != len(embeddings):
        log.warning(f"Mismatch: {len(records)} records vs {len(embeddings)} embeddings")
        log.warning(f"Truncating to {len(embeddings)}")
        records = records[:len(embeddings)]

    C, S, node_names = load_coredrill(args.coredrill)

    # Score
    all_categories = score_all(
        embeddings, C, S, node_names,
        cos_threshold=args.cos_threshold,
    )

    # Build and write
    build_rhizome_sentences(records, all_categories, args.out)

    # Stats
    if args.stats:
        print_stats(args.out, node_names)

    print()
    print("=" * 70)
    print("DONE")
    print(f"  sentences_rhizome.json: {args.out}")
    print()
    print("  Next: build v4 coredrill from this dataset")
    print("  python build_rhizome_v4.py \\")
    print(f"    --sentences-rhizome {args.out} \\")
    print(f"    --embeddings {args.embeddings} \\")
    print(f"    --coredrill coredrill/coredrill_v4.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
