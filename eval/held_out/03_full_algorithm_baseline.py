#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_full_algorithm_baseline.py - reproduces the decision rules from the
original research pipeline (publish/rhizome.py, not in this repo:
_score_node_analysis, _consensus_and_recommended), evaluated on the same
held-out test set as everything else, using direct-record, TRAIN-only
centroids/sigmas from artifacts/split.npz. scripts/rhizome_engine.py's
version of the formula differs only in flooring sigma at 0.25 instead of
1e-6. For the deployed engine end to end (its own build, GRAM-INVERSE,
real API precedence), see 05_full_pipeline.py.

Three candidates, all operating across all 199 classes simultaneously
(SKIP_NODES excluded from candidacy, matching production):

  cosine_argmax       - raw cosine similarity, no sigma adjustment at all.
  production_argmax   - the full blended formula (sigma-cap + diffuse_pen):
                         sigma_z = min(sigma_eff, 0.45)
                         z = (1 - cos) / sigma_z
                         lap = -z ; gau = -0.5*z^2
                         man = lap + 0.15*cos - 0.05*log(sigma_eff)
                         diffuse_pen = max(0, sigma_eff - 0.40) * 0.60
                         prod = 0.62*man + 0.18*lap + 0.10*gau + 0.08*cos - diffuse_pen
  recommended         - what _consensus_and_recommended() actually always
                         reduces to: argmin(z) (laplace_score = -z, so the
                         "laplace" and "smallest_z" families are the same
                         node by construction; the gaussian tie-break in
                         the real code only ever confirms that same node -
                         see commit notes / chat log for the derivation).

predict()'s hierarchical per-level walk is deliberately NOT reproduced
here: reading it closely, the per-level gate states (confident/weak/
ambiguous/deep_space) are computed and stored but never used to stop the
loop or override best_node - the loop runs every level unconditionally
and best_node always ends up being the deepest level's raw nearest-
neighbor pick. For our 199-class set (spanning levels 0-5), that would
mean predict() can only ever answer with one of the ~19 level-5 nodes,
which isn't a meaningful general classifier for this comparison.
"""
import json
import numpy as np
from common import load_embeddings, load_split, OUT_DIR

SIGMA_CAP = 0.45
SIGMA_BASE = 0.40
SIGMA_PENALTY = 0.60

SKIP_NODE_NAMES = {"Dialogue Scene", "Request–Response Dialogue"}


def main():
    split = load_split()
    labels = list(split["labels"])
    y = split["y"]
    test_idx = split["test_idx"]
    centroids = split["centroids"]          # (n_classes, 768), train-only, L2-normalized
    sigmas = split["sigmas"]                # (n_classes,), train-only

    embeddings = load_embeddings()
    X_test, y_test = embeddings[test_idx], y[test_idx]

    skip_mask = np.array([l in SKIP_NODE_NAMES for l in labels])
    print(f"SKIP_NODES excluded from candidacy: {[l for l in labels if l in SKIP_NODE_NAMES]}")

    cos = X_test @ centroids.T                               # (n_test, n_classes)
    sigma_eff = np.maximum(sigmas, 1e-6)
    sigma_z = np.minimum(sigma_eff, SIGMA_CAP)
    z = (1.0 - cos) / sigma_z[None, :]
    lap = -z
    gau = -0.5 * z * z
    man = lap + 0.15 * cos - 0.05 * np.log(sigma_eff)[None, :]
    diffuse_pen = np.maximum(0.0, sigma_eff - SIGMA_BASE) * SIGMA_PENALTY
    prod = 0.62 * man + 0.18 * lap + 0.10 * gau + 0.08 * cos - diffuse_pen[None, :]

    # Exclude SKIP_NODES from candidacy everywhere, matching production.
    NEG_INF = -1e9
    cos_c = np.where(skip_mask[None, :], NEG_INF, cos)
    prod_c = np.where(skip_mask[None, :], NEG_INF, prod)
    neg_z_c = np.where(skip_mask[None, :], NEG_INF, -z)       # argmax(-z) == argmin(z)

    results = {}
    for name, matrix in [("cosine_argmax", cos_c), ("production_argmax", prod_c), ("recommended", neg_z_c)]:
        preds = matrix.argmax(axis=1)
        acc = float((preds == y_test).mean())
        results[name] = acc
        print(f"  {name:<20s} accuracy={acc:.4f}")

    # How much of the gap, if any, is explained by SKIP_NODES examples
    # being unwinnable by construction?
    skip_class_idx = {i for i, l in enumerate(labels) if l in SKIP_NODE_NAMES}
    n_skip_test = int(np.isin(y_test, list(skip_class_idx)).sum())
    print(f"\n{n_skip_test} test examples belong to a SKIP_NODES class "
          f"(unwinnable by construction, {n_skip_test/len(y_test)*100:.2f}% of test set)")

    (OUT_DIR / "full_algorithm_baseline.json").write_text(json.dumps({
        "skip_nodes": sorted(SKIP_NODE_NAMES & set(labels)),
        "n_skip_test_examples": n_skip_test,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_DIR / 'full_algorithm_baseline.json'}")


if __name__ == "__main__":
    main()
