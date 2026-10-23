#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_baseline.py - build the fixed train/test split and recompute the
geometric (nearest-centroid, no training) baseline on a genuine held-out
set. This is the number everything else has to beat.

Writes artifacts/split.npz - every later stage loads this same split, so
the comparison is apples-to-apples.
"""
import json
import numpy as np
from common import (
    load_sentences, load_embeddings, build_label_set, make_split,
    compute_centroids, compute_sigmas, save_split, OUT_DIR, MIN_COUNT, TEST_FRAC, SEED,
)


def main():
    sentences = load_sentences()
    embeddings = load_embeddings()
    assert len(sentences) == embeddings.shape[0], "sentences.json / embeddings.npz row mismatch"

    labels = build_label_set(sentences, min_count=MIN_COUNT)
    y, train_idx, test_idx = make_split(sentences, labels, test_frac=TEST_FRAC, seed=SEED)

    kept = (y >= 0)
    dropped = (~kept).sum()
    print(f"{len(labels)} classes kept (min_count={MIN_COUNT}), "
          f"{dropped} records dropped (below threshold or unlabeled) out of {len(sentences)}")
    print(f"train: {len(train_idx)}  test: {len(test_idx)}  "
          f"(test_frac={TEST_FRAC}, seed={SEED})")

    centroids = compute_centroids(embeddings, y, train_idx, len(labels))
    sigmas = compute_sigmas(embeddings, y, train_idx, centroids)
    save_split(labels, y, train_idx, test_idx, centroids, sigmas)
    print(f"sigma range: min={sigmas.min():.3f} max={sigmas.max():.3f} "
          f"mean={sigmas.mean():.3f}")

    # Geometric baseline: nearest-centroid by cosine similarity, on the
    # held-out test set, using centroids built ONLY from train records.
    test_emb = embeddings[test_idx]
    test_y = y[test_idx]
    sims = test_emb @ centroids.T  # (n_test, n_classes), both L2-normalized
    preds = sims.argmax(axis=1)
    acc = float((preds == test_y).mean())
    print(f"\nGeometric baseline (nearest-centroid, held-out test set): {acc:.4f}")

    # Per-class breakdown, worst 10, to sanity-check nothing degenerate.
    per_class_acc = []
    for cls_idx in range(len(labels)):
        idx = np.where(test_y == cls_idx)[0]
        if len(idx) == 0:
            continue
        a = float((preds[idx] == cls_idx).mean())
        per_class_acc.append((labels[cls_idx], a, len(idx)))
    per_class_acc.sort(key=lambda x: x[1])
    print("\nWorst 10 classes (geometric baseline):")
    for name, a, n in per_class_acc[:10]:
        print(f"  {name:<32s} acc={a:.3f}  n_test={n}")

    (OUT_DIR / "baseline.json").write_text(json.dumps({
        "n_classes": len(labels), "min_count": MIN_COUNT, "test_frac": TEST_FRAC, "seed": SEED,
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "geometric_accuracy": acc,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
