#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train.py - train a linear classifier head on frozen LaBSE embeddings,
in two configurations, evaluated on the exact same held-out test set as
01_baseline.py's geometric number:

  plain      - random init, one-hot cross-entropy, no sigma weighting.
               Isolates "does training at all help".
  integrated - centroid-initialized weights + sigma-aware loss weighting
               + multi-label rhizome soft targets. Isolates "does folding
               the geometric method's own structure into training help
               beyond training alone".

Architecture is deliberately a single nn.Linear(768, n_classes) - no
hidden layer - so centroid-init is a direct, literal correspondence
(logit_c = x . W[c], same dot product the geometric baseline uses,
now with W trainable instead of frozen).

Sigma-aware weighting reuses the original study's own diffuse_pen
formula verbatim (Appendix B, study_v5.md):
  diffuse_pen = max(0, sigma - 0.40) * 0.60
applied as a per-class training-loss weight = 1 - diffuse_pen, floored at
0.1 so no class's gradient is ever fully silenced.

Multi-label rhizome targets: sentences_rhizome.json's per-record
`categories` list (node, weight pairs, already computed by the original
pipeline) becomes a multi-label BCE target instead of one-hot - a record
activating 3 nodes trains against all 3, weighted, with the record's own
hard label forced to weight 1.0.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import load_embeddings, load_split, DATA_DIR, OUT_DIR

SIGMA_BASE = 0.40
SIGMA_PENALTY = 0.60
MIN_CLASS_WEIGHT = 0.10

EPOCHS = 40
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_multilabel_targets(labels: list[str], y: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """(n_train, n_classes) soft target matrix from sentences_rhizome.json,
    restricted to the kept label set, hard label forced to 1.0."""
    rhizome = json.loads((DATA_DIR / "sentences_rhizome.json").read_text(encoding="utf-8"))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n_classes = len(labels)
    targets = np.zeros((len(train_idx), n_classes), dtype=np.float32)
    for row, global_i in enumerate(train_idx):
        hard_c = y[global_i]
        targets[row, hard_c] = 1.0
        for entry in rhizome[global_i].get("categories", []):
            ci = label_to_idx.get(entry["node"])
            if ci is not None:
                targets[row, ci] = max(targets[row, ci], float(entry["weight"]))
    return targets


def diffuse_pen(sigma: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, sigma - SIGMA_BASE) * SIGMA_PENALTY


def train_one(config: str, X_train, y_train, X_test, y_test, centroids, sigmas,
              multilabel_targets=None) -> dict:
    """config is one of:
      plain                - random init, one-hot CE, no sigma weighting.
      centroid_only        - centroid-initialized weights, one-hot CE,
                              nothing else. Isolates the warm-start effect.
      centroid_sigma       - centroid init + sigma-weighted one-hot CE.
                              Isolates the sigma-weighting mechanism alone.
      centroid_multilabel  - centroid init + multi-label rhizome
                              soft-target CE, no sigma weighting. Isolates
                              the multi-label mechanism alone.
      integrated           - centroid init + sigma-weighted loss +
                              multi-label rhizome soft-target CE (all
                              three mechanisms combined).
    """
    set_seed(SEED)
    n_classes = centroids.shape[0]
    model = nn.Linear(768, n_classes, bias=True)

    use_centroid_init = config != "plain"
    use_sigma_weight = config in ("centroid_sigma", "integrated")
    use_multilabel = config in ("centroid_multilabel", "integrated")

    if use_centroid_init:
        with torch.no_grad():
            model.weight.copy_(torch.from_numpy(centroids))
            model.bias.zero_()

    if use_sigma_weight:
        class_weight = torch.from_numpy(
            np.clip(1.0 - diffuse_pen(sigmas), MIN_CLASS_WEIGHT, 1.0).astype(np.float32)
        )
    else:
        class_weight = None

    if use_multilabel:
        # Normalize each example's rhizome weights to a probability
        # distribution, so classes stay in competition via softmax
        # instead of being scored as independent binary decisions (the
        # independent-BCE version of this collapsed to 6.8% test accuracy -
        # averaging BCE over 199 mostly-negative classes trivially rewards
        # pushing every logit toward negative, wrecking the argmax ranking
        # a single-label prediction needs. Soft cross-entropy keeps the
        # same "train against all co-activated nodes, weighted" idea
        # without that failure mode.).
        row_sums = multilabel_targets.sum(axis=1, keepdims=True)
        soft_targets = multilabel_targets / np.clip(row_sums, 1e-8, None)
        targets = torch.from_numpy(soft_targets.astype(np.float32))
    else:
        targets = torch.from_numpy(y_train)

    Xt = torch.from_numpy(X_train)
    ds = TensorDataset(Xt, targets, torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, tb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            if use_multilabel:
                log_probs = torch.log_softmax(logits, dim=1)
                per_example = -(tb * log_probs).sum(dim=1)      # soft cross-entropy
            else:
                per_example = nn.functional.cross_entropy(logits, tb, reduction="none")
            if use_sigma_weight:
                weight = class_weight[yb]                        # weight by the TRUE class's sigma
                loss = (per_example * weight).mean()
            else:
                loss = per_example.mean()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{config}] epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(ds):.4f}")
    train_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test))
        preds = test_logits.argmax(dim=1).numpy()
        top5 = test_logits.topk(5, dim=1).indices.numpy()
    y_test_t = y_test.reshape(-1, 1)
    acc = float((preds == y_test).mean())
    top3_acc = float((top5[:, :3] == y_test_t).any(axis=1).mean())
    top5_acc = float((top5 == y_test_t).any(axis=1).mean())
    print(f"  [{config}] test accuracy: {acc:.4f}  top3={top3_acc:.4f}  top5={top5_acc:.4f}"
          f"  (trained {EPOCHS} epochs in {train_s:.1f}s)")

    return {"config": config, "test_accuracy": acc, "top3_accuracy": top3_acc,
            "top5_accuracy": top5_acc, "train_seconds": round(train_s, 1), "preds": preds}


def main():
    split = load_split()
    labels = list(split["labels"])
    y = split["y"]
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    centroids = split["centroids"]
    sigmas = split["sigmas"]

    embeddings = load_embeddings()
    X_train, y_train = embeddings[train_idx], y[train_idx]
    X_test, y_test = embeddings[test_idx], y[test_idx]

    print(f"{len(labels)} classes, train={len(train_idx)} test={len(test_idx)}\n")

    results = []
    ml_targets = build_multilabel_targets(labels, y, train_idx)

    configs = [
        ("plain", None),
        ("centroid_only", None),
        ("centroid_sigma", None),
        ("centroid_multilabel", ml_targets),
        ("integrated", ml_targets),
    ]
    for name, ml in configs:
        print(f"\n=== {name} ===")
        results.append(train_one(name, X_train, y_train, X_test, y_test, centroids, sigmas,
                                  multilabel_targets=ml))

    print("\n=== summary ===")
    print(f"  geometric baseline (01_baseline.py): see artifacts/baseline.json")
    for r in results:
        print(f"  {r['config']:<20s} top1={r['test_accuracy']:.4f}  top3={r['top3_accuracy']:.4f}  "
              f"top5={r['top5_accuracy']:.4f}  train_time={r['train_seconds']}s")

    out = {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "sigma_base": SIGMA_BASE, "sigma_penalty": SIGMA_PENALTY, "min_class_weight": MIN_CLASS_WEIGHT,
        "results": [{"config": r["config"], "test_accuracy": r["test_accuracy"],
                     "top3_accuracy": r["top3_accuracy"], "top5_accuracy": r["top5_accuracy"],
                     "train_seconds": r["train_seconds"]} for r in results],
    }
    (OUT_DIR / "train_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez(OUT_DIR / "train_preds.npz", **{r["config"]: r["preds"] for r in results}, y_test=y_test)
    print(f"\nSaved -> {OUT_DIR / 'train_results.json'}")


if __name__ == "__main__":
    main()
