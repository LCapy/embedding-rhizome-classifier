#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_review_checks.py - two checks for "is the centroid approach relevant at all":

  1. Few-shot regime: nearest-centroid vs trained linear head when each
     class has only k training examples (k = 1, 2, 5, 10, 20, all).
     Prototype methods are expected to win at small k; training at large k.
  2. Error structure: for the plain trained classifier's mistakes, how many
     are taxonomy near-misses (ancestor/descendant, shared parent) versus
     unrelated. A high near-miss rate means the label set caps accuracy,
     not the method.
"""
import numpy as np
import torch
import torch.nn as nn
from common import load_embeddings, load_split, load_coredrill

SEEDS = [0, 1, 2]
KS = [1, 2, 5, 10, 20, None]   # None = all training data


def train_head(X, y, n_classes, epochs=40, seed=0):
    torch.manual_seed(seed)
    model = nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    bs = 256
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            nn.functional.cross_entropy(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    return model


def few_shot(emb, y, train_idx, test_idx, n_classes):
    X_test, y_test = emb[test_idx], y[test_idx]
    print("=== 1. Few-shot: nearest-centroid vs trained head (mean of 3 seeds) ===")
    print(f"  {'k/class':>8}  {'centroid':>9}  {'trained':>9}  {'winner':>9}")
    for k in KS:
        c_accs, t_accs = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            parts = []
            for c in range(n_classes):
                idx = train_idx[y[train_idx] == c]
                if k is not None and len(idx) > k:
                    idx = rng.choice(idx, size=k, replace=False)
                parts.append(idx)
            sub = np.concatenate(parts)
            X_tr, y_tr = emb[sub], y[sub]

            cents = np.zeros((n_classes, emb.shape[1]), dtype=np.float32)
            for c in range(n_classes):
                mu = X_tr[y_tr == c].mean(axis=0)
                cents[c] = mu / (np.linalg.norm(mu) + 1e-12)
            c_accs.append(float(((X_test @ cents.T).argmax(1) == y_test).mean()))

            # Scale epochs so tiny subsets still get a comparable number of steps.
            epochs = 40 if k is None else max(40, int(40 * 27032 / max(len(sub), 1) / 10))
            model = train_head(X_tr, y_tr, n_classes, epochs=min(epochs, 400), seed=seed)
            with torch.no_grad():
                t_accs.append(float((model(torch.from_numpy(X_test)).argmax(1).numpy() == y_test).mean()))
        c, t = np.mean(c_accs), np.mean(t_accs)
        label = "all" if k is None else str(k)
        print(f"  {label:>8}  {c:9.4f}  {t:9.4f}  {'centroid' if c > t else 'trained':>9}")


def ancestors(node, parents_map, memo):
    if node in memo:
        return memo[node]
    out = set()
    for p in parents_map.get(node, []):
        out.add(p)
        out |= ancestors(p, parents_map, memo)
    memo[node] = out
    return out


def error_structure(labels, y_test):
    tree = load_coredrill()["tree"]
    parents_map = {n: (s.get("parents") or []) for n, s in tree.items()}
    memo = {}
    preds = np.load("artifacts/train_preds.npz", allow_pickle=True)["plain"]
    wrong = np.where(preds != y_test)[0]

    counts = {"ancestor_or_descendant": 0, "shared_parent": 0, "unrelated": 0}
    for i in wrong:
        t, p = labels[y_test[i]], labels[preds[i]]
        ta, pa = ancestors(t, parents_map, memo), ancestors(p, parents_map, memo)
        if p in ta or t in pa:
            counts["ancestor_or_descendant"] += 1
        elif set(parents_map.get(t, [])) & set(parents_map.get(p, [])):
            counts["shared_parent"] += 1
        else:
            counts["unrelated"] += 1

    n = len(wrong)
    print(f"\n=== 2. Error structure (plain trained head, {n} errors of {len(y_test)}) ===")
    for k, v in counts.items():
        print(f"  {k:<24s} {v:5d}  ({v / n * 100:.1f}% of errors)")
    near = counts["ancestor_or_descendant"] + counts["shared_parent"]
    lenient = (len(y_test) - n + near) / len(y_test)
    print(f"  accuracy if taxonomy near-misses counted as correct: {lenient:.4f}")


def main():
    split = load_split()
    labels = list(split["labels"])
    y, train_idx, test_idx = split["y"], split["train_idx"], split["test_idx"]
    emb = load_embeddings()
    few_shot(emb, y, train_idx, test_idx, len(labels))
    error_structure(labels, y[test_idx])


if __name__ == "__main__":
    main()
