#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_top_down.py - classify top-down (main class first, then its children),
the way a hierarchical ML classifier is trained, with no skip list.

The coredrill scores every node against every text at once, so a diffuse
node (centroid near the middle of the sphere) competes everywhere; the
26-node skip list exists to stop those hubs. Routing top-down confines each
node to its own branch: a diffuse child can only beat its siblings.

Each node gets two vectors from TRAIN records only:
  route[n] - mean of the records labeled n or any descendant (the branch)
  own[n]   - mean of the records labeled n itself (the "stop here" option)
At node n the options are "stop at n" (if n is a label) and each child with
data; the walk descends until it picks "stop" or reaches a leaf.

Variants, all on the same held-out split with all 199 labels winnable:
  flat_centroid          nearest own-centroid over all labels (01's baseline)
  flat_centroid_skip26   same, with the coredrill's 26 skip nodes removed
  td_centroid            top-down, cosine
  td_centroid_centered   top-down, cosine after subtracting the mean of the
                         options at each step (local centering)
  td_centroid_path       soft top-down: softmax over the options at every
                         step, best full path by product (no early commitment)
  td_centroid_centered_path
  td_trained             one small linear softmax per parent node (a local
                         classifier per parent node), hard top-down
  td_trained_path        same classifiers, best full path by product
  flat_trained           02's plain linear head, for reference
  flat_logreg            one flat logistic regression with the same C as
                         td_trained, so hierarchy is not confused with
                         regularization
tau and C are chosen on a 15% slice of the train split, never on test.
"""
import json
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from common import OUT_DIR, load_embeddings, load_split

TAUS = (0.01, 0.02, 0.05, 0.1)
CS = (1.0, 10.0, 100.0)
VAL_FRAC = 0.15


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def log_softmax(s):
    s = s - s.max(axis=1, keepdims=True)
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


class Tree:
    def __init__(self, parents_map, labels):
        self.parent = {n: (p[0] if p else None) for n, p in parents_map.items()}
        self.labels = labels
        self.paths = {}
        for l in labels:
            path, n = [], l
            while n is not None:
                path.append(n)
                n = self.parent[n]
            self.paths[l] = path[::-1]  # root ... l

    def near_miss(self, true, pred):
        if pred == true or pred in self.paths[true] or true in self.paths[pred]:
            return True
        p = self.parent[true]
        return p is not None and p == self.parent[pred]


def fit(tree, emb, y, fit_idx, C=None):
    """Options, vectors and (if C) per-node classifiers from fit_idx only."""
    labels = tree.labels
    ylab = y[fit_idx]
    present = sorted(set(ylab.tolist()))
    opts = {}
    for li in present:
        prev = None
        for n in tree.paths[labels[li]]:
            opts.setdefault(prev, set()).add(("down", n))
            prev = n
        opts.setdefault(labels[li], set()).add(("stop", labels[li]))
    opts = {n: sorted(o) for n, o in opts.items()}

    decisions, target = {}, {n: {} for n in opts}
    for li in present:
        L, prev, steps = labels[li], None, []
        for n in tree.paths[L]:
            steps.append((prev, opts[prev].index(("down", n))))
            prev = n
        steps.append((L, opts[L].index(("stop", L))))
        decisions[li] = steps
        for node, k in steps:
            target[node][li] = k

    def mean_of(lis):
        return unit(emb[fit_idx[np.isin(ylab, list(lis))]].mean(axis=0))

    V, clf = {}, {}
    for node, o in opts.items():
        V[node] = np.stack([
            mean_of([labels.index(m)]) if kind == "stop"
            else mean_of([li for li in present if m in tree.paths[labels[li]]])
            for kind, m in o])
        if C is not None and len(o) > 1:
            rows = fit_idx[np.isin(ylab, list(target[node]))]
            t = np.array([target[node][li] for li in y[rows]])
            clf[node] = LogisticRegression(C=C, max_iter=2000).fit(emb[rows], t)
    return {"opts": opts, "decisions": decisions, "present": present, "V": V, "clf": clf}


def cos_scorer(model, centered):
    def score(node, X):
        V = model["V"][node]
        if centered:
            m = V.mean(axis=0)
            return unit(X - m) @ unit(V - m).T
        return X @ V.T
    return score


def route_hard(model, X, score):
    pred = np.empty(len(X), dtype=object)
    stack = [(None, np.arange(len(X)))]
    while stack:
        node, idx = stack.pop()
        o = model["opts"][node]
        k = np.zeros(len(idx), dtype=int) if len(o) == 1 else score(node, X[idx]).argmax(axis=1)
        for j, (kind, m) in enumerate(o):
            sub = idx[k == j]
            if len(sub):
                if kind == "stop":
                    pred[sub] = m
                else:
                    stack.append((m, sub))
    return list(pred)


def path_scores(model, X, logp):
    """Log-probability of every full path; columns follow model["present"]."""
    cache = {n: logp(n, X) for n, o in model["opts"].items() if len(o) > 1}
    total = np.zeros((len(X), len(model["present"])))
    for j, li in enumerate(model["present"]):
        for node, k in model["decisions"][li]:
            if node in cache:
                total[:, j] += cache[node][:, k]
    return total


def route_path(model, X, logp, labels):
    return [labels[model["present"][a]] for a in path_scores(model, X, logp).argmax(axis=1)]


def train_flat_head(X, y, n_classes):
    torch.manual_seed(42)
    model = nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(40):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            nn.functional.cross_entropy(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    return model.eval()


def metrics(pred, truths, tree, skip):
    on_skip = [p == t for p, t in zip(pred, truths) if t in skip]
    top4 = Counter(pred).most_common(4)
    return {
        "strict": float(np.mean([p == t for p, t in zip(pred, truths)])),
        "near_miss": float(np.mean([tree.near_miss(t, p) for p, t in zip(pred, truths)])),
        "main_class": float(np.mean([tree.paths[p][0] == tree.paths[t][0] for p, t in zip(pred, truths)])),
        "skip_labels_strict": float(np.mean(on_skip)) if on_skip else None,
        "top4_answer_share": sum(c for _, c in top4) / len(pred),
        "top4_answers": top4,
    }


def inner_split(y, train_idx, n_classes):
    rng = np.random.default_rng(42)
    fit_parts, val_parts = [], []
    for c in range(n_classes):
        idx = train_idx[y[train_idx] == c].copy()
        rng.shuffle(idx)
        k = int(len(idx) * VAL_FRAC)
        val_parts.append(idx[:k])
        fit_parts.append(idx[k:])
    return np.concatenate(fit_parts), np.concatenate(val_parts)


def main():
    t0 = time.perf_counter()
    emb = load_embeddings()
    split = load_split()
    labels = list(split["labels"])
    y, train_idx, test_idx = split["y"], split["train_idx"], split["test_idx"]
    cd = json.loads((OUT_DIR / "coredrill_train_only.json").read_text(encoding="utf-8"))
    tree = Tree(cd["taxonomy_backbone"]["parents_map"], labels)
    skip = set(cd.get("skip_nodes", []))

    # Hyperparameters on an inner slice of train.
    fit_idx, val_idx = inner_split(y, train_idx, len(labels))
    Xv, val_truth = emb[val_idx], [labels[c] for c in y[val_idx]]

    def acc(pred):
        return float(np.mean([p == t for p, t in zip(pred, val_truth)]))

    inner = fit(tree, emb, y, fit_idx)
    best_tau = {}
    for centered in (False, True):
        score = cos_scorer(inner, centered)
        scores = {tau: acc(route_path(inner, Xv, lambda n, X: log_softmax(score(n, X) / tau), labels))
                  for tau in TAUS}
        best_tau[centered] = max(scores, key=scores.get)
        print(f"  tau (centered={centered}): {scores} -> {best_tau[centered]}")
    c_scores = {}
    for C in CS:
        m = fit(tree, emb, y, fit_idx, C=C)
        c_scores[C] = acc(route_hard(m, Xv, lambda n, X: m["clf"][n].predict_log_proba(X)))
    best_c = max(c_scores, key=c_scores.get)
    print(f"  C: {c_scores} -> {best_c}   ({time.perf_counter() - t0:.0f}s)")

    # Final fit on the whole train split, scored on test.
    model = fit(tree, emb, y, train_idx, C=best_c)
    X, truths = emb[test_idx], [labels[c] for c in y[test_idx]]
    cos_plain, cos_cent = cos_scorer(model, False), cos_scorer(model, True)

    def trained_logp(n, X_):
        return model["clf"][n].predict_log_proba(X_)

    flat_cos = X @ split["centroids"].T
    flat_cos_skip = flat_cos.copy()
    flat_cos_skip[:, [labels.index(s) for s in skip if s in labels]] = -np.inf
    head = train_flat_head(emb[train_idx], y[train_idx], len(labels))
    with torch.no_grad():
        flat_logits = head(torch.from_numpy(X)).numpy()

    preds = {
        "flat_centroid": [labels[i] for i in flat_cos.argmax(axis=1)],
        "flat_centroid_skip26": [labels[i] for i in flat_cos_skip.argmax(axis=1)],
        "td_centroid": route_hard(model, X, cos_plain),
        "td_centroid_centered": route_hard(model, X, cos_cent),
        "td_centroid_path": route_path(
            model, X, lambda n, X_: log_softmax(cos_plain(n, X_) / best_tau[False]), labels),
        "td_centroid_centered_path": route_path(
            model, X, lambda n, X_: log_softmax(cos_cent(n, X_) / best_tau[True]), labels),
        "td_trained": route_hard(model, X, trained_logp),
        "td_trained_path": route_path(model, X, trained_logp, labels),
        "flat_trained": [labels[i] for i in flat_logits.argmax(axis=1)],
        "flat_logreg": [labels[i] for i in LogisticRegression(C=best_c, max_iter=2000)
                        .fit(emb[train_idx], y[train_idx]).predict(X)],
    }

    out = {"chosen": {"tau": {str(k): v for k, v in best_tau.items()}, "C": best_c},
           "n_test": len(truths), "skip_label_test_records": sum(t in skip for t in truths),
           "results": {}}
    print(f"\n{'variant':<27}{'strict':>8}{'near':>8}{'main':>8}{'skip-lbl':>10}{'top4':>8}")
    for name, pred in preds.items():
        r = metrics(pred, truths, tree, skip)
        out["results"][name] = r
        print(f"{name:<27}{r['strict']:8.4f}{r['near_miss']:8.4f}{r['main_class']:8.4f}"
              f"{r['skip_labels_strict']:10.4f}{r['top4_answer_share']:8.1%}")
    (OUT_DIR / "top_down.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_DIR / 'top_down.json'}   ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
