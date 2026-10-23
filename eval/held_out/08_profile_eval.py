#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_profile_eval.py - evaluate the rhizome's whole output (its profile), not
just its single best node, against the trained head at matched size.

The rhizome reports an active overlap set (every node within
ACTIVE_POSTERIOR_DELTA of the top production score, at most 10) plus ranked
lists. With single-label ground truth, the fair questions are:
  - set recall: is the true label inside the profile, and how big is it?
  - recall@k for each ranked list (production, gram, cosine) vs the trained
    head's top-k.
  - matched size: for each text, does the trained head's top-|profile| list
    contain the true label? Same list length, so neither side gets credit
    for listing more nodes.

Part A (fair): real analyze_text_rhizome_v2 + train-only coredrill from
05_full_pipeline.py on all 6,746 held-out texts.
Part B (live, optimistic): the live private HF Space on a random 300-text
sample of the same held-out set. Its v4.0 coredrill was built from unknown
data that may include these texts, so treat B as an upper bound.
"""
import json
import os
import random
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from common import ENGINE_DIR, OUT_DIR, load_embeddings, load_sentences, load_split

sys.path.insert(0, str(ENGINE_DIR))
import rhizome_engine as eng  # noqa: E402

KS = (1, 3, 5, 10)
LIVE_SAMPLE = 300
SPACE = os.environ.get("EVAL_SPACE_URL", "https://lcapy-coredrill.hf.space")


class StoredEmbeddingModel:
    def __init__(self):
        self.current = None

    def encode(self, texts, **kwargs):
        return np.tile(self.current, (len(texts), 1)).astype(np.float32)


def train_head(X, y, n_classes):
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


def recall_at(ranked_lists, truths, k):
    return float(np.mean([t in r[:k] for r, t in zip(ranked_lists, truths)]))


def report(title, profiles, ranked, trained_top, truths, logreg_top=None):
    sizes = [len(p) for p in profiles]

    def matched(top):
        return float(np.mean([t in tr[:max(len(p), 1)] for p, tr, t in zip(profiles, top, truths)]))

    out = {
        "n": len(truths),
        "profile_set_recall": float(np.mean([t in p for p, t in zip(profiles, truths)])),
        "profile_mean_size": float(np.mean(sizes)),
        "trained_matched_size_recall": matched(trained_top),
        "recall_at_k": {name: {k: recall_at(lists, truths, k) for k in KS} for name, lists in ranked.items()},
    }
    out["recall_at_k"]["trained_head"] = {k: recall_at(trained_top, truths, k) for k in KS}
    if logreg_top is not None:
        out["logreg_matched_size_recall"] = matched(logreg_top)
        out["recall_at_k"]["logreg"] = {k: recall_at(logreg_top, truths, k) for k in KS}
    print(f"\n=== {title} (n={out['n']}) ===")
    print(f"  rhizome profile (active set): true label inside {out['profile_set_recall']:.4f}, "
          f"mean size {out['profile_mean_size']:.2f}")
    print(f"  trained head, same size per text:  {out['trained_matched_size_recall']:.4f}")
    if logreg_top is not None:
        print(f"  logistic regression, same size:    {out['logreg_matched_size_recall']:.4f}")
    print(f"  {'ranked list':<18}" + "".join(f"{'@' + str(k):>9}" for k in KS))
    for name, by_k in out["recall_at_k"].items():
        print(f"  {name:<18}" + "".join(f"{by_k[k]:9.4f}" for k in KS))
    return out


def part_a(sents, emb, labels, y, test_idx, trained_top, logreg_top):
    cd = json.loads((OUT_DIR / "coredrill_train_only.json").read_text(encoding="utf-8"))
    if "skip_nodes" in cd:  # mirror api/app.py
        eng.SKIP_NODES.clear()
        eng.SKIP_NODES.update(cd["skip_nodes"])
    model, tau = StoredEmbeddingModel(), cd.get("tau", 0.1)
    profiles, prod, gram, cos, truths = [], [], [], [], []
    t0 = time.perf_counter()
    for n, i in enumerate(test_idx):
        model.current = emb[i]
        raw = eng.analyze_text_rhizome_v2(cd, sents[i]["text"], model, tau=tau, print_output=False)
        results = raw.get("results", [])
        profiles.append([r["node"] for r in raw.get("active_overlap_set", [])])
        prod.append([r["node"] for r in results[:10]])
        cos.append([r["node"] for r in sorted(results, key=lambda r: -r["score"])[:10]])
        gram.append([n_ for n_, _ in sorted((raw.get("gram_scores") or {}).items(), key=lambda kv: -kv[1])[:10]])
        truths.append(labels[y[i]])
        if (n + 1) % 1000 == 0:
            print(f"  part A: {n + 1}/{len(test_idx)} ({time.perf_counter() - t0:.0f}s)")
    return report("Part A: held-out, train-only build, real engine code", profiles,
                  {"production": prod, "gram_inverse": gram, "cosine": cos}, trained_top, truths, logreg_top)


def part_b(sents, labels, y, test_idx, trained_top_all):
    token = os.environ["HF_TOKEN"]
    rng = random.Random(42)
    pick = sorted(rng.sample(range(len(test_idx)), LIVE_SAMPLE))
    profiles, ranked, best, truths, trained_top = [], [], [], [], []
    t0 = time.perf_counter()
    for n, j in enumerate(pick):
        i = test_idx[j]
        req = urllib.request.Request(
            SPACE + "/predict", method="POST",
            data=json.dumps({"text": sents[i]["text"], "full": True}).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            live = json.loads(resp.read().decode("utf-8"))
        profiles.append([r["node"] for r in live.get("active_overlap", [])])
        ranked.append([r["node"] for r in (live.get("ranked") or [])[:10]])
        best.append([live.get("best_node")])
        truths.append(labels[y[i]])
        trained_top.append(trained_top_all[j])
        if (n + 1) % 50 == 0:
            print(f"  part B: {n + 1}/{LIVE_SAMPLE} ({time.perf_counter() - t0:.0f}s)")
    return report("Part B: LIVE Space, 300 held-out texts (optimistic, see docstring)", profiles,
                  {"live_best_node": best, "live_production": ranked}, trained_top, truths)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-live", action="store_true", help="run part A only")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    args = ap.parse_args()
    out_path = OUT_DIR / f"profile_eval{'_' + args.tag if args.tag else ''}.json"

    sents, emb = load_sentences(), load_embeddings()
    split = load_split()
    labels = list(split["labels"])
    y, train_idx, test_idx = split["y"], split["train_idx"], split["test_idx"]

    print("Training linear head on the train split (same config as 02_train.py 'plain')...")
    head = train_head(emb[train_idx], y[train_idx], len(labels))
    with torch.no_grad():
        logits = head(torch.from_numpy(emb[test_idx])).numpy()
    trained_top = [[labels[c] for c in row] for row in np.argsort(-logits, axis=1)[:, :10]]
    # The stronger flat classifier from 10_top_down.py, with the C it chose.
    c = json.loads((OUT_DIR / "top_down.json").read_text(encoding="utf-8"))["chosen"]["C"]
    print(f"Fitting logistic regression (C={c})...")
    lr = LogisticRegression(C=c, max_iter=2000).fit(emb[train_idx], y[train_idx])
    lr_proba = lr.predict_log_proba(emb[test_idx])
    logreg_top = [[labels[lr.classes_[j]] for j in row] for row in np.argsort(-lr_proba, axis=1)[:, :10]]

    out = {"part_a": part_a(sents, emb, labels, y, test_idx, trained_top, logreg_top)}
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if not args.skip_live:
        out["part_b"] = part_b(sents, labels, y, test_idx, trained_top)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
