#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_by_article.py - the held-out numbers again, with a split by article.

01's split is by record: 4,863 of its 4,867 Wikipedia test records come from
an article that also has training records, so a test paragraph's neighbours
from the same article sit in the training set. Here every Wikipedia record
is grouped by its concept (all language versions of an article, plus its
Simple English version) and whole concepts go to train or test. Records
with no article (bitext, CFPB, ChatDoctor) are still split by record.

A concept goes to test only if every class it feeds keeps at least
MIN_TRAIN training records, so a class whose only article is in test is not
silently made unlearnable; classes with a single article therefore stay
untested on their Wikipedia side.

Both splits are scored with the same methods: nearest centroid, logistic
regression (C from 10_top_down.py), the plain linear head, and the deployed
pipeline (train-only coredrill built with the real build_coredrill(), real
analyze_text_rhizome_v2(), api/app.py's answer precedence) with its profile.
Results are broken down into Wikipedia-sourced and other test records.
"""
import contextlib
import importlib.util
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from common import OUT_DIR, load_embeddings, load_sentences, load_split

HERE = Path(__file__).parent
TEST_FRAC = 0.2
MAX_TEST_FRAC = 0.35  # a concept may overshoot a class's 20% up to this share
MIN_TRAIN = 5
SEED = 42


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pipeline = load_module("pipeline", HERE / "05_full_pipeline.py")
td = load_module("top_down", HERE / "10_top_down.py")
eng = pipeline.eng


def concept(r):
    """Group key for Wikipedia records, None for records with no article."""
    src = r.get("source") or ""
    if src == "wikipedia_main_article":
        return "wiki:" + (r.get("source_article") or r.get("article_title") or "").lower()
    if src in ("wikipedia", "simple_wikipedia"):
        title = r.get("article_title") or ""
        if title.startswith("simple:"):
            title = title.split(":", 1)[1]
        return "wiki:" + title.lower()
    return None


def by_article_split(sents, y, n_classes):
    rng = np.random.default_rng(SEED)
    labeled = np.where(y >= 0)[0]
    total = np.bincount(y[labeled], minlength=n_classes)
    groups, loose = defaultdict(list), defaultdict(list)
    for i in labeled:
        k = concept(sents[i])
        if k:
            groups[k].append(i)
        else:
            loose[y[i]].append(i)

    test, test_cnt = [], np.zeros(n_classes, dtype=int)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        idx = np.array(groups[k])
        cls = np.bincount(y[idx], minlength=n_classes)
        inv = cls > 0
        after = test_cnt + cls
        wanted = np.any(test_cnt[inv] < TEST_FRAC * total[inv])
        fits = np.all(after[inv] <= MAX_TEST_FRAC * total[inv]) and np.all(total[inv] - after[inv] >= MIN_TRAIN)
        if wanted and fits:
            test.extend(idx.tolist())
            test_cnt = after
    for c, idx in loose.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        take = int(max(0, min(round(TEST_FRAC * total[c]) - test_cnt[c], len(idx) - MIN_TRAIN)))
        test.extend(idx[:take].tolist())
        test_cnt[c] += take
    test_idx = np.array(sorted(test))
    train_idx = np.setdiff1d(labeled, test_idx)
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


def leak_share(sents, train_idx, test_idx):
    seen = {concept(sents[i]) for i in train_idx} - {None}
    wiki_test = [i for i in test_idx if concept(sents[i])]
    return sum(concept(sents[i]) in seen for i in wiki_test), len(wiki_test)


def evaluate(name, sents, emb, labels, y, train_idx, test_idx, C, taxonomy):
    t0 = time.perf_counter()
    truths = [labels[c] for c in y[test_idx]]
    X = emb[test_idx]

    centroids = np.zeros((len(labels), emb.shape[1]), dtype=np.float32)
    for c in range(len(labels)):
        rows = train_idx[y[train_idx] == c]
        if len(rows):
            mu = emb[rows].mean(axis=0)
            centroids[c] = mu / (np.linalg.norm(mu) + 1e-12)
    lr = LogisticRegression(C=C, max_iter=2000).fit(emb[train_idx], y[train_idx])
    lr_rank = [[labels[lr.classes_[j]] for j in row] for row in np.argsort(-lr.predict_log_proba(X), axis=1)[:, :10]]
    head = td.train_flat_head(emb[train_idx], y[train_idx], len(labels))
    with torch.no_grad():
        head_rank = [[labels[j] for j in row]
                     for row in np.argsort(-head(torch.from_numpy(X)).numpy(), axis=1)[:, :10]]

    cd = pipeline.build([sents[i] for i in train_idx], emb[train_idx], taxonomy)
    eng.SKIP_NODES.clear()
    eng.SKIP_NODES.update(cd.get("skip_nodes", []))
    model = pipeline.StoredEmbeddingModel()
    deployed, profiles = [], []
    for i in test_idx:
        model.current = emb[i]
        with contextlib.redirect_stdout(io.StringIO()):
            raw = eng.analyze_text_rhizome_v2(cd, sents[i]["text"], model, tau=cd.get("tau", 0.1), print_output=False)
        deployed.append(pipeline.deployed_answer(raw))
        profiles.append([r["node"] for r in raw.get("active_overlap_set", [])])
    print(f"  {name}: scored {len(test_idx)} texts in {time.perf_counter() - t0:.0f}s")

    # The deployed answer can be any taxonomy node, not only one of the
    # labels, so near-miss walks the full parent map (05's definition).
    return {
        "parents_map": cd["taxonomy_backbone"]["parents_map"],
        "truths": truths, "profiles": profiles, "lr_rank": lr_rank, "head_rank": head_rank,
        "is_wiki": np.array([concept(sents[i]) is not None for i in test_idx]),
        "answers": {
            "deployed": deployed,
            "centroid": [labels[j] for j in (X @ centroids.T).argmax(axis=1)],
            "linear_head": [r[0] for r in head_rank],
            "logreg": [r[0] for r in lr_rank],
        },
    }


def summarize(run, mask):
    sel = np.where(mask)[0]
    t = [run["truths"][j] for j in sel]
    res = {"n": int(len(sel)), "classes": int(len(set(t)))}
    memo = {}
    for m, ans in run["answers"].items():
        a = [ans[j] for j in sel]
        res[m] = {"strict": float(np.mean([p == q for p, q in zip(a, t)])),
                  "near_miss": float(np.mean([pipeline.near_miss(q, p, run["parents_map"], memo)
                                              for p, q in zip(a, t)]))}
    size = [max(len(run["profiles"][j]), 1) for j in sel]
    res["profile"] = {
        "mean_size": float(np.mean([len(run["profiles"][j]) for j in sel])),
        "rhizome": float(np.mean([q in run["profiles"][j] for j, q in zip(sel, t)])),
        "logreg_same_size": float(np.mean([q in run["lr_rank"][j][:k] for j, q, k in zip(sel, t, size)])),
        "linear_head_same_size": float(np.mean([q in run["head_rank"][j][:k] for j, q, k in zip(sel, t, size)])),
    }
    return res


def main():
    sents, emb = load_sentences(), load_embeddings()
    split = load_split()
    labels = list(split["labels"])
    y = split["y"]
    C = json.loads((OUT_DIR / "top_down.json").read_text(encoding="utf-8"))["chosen"]["C"]
    taxonomy = eng.load_taxonomy(pipeline.TAXONOMY_PATH)

    tr_a, te_a = by_article_split(sents, y, len(labels))
    np.savez(OUT_DIR / "split_by_article.npz", train_idx=tr_a, test_idx=te_a)
    splits = {"by_record": (split["train_idx"], split["test_idx"]), "by_article": (tr_a, te_a)}
    runs, out = {}, {}
    for name, (tr, te) in splits.items():
        shared, n_wiki = leak_share(sents, tr, te)
        print(f"{name}: {len(tr)} train / {len(te)} test; Wikipedia test records whose concept "
              f"has training records: {shared}/{n_wiki}")
        runs[name] = evaluate(name, sents, emb, labels, y, tr, te, C, taxonomy)
        out[name] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                     "wiki_test_sharing_concept_with_train": [shared, n_wiki]}

    # The by-article split can only test classes fed by several concepts (or
    # by records with no article), so compare both splits on those classes.
    tested = set(runs["by_article"]["truths"])
    parts = {}
    for name, run in runs.items():
        same = np.array([t in tested for t in run["truths"]])
        wiki = run["is_wiki"]
        parts[name] = {"all classes": np.ones(len(same), bool), "same classes": same,
                       "same classes, Wikipedia": same & wiki, "same classes, other sources": same & ~wiki}
        out[name]["results"] = {p: summarize(run, m) for p, m in parts[name].items() if m.any()}
    out["classes_tested_by_article"] = sorted(tested)
    (OUT_DIR / "by_article.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for part in parts["by_record"]:
        rows = {s: out[s]["results"].get(part) for s in splits}
        print(f"\n== {part}: " + ", ".join(f"{s} n={r['n']} ({r['classes']} classes)" for s, r in rows.items() if r))
        print(f"  {'strict / near-miss':<30}" + "".join(f"{s:>22}" for s in splits))
        for m in ("deployed", "centroid", "linear_head", "logreg"):
            cells = ["%.3f / %.3f" % (r[m]["strict"], r[m]["near_miss"]) if r else "-" for r in rows.values()]
            print(f"  {m:<30}" + "".join(f"{c:>22}" for c in cells))
        for k in ("rhizome", "logreg_same_size", "linear_head_same_size"):
            cells = ["%.3f" % r["profile"][k] if r else "-" for r in rows.values()]
            print(f"  {'profile: ' + k:<30}" + "".join(f"{c:>22}" for c in cells))
    print(f"\nSaved -> {OUT_DIR / 'by_article.json'}")


if __name__ == "__main__":
    main()
