#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_full_pipeline.py - test the deployed system end to end, using its own code.

Imports scripts/rhizome_engine.py unchanged (the same file the Space runs) and:
  1. Validation: rebuilds a coredrill from ALL records with the real
     build_coredrill() (CLI defaults) and compares it to the stored
     coredrill (EVAL_COREDRILL).
  2. Builds a train-only coredrill with the same function and defaults
     (every record except the held-out test set), round-tripped through
     JSON exactly as the deployed API loads it.
  3. Runs the real analyze_text_rhizome_v2() on every held-out test text and
     takes the final answer the way api/app.py's _build_response() does:
     gram_winner unless missing or in SKIP_NODES, else recommended_winner,
     else the top production score, else predict()'s best_node.

The only substitution is a stand-in model whose encode() returns the stored
LaBSE embedding for the current text (verified identical to live encoding,
cosine 1.000000). analyze_text_rhizome() also encodes micro-texts to build a
barycenter, but never uses the barycenter for scoring, so the stand-in
returning the same vector for those calls changes nothing.
"""
import contextlib
import io
import json
import sys
import time
from collections import Counter

import numpy as np

from common import (ENGINE_DIR, OUT_DIR, TAXONOMY_PATH, load_coredrill, load_embeddings,
                    load_sentences, load_split)

sys.path.insert(0, str(ENGINE_DIR))
import rhizome_engine as eng  # noqa: E402


class StoredEmbeddingModel:
    """Stand-in for SentenceTransformer: returns the precomputed embedding."""

    def __init__(self):
        self.current = None

    def encode(self, texts, **kwargs):
        return np.tile(self.current, (len(texts), 1)).astype(np.float32)


def build(records, emb, taxonomy) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        cd = eng.build_coredrill(records, emb, taxonomy)
    # Round-trip through JSON so centroids are plain lists, as the API sees them.
    return json.loads(json.dumps(eng.to_native(cd)))


def validate(records, emb, taxonomy) -> dict:
    stored = load_coredrill()
    rebuilt = build(records, emb, taxonomy)
    cos = []
    for name, st in stored["tree"].items():
        a, b = st.get("centroid"), rebuilt["tree"].get(name, {}).get("centroid")
        if a is not None and b is not None:
            a, b = np.asarray(a), np.asarray(b)
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    cos = np.array(cos)
    out = {
        "stored_n_source_records": stored.get("n_source_records"),
        "rebuilt_n_source_records": rebuilt.get("n_source_records"),
        "stored_flat_prototype_accuracy": stored.get("flat_prototype_accuracy"),
        "rebuilt_flat_prototype_accuracy": rebuilt.get("flat_prototype_accuracy"),
        "centroid_cosine_min": float(cos.min()),
        "centroid_cosine_mean": float(cos.mean()),
        "n_nodes_compared": int(len(cos)),
    }
    print("=== 1. Validation: rebuilt (all records) vs stored coredrill ===")
    for k, v in out.items():
        print(f"  {k:<34s} {v}")
    return out


def ancestors(node, parents_map, memo):
    if node in memo:
        return memo[node]
    out = set()
    for p in parents_map.get(node, []):
        out.add(p)
        out |= ancestors(p, parents_map, memo)
    memo[node] = out
    return out


def near_miss(true, pred, parents_map, memo):
    if pred == true:
        return True
    if pred in ancestors(true, parents_map, memo) or true in ancestors(pred, parents_map, memo):
        return True
    return bool(set(parents_map.get(true, [])) & set(parents_map.get(pred, [])))


def deployed_answer(raw: dict) -> str:
    """Same precedence as api/app.py _build_response()."""
    gram = raw.get("gram_winner")
    if gram and gram not in eng.SKIP_NODES:
        return gram
    rec = raw.get("recommended_winner")
    if rec and isinstance(rec, dict) and rec.get("node"):
        return rec["node"]
    if raw.get("results"):
        return raw["results"][0]["node"]
    return (raw.get("prediction") or {}).get("best_node", "unknown")


def main():
    sents = load_sentences()
    emb = load_embeddings()
    split = load_split()
    labels = list(split["labels"])
    y, test_idx = split["y"], split["test_idx"]
    taxonomy = eng.load_taxonomy(TAXONOMY_PATH)

    t0 = time.perf_counter()
    validation = validate(sents, emb, taxonomy)
    print(f"  (built in {time.perf_counter() - t0:.0f}s)\n")

    test_set = set(test_idx.tolist())
    train_rows = [i for i in range(len(sents)) if i not in test_set]
    t0 = time.perf_counter()
    cd = build([sents[i] for i in train_rows], emb[train_rows], taxonomy)
    (OUT_DIR / "coredrill_train_only.json").write_text(json.dumps(cd), encoding="utf-8")
    # Mirror api/app.py: the coredrill's own skip list replaces the engine default.
    if "skip_nodes" in cd:
        eng.SKIP_NODES.clear()
        eng.SKIP_NODES.update(cd["skip_nodes"])
    print(f"=== 2. Train-only coredrill: {len(train_rows)} records, "
          f"{cd['n_nodes_with_data']}/{cd['n_nodes']} nodes with data "
          f"(built in {time.perf_counter() - t0:.0f}s) ===\n")

    model = StoredEmbeddingModel()
    tau = cd.get("tau", 0.1)
    tree = cd["tree"]
    rows = []
    t0 = time.perf_counter()
    print(f"=== 3. analyze_text_rhizome_v2 on {len(test_idx)} held-out texts ===")
    for n, i in enumerate(test_idx):
        model.current = emb[i]
        raw = eng.analyze_text_rhizome_v2(cd, sents[i]["text"], model, tau=tau, print_output=False)
        results = raw.get("results", [])
        rec = raw.get("recommended_winner") or {}
        pred = raw.get("prediction") or {}
        rows.append({
            "true": labels[y[i]],
            "deployed": deployed_answer(raw),
            "gram_winner": raw.get("gram_winner"),
            "gram_state": raw.get("gram_state"),
            "recommended": rec.get("node"),
            "production_top": results[0]["node"] if results else None,
            "cosine_top": max(results, key=lambda r: r["score"])["node"] if results else None,
            "predict_best": pred.get("best_node"),
            "predict_best_level": pred.get("best_level"),
        })
        if (n + 1) % 500 == 0:
            el = time.perf_counter() - t0
            print(f"  {n + 1}/{len(test_idx)}  ({el:.0f}s elapsed, ~{el / (n + 1) * (len(test_idx) - n - 1):.0f}s left)")

    parents_map = cd["taxonomy_backbone"]["parents_map"]
    memo = {}
    label_set = set(labels)
    rules = ["deployed", "gram_winner", "recommended", "production_top", "cosine_top", "predict_best"]
    metrics = {}
    print(f"\n=== Results (n={len(rows)}) ===")
    print(f"  {'decision rule':<16} {'strict':>8} {'near-miss':>10} {'outside 199':>12}")
    for rule in rules:
        strict = np.mean([r[rule] == r["true"] for r in rows])
        lenient = np.mean([r[rule] is not None and near_miss(r["true"], r[rule], parents_map, memo) for r in rows])
        outside = np.mean([r[rule] not in label_set for r in rows])
        metrics[rule] = {"strict": float(strict), "near_miss": float(lenient), "outside_label_set": float(outside)}
        print(f"  {rule:<16} {strict:8.4f} {lenient:10.4f} {outside:12.4f}")

    states = Counter(r["gram_state"] for r in rows)
    levels = Counter(r["predict_best_level"] for r in rows)
    print(f"\n  gram_state counts: {dict(states)}")
    print(f"  predict() best_node level counts: {dict(sorted(levels.items()))}")

    (OUT_DIR / "full_pipeline.json").write_text(json.dumps({
        "validation": validation,
        "n_test": len(rows),
        "metrics": metrics,
        "gram_state_counts": dict(states),
        "predict_best_level_counts": {str(k): v for k, v in levels.items()},
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "full_pipeline_preds.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {OUT_DIR / 'full_pipeline.json'}")


if __name__ == "__main__":
    main()
