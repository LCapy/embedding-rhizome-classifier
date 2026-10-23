#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_skip_list_check.py - does the 26-node skip list help or hurt, once parent
centroids come from their own records?

The engine and the live v4.0 coredrill carry the same 26-node skip list,
added because those nodes' centroids were diffuse (sigma > 0.70) and won
everywhere. Here the list is scored against no skip list at all on the same
held-out texts, with the train-only coredrill 05_full_pipeline.py built
(run 05 first), so the choice for the rebuilt coredrill is measured rather
than guessed. Reports the deployed answer's strict and near-miss accuracy,
how concentrated its answers are on a few nodes (hubs), and how often the
formerly skipped nodes win.
"""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import OUT_DIR, load_embeddings, load_sentences, load_split

_spec = importlib.util.spec_from_file_location("pipeline", Path(__file__).parent / "05_full_pipeline.py")
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)
eng = pipeline.eng

LIVE_SKIP_26 = [
    "Agree", "Agreement Formation", "Clarify", "Commuting", "Dialogue Scene", "Domestic Planning",
    "Emotion & Social Bonding", "Empathy & Support", "Home Life", "Home Repair Request",
    "Household Maintenance", "Instruction & Procedure", "Jobs & Professions", "Meal Preparation",
    "Meaning Relations", "Move", "Ordering at a Restaurant", "Project Agreement", "Repair", "Report",
    "Request–Response Dialogue", "Restaurant & Ordering", "Task Assignment", "Transactions",
    "Trip Planning", "Visual Arts",
]


def run(cd, sents, emb, labels, y, test_idx, skip):
    eng.SKIP_NODES.clear()
    eng.SKIP_NODES.update(skip)
    model = pipeline.StoredEmbeddingModel()
    parents_map = cd["taxonomy_backbone"]["parents_map"]
    memo = {}
    preds, truths = [], []
    for i in test_idx:
        model.current = emb[i]
        raw = eng.analyze_text_rhizome_v2(cd, sents[i]["text"], model, tau=cd.get("tau", 0.1), print_output=False)
        preds.append(pipeline.deployed_answer(raw))
        truths.append(labels[y[i]])
    strict = float(np.mean([p == t for p, t in zip(preds, truths)]))
    near = float(np.mean([pipeline.near_miss(t, p, parents_map, memo) for p, t in zip(preds, truths)]))
    top4 = Counter(preds).most_common(4)
    top4_share = sum(c for _, c in top4) / len(preds)
    unwinnable = float(np.mean([t in skip for t in truths]))
    on_list = [(p, t) for p, t in zip(preds, truths) if t in LIVE_SKIP_26]
    return {"skip_nodes": len(skip), "strict": strict, "near_miss": near,
            "top4_answer_share": top4_share, "top4_answers": top4,
            "test_labels_unwinnable": unwinnable,
            "answers_on_listed_nodes": float(np.mean([p in LIVE_SKIP_26 for p in preds])),
            "strict_on_listed_labels": float(np.mean([p == t for p, t in on_list])) if on_list else None}


def main():
    sents, emb = load_sentences(), load_embeddings()
    split = load_split()
    labels = list(split["labels"])
    y, test_idx = split["y"], split["test_idx"]
    cd = json.loads((OUT_DIR / "coredrill_train_only.json").read_text(encoding="utf-8"))

    results = {}
    for name, skip in [("skip_26", LIVE_SKIP_26), ("no_skip", [])]:
        r = run(cd, sents, emb, labels, y, test_idx, set(skip))
        results[name] = r
        print(f"{name:<15} skip={r['skip_nodes']:>2}  strict={r['strict']:.4f}  near_miss={r['near_miss']:.4f}  "
              f"top-4 answers take {r['top4_answer_share']:.1%}  unwinnable labels {r['test_labels_unwinnable']:.1%}  "
              f"answers on listed nodes {r['answers_on_listed_nodes']:.1%}")
        print(f"{'':15} top answers: {r['top4_answers']}")
    (OUT_DIR / "skip_list_check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_DIR / 'skip_list_check.json'}")


if __name__ == "__main__":
    main()
