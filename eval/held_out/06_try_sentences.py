#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_try_sentences.py - run real sentences through three systems side by side:

  deployed  - this repo's api/app.py imported in-process and called through
              its real POST /predict endpoint (FastAPI TestClient). Same code
              as the HF Space, same LaBSE model; the coredrill is the local
              EVAL_COREDRILL file, since the live Space pulls its own.
  trained   - plain linear head (02_train.py's "plain" config) fitted on all
              33,778 labeled records, as you would for production use.
  centroid  - plain nearest-centroid over the same 199 classes, centroids
              from all labeled records.

Inputs: the curated multilingual files in EVAL_TEST_TEXTS plus a few short
probes. Writes artifacts/try_sentences.json and artifacts/try_sentences.md.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from common import (COREDRILL_PATH, OUT_DIR, REPO, TEST_TEXTS_DIR, build_label_set,
                    compute_centroids, load_embeddings, load_sentences)

os.environ["COREDRILL_PATH"] = str(COREDRILL_PATH)
os.environ.pop("API_KEY", None)
sys.path.insert(0, str(REPO))

from fastapi.testclient import TestClient  # noqa: E402
import api.app as appmod  # noqa: E402

TEST_TEXTS = TEST_TEXTS_DIR
PROBES = [
    ("en", "I was charged twice for the same order and I want a full refund."),
    ("en", "this email is about the insurance claim missing from client ABC."),
    ("en", "Natural selection acts on heritable variation, favoring traits that increase reproductive fitness."),
    ("en", "My doctor prescribed antibiotics for the infection."),
    ("en", "Influenza is a contagious respiratory illness caused by influenza viruses."),
    ("fr", "Les élections législatives ont donné la majorité au parti d'opposition."),
    ("ru", "Могу ли я отменить заказ?"),
    ("ja", "量子力学は原子や素粒子の振る舞いを記述する物理学の理論である。"),
]


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


def load_inputs():
    items = [(f.stem, f.read_text(encoding="utf-8").strip()) for f in sorted(TEST_TEXTS.glob("*.txt"))]
    items += [(f"probe_{lang}_{k + 1}", text) for k, (lang, text) in enumerate(PROBES)]
    return items


def main():
    client = TestClient(appmod.app)
    health = client.get("/health").json()
    print(f"Deployed API loaded in-process: {health['coredrill_nodes']} nodes, model={health['model']}")
    model = appmod._get_model()

    sents, emb = load_sentences(), load_embeddings()
    labels = build_label_set(sents)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y = np.array([label_to_idx.get(r.get("category", ""), -1) for r in sents])
    rows = np.where(y >= 0)[0]
    centroids = compute_centroids(emb, y, rows, len(labels))
    print(f"Training linear head on {len(rows)} labeled records ({len(labels)} classes)...")
    head = train_head(emb[rows], y[rows], len(labels))

    results = []
    for name, text in load_inputs():
        x = model.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        api = client.post("/predict", json={"text": text, "full": True}).json()
        with torch.no_grad():
            probs = torch.softmax(head(torch.from_numpy(x)), dim=1)[0].numpy()
        top3 = np.argsort(-probs)[:3]
        cos = (x @ centroids.T)[0]
        c_top = int(cos.argmax())
        results.append({
            "name": name,
            "text": text,
            "deployed": {
                "best_node": api.get("best_node"),
                "lineage": api.get("lineage"),
                "confidence": api.get("confidence"),
                "flow": api.get("flow_label"),
                "active_overlap": [r["node"] for r in api.get("active_overlap", [])[:4]],
            },
            "trained_top3": [(labels[i], round(float(probs[i]), 3)) for i in top3],
            "centroid_top1": (labels[c_top], round(float(cos[c_top]), 3)),
        })
        r = results[-1]
        print(f"\n== {name}\n   {text[:90]}{'...' if len(text) > 90 else ''}")
        print(f"   deployed : {r['deployed']['best_node']}   [{' -> '.join(r['deployed']['lineage'] or [])}]")
        print(f"   trained  : " + ", ".join(f"{n} ({p:.2f})" for n, p in r["trained_top3"]))
        print(f"   centroid : {r['centroid_top1'][0]} (cos {r['centroid_top1'][1]:.2f})")

    (OUT_DIR / "try_sentences.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["| Input | Deployed API | Trained head (top 1) | Nearest centroid |", "|---|---|---|---|"]
    for r in results:
        md.append(f"| `{r['name']}` | {r['deployed']['best_node']} | {r['trained_top3'][0][0]} "
                  f"({r['trained_top3'][0][1]:.2f}) | {r['centroid_top1'][0]} |")
    (OUT_DIR / "try_sentences.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUT_DIR / 'try_sentences.json'} and try_sentences.md")


if __name__ == "__main__":
    main()
