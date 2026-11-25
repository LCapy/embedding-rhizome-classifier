"""Holdout scoring for an encoder, the way eval/held_out/12_by_article.py
scores, so every loop's numbers sit next to the existing benchmark
(eval/held_out/results/by_article.json):

  deployed     a coredrill built from the training rows with the real
               build_coredrill(), scored with the real analyze_text_rhizome_v2()
               and api/app.py's answer precedence; its active set is the profile
  centroid     nearest label mean of the training rows
  linear_head  10_top_down.py's flat linear head on the embeddings
  logreg       logistic regression on the embeddings (C from 10_top_down.py)
  head         the run's own prototype head (trained runs only)

Every system sees the same training rows and the same embeddings. Near-miss
walks the coredrill's full parent map (05_full_pipeline.py's definition).
"""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from .data import Corpus, ancestor_depths, concept, eng, english_title, record_keys, text_key, title_key

LOGREG_C = 10.0        # eval/held_out/results/top_down.json, "chosen"
RANK_DEPTH = 10        # ranked answers kept per text (12 keeps 10)
SYSTEMS = ("deployed", "centroid", "linear_head", "logreg", "head")
BENCH_SUBSETS = ("all classes", "same classes", "same classes, Wikipedia", "same classes, other sources")


class StoredEmbeddingModel:
    """Stands in for the SentenceTransformer: returns a precomputed embedding
    (the same device as eval/held_out/05_full_pipeline.py)."""

    def __init__(self):
        self.current = None

    def encode(self, texts, **kwargs):
        return np.tile(self.current, (len(texts), 1)).astype(np.float32)


def deployed_answer(raw: dict) -> str:
    """api/app.py _build_response()'s precedence."""
    gram = raw.get("gram_winner")
    if gram and gram not in eng.SKIP_NODES:
        return gram
    rec = raw.get("recommended_winner")
    if rec and isinstance(rec, dict) and rec.get("node"):
        return rec["node"]
    if raw.get("results"):
        return raw["results"][0]["node"]
    return (raw.get("prediction") or {}).get("best_node", "unknown")


def build_coredrill(records: List[dict], embs: np.ndarray, taxonomy: List[dict], node_cap: int = 0,
                    build: Optional[dict] = None) -> dict:
    """build_coredrill() on the given embeddings, round-tripped through JSON
    the way the API loads it."""
    with contextlib.redirect_stdout(io.StringIO()):
        cd = eng.build_coredrill(records, embs, taxonomy, per_node_cap=node_cap, **(build or {}))
    return json.loads(json.dumps(eng.to_native(cd)))


def train_flat_head(X: np.ndarray, y: np.ndarray, n_classes: int):
    """eval/held_out/10_top_down.py's train_flat_head(), unchanged: a linear
    layer on the embeddings, Adam, 40 epochs, batch 256, seed 42, on the CPU."""
    import torch
    import torch.nn as nn
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


def near_miss_fn(parents_map: Dict[str, List[str]]) -> Callable[[str, Optional[str]], bool]:
    """05_full_pipeline.py's near_miss with a memo: the label itself, an
    ancestor or descendant of it, or a sibling."""
    memo: Dict[str, set] = {}

    def anc(node: str) -> set:
        if node not in memo:
            memo[node] = set(ancestor_depths(node, parents_map))
        return memo[node]

    def near(true: str, pred: Optional[str]) -> bool:
        if pred is None:
            return False
        if pred == true or pred in anc(true) or true in anc(pred):
            return True
        return bool(set(parents_map.get(true, [])) & set(parents_map.get(pred, [])))

    return near


def evaluate(corpus: Corpus, train_records: List[dict], train_embs: np.ndarray, test_idx: np.ndarray,
             test_embs: np.ndarray, head_logits: Optional[np.ndarray] = None,
             extra_systems: Optional[Dict[str, List[str]]] = None, classic: bool = True,
             blocked_titles: Optional[Set[str]] = None, log=print, node_cap: int = 0,
             build: Optional[dict] = None) -> dict:
    """Scores every system on the holdout. Returns
      metrics      metrics[system][subset] = {strict, near_miss, n}; the gate reads these
      benchmark    12_by_article.py's layout, comparable with by_article.json's "by_article"
      coredrill    the deployed system's coredrill
      predictions  one row per holdout text: truth, every system's answer, ranks and profile
    classic=False skips logistic regression and the linear head."""
    labels, index = corpus.labels, corpus.label_index
    truths = [labels[corpus.y[i]] for i in test_idx]
    train_y = np.array([index[r["category"]] for r in train_records], dtype=np.int64)
    X_tr = np.ascontiguousarray(train_embs, dtype=np.float32)
    X_te = np.ascontiguousarray(test_embs, dtype=np.float32)

    log(f"  building the coredrill from {len(train_records):,} training rows ...")
    cd = build_coredrill(train_records, X_tr, corpus.taxonomy, node_cap, build)
    eng.SKIP_NODES.clear()                       # mirror api/app.py: the coredrill's own list
    eng.SKIP_NODES.update(cd.get("skip_nodes", []))
    near = near_miss_fn(cd["taxonomy_backbone"]["parents_map"])

    answers: Dict[str, List[str]] = {}
    ranks: Dict[str, List[List[str]]] = {}

    centroids = np.zeros((len(labels), X_tr.shape[1]), dtype=np.float32)
    for c in range(len(labels)):
        rows = train_y == c
        if rows.any():
            mu = X_tr[rows].mean(axis=0)
            centroids[c] = mu / (np.linalg.norm(mu) + 1e-12)
    answers["centroid"] = [labels[j] for j in (X_te @ centroids.T).argmax(axis=1)]

    if classic:
        import torch
        from sklearn.linear_model import LogisticRegression
        t0 = time.perf_counter()
        log("  fitting logistic regression and the linear head ...")
        lr = LogisticRegression(C=LOGREG_C, max_iter=2000).fit(X_tr, train_y)
        order = np.argsort(-lr.predict_log_proba(X_te), axis=1)[:, :RANK_DEPTH]
        ranks["logreg"] = [[labels[lr.classes_[j]] for j in row] for row in order]
        lin = train_flat_head(X_tr, train_y, len(labels))
        with torch.no_grad():
            order = np.argsort(-lin(torch.from_numpy(X_te)).numpy(), axis=1)[:, :RANK_DEPTH]
        ranks["linear_head"] = [[labels[j] for j in row] for row in order]
        log(f"  logistic regression and linear head fitted ({time.perf_counter() - t0:.0f}s)")
    if head_logits is not None:
        ranks["head"] = [[labels[j] for j in row] for row in np.argsort(-head_logits, axis=1)[:, :RANK_DEPTH]]
    for name, r in ranks.items():
        answers[name] = [row[0] for row in r]
    answers.update(extra_systems or {})

    t0 = time.perf_counter()
    log(f"  scoring the deployed pipeline on {len(test_idx):,} holdout texts ...")
    model, tau = StoredEmbeddingModel(), cd.get("tau", 0.1)
    deployed, profiles = [], []
    for i, e in zip(test_idx, X_te):
        model.current = e
        with contextlib.redirect_stdout(io.StringIO()):
            raw = eng.analyze_text_rhizome_v2(cd, corpus.records[i]["text"], model, tau=tau, print_output=False)
        deployed.append(deployed_answer(raw))
        profiles.append([r["node"] for r in raw.get("active_overlap_set", [])])
    answers["deployed"] = deployed
    log(f"  deployed pipeline scored {len(test_idx):,} texts ({time.perf_counter() - t0:.0f}s)")
    systems = [s for s in SYSTEMS if s in answers] + [s for s in answers if s not in SYSTEMS]

    is_wiki = np.array([concept(corpus.records[i]) is not None for i in test_idx], dtype=bool)
    is_en = np.array([(corpus.records[i].get("lang") or "en") == "en" for i in test_idx], dtype=bool)

    # metrics: what the gate and the console read
    subsets = {"all": np.ones(len(test_idx), dtype=bool), "wikipedia": is_wiki, "other_sources": ~is_wiki,
               "english": is_en, "other_languages": ~is_en}
    metrics: dict = {"n": int(len(test_idx))}
    for name in systems:
        metrics[name] = {}
        for sub, mask in subsets.items():
            if mask.any():
                sel = np.where(mask)[0]
                metrics[name][sub] = {
                    "strict": round(float(np.mean([answers[name][j] == truths[j] for j in sel])), 4),
                    "near_miss": round(float(np.mean([near(truths[j], answers[name][j]) for j in sel])), 4),
                    "n": int(len(sel))}
    if "head" in ranks:
        metrics["head"]["all"]["top3"] = round(float(np.mean([t in r[:3] for t, r in zip(truths, ranks["head"])])), 4)
        metrics["head"]["all"]["top5"] = round(float(np.mean([t in r[:5] for t, r in zip(truths, ranks["head"])])), 4)
    metrics["deployed"]["all"]["profile_recall"] = round(float(np.mean([t in p for t, p in zip(truths, profiles)])), 4)
    metrics["deployed"]["all"]["profile_size"] = round(float(np.mean([len(p) for p in profiles])), 2)

    # benchmark: 12_by_article.py's summarize(), unrounded, same keys
    tested = set(truths)
    same = np.array([t in tested for t in truths], dtype=bool)
    parts = {"all classes": np.ones(len(truths), dtype=bool), "same classes": same,
             "same classes, Wikipedia": same & is_wiki, "same classes, other sources": same & ~is_wiki}
    results = {}
    for part, mask in parts.items():
        if not mask.any():
            continue
        sel = np.where(mask)[0]
        t = [truths[j] for j in sel]
        res: dict = {"n": int(len(sel)), "classes": int(len(set(t)))}
        for name in systems:
            a = [answers[name][j] for j in sel]
            res[name] = {"strict": float(np.mean([p == q for p, q in zip(a, t)])),
                         "near_miss": float(np.mean([near(q, p) for p, q in zip(a, t)]))}
        size = [max(len(profiles[j]), 1) for j in sel]
        res["profile"] = {"mean_size": float(np.mean([len(profiles[j]) for j in sel])),
                          "rhizome": float(np.mean([q in profiles[j] for j, q in zip(sel, t)]))}
        for name in ("logreg", "linear_head", "head"):
            if name in ranks:
                res["profile"][f"{name}_same_size"] = float(np.mean(
                    [q in ranks[name][j][:k] for j, q, k in zip(sel, t, size)]))
        results[part] = res
    seen = {concept(r) for r in train_records} - {None}
    wiki_test = [i for i in test_idx if concept(corpus.records[i])]
    benchmark = {"split": "by_article", "n_train": int(len(train_records)), "n_test": int(len(test_idx)),
                 "wiki_test_sharing_concept_with_train": [sum(concept(corpus.records[i]) in seen for i in wiki_test),
                                                          len(wiki_test)],
                 "results": results}
    if blocked_titles is not None:
        # Harvested pages carry no concept; check them against the holdout's articles by title.
        benchmark["harvested_rows_from_holdout_articles"] = sum(
            1 for r in train_records if r.get("source") == "wikipedia_harvest"
            and title_key(english_title(r) or "") in blocked_titles)
    # Exact text repeats across the split: the corpus has some (the reference split has them too).
    held_texts = {text_key(corpus.records[i]["text"]) for i in test_idx}
    benchmark["train_rows_repeating_a_holdout_text"] = sum(text_key(r["text"]) in held_texts for r in train_records)

    keys = record_keys(corpus.records)
    predictions = []
    for j, i in enumerate(test_idx):
        row = {"key": keys[i], "truth": truths[j], "wikipedia": bool(is_wiki[j]),
               "lang": corpus.records[i].get("lang") or "en", "profile": profiles[j]}
        for name in systems:
            row[name] = ranks[name][j] if name in ranks else answers[name][j]
        predictions.append(row)
    return {"metrics": metrics, "benchmark": benchmark, "coredrill": cd, "predictions": predictions}


def save_predictions(path: Path, predictions: List[dict]) -> None:
    """One JSON line per holdout text, gzipped."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_predictions(run_dir: Path) -> Optional[List[dict]]:
    """A loop's holdout_predictions.jsonl.gz, or None when it has none."""
    path = Path(run_dir) / "holdout_predictions.jsonl.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _answer(row: dict, system: str) -> Optional[str]:
    value = row.get(system)
    return value[0] if isinstance(value, list) else value


def paired_test(new: List[dict], old: List[dict], system: str = "deployed") -> Optional[dict]:
    """McNemar on the texts both scored: how many only the new model gets
    right, how many only the old one, and z = (gained - lost) / sqrt(gained + lost).
    z >= 1.645 is a gain at one-sided 95%."""
    by_key = {r["key"]: r for r in old}
    gained = lost = n = 0
    for r in new:
        o = by_key.get(r["key"])
        if o is None or system not in r or system not in o:
            continue
        n += 1
        a, b = _answer(r, system) == r["truth"], _answer(o, system) == o["truth"]
        gained += a and not b
        lost += b and not a
    if not n:
        return None
    z = (gained - lost) / math.sqrt(gained + lost) if gained + lost else 0.0
    return {"system": system, "n": n, "gained": gained, "lost": lost, "z": round(z, 3)}


def metric_value(metrics: dict, path: str) -> float:
    """'deployed.strict' -> metrics['deployed']['all']['strict']."""
    system, key = path.split(".", 1)
    return float(metrics[system]["all"][key])


def format_metrics(metrics: dict) -> str:
    rows = [f"  holdout: {metrics['n']:,} records"]
    for system in SYSTEMS:
        if system not in metrics:
            continue
        a = metrics[system]["all"]
        extra = ""
        if "top5" in a:
            extra = f"  top-5 {a['top5']:.3f}"
        if "profile_recall" in a:
            extra = f"  profile recall {a['profile_recall']:.3f} (size {a['profile_size']:.1f})"
        wiki = metrics[system].get("wikipedia", {})
        rows.append(f"  {system:<11} strict {a['strict']:.3f}  near-miss {a['near_miss']:.3f}{extra}"
                    + (f"  | Wikipedia strict {wiki['strict']:.3f}" if wiki else ""))
    return "\n".join(rows)
