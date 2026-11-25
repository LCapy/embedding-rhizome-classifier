"""Champion/challenger bookkeeping and the commands: baseline, fit, round,
loop, status, export.

A round harvests new text, labels it with the current champion, keeps only
labels that are confident and agree with a second signal, trains a
challenger on the labeled data plus every pseudo-label kept so far, and
promotes it only if it beats the champion on the fixed holdout. A rejected
round's pseudo-labels are set aside, so the pool only holds data from rounds
that helped.

Every baseline, fit and round is recorded (record.py): its own folder under
runs/ and a start and an end line in runs/ledger.jsonl, whatever happens.
"""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .data import (Corpus, Holdout, load_corpus, load_extra_labeled, load_or_create_holdout, neighborhood,
                   soft_target_matrix, split_indices, text_key)
from .evaluate import evaluate, format_metrics, load_predictions, metric_value, paired_test, save_predictions
from .fit import FitConfig, TrainSet, embed_trainset, fit
from .harvest import (Wikipedia, archive_inbox, inbox_candidates, known_titles, seed_titles,
                      translation_candidates, wiki_candidates)
from .live import LiveTap
from .model import BASE_MODEL, apply_delta, encode, free_gpu, head_probs, load_encoder, load_run, save_run
from .record import (LEDGER, LoopRecord, append_event, compare_to_reference, format_reference_check,
                     load_reference, loops, now, state)

ROUND_DONE = ("no_candidates", "nothing_accepted", "promoted", "not_promoted")


@dataclass
class Paths:
    data: Path
    runs: Path
    taxonomy: Optional[Path] = None
    live: bool = True               # capture the matrices for `python -m train view`

    @property
    def selftrain(self) -> Path:
        return self.data / "selftrain"

    @property
    def pool_records(self) -> Path:
        return self.selftrain / "pool.jsonl"

    @property
    def pool_f0(self) -> Path:
        return self.selftrain / "pool_f0.npy"

    @property
    def rejected(self) -> Path:
        return self.selftrain / "rejected.jsonl"

    @property
    def pending_records(self) -> Path:
        return self.selftrain / "pending.jsonl"

    @property
    def pending_f0(self) -> Path:
        return self.selftrain / "pending_f0.npy"

    @property
    def inbox(self) -> Path:
        return self.data / "inbox"

    @property
    def labeled(self) -> Path:
        return self.data / "labeled"

    @property
    def testset(self) -> Path:
        return self.data / "testset" / "candidates.jsonl"

    @property
    def holdout(self) -> Path:
        return self.runs / "holdout.json"

    @property
    def champion(self) -> Path:
        return self.runs / "champion.json"

    @property
    def baseline(self) -> Path:
        return self.runs / "baseline.json"


@dataclass
class RoundConfig:
    sources: str = "wiki,inbox"
    labels: str = "tested"          # labels that may gain rows: the ones the holdout tests, or "all"
    nodes_per_round: int = 40       # Wikipedia: nodes searched per round, fewest articles first
    per_node: int = 8               # articles per node
    paras_per_article: int = 2
    langs: str = "es,fr,de,pt"      # other-language versions of accepted articles ("" for none)
    conf: float = 0.8               # calibrated head probability a pseudo-label needs
    margin: float = 0.15            # over the runner-up
    per_class_frac: float = 0.5     # new pseudo-labels per label per round, as a share of its labeled rows
    min_per_class: int = 3
    max_new: int = 1500
    total_frac: float = 1.0         # self-labeled rows a label may have in all, as a share of its labeled rows
    unproven: str = "stack"         # a round that is not promoted: "stack" its rows with the next, or "drop" them
    wiki_delay: float = 0.3


@dataclass
class Gate:
    select_metric: str = "deployed.strict"   # what the API answers with
    min_delta: float = 0.002
    max_near_miss_drop: float = 0.01
    min_z: float = 1.645                     # McNemar z on the same texts: a gain at one-sided 95%


# ── small file helpers ───────────────────────────────────────────────────────

def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_champion(paths: Paths) -> Optional[dict]:
    return json.loads(paths.champion.read_text(encoding="utf-8")) if paths.champion.exists() else None


class Pool:
    """Pseudo-labels with their stock LaBSE embeddings: the ones from promoted
    rounds, or (pending=True) the ones no round has proven yet. Those train
    along with each new round's until the stack earns a promotion."""

    def __init__(self, paths: Paths, enabled: bool = True, pending: bool = False):
        self.paths = paths
        self.records_path = paths.pending_records if pending else paths.pool_records
        self.f0_path = paths.pending_f0 if pending else paths.pool_f0
        self.records = read_jsonl(self.records_path) if enabled else []
        self.f0 = (np.load(self.f0_path) if enabled and self.f0_path.exists()
                   else np.zeros((0, 768), dtype=np.float32))
        if len(self.records) != len(self.f0):
            raise SystemExit(f"{self.records_path} and {self.f0_path} are out of step "
                             f"({len(self.records)} vs {len(self.f0)} rows)")

    def clear(self) -> None:
        self.records, self.f0 = [], self.f0[:0]
        self._save()

    def add(self, records: List[dict], f0: np.ndarray) -> None:
        self.records = self.records + records
        self.f0 = np.vstack([self.f0, f0]).astype(np.float32) if len(self.f0) else f0.astype(np.float32)
        self._save()

    def _save(self) -> None:
        self.paths.selftrain.mkdir(parents=True, exist_ok=True)
        tmp = self.records_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.records), encoding="utf-8")
        tmp.replace(self.records_path)
        tmp_npy = self.f0_path.with_name(self.f0_path.stem + ".tmp.npy")
        np.save(tmp_npy, self.f0)
        tmp_npy.replace(self.f0_path)


def considered_titles(paths: Paths, champion: str) -> set:
    """Articles that finished rounds under this champion already judged. The
    champion would say the same again; a new one reconsiders everything."""
    out: set = set()
    for e in loops(paths.runs):
        if (e.get("kind") == "round" and e.get("champion_before") == champion and e.get("folder")
                and e.get("status") in ROUND_DONE):
            out |= known_titles(read_jsonl(Path(e["folder"]) / "candidates.jsonl"))
    return out


def beats(new: dict, old: dict, gate: Gate, new_preds: Optional[List[dict]] = None,
          old_preds: Optional[List[dict]] = None) -> Tuple[bool, str, Optional[dict]]:
    """(promote?, why, the paired test). Two baselines that differ only in
    floating-point rounding already differ by up to 8 of the 5,065 holdout
    texts, so a gain must also hold up text by text (McNemar), not only on
    the average."""
    a, b = metric_value(new, gate.select_metric), metric_value(old, gate.select_metric)
    system, key = gate.select_metric.split(".", 1)
    nm_new, nm_old = new[system]["all"]["near_miss"], old[system]["all"]["near_miss"]
    test = paired_test(new_preds, old_preds, system) if new_preds and old_preds and key == "strict" else None
    if a < b + gate.min_delta:
        return False, f"{gate.select_metric} {a:.4f} vs champion {b:.4f} (needs +{gate.min_delta})", test
    if nm_new < nm_old - gate.max_near_miss_drop:
        return False, f"{system} near-miss fell {nm_old:.4f} -> {nm_new:.4f}", test
    if test and test["z"] < gate.min_z:
        return False, (f"{gate.select_metric} {b:.4f} -> {a:.4f} is not significant: {test['gained']} texts gained, "
                       f"{test['lost']} lost, z {test['z']:.2f} < {gate.min_z}"), test
    sig = f" ({test['gained']} gained, {test['lost']} lost, z {test['z']:.2f})" if test else ""
    return True, f"{gate.select_metric} {b:.4f} -> {a:.4f}{sig}", test


def promote(paths: Paths, result: dict, reason: str, log=print) -> None:
    paths.champion.write_text(json.dumps({
        "run": result["run_dir"].name, "promoted": now(), "reason": reason,
        "metrics": result["metrics"], "benchmark": result["benchmark"]}, indent=2), encoding="utf-8")
    log(f"Promoted {result['run_dir'].name}: {reason}")


def format_against_reference(bench: dict) -> str:
    """One line: each system's strict accuracy minus the reference's, in points."""
    ref = load_reference()
    mine = bench["results"].get("all classes", {})
    theirs = (ref or {}).get("results", {}).get("all classes", {})
    parts = [f"{s} {100 * (mine[s]['strict'] - theirs[s]['strict']):+.1f}"
             for s in ("deployed", "centroid", "linear_head", "logreg") if s in mine and s in theirs]
    return "  strict against the reference benchmark, in points: " + (", ".join(parts) if parts else "n/a")


# ── training one model ───────────────────────────────────────────────────────

def assemble(corpus: Corpus, holdout: Holdout, extra: List[dict], pseudo: List[dict],
             pseudo_f0: np.ndarray, cfg: FitConfig):
    """(TrainSet, the records behind it, holdout indices). Extra hand-labeled
    rows have no stock embedding yet; their f0 rows are NaN until filled."""
    train_idx, test_idx = split_indices(corpus, holdout)
    index = corpus.label_index
    dim = corpus.f0.shape[1]
    records = [corpus.records[i] for i in train_idx] + extra + pseudo
    y = np.concatenate([corpus.y[train_idx],
                        np.array([index[r["category"]] for r in extra + pseudo], dtype=np.int64)])
    weight = np.concatenate([np.ones(len(train_idx) + len(extra)),
                             np.array([cfg.pseudo_weight * float(r.get("confidence", 1.0)) for r in pseudo])])
    f0 = np.vstack([corpus.f0[train_idx], np.full((len(extra), dim), np.nan, dtype=np.float32),
                    pseudo_f0.reshape(-1, dim)]).astype(np.float32)
    labeled = np.concatenate([np.ones(len(train_idx) + len(extra), dtype=bool), np.zeros(len(pseudo), dtype=bool)])
    if cfg.balance > 0:                          # a label's rows share its weight, up to 3x and down to a third
        size = np.bincount(y, weights=weight, minlength=len(corpus.labels))
        typical = float(np.median(size[size > 0]))
        weight = weight * np.clip((typical / np.maximum(size, 1e-9)) ** cfg.balance, 1 / 3, 3.0)[y]
    ts = TrainSet([r["text"] for r in records], y, weight, f0, labeled)
    return ts, records, test_idx


def train_and_evaluate(paths: Paths, corpus: Corpus, holdout: Holdout, cfg: FitConfig, rec: LoopRecord,
                       pool: Pool, pending: Optional[Tuple[List[dict], np.ndarray]], device) -> dict:
    """Trains a model from stock LaBSE, scores it, and saves it into the loop's folder."""
    log = rec.log
    extra = load_extra_labeled(paths.labeled, corpus.labels)
    pseudo, pseudo_f0 = list(pool.records), pool.f0
    if pending:
        pseudo = pseudo + pending[0]
        pseudo_f0 = np.vstack([pseudo_f0, pending[1]]) if len(pseudo_f0) else pending[1]
    # Added rows never repeat a holdout text; pseudo-labels whose category
    # stopped being a label are dropped with their embeddings.
    held = holdout.texts
    n_extra = len(extra)
    extra = [r for r in extra if text_key(r["text"]) not in held]
    keep = np.array([r.get("category") in corpus.label_index and text_key(r["text"]) not in held
                     for r in pseudo], dtype=bool)
    if n_extra - len(extra) or (~keep).sum():
        log(f"  left out: {n_extra - len(extra)} hand-labeled and {int((~keep).sum())} pseudo-labeled rows "
            "(holdout text, or no longer a label)")
    pseudo = [r for r, k in zip(pseudo, keep) if k]
    pseudo_f0 = pseudo_f0[keep] if len(keep) else pseudo_f0[:0]
    ts, records, test_idx = assemble(corpus, holdout, extra, pseudo, pseudo_f0, cfg)
    n_corpus = len(records) - len(extra) - len(pseudo)
    log(f"Training a {rec.kind} model: {n_corpus:,} labeled rows, {len(extra)} hand-added, "
        f"{len(pseudo):,} pseudo-labels; holdout {len(test_idx):,}")

    t0 = time.perf_counter()
    st = load_encoder(BASE_MODEL, device, cfg.max_seq_length)
    missing = np.isnan(ts.f0).any(axis=1)
    if missing.any():
        ts.f0[missing] = encode(st, [ts.texts[i] for i in np.where(missing)[0]], device)
    init = np.zeros((len(corpus.labels), ts.f0.shape[1]), dtype=np.float32)
    for k in range(len(corpus.labels)):
        rows = ts.y == k
        init[k] = ts.f0[rows].mean(axis=0) if rows.any() else np.random.default_rng(k).normal(size=init.shape[1])
    targets = soft_target_matrix(corpus.labels, corpus.parents, cfg.ancestor_eps)

    tap = LiveTap(rec.dir / "live", corpus.labels, log=log) if paths.live else None
    result = fit(ts, st, init, targets, cfg, device, log, tap)
    fit_seconds = time.perf_counter() - t0
    head = result["head"]
    meta = {
        "kind": rec.kind, "loop": rec.id, "created": now(), "base_model": BASE_MODEL,
        "unfreeze_layers": cfg.unfreeze_layers, "fit": asdict(cfg), "labels": corpus.labels,
        "temperature": result["temperature"], "scale": head.scale, "history": result["history"],
        "best_epoch": result["best_epoch"], "val_acc": round(result["val_acc"], 4),
        "data": {"labeled": n_corpus, "hand_added": len(extra), "pseudo": len(pseudo),
                 "holdout": int(len(test_idx))},
        "fit_seconds": round(fit_seconds, 1),
    }
    # Saved before scoring: an error while scoring must not cost the training.
    save_run(rec.dir, st, result["trainable"], head, meta)
    free_gpu()                                   # the optimizer's state, before embedding everything
    rec.update(training={k: meta[k] for k in ("unfreeze_layers", "temperature", "best_epoch", "val_acc",
                                              "history", "fit_seconds")}, rows=meta["data"])

    log(f"  embedding {len(ts.texts):,} training and {len(test_idx):,} holdout texts with the trained encoder ...")
    train_embs = embed_trainset(st, ts, result, device)
    test_embs = encode(st, [corpus.records[i]["text"] for i in test_idx], device)
    protos = head.protos.detach().cpu().numpy()
    head_logits = head.scale * test_embs @ (protos / np.linalg.norm(protos, axis=1, keepdims=True)).T
    log("  scoring the holdout: deployed pipeline, centroid, linear head, logistic regression, prototype head")
    ev = evaluate(corpus, records, train_embs, test_idx, test_embs, head_logits,
                  blocked_titles=holdout.blocked_titles, log=log, node_cap=cfg.node_cap)
    metrics, bench, cd = ev["metrics"], ev["benchmark"], ev["coredrill"]
    bench.update({"n_labeled": n_corpus, "n_hand_added": len(extra), "n_pseudo": len(pseudo)})
    meta.update({"metrics": metrics, "benchmark": bench, "seconds": round(time.perf_counter() - t0, 1)})
    (rec.dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cd["labse_model"] = f"(trained encoder of run {rec.id}: see `python -m train export`)"
    (rec.dir / "coredrill_hierarchical.json").write_text(json.dumps(cd), encoding="utf-8")
    save_predictions(rec.dir / "holdout_predictions.jsonl.gz", ev["predictions"])
    rec.update(metrics=metrics, benchmark=bench, reference_check=compare_to_reference(bench, load_reference()))
    log(format_metrics(metrics))
    log(format_against_reference(bench))
    log(f"  model saved in {rec.dir} ({meta['seconds']:.0f}s)")
    del st, result
    free_gpu()
    return {"run_dir": rec.dir, "metrics": metrics, "benchmark": bench, "meta": meta,
            "predictions": ev["predictions"]}


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_score(paths: Paths, run_dir: Path, device, node_cap: int = 0) -> dict:
    """An earlier model on this corpus and holdout: its encoder embeds the
    training rows here, and the coredrill is rebuilt from them."""
    run_dir = Path(run_dir).resolve()
    with LoopRecord(paths, "score", {"scored": str(run_dir), "node_cap": node_cap}) as rec:
        corpus = load_corpus(paths.data, paths.taxonomy)
        holdout = load_or_create_holdout(corpus, paths.holdout, rec.log)
        train_idx, test_idx = split_indices(corpus, holdout)
        st, head, meta = load_run(run_dir, device)
        rec.log(f"Scoring {run_dir.name} on {len(test_idx):,} holdout records; embedding {len(train_idx) + len(test_idx):,} texts ...")
        E = encode(st, [corpus.records[i]["text"] for i in np.concatenate([train_idx, test_idx])], device)
        train_embs, test_embs = E[:len(train_idx)], E[len(train_idx):]
        protos = head.protos.detach().cpu().numpy()
        mine = {l: i for i, l in enumerate(meta["labels"])}
        logits = np.full((len(test_idx), len(corpus.labels)), -1e9, dtype=np.float32)
        unit = protos / np.linalg.norm(protos, axis=1, keepdims=True)
        for k, l in enumerate(corpus.labels):
            if l in mine:
                logits[:, k] = head.scale * test_embs @ unit[mine[l]]
        del st, head
        free_gpu()
        ev = evaluate(corpus, [corpus.records[i] for i in train_idx], train_embs, test_idx, test_embs, logits,
                      blocked_titles=holdout.blocked_titles, log=rec.log, node_cap=node_cap)
        ev["benchmark"].update({"n_labeled": int(len(train_idx)), "n_hand_added": 0, "n_pseudo": 0})
        save_predictions(rec.dir / "holdout_predictions.jsonl.gz", ev["predictions"])
        rec.update(metrics=ev["metrics"], benchmark=ev["benchmark"])
        rec.log(format_metrics(ev["metrics"]))
        return ev["metrics"]


def cmd_baseline(paths: Paths, device, node_cap: int = 0) -> dict:
    """Stock LaBSE on the same holdout, every system 12_by_article.py scores:
    the deployed pipeline, nearest centroid, the linear head and logistic
    regression. Its numbers should match eval/held_out/results/by_article.json;
    the record says by how much they differ."""
    with LoopRecord(paths, "baseline", {"node_cap": node_cap}) as rec:
        corpus = load_corpus(paths.data, paths.taxonomy)
        holdout = load_or_create_holdout(corpus, paths.holdout, rec.log)
        train_idx, test_idx = split_indices(corpus, holdout)
        rec.log(f"Baseline: stock LaBSE, {len(train_idx):,} training records, {len(test_idx):,} holdout records")
        ev = evaluate(corpus, [corpus.records[i] for i in train_idx], corpus.f0[train_idx], test_idx,
                      corpus.f0[test_idx], blocked_titles=holdout.blocked_titles, log=rec.log, node_cap=node_cap)
        bench = ev["benchmark"]
        bench.update({"n_labeled": int(len(train_idx)), "n_hand_added": 0, "n_pseudo": 0})
        save_predictions(rec.dir / "holdout_predictions.jsonl.gz", ev["predictions"])
        check = compare_to_reference(bench, load_reference())
        rec.update(metrics=ev["metrics"], benchmark=bench, reference_check=check)
        paths.baseline.write_text(json.dumps({"created": now(), "loop": rec.id, "metrics": ev["metrics"],
                                              "benchmark": bench}, indent=2), encoding="utf-8")
        rec.log(format_metrics(ev["metrics"]))
        rec.log(format_reference_check(check))
        return ev["metrics"]


def cmd_fit(paths: Paths, cfg: FitConfig, gate: Gate, device, use_pool: bool = True) -> dict:
    with LoopRecord(paths, "fit", {"fit": asdict(cfg), "gate": asdict(gate), "use_pool": use_pool}) as rec:
        corpus = load_corpus(paths.data, paths.taxonomy)
        holdout = load_or_create_holdout(corpus, paths.holdout, rec.log)
        champ = read_champion(paths)
        rec.update(champion_before=(champ or {}).get("run"))
        result = train_and_evaluate(paths, corpus, holdout, cfg, rec, Pool(paths, enabled=use_pool), None, device)
        if champ is None:
            promoted, reason, test = True, "first trained model", None
        else:
            promoted, reason, test = beats(result["metrics"], champ["metrics"], gate, result["predictions"],
                                           load_predictions(paths.runs / champ["run"]))
        if promoted:
            promote(paths, result, reason, rec.log)
        else:
            rec.log(f"Not promoted: {reason}")
        rec.update(status="promoted" if promoted else "not_promoted", promoted=promoted, reason=reason,
                   paired_test=test, champion_after=(read_champion(paths) or {}).get("run"))
        return result


def _select(cands: List[dict], probs: np.ndarray, cent_pred: List[str], corpus: Corpus,
            rcfg: RoundConfig) -> Tuple[List[Tuple[int, int, float]], List[str]]:
    """Candidates whose label is confident and agrees with a second signal:
    for a Wikipedia page, the node whose articles led to it (the label must be
    that node or a relative of it); for an inbox text, the nearest centroid
    (it must be the label or a relative). Returns the chosen ones, best
    first, and the decision taken on every candidate."""
    order = np.argsort(-probs, axis=1)
    rows = np.arange(len(probs))
    p1, p2, pred = probs[rows, order[:, 0]], probs[rows, order[:, 1]], order[:, 0]
    cache: Dict[str, set] = {}

    def near(node: str) -> set:
        if node not in cache:
            cache[node] = neighborhood(node, corpus.parents, corpus.children)
        return cache[node]

    chosen, decisions = [], [""] * len(cands)
    for i in np.argsort(-p1):
        label = corpus.labels[pred[i]]
        seed = cands[i].get("seed_node")
        if p1[i] < rcfg.conf:
            decisions[i] = f"confidence {p1[i]:.3f} below {rcfg.conf}"
        elif p1[i] - p2[i] < rcfg.margin:
            decisions[i] = f"margin {p1[i] - p2[i]:.3f} below {rcfg.margin}"
        elif seed and label not in near(seed):
            decisions[i] = f"{label} is outside the neighborhood of seed node {seed}"
        elif not seed and cent_pred[i] not in near(label):
            decisions[i] = f"nearest centroid {cent_pred[i]} is outside the neighborhood of {label}"
        else:
            decisions[i] = "selected"
            chosen.append((int(i), int(pred[i]), float(p1[i])))
    return chosen, decisions


def _candidate_row(cand: dict, p: np.ndarray, centroid: Optional[str], decision: str, labels: List[str]) -> dict:
    top = np.argsort(-p)[:2]
    return {"text": cand["text"], "source": cand.get("source"), "lang": cand.get("lang"),
            "article_title": cand.get("article_title"), "source_article": cand.get("source_article"),
            "seed_node": cand.get("seed_node"), "translated_from": cand.get("translated_from"),
            "inherited_category": cand.get("category"),
            "top1": labels[top[0]], "p1": round(float(p[top[0]]), 4),
            "top2": labels[top[1]] if len(top) > 1 else None,
            "p2": round(float(p[top[1]]), 4) if len(top) > 1 else None,
            "nearest_centroid": centroid, "decision": decision}


def cmd_round(paths: Paths, rcfg: RoundConfig, cfg: FitConfig, gate: Gate, device,
              batch: Optional[str] = None) -> dict:
    configs = {"fit": asdict(cfg), "round": asdict(rcfg), "gate": asdict(gate)}
    with LoopRecord(paths, "round", configs, batch=batch) as rec:
        return _round(paths, rcfg, cfg, gate, device, rec)


def _round(paths: Paths, rcfg: RoundConfig, cfg: FitConfig, gate: Gate, device, rec: LoopRecord) -> dict:
    log = rec.log
    corpus = load_corpus(paths.data, paths.taxonomy)
    holdout = load_or_create_holdout(corpus, paths.holdout, log)
    champ = read_champion(paths)
    if champ is None:
        raise SystemExit("No champion yet: run `python -m train fit` first.")
    champ_dir = paths.runs / champ["run"]
    champ_meta = json.loads((champ_dir / "run.json").read_text(encoding="utf-8"))
    if champ_meta["labels"] != corpus.labels:
        raise SystemExit("The corpus's label set changed since the champion was trained; run `fit` again.")
    pool = Pool(paths)
    stacked = Pool(paths, enabled=rcfg.unproven == "stack", pending=True)
    rejected = read_jsonl(paths.rejected)
    round_no = 1 + sum(1 for e in loops(paths.runs)
                       if e.get("kind") == "round" and e["id"] != rec.id and e.get("status") in ROUND_DONE)
    name = f"round{round_no:03d}"
    rec.update(round=round_no, champion_before=champ["run"], pool_before=len(pool.records),
               stacked_before=len(stacked.records))
    log(f"Round {round_no}: champion {champ['run']} "
        f"({gate.select_metric} {metric_value(champ['metrics'], gate.select_metric):.4f}), "
        f"pool {len(pool.records):,} pseudo-labels, {len(stacked.records):,} unproven")

    # 1. candidates
    if rcfg.labels not in ("tested", "all"):
        raise SystemExit("--labels must be tested or all")
    allowed = (set(corpus.labels) if rcfg.labels == "all"
               else {corpus.labels[k] for k in corpus.y[split_indices(corpus, holdout)[1]]})
    log(f"  labels that may gain rows: {len(allowed)} of {len(corpus.labels)} ({rcfg.labels})")
    sources = {s.strip() for s in rcfg.sources.split(",") if s.strip()}
    wiki = Wikipedia(delay=rcfg.wiki_delay, log=log)
    cands: List[dict] = []
    inbox_files: List[Path] = []
    if "wiki" in sources:
        skip = holdout.blocked_titles | known_titles(corpus.records) | known_titles(pool.records) | known_titles(rejected)
        skip |= known_titles(read_jsonl(paths.testset)) | known_titles(stacked.records)
        skip |= considered_titles(paths, champ["run"])
        seeds, n_articles = seed_titles(corpus)
        order = sorted(allowed, key=lambda l: (n_articles.get(l, 0), l))
        start = ((round_no - 1) * rcfg.nodes_per_round) % len(order)
        nodes = (order + order)[start:start + min(rcfg.nodes_per_round, len(order))]
        rec.update(nodes_searched=nodes)
        log(f"  Wikipedia: {len(nodes)} nodes, fewest articles first")
        cands += wiki_candidates(nodes, seeds, skip, wiki, rcfg.per_node, rcfg.paras_per_article, log)
    if "inbox" in sources:
        inbox_cands, inbox_files = inbox_candidates(paths.inbox)
        if inbox_cands:
            log(f"  inbox: {len(inbox_cands):,} texts from {len(inbox_files)} files")
        cands += inbox_cands
    rec.update(inbox_files=[f.name for f in inbox_files])
    known = {text_key(r["text"]) for r in corpus.records + pool.records + stacked.records + rejected
             + read_jsonl(paths.testset)}
    unique = []
    for c in cands:
        k = text_key(c["text"])
        if k not in known:
            known.add(k)
            unique.append(c)
    cands = unique
    if not cands:
        log("  no new text found")
        archive_inbox(inbox_files, paths.inbox, name)
        rec.update(status="no_candidates", promoted=False, candidates=0, wikipedia_requests=wiki.requests)
        return {"status": "no_candidates", "round": round_no, "promoted": False}

    # 2. stock embeddings, then the champion's labels
    st = load_encoder(champ_meta["base_model"], device, champ_meta["fit"]["max_seq_length"])
    texts = [c["text"] for c in cands]
    f0 = encode(st, texts, device)
    weights = torch.load(champ_dir / "weights.pt", map_location="cpu", weights_only=True)
    stock = {k: v.detach().clone() for k, v in st.state_dict().items() if k in weights["encoder_delta"]}
    apply_delta(st, weights["encoder_delta"])
    protos, scale, temp = weights["protos"].numpy(), float(weights["scale"]), float(champ_meta["temperature"])
    emb = encode(st, texts, device)
    probs = head_probs(emb, protos, scale, temp)
    cd = json.loads((champ_dir / "coredrill_hierarchical.json").read_text(encoding="utf-8"))
    label_nodes = [l for l in corpus.labels if cd["tree"].get(l, {}).get("centroid") is not None]
    C = np.array([cd["tree"][l]["centroid"] for l in label_nodes], dtype=np.float32)
    cent_pred = [label_nodes[j] for j in (emb @ C.T).argmax(axis=1)]

    # 3. selection, then translations of accepted articles
    picks, decisions = _select(cands, probs, cent_pred, corpus, rcfg)
    for i, k, _ in picks:
        if corpus.labels[k] not in allowed:
            decisions[i] = f"the holdout does not test {corpus.labels[k]}"
    picks = [p for p in picks if corpus.labels[p[1]] in allowed]
    log_rows = [_candidate_row(c, probs[i], cent_pred[i], decisions[i], corpus.labels) for i, c in enumerate(cands)]
    accepted = [{**cands[i], "category": corpus.labels[k], "confidence": round(p, 4), "round": round_no,
                 "loop": rec.id, "_row": i} for i, k, p in picks]
    accepted_f0 = f0[[i for i, _, _ in picks]] if picks else np.zeros((0, f0.shape[1]), dtype=np.float32)
    langs = [l.strip() for l in rcfg.langs.split(",") if l.strip()]
    n_trans = n_trans_kept = 0
    if langs and "wiki" in sources and accepted:
        trans = [t for t in translation_candidates(wiki, accepted, langs, rcfg.paras_per_article, log)
                 if text_key(t["text"]) not in known]
        if trans:
            t_texts = [t["text"] for t in trans]
            t_probs = head_probs(encode(st, t_texts, device), protos, scale, temp)
            apply_delta(st, stock)                     # back to stock LaBSE for the canonical embedding
            t_f0 = encode(st, t_texts, device)
            index = corpus.label_index
            keep = []
            for j, t in enumerate(trans):
                pred = corpus.labels[int(t_probs[j].argmax())]
                ok = pred in neighborhood(t["category"], corpus.parents, corpus.children)
                log_rows.append(_candidate_row(t, t_probs[j], None, "selected (translation)" if ok else
                                               f"translation: the champion says {pred}, outside the "
                                               f"neighborhood of {t['category']}", corpus.labels))
                if ok:
                    keep.append(j)
                    known.add(text_key(t["text"]))
                    t.update({"confidence": round(float(t_probs[j, index[t["category"]]]), 4), "round": round_no,
                              "loop": rec.id, "_row": len(log_rows) - 1})
            accepted += [trans[j] for j in keep]
            accepted_f0 = np.vstack([accepted_f0, t_f0[keep]]) if keep else accepted_f0
            n_trans, n_trans_kept = len(trans), len(keep)
            log(f"  translations: {len(keep)} of {len(trans)} kept")
    del st, stock, weights
    free_gpu()                                   # or the challenger trains next to a second LaBSE

    # per-label caps (English first, best first), then the global cap
    labeled_counts = Counter(corpus.labels[k] for k in corpus.y[split_indices(corpus, holdout)[0]])
    taken: Counter = Counter()
    have = Counter(r["category"] for r in pool.records + stacked.records)
    keep_rows = []
    for j in sorted(range(len(accepted)), key=lambda j: (accepted[j].get("translated_from") is not None,
                                                           -accepted[j]["confidence"])):
        label = accepted[j]["category"]
        cap = max(rcfg.min_per_class, int(rcfg.per_class_frac * labeled_counts[label]))
        row = log_rows[accepted[j]["_row"]]
        if len(keep_rows) >= rcfg.max_new:
            row["decision"] = f"round cap ({rcfg.max_new}) reached"
        elif taken[label] >= cap:
            row["decision"] = f"per-label cap ({cap}) reached for {label}"
        elif have[label] + taken[label] >= max(cap, int(rcfg.total_frac * labeled_counts[label])):
            row["decision"] = f"{label} already has as many self-labeled rows as it may"
        else:
            taken[label] += 1
            keep_rows.append(j)
            row.update({"decision": "accepted", "category": label, "confidence": accepted[j]["confidence"]})
    accepted = [{k: v for k, v in accepted[j].items() if k != "_row"} for j in keep_rows]
    accepted_f0 = accepted_f0[keep_rows] if keep_rows else accepted_f0[:0]
    append_jsonl(rec.dir / "candidates.jsonl", log_rows)
    rec.update(candidates=len(cands), selected=len(picks), translations=n_trans, translations_kept=n_trans_kept,
               accepted=len(accepted), labels_touched=len(taken), wikipedia_requests=wiki.requests)
    log(f"  {len(cands):,} candidates -> {len(accepted):,} pseudo-labels across {len(taken)} labels "
        f"(every decision: {rec.dir / 'candidates.jsonl'})")
    if not accepted:
        archive_inbox(inbox_files, paths.inbox, name)
        rec.update(status="nothing_accepted", promoted=False)
        return {"status": "nothing_accepted", "round": round_no, "promoted": False, "candidates": len(cands)}

    # 4. challenger
    trial = stacked.records + accepted
    trial_f0 = np.vstack([stacked.f0, accepted_f0]) if len(stacked.f0) else accepted_f0
    result = train_and_evaluate(paths, corpus, holdout, cfg, rec, pool, (trial, trial_f0), device)
    promoted, reason, test = beats(result["metrics"], champ["metrics"], gate, result["predictions"],
                                   load_predictions(champ_dir))
    worse = bool(test) and test["z"] <= -1.645
    if promoted:
        promote(paths, result, reason, log)
        pool.add(trial, trial_f0)
        stacked.clear()
    elif rcfg.unproven == "stack" and not worse:
        stacked.add(accepted, accepted_f0)
        log(f"Not promoted: {reason}. Its {len(accepted):,} pseudo-labels stack with the next round's "
            f"({len(stacked.records):,} unproven).")
    else:
        # Significantly worse: this round's rows go. Twice in a row under one
        # champion, and the stack under them goes too.
        before = [e for e in loops(paths.runs) if e.get("kind") == "round" and e["id"] != rec.id
                  and e.get("champion_before") == champ["run"] and e.get("paired_test")]
        dropped = list(accepted)
        if worse and stacked.records and before and before[-1]["paired_test"]["z"] <= -1.645:
            dropped += stacked.records
            stacked.clear()
        append_jsonl(paths.rejected, dropped)
        log(f"Not promoted: {reason}. {len(dropped):,} pseudo-labels are set aside.")
    rec.update(stacked_after=len(stacked.records))
    archive_inbox(inbox_files, paths.inbox, name)
    rec.update(status="promoted" if promoted else "not_promoted", promoted=promoted, reason=reason,
               paired_test=test, champion_after=(read_champion(paths) or {}).get("run"))
    return {"status": "promoted" if promoted else "not_promoted", "round": round_no, "promoted": promoted,
            "reason": reason, "candidates": len(cands), "accepted": len(accepted)}


def cmd_loop(paths: Paths, rcfg: RoundConfig, cfg: FitConfig, gate: Gate, device, rounds: int,
             patience: int, log=print) -> None:
    """Rounds until `patience` in a row fail to promote. Each round is its own
    loop in the ledger; `batch` ties together the rounds of one call."""
    batch = time.strftime("%Y%m%d-%H%M%S") + "-loop"
    misses = 0
    for r in range(1, rounds + 1):
        log(f"\n=== {batch}: round {r} of {rounds} ===")
        summary = cmd_round(paths, rcfg, cfg, gate, device, batch=batch)
        misses = 0 if summary.get("promoted") else misses + 1
        if misses >= patience:
            log(f"Stopping: {misses} rounds in a row without a promotion.")
            break


def cmd_status(paths: Paths, log=print) -> None:
    champ = read_champion(paths)
    if paths.baseline.exists():
        log("Baseline (stock LaBSE):")
        log(format_metrics(json.loads(paths.baseline.read_text(encoding="utf-8"))["metrics"]))
    if champ:
        log(f"\nChampion: {champ['run']} (promoted {champ['promoted']}: {champ['reason']})")
        log(format_metrics(champ["metrics"]))
    else:
        log("\nNo champion yet.")
    log(f"\nPseudo-label pool: {len(read_jsonl(paths.pool_records)):,}; "
        f"unproven, stacking: {len(read_jsonl(paths.pending_records)):,}; "
        f"set aside: {len(read_jsonl(paths.rejected)):,}")
    items = loops(paths.runs)
    log(f"\n{len(items)} loops recorded in {paths.runs / LEDGER}" + ("; the last 10:" if items else "."))
    for it in items[-10:]:
        res = ((it.get("benchmark") or {}).get("results") or {}).get("all classes")
        score = f"deployed strict {res['deployed']['strict']:.3f}" if res and "deployed" in res else ""
        log(f"  {it.get('seq', ''):>4}  {it['id']:<30} {state(it):<16} {score}")
    if items:
        log("Every loop next to the reference benchmark: python -m train report")


def cmd_testset(paths: Paths, per_label: int, log=print) -> None:
    """Real Wikipedia paragraphs, from articles in neither the corpus nor the
    pool, for the labels the holdout cannot test. A person fills in
    correct_label; suggested_label is only the search's starting point. Every
    article listed is kept out of later harvests."""
    out = paths.testset
    if out.exists():
        raise SystemExit(f"{out} exists; move it away to gather a new set")
    corpus = load_corpus(paths.data, paths.taxonomy)
    holdout = load_or_create_holdout(corpus, paths.holdout, log)
    tested = {corpus.labels[k] for k in corpus.y[split_indices(corpus, holdout)[1]]}
    pool = read_jsonl(paths.pool_records) + read_jsonl(paths.rejected)
    skip = holdout.blocked_titles | known_titles(corpus.records) | known_titles(pool)
    seeds, _ = seed_titles(corpus)
    nodes = [l for l in corpus.labels if l not in tested]
    log(f"Test-set candidates for {len(nodes)} labels the holdout does not test")
    rows = wiki_candidates(nodes, seeds, skip, Wikipedia(delay=0.3, log=log), per_label, 1, log)
    known = {text_key(r["text"]) for r in corpus.records + pool}
    rows = [r for r in rows if text_key(r["text"]) not in known]
    append_jsonl(out, rows)
    sheet = out.with_name("to_check.csv")
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "suggested_label", "correct_label", "article", "text"])
        for n, r in enumerate(rows, 1):
            w.writerow([n, r["seed_node"], "", r["source_article"], r["text"]])
    got = Counter(r["seed_node"] for r in rows)
    log(f"{len(rows)} paragraphs, {len(got)} labels -> {sheet}")
    log(f"no candidates: {[l for l in nodes if not got[l]]}")


def cmd_export(paths: Paths, run: str, out: Path, model_id: Optional[str], device, log=print) -> None:
    """A deployable bundle: the full encoder (sentence-transformers format) and
    the coredrill built with it, its labse_model pointing at the encoder.
    Point the API at both: COREDRILL_PATH=<out>/coredrill_hierarchical.json
    and MODEL_NAME=<out>/encoder (or the --model-id it was uploaded under)."""
    champ = read_champion(paths)
    name = (champ or {}).get("run") if run == "champion" else run
    if not name:
        raise SystemExit("No champion yet: run `python -m train fit` first.")
    run_dir = paths.runs / name
    st, head, meta = load_run(run_dir, device)
    out.mkdir(parents=True, exist_ok=True)
    st.save(str(out / "encoder"))
    cd = json.loads((run_dir / "coredrill_hierarchical.json").read_text(encoding="utf-8"))
    cd["labse_model"] = model_id or str((out / "encoder").resolve())
    cd["trained_encoder"] = {"run": name, "base_model": meta["base_model"],
                             "unfreeze_layers": meta["unfreeze_layers"], "holdout": meta.get("metrics")}
    (out / "coredrill_hierarchical.json").write_text(json.dumps(cd), encoding="utf-8")
    append_event(paths.runs, {"event": "export", "time": now(), "run": name, "out": str(out.resolve()),
                              "model_id": model_id})
    log(f"Exported {name} to {out}")
    log(f"  COREDRILL_PATH={out / 'coredrill_hierarchical.json'}")
    log(f"  MODEL_NAME={cd['labse_model']}")
