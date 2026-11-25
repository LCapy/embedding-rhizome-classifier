"""What each loop did to the rhizome, for the view's Loops page; NumPy only.

The map is the territories, placed by which of them resemble each other, not by
the taxonomy: no center and no root. The taxonomy is one tracing over it.
 - alliances: likenesses the coredrill found between different top branches
 - flights: test texts that left their territory for another one
 - live: for a training that is running, the likenesses it makes and unmakes
"""
from __future__ import annotations

import gzip
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import coredrills
from .record import loops, state

PREDICTIONS = "holdout_predictions.jsonl.gz"
MAX_FLIGHTS = 260


def layout(C: np.ndarray, seed: int = 3, steps: int = 350, k: int = 5) -> np.ndarray:
    """Places on a thin sheet: likes pull together (the k nearest each), all
    push apart, so nothing is the middle of anything."""
    n = len(C)
    S = C @ C.T
    np.fill_diagonal(S, -1.0)
    W = np.zeros((n, n))
    near = np.argsort(-S, axis=1)[:, :k]
    for i in range(n):
        W[i, near[i]] = 1.0
    W = np.maximum(W, W.T)
    X = np.random.default_rng(seed).normal(size=(n, 3)) * [4.0, 0.6, 4.0]
    for s in range(steps):
        d = X[:, None] - X[None]
        r2 = (d ** 2).sum(-1) + 0.05
        F = (d * (0.9 / r2 - 0.10 * W * np.sqrt(r2))[..., None]).sum(1) - 0.06 * X
        F[:, 1] -= 0.5 * X[:, 1]
        size = np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
        X += F / size * np.minimum(size, 0.35 * (1 - s / steps) + 0.005)
    X -= X.mean(axis=0)
    return X * (10.0 / np.percentile(np.linalg.norm(X, axis=1), 95))


@lru_cache(maxsize=8)
def _ground(folder: str) -> Tuple[dict, dict, List[list]]:
    """(paths, places, alliances) of one loop's coredrill."""
    cd = json.loads((Path(folder) / coredrills.FILE).read_text(encoding="utf-8"))
    paths = {n: p[0] for n, p in cd["taxonomy_backbone"]["lineage_paths"].items() if p}
    names = [n for n, s in cd["tree"].items() if s.get("centroid") is not None]
    C = np.array([cd["tree"][n]["centroid"] for n in names], dtype=np.float64)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    at = {n: [round(float(v), 3) for v in x] for n, x in zip(names, layout(C))}
    best: Dict[tuple, float] = {}
    for n, others in (cd.get("rhizome") or {}).get("overlay", {}).items():
        across = [o for o in others if n in at and o["name"] in at and paths[n][0] != paths[o["name"]][0]]
        for o in across[:2]:
            best[tuple(sorted((n, o["name"])))] = o["sim"]
    return paths, at, [[a, b, round(s, 3)] for (a, b), s in best.items()]


@lru_cache(maxsize=80)
def _answers(file: str, stamp: float) -> Tuple[dict, Counter]:
    """(territory -> (right, all), (from, to) -> texts) on the test."""
    seen: Dict[str, list] = {}
    flights: Counter = Counter()
    with gzip.open(file, "rt", encoding="utf-8") as lines:
        for line in lines:
            r = json.loads(line)
            if r.get("deployed"):
                s = seen.setdefault(r["truth"], [0, 0])
                s[0] += r["deployed"] == r["truth"]
                s[1] += 1
                if r["deployed"] != r["truth"]:
                    flights[(r["truth"], r["deployed"])] += 1
    return seen, flights


def answers(folder: Path) -> Optional[Tuple[dict, Counter]]:
    f = folder / PREDICTIONS
    return _answers(str(f), f.stat().st_mtime) if f.exists() else None


def _live(folder: Path, names: set) -> Optional[dict]:
    meta = coredrills.live_meta(folder)
    frame_f = folder / "live" / "frame.json"
    if meta is None or not frame_f.exists():
        return None
    frame = json.loads(frame_f.read_text(encoding="utf-8"))
    q = frame.get("protos_cos")
    if not q:
        return None
    labels, n = meta["labels"], len(meta["labels"])
    P = coredrills._bytes_to(meta["protos0"], "<f2", (n, -1)).astype(np.float64)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    was, now = P @ P.T, coredrills._bytes_to(q["q"], "i1", (n, n)).astype(np.float64) / 127
    pairs = set()
    for M in (was, now):
        S = M.copy()
        np.fill_diagonal(S, -1.0)
        for i, row in enumerate(np.argsort(-S, axis=1)[:, :2]):
            pairs.update(tuple(sorted((i, int(j)))) for j in row)
    links = [[labels[i], labels[j], round(float(was[i, j]), 3), round(float(now[i, j]), 3)]
             for i, j in pairs if labels[i] in names and labels[j] in names]
    return {"links": links, "epoch": frame.get("epoch"), "step": frame.get("step")}


def rhizome(runs: Path, loop: str) -> dict:
    folder = coredrills.find(runs, loop)
    if folder is None:
        raise ValueError("unknown loop")
    root = folder.parent
    items = loops(root)
    # one map per corpus, from its first coredrill, so loops can be compared on it
    first = next((root / it["id"] for it in items if (root / it["id"] / coredrills.FILE).exists()), None)
    if first is None:
        any_root = [r / it["id"] for r in coredrills.roots(runs) for it in loops(r) if (r / it["id"] / coredrills.FILE).exists()]
        if not any_root:
            raise ValueError("no coredrill has been built yet")
        first = any_root[-1]
    paths, at, alliances = _ground(str(first))
    index = {n: i for i, n in enumerate(at)}
    place = next(i for i, it in enumerate(items) if it["id"] == loop)
    # what it is compared with: the last one before it that was kept, else that was scored
    earlier = [it for it in items[:place] if answers(root / it["id"])]
    kept = [it for it in earlier if state(it) == "promoted"]
    before = (kept or earlier or [None])[-1]
    mine = answers(folder)
    shown = loop if mine else before["id"] if before else None
    base = answers(root / before["id"]) if before else None
    now = mine or base or ({}, Counter())
    old = base or now
    flights = []
    for a, b in set(now[1]) | set(old[1]):
        if a in index and b in index:
            flights.append([index[a], index[b], now[1][(a, b)], old[1][(a, b)]])
    flights.sort(key=lambda f: -max(f[2], f[3]))
    live = _live(folder, set(index)) if state(items[place]) == "running" else None
    if live:
        live["links"] = [[index[a], index[b], w, n] for a, b, w, n in live["links"]]
    tree = [[index[n], index[p[-2]]] for n, p in paths.items() if n in index and len(p) > 1 and p[-2] in index]
    return {"loop": loop, "shown": shown, "before": before and {"id": before["id"], "seq": before.get("seq")},
            "nodes": [{"name": n, "path": paths.get(n, [n]), "at": p, "now": now[0].get(n), "old": old[0].get(n)} for n, p in at.items()],
            "alliances": [[index[a], index[b], s] for a, b, s in alliances], "tree": tree, "flights": flights[:MAX_FLIGHTS],
            "gained": sum(max(0, f[3] - f[2]) for f in flights), "left": sum(f[2] for f in flights), "live": live}
