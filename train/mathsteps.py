"""python -m train math: the coredrill's calculations, step by step, with one
loop's real numbers, for the view's Math page.

Every step of build_coredrill() and analyze_text_rhizome_v2() is done again
here in NumPy and checked against what the loop saved (its centroids and radii,
its answer to every test text), so the page shows the engine's own arithmetic.
Writes <loop>/math.json. No PyTorch.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import coredrills
from .data import eng

FILE = "math.json"
TOP_M, SVD_CUT, TAU = 20, 0.05, 0.1
FACTORS = ("alpha", "s", "NS", "PC", "SW", "mass")


class Coredrill:
    """A coredrill as arrays, in the engine's node order."""

    def __init__(self, cd: dict):
        self.cd, self.tree = cd, cd["tree"]
        skip = set(cd.get("skip_nodes") or ())
        self.names = [n for n, s in self.tree.items() if s.get("centroid") is not None and n not in skip]
        self.index = {n: i for i, n in enumerate(self.names)}
        st = [self.tree[n] for n in self.names]
        self.C = np.array([s["centroid"] for s in st], dtype=np.float32)
        self.sigma = np.array([float(s.get("sigma") or 0.35) for s in st])
        self.level = np.array([int(s["level"]) for s in st])
        heads, owner = [], []
        for i, s in enumerate(st):
            many = s.get("centers") or [s["centroid"]]
            heads += many
            owner += [i] * len(many)
        self.heads, self.owner = np.array(heads, dtype=np.float32), np.array(owner)
        self.first = np.r_[0, np.flatnonzero(np.diff(self.owner)) + 1]
        lineage = cd["taxonomy_backbone"]["lineage_paths"]
        self.path = [[n for n in (lineage.get(a) or [[]])[0] if n not in skip and n != a] for a in self.names]
        self.ancestors = [[self.index[n] for n in p if n in self.index] for p in self.path]

    def scores(self, X: np.ndarray, centers: bool = True) -> np.ndarray:
        if not centers:
            return X @ self.C.T
        return np.maximum.reduceat(X @ self.heads.T, self.first, axis=1)

    def production(self, s: np.ndarray) -> dict:
        eff = np.maximum(self.sigma, 0.25)
        z = (1.0 - s) / np.minimum(eff, 0.45)
        pen = np.maximum(0.0, eff - 0.40) * 0.60
        # 0.62 (-z + .15 s - .05 ln eff) + 0.18 (-z) + 0.10 (-z^2/2) + 0.08 s - pen
        parts = {"distance": -0.80 * z - 0.05 * z * z, "cosine": 0.173 * s,
                 "radius": np.broadcast_to(-0.031 * np.log(eff) - pen, s.shape)}
        return {"z": z, "parts": parts, "prod": sum(parts.values())}

    def gram(self, s_all: np.ndarray, prod: np.ndarray, off: frozenset = frozenset()) -> dict:
        """The final score's every factor, for one text."""
        order = np.lexsort((-s_all, -prod))[:TOP_M]
        s, lv = s_all[order].astype(np.float64), self.level[order]
        M = self.C[order].astype(np.float64)
        G = M @ M.T
        U, sv, Vt = np.linalg.svd(G)
        keep = sv > SVD_CUT * sv[0]
        alpha = ((Vt.T * np.where(keep, 1.0 / np.where(keep, sv, 1.0), 0.0)) @ U.T) @ s
        peer = (G > 0.5) & (np.abs(lv[:, None] - lv[None]) <= 1)
        np.fill_diagonal(peer, False)
        n = peer.sum(1)
        NS = np.where(n > 0, (peer @ s) / np.maximum(n, 1), 0.8 * s) / (s.max() if s.max() > 0 else 1.0)
        PC = np.ones(len(order))
        for i, j in enumerate(order):
            if not self.path[j]:
                continue
            r = [min(s_all[a] / s[i], 1.0) for a in self.ancestors[j] if s[i] > 1e-9]
            PC[i] = np.mean(r) if len(r) >= 2 else 0.85 * r[0] if r else 0.60
        SW = np.maximum(self.sigma[order], 0.30) ** -0.5
        SW = SW / SW.mean()
        mass = np.sqrt(np.maximum(np.exp((prod[order] - prod[order[0]]) / TAU), 1e-6))
        f = {"alpha": alpha, "s": s, "NS": NS, "PC": PC, "SW": SW, "mass": mass}
        R = np.ones(len(order))
        for k, v in f.items():
            if k not in off:
                R = R * v
        return {"order": order, "G": G, "sv": sv, "kept": int(keep.sum()), "R": R, **f}

    def answer(self, g: dict, z: np.ndarray) -> int:
        j = int(g["order"][int(np.argmax(g["R"]))])
        return int(np.argmin(z)) if self.level[j] == 0 else j


def _round(v, k=4):
    return [round(float(x), k) for x in np.ravel(v)]


def _plane(X: np.ndarray, mu: np.ndarray, extra: np.ndarray):
    """Across: how far a text is from the center, along its own first direction;
    up: the second. The center is at 0, 0."""
    Y = X - (X @ mu)[:, None] * mu
    _, _, vt = np.linalg.svd(Y - Y.mean(0), full_matrices=False)
    B = vt[:2] if len(vt) >= 2 else np.vstack([vt, np.zeros_like(vt)])[:2]
    put = lambda Z: (Z - (Z @ mu)[:, None] * mu) @ B.T  # noqa: E731
    return put(X), put(extra)


def measure(runs: Path, loop: str, data: Path, log=print) -> dict:
    folder = coredrills.find(runs, loop)
    if folder is None or not (folder / "embeddings.npz").exists():
        raise SystemExit(f"{loop} saved no embeddings")
    cd = json.loads((folder / coredrills.FILE).read_text(encoding="utf-8"))
    with np.load(folder / "embeddings.npz") as f:
        X, T, test_idx = f["train"].astype(np.float32), f["test"].astype(np.float32), f["test_idx"]
    for A in (X, T):
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    with gzip.open(folder / "train_rows.json.gz", "rt", encoding="utf-8") as f:
        rows = json.load(f)
    with gzip.open(folder / "holdout_predictions.jsonl.gz", "rt", encoding="utf-8") as f:
        saved = [json.loads(z) for z in f]
    k = Coredrill(cd)
    label = np.array([r["category"] for r in rows], dtype=object)
    source = np.array([r.get("source", "unknown") for r in rows], dtype=object)
    out: dict = {"loop": loop, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "build": cd.get("build"), "dim": X.shape[1]}

    # ── the build ────────────────────────────────────────────────────────────
    S = k.C @ k.C.T
    np.fill_diagonal(S, -1)
    cap = int((cd.get("build") or {}).get("per_node_cap") or 0)
    territories, worst_mu, worst_sigma, dropped = [], 0.0, 0.0, 0
    raw = np.array([float(k.tree[n].get("sigma_raw", k.tree[n]["sigma"])) for n in k.names])
    for i, n in enumerate(k.names):
        st, all_idx = k.tree[n], np.where(label == n)[0]
        t = {"name": n, "path": k.path[i] + [n], "mode": st["centroid_mode"], "rows": int(len(all_idx)), "used": st["support_n"],
             "sigma_raw": round(float(raw[i]), 4), "sigma": round(float(k.sigma[i]), 4), "kappa": st["kappa_vmf"],
             "centers": len(st.get("centers") or []) or 1,
             "near": [[k.names[j], round(float(S[i, j]), 4)] for j in np.argsort(-S[i])[:6]],
             "right": [0, 0]}
        if st["centroid_mode"] != "child_only" and cap == 0:
            idx = eng._balanced_source_sample_indices(all_idx.astype(np.int32), source)
            dropped += len(all_idx) - len(idx)
            mean = X[idx].astype(np.float64).mean(0)
            mu = mean / np.linalg.norm(mean)
            cos = X[idx] @ mu
            worst_mu = max(worst_mu, float(1 - mu @ k.C[i]))
            worst_sigma = max(worst_sigma, abs(float(1 - cos.mean()) - raw[i]))
            m = len(idx)
            R2 = float(mean @ mean)
            t["length"] = round(float(np.sqrt(R2)), 4)
            # the mean of m texts looks longer than the territory's true one
            t["sigma_unbiased"] = round(1 - float(np.sqrt(max(0.0, (m * R2 - 1) / (m - 1)))), 4) if m > 1 else None
            pick = idx[np.linspace(0, m - 1, min(m, 110)).astype(int)]
            H = k.heads[k.owner == i]
            P, Q = _plane(X[pick].astype(np.float64), mu, H.astype(np.float64))
            own = (X[pick] @ H.T).argmax(1)
            t["points"] = [[round(float(a), 3), round(float(b), 3), int(o)] for (a, b), o in zip(P, own)]
            t["heads"] = [[round(float(a), 3), round(float(b), 3)] for a, b in Q]
            t["texts"] = [str(rows[p].get("text") or "")[:110] for p in pick]
            h, e = np.histogram(1 - cos, bins=24, range=(0, max(0.6, float((1 - cos).max()))))
            t["hist"] = {"n": h.tolist(), "to": round(float(e[-1]), 3)}
            t["mean"], t["first"] = _round(mean[:96], 3), _round(X[pick[0]][:96], 3)
            t["sources"] = [[str(a), int((source[all_idx] == a).sum()), int((source[idx] == a).sum())]
                            for a in sorted(set(source[all_idx]))]
        territories.append(t)
    out["check"] = {"centroids_off_by_at_most": worst_mu, "radii_off_by_at_most": worst_sigma, "rows": len(rows),
                    "rows_not_used": int(dropped)}
    out["sigmas"] = {"raw": _round(raw), "used": _round(k.sigma), "median": round(float(np.median(raw)), 4),
                     "under_floor": int((k.sigma < 0.25).sum()), "under_floor_final": int((k.sigma < 0.30).sum()),
                     "over_cap": int((k.sigma > 0.45).sum()), "n": len(k.names)}

    # ── the answers: every step, on every test text ──────────────────────────
    truth = [p["truth"] for p in saved]
    y = np.array([k.index.get(t, -1) for t in truth])
    s0, s1 = k.scores(T, False), k.scores(T, True)
    p1 = k.production(s1)
    same, hits = 0, {f"without_{f}": 0 for f in FACTORS}
    ladder = {"nearest mean": float((s0.argmax(1) == y).mean()), "nearest center": float((s1.argmax(1) == y).mean()),
              "smallest z": float((p1["z"].argmin(1) == y).mean()), "production score": float((p1["prod"].argmax(1) == y).mean())}
    final, only, kept, cand, void, neg = 0, 0, [], 0, 0, 0
    mine = []
    for j in range(len(T)):
        s, z, pr = s1[j], p1["z"][j], p1["prod"][j]
        g = k.gram(s, pr)
        a = k.answer(g, z)
        mine.append(a)
        same += k.names[a] == saved[j]["deployed"]
        final += a == y[j]
        void += k.level[int(g["order"][int(np.argmax(g["R"]))])] == 0
        cand += y[j] in g["order"]
        kept.append(g["kept"])
        neg += float((g["alpha"] < 0).mean())
        for f in FACTORS:
            R = np.ones(len(g["R"]))
            for q in FACTORS:
                if q != f:
                    R = R * g[q]
            hits[f"without_{f}"] += k.answer({"order": g["order"], "R": R}, z) == y[j]
        R = g["alpha"]
        only += k.answer({"order": g["order"], "R": R}, z) == y[j]
    N = len(T)
    ladder["final score"] = final / N
    out["ladder"] = {a: round(b, 4) for a, b in ladder.items()}
    out["ablation"] = {a: round(b / N, 4) for a, b in hits.items()}
    out["ablation"]["alpha_only"] = round(only / N, 4)
    out["answers"] = {"n": N, "same_as_engine": int(same), "truth_in_top_20": round(cand / N, 4), "void": int(void),
                      "directions_kept": round(float(np.mean(kept)), 2), "negative_alpha": round(neg / N, 3)}
    for t, a in zip(y, mine):
        if t >= 0:
            territories[t]["right"][1] += 1
            territories[t]["right"][0] += int(t == a)
    out["territories"] = territories

    # ── texts to follow: some right at once, some the final score rescued, lost or missed ──
    sent = json.loads((data / "sentences.json").read_text(encoding="utf-8"))
    best = p1["prod"].argmax(1)
    kinds: Dict[str, List[int]] = {"right": [], "rescued": [], "lost": [], "wrong": []}
    for j in np.random.default_rng(5).permutation(N):
        if y[j] < 0 or (saved[j].get("lang") or "en") != "en":
            continue
        kind = ("right" if mine[j] == y[j] else "lost") if best[j] == y[j] else ("rescued" if mine[j] == y[j] else "wrong")
        body = str(sent[int(test_idx[j])]["text"])
        if len(kinds[kind]) < 12 and 60 < len(body) < 420:
            kinds[kind].append(int(j))
    examples = []
    for kind, js in kinds.items():
        for j in js:
            g = k.gram(s1[j], p1["prod"][j])
            o = g["order"]
            examples.append({
                "kind": kind, "text": str(sent[int(test_idx[j])]["text"]), "truth": truth[j], "answer": k.names[mine[j]],
                "x": _round(T[j][:96], 3), "names": [k.names[q] for q in o], "levels": k.level[o].tolist(),
                "mean_cos": _round(s0[j][o]), "cos": _round(s1[j][o]), "sigma": _round(k.sigma[o]), "z": _round(p1["z"][j][o]),
                "parts": {a: _round(b[j][o]) for a, b in p1["parts"].items()}, "prod": _round(p1["prod"][j][o]),
                "G": [_round(r, 3) for r in g["G"]], "sv": _round(g["sv"]), "kept": g["kept"],
                **{f: _round(g[f]) for f in FACTORS if f != "s"}, "R": _round(g["R"], 5)})
    out["examples"] = examples
    (folder / FILE).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log(f"checked: centroids within {worst_mu:.1e}, radii within {worst_sigma:.1e}, "
        f"{same:,} of {N:,} answers the engine's -> {folder / FILE}")
    return out


def listing(runs: Path) -> List[dict]:
    """Loops whose calculations were written out, newest first."""
    found = [(r / d.name) for r in coredrills.roots(runs) for d in r.iterdir() if d.is_dir() and (d / FILE).exists()]
    return [{"id": f.name, "root": coredrills.root_name(runs, f)} for f in sorted(found, key=lambda f: f.name, reverse=True)]


def read(runs: Path, loop: str) -> Optional[bytes]:
    folder = coredrills.find(runs, loop)
    return (folder / FILE).read_bytes() if folder and (folder / FILE).exists() else None
