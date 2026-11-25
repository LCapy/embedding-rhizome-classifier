"""The coredrills the loops have built, and what changed from one to another:
where each territory's center went, how wide it is, how much data is behind it
and how far its nearest neighbor is. For the view's Coredrill page; NumPy only.
"""
from __future__ import annotations

import base64
import gzip
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .record import LEDGER, REPO, loops

FILE = "coredrill_hierarchical.json"
TWIN = 0.999            # centroids this alike are one territory under two names
DEPLOYED = "deployed"   # coredrill/coredrill_hierarchical.json: the one the repository came with


def roots(runs: Path) -> List[Path]:
    """The runs folder, then the ones inside it with a ledger of their own."""
    if not runs.exists():
        return []
    inner = sorted(d for d in runs.iterdir() if d.is_dir() and (d / LEDGER).exists())
    return [runs] + inner


def find(runs: Path, loop: str) -> Optional[Path]:
    """A loop's folder, in whichever runs folder it is."""
    for root in roots(runs):
        if (root / loop).is_dir():
            return root / loop
    return None


def root_name(runs: Path, folder: Path) -> str:
    """"" for the runs folder itself, else the inner one's name."""
    return "" if folder.parent == runs else folder.parent.name


def live_meta(folder: Path) -> Optional[dict]:
    f = folder / "live" / "meta.json"
    if not f.exists():
        return None
    meta = json.loads(f.read_text(encoding="utf-8"))
    return meta if "protos0" in meta else None


def listing(runs: Path) -> List[dict]:
    """Every coredrill, oldest first, and every training whose territories were
    captured as it ran (live: true)."""
    out = []
    from .record import state
    for root in roots(runs):
        for it in loops(root):
            folder = root / it["id"]
            if it.get("kind") in ("fit", "round") and (folder / "live" / "meta.json").exists() and live_meta(folder):
                running = state(it) == "running"
                out.append({"id": it["id"], "live": True, "running": running, "root": root_name(runs, folder),
                            "seq": it.get("seq"), "kind": it.get("kind"),
                            "name": f"Training {'now' if running else 'as it ran'}: loop {it.get('seq')}"})
    deployed = REPO / "coredrill" / FILE
    if deployed.exists():
        out.append({"id": DEPLOYED, "name": "The one this began with", "root": "", "seq": 0, "champion": False, "strict": None})
    for root in roots(runs):
        best = root / "champion.json"
        champion = json.loads(best.read_text(encoding="utf-8"))["run"] if best.exists() else None
        for it in loops(root):
            folder = root / it["id"]
            if (folder / FILE).exists():
                res = ((it.get("benchmark") or {}).get("results") or {}).get("all classes") or {}
                out.append({"id": it["id"], "name": f"Loop {it.get('seq')}", "root": root_name(runs, folder), "seq": it.get("seq"),
                            "kind": it.get("kind"), "champion": it["id"] == champion,
                            "strict": (res.get("deployed") or {}).get("strict")})
    return sorted(out, key=lambda d: (d["id"] != DEPLOYED, d["id"]))


def path_of(runs: Path, which: str) -> Path:
    folder = REPO / "coredrill" if which == DEPLOYED else find(runs, which)
    if folder is None or not (folder / FILE).exists():
        raise ValueError(f"{which} has no coredrill")
    return folder


class Coredrill:
    def __init__(self, folder: Path):
        cd = json.loads((folder / FILE).read_text(encoding="utf-8"))
        self.tree: Dict[str, dict] = cd["tree"]
        self.paths = {n: p[0] for n, p in cd["taxonomy_backbone"]["lineage_paths"].items() if p}
        self.names = [n for n, s in self.tree.items() if s.get("centroid") is not None]
        C = np.array([self.tree[n]["centroid"] for n in self.names], dtype=np.float64)
        self.C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        S = self.C @ self.C.T
        np.fill_diagonal(S, -1.0)
        near = S.argmax(axis=1)
        self.near = {n: (self.names[near[i]], 1.0 - float(S[i, near[i]])) for i, n in enumerate(self.names)}
        self.twins = {n for i, n in enumerate(self.names) if S[i].max() >= TWIN}
        self.row = {n: i for i, n in enumerate(self.names)}
        self.recall = self._recall(folder)

    @staticmethod
    def _recall(folder: Path) -> Dict[str, Tuple[int, int]]:
        """(right, all) per true territory on the holdout, for the deployed answer."""
        f = folder / "holdout_predictions.jsonl.gz"
        out: Dict[str, Tuple[int, int]] = {}
        if f.exists():
            with gzip.open(f, "rt", encoding="utf-8") as lines:
                for line in lines:
                    r = json.loads(line)
                    if r.get("deployed"):
                        right, n = out.get(r["truth"], (0, 0))
                        out[r["truth"]] = (right + (r["deployed"] == r["truth"]), n + 1)
        return out

    def support(self, n: str) -> int:
        return int(self.tree[n].get("support_n") or 0)

    def summary(self) -> dict:
        sizes = np.array([self.tree[n].get("direct_n") or 0 for n in self.names], dtype=float)
        used = np.array([self.support(n) for n in self.names], dtype=float)
        sep = np.array([self.near[n][1] for n in self.names])
        sigma = np.array([self.tree[n]["sigma"] for n in self.names])
        right, n = sum(r for r, _ in self.recall.values()), sum(k for _, k in self.recall.values())

        def spread(v: np.ndarray) -> Optional[float]:
            """How many times the 95th percentile is the 5th."""
            v = v[v > 0]
            return round(float(np.percentile(v, 95) / max(np.percentile(v, 5), 1)), 1) if len(v) else None
        return {"territories": len(self.names), "twins": len(self.twins), "texts_spread": spread(sizes), "used_spread": spread(used),
                "separation": round(float(np.median(sep)), 4), "crowded": int((sep < 0.1).sum()),
                "radius": round(float(np.median(sigma)), 4), "strict": round(right / n, 4) if n else None, "tested": len(self.recall)}


def _follow(start: np.ndarray, before: np.ndarray, after: np.ndarray, steps: int = 80) -> np.ndarray:
    """Positions for the distances `after`, moved as little as possible from
    `start`, which stands for the distances `before`: same scale, no turning."""
    if len(start) < 3:
        return start.copy()
    gap = np.linalg.norm(start[:, None] - start[None], axis=-1)
    scale = float((gap * before).sum() / max((before ** 2).sum(), 1e-12))
    # what the picture got wrong before stays: only the change in distance moves it
    D = np.maximum(gap + scale * (after - before), 1e-3)
    np.fill_diagonal(D, 0.0)
    X, n = start.copy(), len(start)
    for _ in range(steps):
        d = np.linalg.norm(X[:, None] - X[None], axis=-1)
        np.fill_diagonal(d, 1.0)
        B = -D / np.maximum(d, 1e-9)
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(axis=1))
        X = B @ X / n
    X -= X.mean(axis=0)
    Y = start - start.mean(axis=0)
    u, _, vt = np.linalg.svd(X.T @ Y)
    return X @ (u @ vt) + start.mean(axis=0)


def _apart(C: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(2.0 - 2.0 * (C @ C.T), 0.0))


def compare(a: Coredrill, b: Coredrill, at: Dict[str, np.ndarray]) -> dict:
    """What changed from a to b. `at` is where b's territories stand."""
    common = [n for n in b.names if n in a.row]
    A = a.C[[a.row[n] for n in common]]
    B = b.C[[b.row[n] for n in common]]
    u, _, vt = np.linalg.svd(A.T @ B)            # the turn of the whole space between the two encoders
    moved = dict(zip(common, 1.0 - np.einsum("ij,ij->i", A @ (u @ vt), B)))

    was = {n: p.copy() for n, p in at.items()}
    domain = {n: b.paths[n][1] if len(b.paths.get(n, [])) > 1 else None for n in b.tree}
    for d in {v for v in domain.values() if v}:
        ns = [n for n in common if domain[n] == d]
        if len(ns) >= 3:
            X = _follow(np.array([at[n] for n in ns]), _apart(b.C[[b.row[n] for n in ns]]), _apart(a.C[[a.row[n] for n in ns]]))
            was.update(dict(zip(ns, X)))
    index = {n: i for i, n in enumerate(b.tree)}

    def side(c: Coredrill, n: str) -> Optional[dict]:
        if n not in c.row:
            return None
        s = c.tree[n]
        r = c.recall.get(n)
        return {"sigma": s.get("sigma"), "texts": int(s.get("direct_n") or 0), "used": c.support(n), "near": c.near[n][0],
                "gap": round(c.near[n][1], 4), "twin": n in c.twins, "right": r[0] if r else None, "tested": r[1] if r else None}

    nodes = []
    for n in b.tree:
        p = b.paths.get(n, [n])
        nodes.append({"name": n, "level": b.tree[n].get("level"), "path": p, "branch": p[0], "domain": domain[n],
                      "from": [round(float(v), 3) for v in was[n]], "to": [round(float(v), 3) for v in at[n]],
                      "a": side(a, n) if n in a.tree else None, "b": side(b, n),
                      "moved": round(float(moved[n]), 4) if n in moved else None})
    tree = [[index[n], index[p[-2]]] for n in b.tree if len(p := b.paths.get(n, [n])) > 1 and p[-2] in index]
    return {"nodes": nodes, "tree": tree, "domains": [n for n in b.tree if len(b.paths.get(n, [n])) == 2],
            "a": a.summary(), "b": b.summary()}


_LAYOUTS: Dict[str, tuple] = {}
_LOCK = threading.Lock()


def _bytes_to(meta_or_frame: str, dtype: str, shape: tuple) -> np.ndarray:
    return np.frombuffer(base64.b64decode(meta_or_frame), dtype=dtype).reshape(shape)


def live(runs: Path, loop: str) -> dict:
    """A training's territories: where they started, where they are now."""
    from .classify import universe
    folder = find(runs, loop)
    meta = live_meta(folder) if folder else None
    if meta is None:
        raise ValueError("this training did not record its territories (it began before they were captured)")
    labels, paths = meta["labels"], meta["taxonomy_paths"]
    with _LOCK:
        if loop not in _LAYOUTS:
            P0 = _bytes_to(meta["protos0"], "<f2", (len(labels), -1)).astype(np.float64)
            P0 /= np.linalg.norm(P0, axis=1, keepdims=True) + 1e-12
            row = {l: i for i, l in enumerate(labels)}
            tree = {n: {"centroid": P0[row[n]].tolist() if n in row else None} for n in paths}
            _LAYOUTS[loop] = (universe(tree, paths), P0 @ P0.T)
        at, G0 = _LAYOUTS[loop]
    frame_f = folder / "live" / "frame.json"
    frame = json.loads(frame_f.read_text(encoding="utf-8")) if frame_f.exists() else {}
    q = frame.get("protos_cos")
    G = _bytes_to(q["q"], "i1", (q["n"], q["n"])).astype(np.float64) / 127 if q else G0
    row = {l: i for i, l in enumerate(labels)}
    domain = {n: p[1] if len(p) > 1 else None for n, p in paths.items()}
    now = {n: np.asarray(p, dtype=np.float64) for n, p in at.items()}
    for d in {v for v in domain.values() if v}:
        ns = [n for n in labels if domain.get(n) == d]
        if len(ns) >= 3:
            ix = [row[n] for n in ns]
            dist = lambda M: np.sqrt(np.maximum(2.0 - 2.0 * M[np.ix_(ix, ix)], 0.0))  # noqa: E731
            now.update(zip(ns, _follow(np.array([at[n] for n in ns]), dist(G0), dist(G))))

    def side(M: np.ndarray, n: str) -> Optional[dict]:
        if n not in row:
            return None
        s = M[row[n]].copy()
        s[row[n]] = -1.0
        j = int(s.argmax())
        return {"sigma": 0.11, "texts": None, "used": None, "near": labels[j], "gap": round(1.0 - float(s[j]), 4),
                "twin": bool(s[j] >= TWIN), "right": None, "tested": None}

    def summary(M: np.ndarray) -> dict:
        S = M.copy()
        np.fill_diagonal(S, -1.0)
        gap = 1.0 - S.max(axis=1)
        return {"territories": len(labels), "twins": int((S.max(axis=1) >= TWIN).sum()), "crowded": int((gap < 0.1).sum()),
                "separation": round(float(np.median(gap)), 4), "texts_spread": None, "used_spread": None, "radius": None,
                "strict": None, "tested": 0}
    index = {n: i for i, n in enumerate(paths)}
    nodes = [{"name": n, "level": len(p) - 1, "path": p, "branch": p[0], "domain": domain[n],
              "from": [round(float(v), 3) for v in at[n]], "to": [round(float(v), 3) for v in now[n]],
              "a": side(G0, n), "b": side(G, n), "moved": None} for n, p in paths.items()]
    tree = [[index[n], index[p[-2]]] for n, p in paths.items() if len(p) > 1 and p[-2] in index]
    return {"from": "start", "to": loop, "live": {"epoch": frame.get("epoch"), "step": frame.get("step"),
            "steps": meta.get("steps"), "epochs": meta.get("epochs"), "loss": frame.get("loss"), "t": frame.get("t")},
            "nodes": nodes, "tree": tree, "domains": [n for n, p in paths.items() if len(p) == 2],
            "a": summary(G0), "b": summary(G)}
