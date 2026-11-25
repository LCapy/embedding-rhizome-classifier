"""The rhizome classifier, with any trained loop's model: what `python -m train
classify` prints and what the view's classifier page shows.

Answers come from the engine's own code, the way the API gives them, with the
model's encoder (encoder_np.py, no PyTorch) and its coredrill. Nothing leaves
the machine.

`python -m train classify --serve` is the process the view starts: one JSON
request per line on stdin, one JSON answer per line on stdout.
"""
from __future__ import annotations

import base64
import contextlib
import gzip
import io
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Before NumPy loads. Each BLAS thread commits memory of its own, and when the
# CPU is busy many threads are far slower than a few: they wait for each other.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402

from . import coredrills  # noqa: E402
from .data import eng, record_keys  # noqa: E402
from .encoder_np import BASE_MODEL, Encoder, base_dir, read_weights  # noqa: E402
from .evaluate import StoredEmbeddingModel, deployed_answer  # noqa: E402

MAX_CHARS = 8000        # api/app.py's limit


def classifiable(runs: Path) -> List[str]:
    """Loops with a complete trained model, newest first."""
    return sorted((p.name for root in coredrills.roots(runs) for p in root.iterdir() if p.is_dir() and all(
        (p / f).exists() for f in ("run.json", "weights.pt", "coredrill_hierarchical.json"))), reverse=True)


def champion(runs: Path) -> Optional[str]:
    f = runs / "champion.json"
    return json.loads(f.read_text(encoding="utf-8"))["run"] if f.exists() else None


# ── the universe: a map of the taxonomy ──────────────────────────────────────

def _distances(points: np.ndarray) -> np.ndarray:
    d = points[:, None, :] - points[None, :, :]
    return np.sqrt((d ** 2).sum(axis=-1))


def layout(D: np.ndarray, steps: int = 250) -> np.ndarray:
    """Points in 3D whose distances come as close as they can to D (metric
    multidimensional scaling: SMACOF from the classical solution)."""
    n = len(D)
    J = np.eye(n) - 1.0 / n
    values, vectors = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    X = vectors[:, -3:] * np.sqrt(np.maximum(values[-3:], 1e-9))
    for _ in range(steps):
        d = _distances(X)
        np.fill_diagonal(d, 1.0)
        B = -D / np.maximum(d, 1e-9)
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(axis=1))
        X = B @ X / n
    return X


def _turned(X: np.ndarray, thin: float) -> np.ndarray:
    """X around its center on its own axes (widest first), the narrowest
    flattened by `thin`: a disk rather than a ball."""
    X = X - X.mean(axis=0)
    _, _, axes = np.linalg.svd(X, full_matrices=False)
    Y = X @ axes.T
    return np.stack([Y[:, 0], Y[:, 2] * thin, Y[:, 1]], axis=1)


def _tilt(seed: int) -> np.ndarray:
    """A fixed random rotation, so that the galaxies do not lie all one way."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def universe(tree: Dict[str, dict], paths: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """Where every node of the taxonomy stands.

    The taxonomy decides first: each domain (level 1) is a galaxy, the domains
    of a branch lie near each other, and inside a galaxy nodes lie near their
    parents and siblings. The model decides the rest: domains, and territories
    within one, stand closer the more alike their centroids are. A single
    layout of all 768 dimensions is a ball with half of the neighbors wrong."""
    domain = {n: paths[n][1] if len(paths[n]) > 1 else None for n in tree}
    domains = [n for n in tree if len(paths[n]) == 2]
    members = {d: [n for n in tree if domain[n] == d] for d in domains}
    centroid = {n: np.array(s["centroid"], dtype=np.float64) for n, s in tree.items() if s.get("centroid") is not None}

    def apart(A: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(2.0 - 2.0 * (A @ A.T), 0.0))

    middle = []
    for d in domains:
        vs = [centroid[n] for n in members[d] if n in centroid]
        middle.append(np.mean(vs, axis=0) / np.linalg.norm(np.mean(vs, axis=0)) if vs else None)
    have = [i for i, m in enumerate(middle) if m is not None]
    D = np.ones((len(domains), len(domains)))   # a domain with no texts: as far as a typical one
    if len(have) > 1:
        sub = apart(np.array([middle[i] for i in have]))
        D *= float(np.median(sub[np.triu_indices(len(have), 1)])) / max(float(sub.max()), 1e-9)
        D[np.ix_(have, have)] = sub / max(float(sub.max()), 1e-9)
    other = np.array([[paths[a][0] != paths[b][0] for b in domains] for a in domains])
    D = D + 0.45 * other
    np.fill_diagonal(D, 0.0)
    X = _turned(layout(D), 0.55) * 11.0
    radius = np.array([0.55 * np.sqrt(len(members[d])) for d in domains])
    for _ in range(200):                         # no galaxy may reach into another
        moved = False
        for i in range(len(domains)):
            for j in range(i):
                gap, least = X[i] - X[j], 1.25 * (radius[i] + radius[j])
                d = np.linalg.norm(gap) + 1e-9
                if d < least:
                    X[i] += gap / d * (least - d) / 2
                    X[j] -= gap / d * (least - d) / 2
                    moved = True
        if not moved:
            break

    at: Dict[str, np.ndarray] = {}
    for k, d in enumerate(domains):
        names = members[d]
        hops = np.array([[len(set(paths[a]) ^ set(paths[b])) for b in names] for a in names], dtype=np.float64)
        T = hops / max(hops.max(), 1.0)
        M = T.copy()                             # a node without texts has only the taxonomy
        has = [i for i, n in enumerate(names) if n in centroid]
        if len(has) > 1:
            m = apart(np.array([centroid[names[i]] for i in has]))
            M[np.ix_(has, has)] = m / max(m.max(), 1e-9)
        L = 0.6 * T + 0.4 * M
        np.fill_diagonal(L, 0.0)
        Y = _turned(layout(L), 0.5) if len(names) > 3 else layout(L)
        Y *= radius[k] / max(np.linalg.norm(Y, axis=1).max(), 1e-9)
        for n, y in zip(names, Y @ _tilt(k + 1).T):
            at[n] = X[k] + y
    for n in tree:                               # a branch stands amid its domains
        if n not in at:
            at[n] = np.mean([X[i] for i, d in enumerate(domains) if paths[d][0] == n], axis=0)
    center = np.mean(X, axis=0)
    return {n: v - center for n, v in at.items()}


class Classifier:
    def __init__(self, runs: Path, data: Path):
        self.runs, self.data = runs, data
        self.encoder: Optional[Encoder] = None
        self.id = ""
        self.samples: Optional[List[dict]] = None
        self.samples_of: Optional[Path] = None
        self.coredrills: Dict[str, coredrills.Coredrill] = {}
        self.positions: Dict[str, Dict[str, np.ndarray]] = {}
        self.summaries: Dict[str, dict] = {}

    def load(self, run: Optional[str] = None) -> None:
        run = run or champion(self.runs)
        if not run:
            raise ValueError("No trained model yet: run `python -m train fit` first.")
        if run == self.id:
            return
        if run not in classifiable(self.runs):
            raise ValueError(f"{run} has no complete trained model")
        folder = coredrills.find(self.runs, run)
        self.meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if self.encoder is None:
            self.encoder = Encoder(base_dir(self.meta.get("base_model", BASE_MODEL)))
        self.id = ""                             # nothing is loaded if what follows fails
        w = read_weights(folder / "weights.pt")
        self.encoder.set_weights(w["encoder_delta"])
        self.encoder.set_max_tokens(self.meta.get("fit", {}).get("max_seq_length") or 256)
        self.protos = w["protos"] / (np.linalg.norm(w["protos"], axis=1, keepdims=True) + 1e-12)
        self.scale, self.temperature = float(w["scale"]), float(self.meta.get("temperature") or 1.0)
        self.cd = json.loads((folder / "coredrill_hierarchical.json").read_text(encoding="utf-8"))
        eng.SKIP_NODES.clear()                   # mirror api/app.py: the coredrill's own list
        eng.SKIP_NODES.update(self.cd.get("skip_nodes", []))
        self.tree: Dict[str, dict] = self.cd["tree"]
        self.paths = {n: p[0] for n, p in self.cd["taxonomy_backbone"]["lineage_paths"].items() if p}
        self.folder = folder
        self._map()
        self.id = run

    def _map(self) -> None:
        self.placed = [n for n, s in self.tree.items() if s.get("centroid") is not None]
        self.C = np.array([self.tree[n]["centroid"] for n in self.placed], dtype=np.float32)
        at = universe(self.tree, self.paths)
        self.names = list(self.tree)
        index = {n: i for i, n in enumerate(self.names)}
        self.at = np.array([at[n] for n in self.names])
        self.X = self.at[[index[n] for n in self.placed]]
        S = self.C @ self.C.T
        np.fill_diagonal(S, -1.0)
        near = {n: np.argsort(-S[i])[:5] for i, n in enumerate(self.placed)}
        twin = {n: np.where(S[i] >= 0.999)[0] for i, n in enumerate(self.placed)}
        gaps = _distances(self.at)
        np.fill_diagonal(gaps, np.inf)
        self.reach = np.clip(gaps.min(axis=1) * 0.9, 0.18, 0.9)    # how far a territory's texts spread on the map
        row = {n: i for i, n in enumerate(self.placed)}
        nodes = []
        for i, n in enumerate(self.names):
            s, path = self.tree[n], self.paths.get(n, [n])
            nodes.append({"name": n, "level": s.get("level"), "path": path, "branch": path[0],
                          "domain": path[1] if len(path) > 1 else None, "n": int(s.get("n") or 0),
                          "texts": int(s.get("direct_n") or 0), "data": n in row,
                          "xyz": [round(float(v), 3) for v in self.at[i]],
                          "near": [[index[self.placed[j]], round(float(S[row[n], j]), 3)] for j in near[n]] if n in row else [],
                          "twins": [index[self.placed[j]] for j in twin[n]] if n in row else []})
        parents = [[index[n], index[p[-2]]] for n in self.names if len(p := self.paths.get(n, [n])) > 1 and p[-2] in index]
        threads, seen = [], set()
        for d in nodes:                          # alike to the model, apart in the taxonomy
            for j, c in d["near"][:3]:
                pair = tuple(sorted((index[d["name"]], j)))
                if c >= 0.45 and pair not in seen and nodes[j]["domain"] != d["domain"]:
                    seen.add(pair)
                    threads.append([*pair, c])
        spread = np.linalg.svd(self.C - self.C.mean(axis=0), compute_uv=False) ** 2
        share = np.cumsum(spread / spread.sum())
        self.map = {"nodes": nodes, "tree": parents, "threads": threads,
                    "domains": [n for n in self.names if len(self.paths.get(n, [n])) == 2],
                    "geometry": {"dims": int(self.C.shape[1]), "three": round(float(share[2]), 3),
                                 "for80": int(np.searchsorted(share, 0.8)) + 1},
                    "trust": self._trust()}

    def _trust(self) -> Optional[dict]:
        """How often the answer was right on the holdout, when the prototypes
        agreed with it and when they did not."""
        f = self.folder / "holdout_predictions.jsonl.gz"
        if not f.exists():
            return None
        count = {True: [0, 0], False: [0, 0]}
        with gzip.open(f, "rt", encoding="utf-8") as lines:
            for line in lines:
                r = json.loads(line)
                if r.get("head") and r.get("deployed"):
                    c = count[r["deployed"] == r["head"][0]]
                    c[0] += 1
                    c[1] += r["deployed"] == r["truth"]
        if not count[True][0] or not count[False][0]:
            return None
        return {"agree": round(count[True][1] / count[True][0], 3), "differ": round(count[False][1] / count[False][0], 3)}

    def stars(self) -> dict:
        """Every labeled text as a star near its territory. Where it stands in
        its cloud is its own stock LaBSE embedding, on the three axes along
        which its territory's texts differ most. Computed once and kept."""
        data = self._data()
        cache, source = self.folder.parent / "stars.npz", data / "embeddings.npz"
        stamp = f"{source.stat().st_mtime_ns} {(data / 'sentences.json').stat().st_mtime_ns}"
        if cache.exists():
            with np.load(cache) as z:
                kept = {k: z[k] for k in z.files}
            if str(kept.get("stamp")) != stamp:
                kept = None
        else:
            kept = None
        if kept is None:
            labels = [r["category"] for r in json.loads((data / "sentences.json").read_text(encoding="utf-8"))]
            with np.load(source) as z:
                E = z[z.files[0]]
            names = sorted(set(labels))
            y = np.array([names.index(v) for v in labels], dtype=np.int16)
            offset = np.zeros((len(y), 3), dtype=np.float32)
            for k in range(len(names)):
                rows = np.where(y == k)[0]
                A = E[rows] - E[rows].mean(axis=0)
                if len(rows) < 4:
                    continue
                values, axes = np.linalg.eigh(A.T @ A) if len(rows) > A.shape[1] else (None, None)
                if axes is None:
                    _, _, vt = np.linalg.svd(A, full_matrices=False)
                    axes = vt[:3].T
                else:
                    axes = axes[:, -3:]
                P = A @ axes
                offset[rows] = P / max(float(np.percentile(np.linalg.norm(P, axis=1), 90)), 1e-9)
            kept = {"stamp": np.array(stamp), "names": np.array(names), "y": y, "offset": offset}
            tmp = cache.with_name("stars.tmp.npz")
            np.savez_compressed(tmp, **kept)
            tmp.replace(cache)
        index = {n: i for i, n in enumerate(self.names)}
        node = np.array([index.get(str(n), -1) for n in kept["names"]])[kept["y"]]
        shown = node >= 0
        node = node[shown]
        at = self.at[node] + kept["offset"][shown] * self.reach[node][:, None]
        scale = float(np.abs(at).max()) / 32767
        return {"count": int(len(node)), "scale": scale,
                "node": base64.b64encode(node.astype("<u2").tobytes()).decode(),
                "xyz": base64.b64encode(np.round(at / scale).astype("<i2").tobytes()).decode()}

    def info(self) -> dict:
        return {"run": self.id, "kind": self.meta.get("kind"), "created": self.meta.get("created"),
                "labels": len(self.meta["labels"]), "territories": len(self.placed), "nodes": len(self.tree),
                "max_tokens": self.encoder.max_tokens}

    def classify(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("Write a text first.")
        if len(text) > MAX_CHARS:
            raise ValueError(f"The text has {len(text):,} characters; the limit is {MAX_CHARS:,}.")
        t0 = time.perf_counter()
        tokens, truncated = self.encoder.count_tokens(text)
        x = self.encoder.encode([text])[0]
        model = StoredEmbeddingModel()
        model.current = x
        with contextlib.redirect_stdout(io.StringIO()):
            raw = eng.analyze_text_rhizome_v2(self.cd, text, model, tau=self.cd.get("tau", 0.1), print_output=False)
        active, ranked = raw.get("active_overlap_set", []), raw.get("results", [])
        best = deployed_answer(raw)
        flow = eng._deleuzian_flow(active, ranked, raw.get("prediction") or {}) or {}
        gram = raw.get("gram_scores") or {}

        logits = self.scale * (self.protos @ x) / self.temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        head = [{"node": self.meta["labels"][i], "p": round(float(p[i]), 4)} for i in np.argsort(-p)[:5]]

        def row(r: dict) -> dict:
            n = r["node"]
            return {"node": n, "level": r.get("level"), "path": self.paths.get(n, [n]),
                    "mass": round(float(r.get("mass", 0.0)), 2), "cosine": round(float(r.get("score", 0.0)), 4),
                    "sigma_dist": round(float(r.get("sigma_dist", 0.0)), 3),
                    "production": round(float(r.get("production_score", 0.0)), 4),
                    "gram": round(float(gram[n]), 5) if n in gram else None}

        rows = [row(r) for r in active if r["node"] not in eng.SKIP_NODES]
        cos = (self.C @ x).astype(np.float64)    # the text stands among the territories most like it
        w = np.exp((cos - cos.max()) / 0.035)
        at = (w[:, None] * self.X).sum(axis=0) / w.sum()
        return {"run": self.id, "text": text, "tokens": tokens, "truncated": truncated,
                "best": {"node": best, "level": self.tree.get(best, {}).get("level"),
                         "path": self.paths.get(best, [best]),
                         "description": self.tree.get(best, {}).get("description") or ""},
                "areas": self._areas(best, rows), "active": rows, "ranked": [row(r) for r in ranked[:15]],
                "flow": {"type": flow.get("flow_type"), "label": flow.get("flow_label"),
                         "intensities": flow.get("intensities", {})},
                "head": head, "xyz": [round(float(v), 3) for v in at],
                "ms": round((time.perf_counter() - t0) * 1000)}

    def _areas(self, best: str, rows: List[dict]) -> List[dict]:
        """The active territories by domain (level 1), the answer's own first."""
        def domain(path: List[str]) -> str:
            return path[1] if len(path) > 1 else path[0]

        groups: Dict[str, List[dict]] = defaultdict(list)
        for r in rows:
            groups[domain(r["path"])].append(r)
        mine = domain(self.paths.get(best, [best]))
        groups.setdefault(mine, [])
        out = [{"area": a, "mass": round(sum(r["mass"] for r in rs), 2), "nodes": [r["node"] for r in rs]}
               for a, rs in groups.items()]
        return sorted(out, key=lambda a: (a["area"] != mine, -a["mass"]))

    def _data(self) -> Path:
        """The folder of the corpus the loaded model was trained on."""
        f = self.folder / "record.json"
        info = (json.loads(f.read_text(encoding="utf-8")).get("data") or {}).get("sentences") if f.exists() else None
        folder = Path(info["path"]).parent if info else self.data
        return folder if (folder / "sentences.json").exists() else self.data

    def coredrill(self, which: str) -> "coredrills.Coredrill":
        if which not in self.coredrills:
            if len(self.coredrills) > 5:         # about 20 MB each
                self.coredrills.clear()
            self.coredrills[which] = coredrills.Coredrill(coredrills.path_of(self.runs, which))
        return self.coredrills[which]

    def compare(self, a: str, b: str) -> dict:
        left, right = self.coredrill(a), self.coredrill(b)
        if b not in self.positions:
            self.positions[b] = universe(right.tree, right.paths)
        return {"from": a, "to": b, **coredrills.compare(left, right, self.positions[b])}

    def timeline(self) -> List[dict]:
        for item in (c for c in coredrills.listing(self.runs) if not c.get("live")):
            if item["id"] not in self.summaries:
                self.summaries[item["id"]] = coredrills.Coredrill(coredrills.path_of(self.runs, item["id"])).summary()
        return [{**item, **self.summaries[item["id"]]} for item in coredrills.listing(self.runs) if not item.get("live")]

    def sample(self) -> dict:
        """A random holdout text and its true label: the model never trained on it."""
        root = self.folder.parent
        if self.samples is None or self.samples_of != root:
            keys = set(json.loads((root / "holdout.json").read_text(encoding="utf-8"))["keys"])
            records = json.loads((self._data() / "sentences.json").read_text(encoding="utf-8"))
            self.samples_of = root
            self.samples = [{"text": r["text"], "truth": r["category"], "lang": r.get("lang") or "en",
                             "source": r.get("article_title") or r.get("source") or ""}
                            for r, k in zip(records, record_keys(records))
                            if k in keys and len(r["text"]) <= MAX_CHARS]
        if not self.samples:
            raise ValueError("The holdout is empty.")
        return random.choice(self.samples)

    def handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "compare":                      # these need no encoder
            return self.compare(request.get("a", ""), request.get("b", ""))
        if op == "timeline":
            return {"coredrills": self.timeline()}
        self.load(request.get("run"))
        if op == "classify":
            return self.classify(request.get("text", ""))
        if op == "map":
            return {**self.info(), **self.map}
        if op == "sample":
            return self.sample()
        if op == "stars":
            return self.stars()
        raise ValueError(f"unknown request {op!r}")


# ── commands ─────────────────────────────────────────────────────────────────

def serve(runs: Path, data: Path) -> None:
    """Until stdin closes, which is also when the view that started it ends."""
    answers = sys.stdout
    sys.stdout = sys.stderr                      # nothing the engine prints may reach the answers
    c = Classifier(runs, data)
    for line in sys.stdin:
        try:
            out = c.handle(json.loads(line))
        except Exception as e:  # noqa: BLE001 - the page shows it; the process goes on
            out = {"error": str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"}
        answers.write(json.dumps(out, ensure_ascii=True) + "\n")
        answers.flush()


def format_result(r: dict) -> str:
    out = [f"  {r['best']['node']}", f"  {' -> '.join(r['best']['path'])}", ""]
    out += [f"  {a['area']}: {', '.join(a['nodes']) or '(no active territory)'}" for a in r["areas"]]
    out += ["", f"  {'territory':<30} {'mass':>7} {'cosine':>7} {'sigma':>6}"]
    out += [f"  {n['node']:<30} {n['mass']:6.1f}% {n['cosine']:7.3f} {n['sigma_dist']:6.2f}" for n in r["active"]]
    out += ["", f"  flow: {r['flow']['label']}",
            "  prototypes: " + ", ".join(f"{h['node']} {100 * h['p']:.0f}%" for h in r["head"][:3])]
    if r["truncated"]:
        out.append(f"  only the first {r['tokens']} tokens of the text were read")
    return "\n".join(out)


def cmd_classify(runs: Path, data: Path, text: Optional[str], run: Optional[str], as_json: bool, log=print) -> None:
    c = Classifier(runs, data)
    try:
        c.load(run)
        result = c.classify(text or "")
    except ValueError as e:
        raise SystemExit(str(e)) from None
    log(json.dumps(result, ensure_ascii=False, indent=2) if as_json else format_result(result))
