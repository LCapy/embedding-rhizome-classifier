"""python -m train density: what density clustering (HDBSCAN) finds in one
encoder's training texts, and whether it builds a better coredrill.

Three readings, all on the training texts, none on the holdout:
 - inside each territory: its dense core (HDBSCAN, eom) and its finer cores
   (leaf); are there several modes, or one cloud with a tail?
 - across territories, labels ignored: dense regions of texts. A region
   shared by several territories is a family the encoder cannot separate.
 - texts outside their territory's core and nearer another territory's
   mean: a list for a person to review (possible mislabels or texts that
   belong elsewhere). Nothing is relabeled.

Then two builds, set before anything is scored, each against the encoder's
own build on the same test (McNemar, as `builds`):
 - density centers: a territory's centers are its leaf cores, not k-means
 - without strict noise: the review list's texts left out of the build
Nothing is adopted: build.json and champion.json are not touched.

No PyTorch: it runs next to a training.
"""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import coredrills
from .builds import PLAIN, Paths, describe, encoder_embeddings
from .data import eng, load_corpus, load_or_create_holdout, record_keys
from .evaluate import evaluate, load_predictions, paired_test, save_predictions
from .record import LoopRecord

SETTINGS = {
    "territory_pca": 10,          # dimensions a territory's texts are reduced to, fitted on them
    "min_cluster_share": 0.05,    # HDBSCAN min_cluster_size: this share of a territory's texts ...
    "min_cluster_floor": 15,      # ... and at least this many; smaller territories are one cloud
    "min_samples": 5,
    "global_pca": 32,             # every text, labels ignored
    "global_min_cluster": 40,
    "global_min_samples": 10,
    "family_share": 0.8,          # a region whose largest territory has less than this is shared
    "max_centers": 6,             # as the champion's build
    "min_center_size": 8,         # as spherical_kmeans
}
REVIEW = "review.jsonl.gz"
REPORT = "density.json"
MAP = "map.json"            # the texts placed by likeness, for the Density page
ANSWERS = "answers.json"    # the test's answers so far, build by build, while it scores
MAP_PER_TERRITORY = 40


def _write(path: Path, data) -> None:
    """Whole or not at all: the page may be reading it."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for _ in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.2)


def text_map(X: np.ndarray, y: np.ndarray, X_te: np.ndarray, seed: int = 0):
    """(sampled training rows, their 2-D places, the test texts' places).
    At most MAP_PER_TERRITORY texts of each territory, placed by t-SNE; a test
    text sits at the mean place of its five most alike sampled texts."""
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    pick = np.concatenate([rng.permutation(np.where(y == k)[0])[:MAP_PER_TERRITORY] for k in np.unique(y)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Y = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed, n_jobs=2).fit_transform(
            _reduce(X[pick], 50))
    Y = (Y - Y.mean(axis=0)) / np.abs(Y).max()
    near = np.zeros((len(X_te), 2))
    Xs = X[pick]
    for a in range(0, len(X_te), 1000):
        S = X_te[a:a + 1000] @ Xs.T
        top = np.argpartition(-S, 5, axis=1)[:, :5]
        near[a:a + 1000] = Y[top].mean(axis=1)
    return pick, Y, near


def _hdbscan(P: np.ndarray, min_cluster: int, min_samples: int, method: str = "eom", single: bool = True):
    from sklearn.cluster import HDBSCAN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samples, allow_single_cluster=single,
                       cluster_selection_method=method, copy=True).fit(P)


def _reduce(X: np.ndarray, dims: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    P = PCA(n_components=min(dims, len(X) - 1, X.shape[1]), random_state=0).fit_transform(X)
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)


def territory_cores(X: np.ndarray, s=SETTINGS) -> Optional[dict]:
    """{core: bool per text (eom), leaf: label per text (-1 noise)} for one
    territory's unit rows, or None when it is too small to have a density."""
    if len(X) < 2 * s["min_cluster_floor"]:
        return None
    P = _reduce(X, s["territory_pca"])
    size = max(s["min_cluster_floor"], int(round(s["min_cluster_share"] * len(X))))
    eom = _hdbscan(P, size, s["min_samples"]).labels_
    leaf = _hdbscan(P, size, s["min_samples"], "leaf", single=False).labels_
    return {"core": eom >= 0, "leaf": leaf}


def leaf_centers(X: np.ndarray, s=SETTINGS) -> Optional[np.ndarray]:
    """A territory's leaf cores as unit centers (largest first, at most
    max_centers, each of at least min_center_size texts); None when fewer
    than two: the territory keeps one center."""
    found = territory_cores(X, s)
    if found is None:
        return None
    lab = found["leaf"]
    sizes = Counter(int(v) for v in lab if v >= 0)
    keep = [k for k, n in sizes.most_common(s["max_centers"]) if n >= s["min_center_size"]]
    if len(keep) < 2:
        return None
    M = np.array([X[lab == k].mean(axis=0) for k in keep])
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def cmd_density(paths: Paths, run: str, log=print) -> dict:
    s = SETTINGS
    with LoopRecord(paths, "density", {"encoder": run, "settings": s}) as rec:
        corpus = load_corpus(paths.data, paths.taxonomy)
        holdout = load_or_create_holdout(corpus, paths.holdout, rec.log)
        rows, X, test_idx, X_te, n_corpus = encoder_embeddings(paths, corpus, holdout, run)
        folder = coredrills.find(paths.runs, run)
        own = json.loads((folder / coredrills.FILE).read_text(encoding="utf-8"))
        build = {k: v for k, v in (own.get("build") or {}).items() if k in ("radius_pull", "centers_per", "centers_max")}
        base = load_predictions(folder)
        if base is None:
            raise SystemExit(f"{run} has no holdout predictions to compare with")
        cats = np.array([r["category"] for r in rows], dtype=object)
        names = sorted(set(cats))
        y = np.searchsorted(np.array(names, dtype=object), cats)
        rec.log(f"Density in the training texts of {run}: {len(rows):,} texts, {len(names)} territories; "
                f"its build: {describe({**PLAIN, **build})}")

        # 1. inside each territory
        t0 = time.perf_counter()
        core = np.ones(len(rows), dtype=bool)
        per: Dict[str, dict] = {}
        for k, name in enumerate(names):
            idx = np.where(y == k)[0]
            found = territory_cores(X[idx], s)
            if found is None:
                per[name] = {"texts": int(len(idx)), "too_small": True}
                continue
            core[idx] = found["core"]
            leaf = Counter(int(v) for v in found["leaf"] if v >= 0)
            per[name] = {"texts": int(len(idx)), "core_share": round(float(found["core"].mean()), 3),
                         "leaf_cores": sorted(leaf.values(), reverse=True)}
        sized = [p for p in per.values() if not p.get("too_small")]
        modes = Counter(len(p["leaf_cores"]) for p in sized)
        rec.log(f"  inside territories ({time.perf_counter() - t0:.0f}s): {len(sized)} large enough; "
                f"{100 * (1 - core.mean()):.0f}% of texts outside their territory's core; "
                f"finer cores per territory: {dict(sorted(modes.items()))}")

        # 2. across territories, labels ignored
        t0 = time.perf_counter()
        g = _hdbscan(_reduce(X, s["global_pca"]), s["global_min_cluster"], s["global_min_samples"], single=False)
        region = g.labels_
        regions = []
        for r in range(region.max() + 1):
            m = region == r
            cnt = Counter(cats[m])
            regions.append({"region": r, "texts": int(m.sum()),
                            "territories": [[t, n] for t, n in cnt.most_common(6)],
                            "shared": bool(cnt.most_common(1)[0][1] / m.sum() < s["family_share"])})
        home: Dict[str, Optional[int]] = {}
        for name in names:
            m = (cats == name) & (region >= 0)
            home[name] = int(Counter(region[m]).most_common(1)[0][0]) if m.any() else None
            per[name]["region"] = home[name]
            per[name]["in_regions"] = round(float(m.sum() / (cats == name).sum()), 3)
        shared = [r for r in regions if r["shared"]]
        rec.log(f"  across territories ({time.perf_counter() - t0:.0f}s): {len(regions)} regions, {len(shared)} shared "
                f"by several territories; {100 * (region < 0).mean():.0f}% of texts in no region")
        for r in sorted(shared, key=lambda r: -r["texts"])[:10]:
            rec.log(f"    {r['texts']:>5} texts: " + ", ".join(f"{t} {n}" for t, n in r["territories"][:4]))

        # do the champion's test errors stay inside a shared region?
        wrong = [(p["truth"], p["deployed"] if isinstance(p["deployed"], str) else p["deployed"][0]) for p in base]
        wrong = [(t, a) for t, a in wrong if t != a]
        same = sum(1 for t, a in wrong if home.get(t) is not None and home.get(t) == home.get(a))
        placed = [n for n in names if home[n] is not None]
        pairs = sum(1 for i, a in enumerate(placed) for b in placed[i + 1:] if home[a] == home[b])
        chance = pairs / max(1, len(placed) * (len(placed) - 1) / 2)
        errors = {"wrong": len(wrong), "same_region": same, "share": round(same / max(1, len(wrong)), 3),
                  "chance": round(chance, 4)}
        rec.log(f"  the encoder's test errors: {same:,} of {len(wrong):,} ({100 * errors['share']:.0f}%) answer a territory "
                f"of the same region; two territories at random share one {100 * chance:.1f}% of the time")

        # 3. the review list
        C = np.array([X[y == k].mean(axis=0) for k in range(len(names))])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        S = X @ C.T
        mine = S[np.arange(len(y)), y]
        S[np.arange(len(y)), y] = -9.0
        nearer, other = S.argmax(axis=1), S.max(axis=1)
        strict = ~core & (other > mine)
        with gzip.open(rec.dir / REVIEW, "wt", encoding="utf-8") as f:
            for i in np.where(strict)[0]:
                r = rows[i]
                f.write(json.dumps({"territory": r["category"], "nearer": names[nearer[i]], "own_cos": round(float(mine[i]), 3),
                                    "nearer_cos": round(float(other[i]), 3), "source": r.get("source"),
                                    "article": eng.record_group(r), "text": r.get("text", "")}, ensure_ascii=False) + "\n")
        by_terr = Counter(cats[strict])
        rec.log(f"  review list: {int(strict.sum()):,} texts outside their core and nearer another territory "
                f"({100 * strict.mean():.1f}%), in {REVIEW}; most in " + ", ".join(f"{t} {n}" for t, n in by_terr.most_common(5)))

        # the map the Density page draws, written before the test is scored
        t0 = time.perf_counter()
        pick, Y, placed_te = text_map(X, y, X_te)
        truth_te = [corpus.records[i]["category"] for i in test_idx]
        where = {n: k for k, n in enumerate(names)}
        _write(rec.dir / MAP, {
            "encoder": run, "names": names, "home": [home[n] for n in names],
            "regions": [{"texts": r["texts"], "shared": r["shared"], "top": [t for t, _ in r["territories"][:3]]} for r in regions],
            "train": [[round(float(a), 4), round(float(b), 4), int(y[i]), int(region[i]), int(core[i]), int(strict[i])]
                      for (a, b), i in zip(Y, pick)],
            "test": [[round(float(a), 4), round(float(b), 4), where.get(t, -1)] for (a, b), t in zip(placed_te, truth_te)]})
        rec.log(f"  a map of {len(pick):,} training texts and the {len(test_idx):,} test texts ({time.perf_counter() - t0:.0f}s)")
        by_key = {p["key"]: p["deployed"] if isinstance(p["deployed"], str) else p["deployed"][0] for p in base}
        keys = record_keys(corpus.records)
        live = {"now": None, "total": len(test_idx), "builds": {
            "own": {"label": f"its own build: {describe({**PLAIN, **build})}",
                    "answers": [where.get(by_key.get(keys[i], ""), -1) for i in test_idx]}}}
        _write(rec.dir / ANSWERS, live)

        # 4. two builds, against the encoder's own (its saved predictions)
        def score(label: str, recs, Xb, **kw) -> dict:
            t0 = time.perf_counter()
            key = label.split()[0]
            got: List[int] = []
            live["now"] = key
            live["builds"][key] = {"label": label, "answers": got}

            def heard(j: int, answer: str) -> None:
                got.append(where.get(answer, -1))
                if len(got) % 150 == 0:
                    _write(rec.dir / ANSWERS, live)

            with contextlib.redirect_stdout(io.StringIO()):
                ev = evaluate(corpus, recs, Xb, test_idx, X_te, classic=False, log=lambda m: None,
                              blocked_titles=holdout.blocked_titles, on_answer=heard, **kw)
            live["now"] = None
            _write(rec.dir / ANSWERS, live)
            m = ev["metrics"]["deployed"]["all"]
            test = paired_test(ev["predictions"], base, "deployed")
            rec.log(f"  {label:<44} {100 * m['strict']:5.1f}% right, {100 * m['near_miss']:5.1f}% near; vs its own build "
                    f"{test['gained']} gained, {test['lost']} lost, z {test['z']:.2f}  ({time.perf_counter() - t0:.0f}s)")
            save_predictions(rec.dir / f"predictions_{label.split()[0]}_{label.split()[-1]}.jsonl.gz", ev["predictions"])
            return {"build": label, "strict": m["strict"], "near_miss": m["near_miss"], "paired_test": test,
                    "_ev": ev}

        src = np.array([r.get("source", "unknown") for r in rows], dtype=object)
        made = {"with": 0}

        def density_centers(cd: dict) -> dict:
            for name, st in cd["tree"].items():
                st.pop("centers", None)
                if st.get("centroid") is None or st.get("centroid_mode") not in ("leaf_direct", "parent_direct"):
                    continue
                idx = np.where(cats == name)[0]
                support = eng._balanced_source_sample_indices(idx, src, 400, 3000)
                M = leaf_centers(X[support], s)
                if M is not None:
                    st["centers"] = M.round(6).tolist()
                    made["with"] += 1
            return cd

        own_strict = float(np.mean([(p["deployed"] if isinstance(p["deployed"], str) else p["deployed"][0]) == p["truth"]
                                    for p in base]))
        rec.log(f"Two builds on the test ({len(test_idx):,} texts); its own build scored {100 * own_strict:.1f}%:")
        tried = [score("density centers", rows, X, build={**build, "centers_per": 0, "centers_max": 1}, adjust=density_centers)]
        rec.log(f"    ({made['with']} territories got several density centers; the others one)")
        keep = np.where(~strict)[0]
        tried.append(score("without strict noise", [rows[i] for i in keep], X[keep], build=build))
        best = max(tried, key=lambda t: t["strict"])
        save_predictions(rec.dir / "holdout_predictions.jsonl.gz", best["_ev"]["predictions"])
        (rec.dir / coredrills.FILE).write_text(json.dumps(best["_ev"]["coredrill"]), encoding="utf-8")
        tried = [{k: v for k, v in t.items() if k != "_ev"} for t in tried]
        report = {"encoder": run, "settings": s, "own_build": build, "own_strict": round(own_strict, 4),
                  "territories": per, "regions": regions, "errors": errors, "review": int(strict.sum()),
                  "review_by_territory": dict(by_terr.most_common()), "builds": tried,
                  "core_outside": round(float(1 - core.mean()), 4), "leaf_modes": {str(k): v for k, v in sorted(modes.items())},
                  "no_region": round(float((region < 0).mean()), 4), "density_center_territories": made["with"]}
        (rec.dir / REPORT).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        rec.update(status="ok", reason="; ".join(f"{t['build']} {100 * t['strict']:.1f}% (z {t['paired_test']['z']:.2f})"
                                                for t in tried), builds=tried, errors=errors,
                   metrics=best["_ev"]["metrics"])
        return report
