"""python -m train builds: better ways of building the coredrill, for one encoder.

The encoder is stock LaBSE, or a trained loop's (its embeddings are saved with
it). Ways of building are tried on a validation slice of the training data,
whole articles set aside, so the holdout is used once: to confirm the best one
against the plain build. The search is one loop in the ledger, with every way
it tried; its coredrill shows on the Coredrill page.

When the best way beats the plain build clearly (McNemar z of at least 1.645
on the holdout), it becomes the way later trainings build theirs:
<runs>/build.json, which their own flags override.

No PyTorch: it runs next to a training.
"""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import coredrills
from .data import by_article_split, load_corpus, load_or_create_holdout, split_indices
from .evaluate import evaluate, format_metrics, paired_test, save_predictions
from .record import LoopRecord, now

PLAIN = {"radius_pull": 0.0, "centers_per": 0, "centers_max": 1}
RADII = [0.0, 0.5, 0.75, 0.9, 1.0]
CENTERS = [(0, 1), (100, 4), (60, 4), (60, 6), (40, 6)]
SETTINGS_FILE = "build.json"
MIN_Z = 1.645


@dataclass
class Paths:
    data: Path
    runs: Path
    taxonomy: Optional[Path] = None

    @property
    def holdout(self) -> Path:
        return self.runs / "holdout.json"

    @property
    def labeled(self) -> Path:
        return self.data / "labeled"

    # what LoopRecord's fingerprint reads; builds change none of them
    pool_records = pool_f0 = rejected = pending_records = property(lambda self: self.runs / "-")


def describe(b: dict) -> str:
    centers = f"one center per {b['centers_per']} texts, at most {b['centers_max']}" if b["centers_per"] else "one center"
    return f"radius {int(100 * b['radius_pull'])}% toward the median, {centers}"


def default_build(runs: Path) -> dict:
    """The build later trainings use unless told otherwise."""
    f = runs / SETTINGS_FILE
    return {**PLAIN, **json.loads(f.read_text(encoding="utf-8"))["build"]} if f.exists() else dict(PLAIN)


def encoder_embeddings(paths: Paths, corpus, holdout, run: Optional[str]) -> Tuple[List[dict], np.ndarray, np.ndarray, np.ndarray, int]:
    """(training rows, their embeddings, holdout indices, their embeddings,
    how many of the training rows are corpus rows, in split order)."""
    train_idx, test_idx = split_indices(corpus, holdout)
    if run is None:
        return [corpus.records[i] for i in train_idx], corpus.f0[train_idx], test_idx, corpus.f0[test_idx], len(train_idx)
    folder = coredrills.find(paths.runs, run)
    if folder is None or not (folder / "embeddings.npz").exists():
        raise SystemExit(f"{run} saved no embeddings (only trainings since today do); use --run stock, or train again")
    with np.load(folder / "embeddings.npz") as z:
        X_tr, X_te, saved_test = z["train"].astype(np.float32), z["test"].astype(np.float32), z["test_idx"]
    with gzip.open(folder / "train_rows.json.gz", "rt", encoding="utf-8") as f:
        rows = json.load(f)
    if not np.array_equal(saved_test, test_idx) or [r["category"] for r in rows[:len(train_idx)]] != \
            [corpus.records[i]["category"] for i in train_idx]:
        raise SystemExit(f"{run} was trained on another corpus or holdout than {paths.data}")
    rows = [corpus.records[i] for i in train_idx] + [{"text": "", **r} for r in rows[len(train_idx):]]
    for norm in (X_tr, X_te):                     # float16 on disk: unit length again
        norm /= np.linalg.norm(norm, axis=1, keepdims=True) + 1e-12
    return rows, X_tr, test_idx, X_te, len(train_idx)


def cmd_builds(paths: Paths, run: Optional[str], quick: bool = False, log=print) -> dict:
    planned = 5 if quick else len(RADII) + len(CENTERS) - 1
    with LoopRecord(paths, "build", {"encoder": run or "stock LaBSE", "radii": RADII, "centers": CENTERS, "quick": quick,
                                     "planned": planned}) as rec:
        corpus = load_corpus(paths.data, paths.taxonomy)
        holdout = load_or_create_holdout(corpus, paths.holdout, rec.log)
        rows, X_tr, test_idx, X_te, n_corpus = encoder_embeddings(paths, corpus, holdout, run)
        train_idx, _ = split_indices(corpus, holdout)
        # validation: whole articles of the training data, as the holdout was drawn
        y_tr = corpus.y[train_idx]
        keep_pos, val_pos = by_article_split([corpus.records[i] for i in train_idx], y_tr, len(corpus.labels),
                                             test_frac=0.15, seed=7)
        fit_pos = np.concatenate([np.sort(keep_pos), np.arange(n_corpus, len(rows))])
        fit_rows, fit_X = [rows[i] for i in fit_pos], X_tr[fit_pos]
        val_idx, val_X = train_idx[val_pos], X_tr[val_pos]
        rec.log(f"Ways of building the coredrill on {run or 'stock LaBSE'}: {len(fit_rows):,} rows build it, "
                f"{len(val_idx):,} rows of other articles judge it")

        def judge(build: dict, records, X, idx, Xt, classic=False, quiet=True) -> dict:
            t0 = time.perf_counter()
            out = io.StringIO()
            with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
                ev = evaluate(corpus, records, X, idx, Xt, classic=classic, log=(lambda m: None) if quiet else rec.log,
                              build=build, blocked_titles=holdout.blocked_titles)
            ev["seconds"] = round(time.perf_counter() - t0)
            return ev

        tried = []

        def attempt(build: dict) -> dict:
            ev = judge(build, fit_rows, fit_X, val_idx, val_X)
            m = ev["metrics"]["deployed"]["all"]
            tried.append({"build": build, "strict": m["strict"], "near_miss": m["near_miss"],
                          "profile_recall": m.get("profile_recall"), "seconds": ev["seconds"], "predictions": ev["predictions"]})
            rec.log(f"  {describe(build):<62} {100 * m['strict']:5.1f}% right, {100 * m['near_miss']:5.1f}% near  ({ev['seconds']}s)")
            return tried[-1]

        radii = [0.0, 0.75, 1.0] if quick else RADII
        best = max((attempt({**PLAIN, "radius_pull": r}) for r in radii), key=lambda t: t["strict"])
        for per, most in (CENTERS[1:3] if quick else CENTERS[1:]):
            t = attempt({**best["build"], "centers_per": per, "centers_max": most})
            if t["strict"] > best["strict"]:
                best = t
        chosen = best["build"]
        plain = next(t for t in tried if t["build"] == PLAIN)
        rec.log(f"Best on validation: {describe(chosen)}: {100 * best['strict']:.1f}%, plain {100 * plain['strict']:.1f}%")

        rec.log("Confirming once on the holdout, with every training row ...")
        ev_plain = judge(PLAIN, rows, X_tr, test_idx, X_te, classic=False)
        ev_best = judge(chosen, rows, X_tr, test_idx, X_te, classic=True) if chosen != PLAIN else judge(PLAIN, rows, X_tr, test_idx, X_te, classic=True)
        test = paired_test(ev_best["predictions"], ev_plain["predictions"], "deployed")
        a, b = ev_plain["metrics"]["deployed"]["all"]["strict"], ev_best["metrics"]["deployed"]["all"]["strict"]
        wins = chosen != PLAIN and test is not None and test["z"] >= MIN_Z and b >= a + 0.002
        reason = (f"holdout {100 * a:.1f}% -> {100 * b:.1f}%" + (f" ({test['gained']} gained, {test['lost']} lost, z {test['z']:.2f})"
                                                                  if test else ""))
        rec.log(format_metrics(ev_best["metrics"]))
        rec.log(f"{'Adopted' if wins else 'Not adopted'}: {describe(chosen)}; {reason}")
        cd = ev_best["coredrill"]
        cd["labse_model"] = f"(the encoder of {run})" if run else cd.get("labse_model")
        (rec.dir / "coredrill_hierarchical.json").write_text(json.dumps(cd), encoding="utf-8")
        save_predictions(rec.dir / "holdout_predictions.jsonl.gz", ev_best["predictions"])
        if wins:
            (paths.runs / SETTINGS_FILE).write_text(json.dumps({"build": chosen, "adopted": now(), "loop": rec.id,
                                                                "encoder": run or "stock LaBSE", "reason": reason}, indent=2),
                                                    encoding="utf-8")
        rec.update(status="promoted" if wins else "not_promoted", promoted=wins, reason=reason, chosen=chosen,
                   tried=[{k: v for k, v in t.items() if k != "predictions"} for t in tried],
                   metrics=ev_best["metrics"], benchmark=ev_best["benchmark"], paired_test=test,
                   plain_holdout=ev_plain["metrics"]["deployed"]["all"])
        return {"chosen": chosen, "adopted": wins, "reason": reason}
