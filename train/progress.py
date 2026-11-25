"""What every loop is doing and has done, for the view's Loops page: read from
the ledgers and the running loops' logs. No PyTorch, no NumPy."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Optional

from .coredrills import roots
from .record import REPO, loops, state

STEP = re.compile(r"epoch (\d+)/(\d+)\s+step (\d+)/(\d+) \((\d+)%\)\s+loss ([\d.]+)\s+([\d.]+) steps/s")
EPOCH = re.compile(r"epoch (\d+)/(\d+): loss ([\d.]+)\s+val acc ([\d.]+)\s+\((\d+)s\)")
SEARCHED = re.compile(r"(\d+)/(\d+) nodes searched")
TRIED = re.compile(r"% right, .*% near")

# per kind of loop: what its log says as it goes, the phase that starts, and
# how far along the loop is then; and the part of it that training takes
PHASES = {
    "fit": [("Training a ", "Preparing the training", .01), ("training:", "Training", .02),
            ("embedding ", "Embedding every text with the trained encoder", .78)],
    "round": [("Round ", "Finding new text on Wikipedia", 0), ("candidates ->", "Labeling what it found", .16),
              ("Training a ", "Preparing the training", .18), ("training:", "Training", .2),
              ("embedding ", "Embedding every text with the trained encoder", .8)],
    "score": [("Scoring ", "Embedding every text with that model", .02)],
    "baseline": [("Baseline:", "Preparing", .02)],
    "build": [("Ways of building", "Trying ways of building the coredrill", .02),
              ("Confirming once", "Confirming the best way on the test", .82)],
}
SCORING = {"fit": (.85, .87, .9), "round": (.86, .88, .91), "score": (.6, .65, .75), "baseline": (.1, .2, .35)}
TRAINING = {"fit": (.02, .78), "round": (.2, .8)}
NAMES = {"": "The first corpus (209 labels)", "v3-trial": "Rebuilt corpus, rehearsal on partial data",
         "v3": "Rebuilt corpus"}


def tail(path: Path, lines: int = 8) -> List[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 6000))
        text = f.read().decode("utf-8", errors="replace")
    return [line.rstrip() for line in text.splitlines()[1:] if line.strip()][-lines:]


def running(folder: Path, it: dict) -> dict:
    """Where a running loop is, from its log."""
    log = (folder / "log.txt").read_text(encoding="utf-8", errors="replace") if (folder / "log.txt").exists() else ""
    kind = it.get("kind") or ""
    phases = list(PHASES.get(kind, []))
    if kind in SCORING:
        b, f, s = SCORING[kind]
        phases += [("building the coredrill from", "Building the coredrill", b),
                   ("fitting logistic regression", "Fitting the classifiers it is compared with", f),
                   ("scoring the deployed pipeline", "Answering every test text with the coredrill", s),
                   ("deployed pipeline scored", "Writing the results", .98)]
    phase, done = "Starting", 0.0
    for line in log.splitlines():
        for key, name, at in phases:
            if key in line and (key != "Round " or line.startswith("Round ")) and at >= done:
                phase, done = name, at
    out: dict = {"phase": phase}
    steps, epochs = STEP.findall(log), EPOCH.findall(log)
    if epochs:
        out["val"] = [{"epoch": int(a), "loss": float(b), "val_acc": float(c)} for a, _, b, c, _ in epochs]
    if steps and phase == "Training":
        e, n, s, total, _, loss, rate = steps[-1]
        e, n, s, total, rate = int(e), int(n), int(s), int(total), float(rate)
        if epochs and int(epochs[-1][0]) >= e:      # an epoch ended after that line
            e, s = int(epochs[-1][0]) + 1, 0
        e = min(e, n)
        lo, hi = TRAINING[kind]
        done = lo + (hi - lo) * min(1.0, ((e - 1) + s / max(total, 1)) / n)
        out.update({"epoch": e, "epochs": n, "step": s, "steps": total, "loss": float(loss), "rate": rate,
                    "left_s": round(((n - e) * total + total - s) / max(rate, 1e-6) + 6 * 60)})
    nodes = SEARCHED.findall(log)
    if kind == "round" and nodes and done < .16:
        a, b = map(int, nodes[-1])
        done = .15 * a / max(b, 1)
        out["searched"] = f"{a} of {b} territories searched"
    if kind == "build" and done < .82:
        tried, planned = len(TRIED.findall(log)), (it.get("config") or {}).get("planned") or 9
        done = .02 + .8 * min(1.0, tried / planned)
        out.update({"tried": tried, "planned": planned})
    started = time.mktime(time.strptime(it["started"], "%Y-%m-%dT%H:%M:%S"))
    elapsed = time.time() - started
    out.update({"done": round(min(done, .99), 3), "elapsed_s": round(elapsed), "log": tail(folder / "log.txt")})
    if "left_s" not in out and done > .05:
        out["left_s"] = round(elapsed * (1 - done) / done)
    return out


def summary(it: dict) -> dict:
    m = ((it.get("metrics") or {}).get("deployed") or {}).get("all") or {}
    test = it.get("paired_test") or {}
    error = (it.get("error") or "").strip().splitlines()
    return {"id": it["id"], "seq": it.get("seq"), "kind": it.get("kind"), "state": state(it), "started": it.get("started"),
            "seconds": it.get("seconds"), "strict": m.get("strict"), "near_miss": m.get("near_miss"),
            "n": (it.get("metrics") or {}).get("n"), "reason": it.get("reason") or "", "accepted": it.get("accepted"),
            "z": test.get("z"), "error": error[-1] if error else None}


def gathering() -> Optional[dict]:
    """The Wikipedia gathering for thin territories (scripts/gather_territories.py)."""
    log = REPO / "data" / "sources" / "gather.log"
    if not log.exists():
        return None
    lines = [l.split("\t") for l in log.read_text(encoding="utf-8").splitlines() if l.count("\t") >= 3]
    if not lines:
        return None
    last = time.mktime(time.strptime(lines[-1][0], "%Y-%m-%dT%H:%M:%S"))
    listed = sum(1 for l in (REPO / "scripts" / "territory_articles.txt").read_text(encoding="utf-8").splitlines()
                 if ": " in l and not l.startswith("#"))
    return {"active": time.time() - last < 300, "articles": len(lines), "texts": sum(int(l[3] or 0) for l in lines),
            "territories": len({l[1] for l in lines}), "listed": listed, "last": lines[-1][2], "for": lines[-1][1],
            "at": lines[-1][0]}


def cmd_progress(runs: Path) -> dict:
    folders, now = [], []
    for root in roots(runs):
        name = "" if root == runs else root.name
        best = root / "champion.json"
        items = loops(root)
        for it in items:
            if state(it) == "running":
                now.append({**summary(it), "root": name, **running(root / it["id"], it)})
        champion = json.loads(best.read_text(encoding="utf-8"))["run"] if best.exists() else None
        folders.append({"root": name, "name": NAMES.get(name, name), "champion": champion,
                        "loops": [summary(it) for it in items][::-1]})
    folders.sort(key=lambda r: r["loops"][0]["id"] if r["loops"] else "", reverse=True)
    return {"time": time.time(), "running": now, "roots": folders, "gathering": gathering()}
