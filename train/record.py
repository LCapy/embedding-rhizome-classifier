"""Every loop is recorded, whatever its outcome, so each one can be compared
later with the existing benchmarks (eval/held_out/results/).

A loop is one baseline, one fit or one self-training round. Each gets its
own folder under runs/ (<date>-<time>-<kind>) and two lines in
runs/ledger.jsonl: one when it starts and one when it ends, with its status:

  ok               a baseline that finished
  promoted         a trained model that became the champion
  not_promoted     a trained model that lost to the champion (kept all the same)
  no_candidates    a round that found no new text
  nothing_accepted a round whose candidates all failed selection
  failed           an error; the traceback is in the record
  interrupted      Ctrl+C
A start line with no end line is a loop that was killed (or is still running).

The trainer never overwrites or deletes a loop's folder. champion.json and
baseline.json only point at the latest.

Folder contents: record.json (the ledger's end line), log.txt (everything the
loop printed), code/ (the trainer and engine source that ran). A loop that
scored the holdout adds holdout_predictions.jsonl.gz; a trained one adds
run.json, weights.pt and coredrill_hierarchical.json; a round adds
candidates.jsonl (every text it considered, and what was decided).

`python -m train report` lays every loop next to the reference benchmark;
`python -m train watch` follows the running loop live.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
import traceback
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional

from .data import REPO, default_taxonomy

LEDGER = "ledger.jsonl"
REFERENCE = REPO / "eval" / "held_out" / "results" / "by_article.json"
CODE_FILES = ("train/*.py", "scripts/rhizome_engine.py")
REPORT_SYSTEMS = ("deployed", "centroid", "linear_head", "logreg", "head")
REPORT_SUBSETS = ("all classes", "same classes, Wikipedia", "same classes, other sources")
PROFILE_KEYS = ("mean_size", "rhizome", "logreg_same_size", "linear_head_same_size", "head_same_size")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def new_run_dir(runs: Path, kind: str) -> Path:
    d = runs / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}"
    while d.exists():
        time.sleep(1)
        d = runs / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}"
    return d


# ── the ledger ───────────────────────────────────────────────────────────────

def append_event(runs: Path, event: dict) -> None:
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / LEDGER).open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def read_events(runs: Path) -> List[dict]:
    path = runs / LEDGER
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass        # a line cut short by a crash mid-write
    return events


def loops(runs: Path) -> List[dict]:
    """Every loop in start order: its end record, or its start line with
    status "did_not_finish" when no end line was written."""
    starts: Dict[str, dict] = {}
    ends: Dict[str, dict] = {}
    for e in read_events(runs):
        if e.get("event") == "start":
            starts[e["id"]] = e
        elif e.get("event") == "end":
            ends[e["id"]] = e
    return [ends.get(i) or {**s, "status": "did_not_finish"} for i, s in starts.items()]



def _created(pid) -> Optional[float]:
    """When a Windows process started (seconds since the epoch); None when that
    cannot be read."""
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k.GetProcessTimes.argtypes = (wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4)
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = k.OpenProcess(0x1000, False, int(pid))     # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        t = [wintypes.FILETIME() for _ in range(4)]
        if not k.GetProcessTimes(handle, *map(ctypes.byref, t)):
            return None
        return ((t[0].dwHighDateTime << 32) + t[0].dwLowDateTime) / 1e7 - 11644473600
    finally:
        k.CloseHandle(handle)


def _alive(pid, since: Optional[float] = None) -> Optional[bool]:
    """Whether a process still runs; None when it cannot be told. (Not
    os.kill(pid, 0): on Windows that terminates the process.)

    since: a time (seconds since the epoch) at which the process was already
    running. Windows gives a dead process's number to other programs, above
    all after a restart, so a process that began later is another one."""
    if not pid:
        return None
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"],
                                 capture_output=True, text=True, timeout=15).stdout
            row = next(csv.reader(line for line in out.splitlines() if f'"{int(pid)}"' in line), None)
            if row is None:
                return False
            if since is None:
                return True
            if not row[0].lower().startswith("python"):
                return False
            created = _created(pid)
            return None if created is None else created <= since + 5
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def state(loop: dict) -> str:
    """A loop's status, with "did_not_finish" told apart: "running" while its
    process is alive, "killed" once it is gone."""
    status = loop.get("status", "")
    if status != "did_not_finish":
        return status
    try:
        since = datetime.fromisoformat(loop["started"]).timestamp()
    except (KeyError, TypeError, ValueError):
        since = None
    alive = _alive(loop.get("pid"), since)
    return "running" if alive else ("killed" if alive is False else status)


# ── what a record holds besides results ──────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_info(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def data_fingerprint(paths, inputs: bool = True) -> dict:
    """Hashes of what a loop read (inputs) or what it can change (the pool)."""
    files = {}
    if inputs:
        files.update({"sentences": paths.data / "sentences.json", "embeddings": paths.data / "embeddings.npz",
                      "taxonomy": Path(paths.taxonomy) if paths.taxonomy else default_taxonomy(paths.data),
                      "holdout": paths.holdout})
    files.update({"pool": paths.pool_records, "pool_f0": paths.pool_f0, "rejected": paths.rejected,
                  "pending": paths.pending_records})
    out = {}
    try:
        out = {name: _file_info(p) for name, p in files.items()}
        if inputs:
            labeled = sorted(paths.labeled.glob("*.jsonl")) if paths.labeled.exists() else []
            out["labeled"] = [_file_info(p) for p in labeled]
    except OSError as e:
        out["error"] = str(e)
    return out


def _git() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    try:
        changes = [l for l in run("status", "--porcelain").splitlines() if l.strip()]
        return {"git_commit": run("rev-parse", "HEAD") or None, "git_changes": changes[:50]}
    except (OSError, subprocess.SubprocessError) as e:
        return {"git_error": str(e)}


def code_snapshot(dest: Path) -> dict:
    """Copies the trainer and the engine into the loop's folder, so a loop can
    be rerun with the code that ran it even if that code was never committed."""
    info: dict = {"files": {}}
    try:
        for pattern in CODE_FILES:
            for src in sorted(REPO.glob(pattern)):
                rel = src.relative_to(REPO).as_posix()
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / rel)
                info["files"][rel] = _sha256(src)
    except OSError as e:
        info["error"] = str(e)
    info.update(_git())
    return info


def _hf_revision(model_id: str) -> Optional[str]:
    """The commit of the cached base model (what "sentence-transformers/LaBSE" meant)."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        ref = Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--")) / "refs" / "main"
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    except Exception:   # noqa: BLE001 - bookkeeping must never stop a loop
        return None


def environment() -> dict:
    from .model import BASE_MODEL
    env: dict = {"python": platform.python_version(), "platform": platform.platform(), "packages": {}}
    for p in ("torch", "sentence-transformers", "transformers", "numpy", "scikit-learn"):
        try:
            env["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    try:
        if "torch" not in sys.modules:       # a loop without PyTorch does not load it for this
            raise RuntimeError("not used by this loop")
        import torch
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:  # noqa: BLE001
        env["torch_error"] = str(e)
    env["base_model"] = BASE_MODEL
    env["base_model_revision"] = _hf_revision(BASE_MODEL)
    return env


def memory() -> Optional[dict]:
    """Free RAM and free commit (RAM plus page file) in GB, on Windows. GPU
    memory is backed by commit there, so running out of it kills a loop with a
    CUDA out-of-memory error while the card still has room."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("ram", "ram_free", "commit", "commit_free", "v", "v_free", "ext")]
        st = Status()
        st.length = ctypes.sizeof(Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return {"ram_free_gb": round(st.ram_free / 2 ** 30, 1), "commit_free_gb": round(st.commit_free / 2 ** 30, 1),
                "commit_limit_gb": round(st.commit / 2 ** 30, 1)}
    except Exception:  # noqa: BLE001
        return None


def keep_awake(on: bool) -> None:
    """Windows must not go to sleep while a loop runs. Asks for this process
    only, changes no setting; closing the lid still sleeps."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if on else 0x80000000)
        except Exception:  # noqa: BLE001
            pass


# ── one loop ─────────────────────────────────────────────────────────────────

class LoopRecord:
    """with LoopRecord(paths, "fit", configs) as rec: ... rec.log(...); rec.update(...)

    Writes the start line on entry and the end line on exit, including when
    the body raises (status failed or interrupted, with the traceback)."""

    def __init__(self, paths, kind: str, configs: Optional[dict] = None, batch: Optional[str] = None,
                 command: Optional[List[str]] = None):
        self.paths, self.kind, self.batch = paths, kind, batch
        self.configs = configs or {}
        self.command = list(command if command is not None else sys.argv)
        self.dir: Path = Path()
        self.id = ""
        self.seq = 0
        self.entry: dict = {}
        self._log = None
        self._t0 = 0.0

    def __enter__(self) -> "LoopRecord":
        runs = self.paths.runs
        runs.mkdir(parents=True, exist_ok=True)
        self.dir = new_run_dir(runs, self.kind)
        self.dir.mkdir(parents=True)
        self.id = self.dir.name
        self.seq = 1 + sum(1 for e in read_events(runs) if e.get("event") == "start")
        self._t0 = time.perf_counter()
        self._log = (self.dir / "log.txt").open("a", encoding="utf-8")
        self.entry = {"id": self.id, "seq": self.seq, "kind": self.kind, "batch": self.batch,
                      "started": now(), "folder": str(self.dir), "pid": os.getpid(), "command": self.command}
        append_event(runs, {"event": "start", **self.entry})
        keep_awake(True)
        self.log(f"[loop {self.seq}] {self.id}: {' '.join(self.command)}")
        self.entry["config"] = self.configs
        self.entry["code"] = code_snapshot(self.dir / "code")
        self.entry["environment"] = environment()
        mem = memory()
        self.entry["memory_at_start"] = mem
        if mem and self.kind != "baseline" and mem["commit_free_gb"] < 8:
            self.log(f"  low memory: {mem['commit_free_gb']} GB can still be committed (RAM plus page file); "
                     "a fit needs about 8. Close other programs if it fails with out of memory.")
        self.entry["data"] = data_fingerprint(self.paths)
        try:
            if "torch" not in sys.modules:
                raise RuntimeError
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass
        return self

    def log(self, message: str) -> None:
        print(message, flush=True)
        if self._log:
            self._log.write(message + "\n")
            self._log.flush()

    def update(self, **fields) -> None:
        self.entry.update(fields)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            status = self.entry.get("status") or "ok"
        elif issubclass(exc_type, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = "failed"
        if exc_type is not None:
            self.entry["error"] = (str(exc) if issubclass(exc_type, SystemExit)
                                   else "".join(traceback.format_exception(exc_type, exc, tb))[-8000:])
        self.entry.update({"status": status, "finished": now(),
                           "seconds": round(time.perf_counter() - self._t0, 1)})
        try:
            if "torch" not in sys.modules:
                raise RuntimeError
            import torch
            if torch.cuda.is_available():
                self.entry["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
                self.entry["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2 ** 20)
        except Exception:  # noqa: BLE001
            pass
        keep_awake(False)
        self.entry["pool_after"] = data_fingerprint(self.paths, inputs=False)
        (self.dir / "record.json").write_text(json.dumps(self.entry, indent=2, ensure_ascii=False, default=str),
                                              encoding="utf-8")
        append_event(self.paths.runs, {"event": "end", **self.entry})
        last = (self.entry.get("error") or "").strip().splitlines()
        self.log(f"[loop {self.seq}] {status}" + (f": {last[-1]}" if last else "")
                 + f" ({self.entry['seconds']:.0f}s) -> {self.dir}")
        if self._log:
            self._log.close()
            self._log = None
        return False        # an exception still propagates


# ── comparison with the reference benchmark ──────────────────────────────────

def load_reference(path: Optional[Path] = REFERENCE) -> Optional[dict]:
    """The by-article section of eval/held_out/results/by_article.json: stock
    LaBSE on the same split, the numbers in docs/evaluation.md."""
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8")).get("by_article")


def compare_to_reference(bench: dict, reference: Optional[dict]) -> Optional[dict]:
    """Differences, benchmark minus reference, for every number both have."""
    if not reference or not bench:
        return None
    diffs: Dict[str, float] = {}
    for sub, ref in reference.get("results", {}).items():
        mine = bench.get("results", {}).get(sub)
        if not mine:
            continue
        for system in ("deployed", "centroid", "linear_head", "logreg"):
            if system in ref and system in mine:
                for k in ("strict", "near_miss"):
                    diffs[f"{sub} | {system} | {k}"] = round(mine[system][k] - ref[system][k], 6)
        for k in PROFILE_KEYS:
            if k in ref.get("profile", {}) and k in mine.get("profile", {}):
                diffs[f"{sub} | profile | {k}"] = round(mine["profile"][k] - ref["profile"][k], 6)
    shares = {k: v for k, v in diffs.items() if not k.endswith("mean_size")}
    worst = max(shares.items(), key=lambda kv: abs(kv[1])) if shares else (None, None)
    ref_all, mine_all = reference["results"].get("all classes", {}), bench["results"].get("all classes", {})
    return {"n_test": [bench.get("n_test"), reference.get("n_test")],
            "classes": [mine_all.get("classes"), ref_all.get("classes")],
            "same_holdout_size": bench.get("n_test") == reference.get("n_test")
                                 and mine_all.get("classes") == ref_all.get("classes"),
            "max_abs_diff": abs(worst[1]) if worst[1] is not None else None, "worst": worst[0],
            "diffs": diffs}


def format_reference_check(check: Optional[dict]) -> str:
    if not check:
        return f"  no reference benchmark at {REFERENCE}"
    head = (f"  against {REFERENCE.relative_to(REPO).as_posix()}: holdout {check['n_test'][0]:,} vs "
            f"{check['n_test'][1]:,} records, {check['classes'][0]} vs {check['classes'][1]} classes")
    if check["max_abs_diff"] is None:
        return head
    return head + f"; largest difference {check['max_abs_diff'] * 100:.2f} points ({check['worst']})"


# ── the report ───────────────────────────────────────────────────────────────

def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def _cell(res: Optional[dict], ref: Optional[dict], system: str, key: str = "strict") -> str:
    if not res or system not in res:
        return "-"
    v = res[system][key]
    if ref and system in ref:
        return f"{_pct(v)} ({100 * (v - ref[system][key]):+.1f})"
    return _pct(v)


def _note(loop: dict) -> str:
    """The gate's reason or the error's last line, folders stripped from paths."""
    text = loop.get("reason") or loop.get("error") or ""
    lines = [l for l in str(text).strip().splitlines() if l.strip()]
    note = re.sub(r"(?:[A-Za-z]:)?(?:[\\/][^\s\\/:|]+)+[\\/]", "", lines[-1]) if lines else ""
    note = note.replace("|", "/")
    return (note[:90] + "...") if len(note) > 93 else note


def _rows(loop: dict) -> str:
    b = loop.get("benchmark") or {}
    if not b:
        return "-"
    return f"{b.get('n_labeled', b.get('n_train', 0)):,} + {b.get('n_pseudo', 0):,}"


def cmd_report(paths, reference_path: Optional[Path] = REFERENCE, out: Optional[Path] = None, log=print) -> Path:
    items = loops(paths.runs)
    ref = load_reference(reference_path)
    out = out or paths.runs / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    ref_res = (ref or {}).get("results", {})
    counts: Dict[str, int] = {}
    for it in items:
        counts[it.get("status", "?")] = counts.get(it.get("status", "?"), 0) + 1

    md = ["# Training loops", "",
          f"{now()}: {len(items)} loops recorded in `{(paths.runs / LEDGER).as_posix()}` ("
          + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) + ")."]
    if ref:
        rel = Path(reference_path).resolve()
        try:
            rel = rel.relative_to(REPO)
        except ValueError:
            pass
        md += ["", f"Reference: `{rel.as_posix()}`, by-article split, stock LaBSE "
                   f"({ref.get('n_test', 0):,} holdout texts, {ref_res.get('all classes', {}).get('classes')} classes): "
                   "the numbers in docs/evaluation.md. In the tables a loop's value is followed by its difference "
                   "from the reference in points; the prototype head has no reference. \"Text by text\" counts "
                   "the holdout texts a loop gets right and the latest baseline does not (+) and the reverse (-); "
                   "z of 2 or more is a real difference, and two baselines differing only in floating-point "
                   "rounding are 8 texts apart.", "",
               "| reference | " + " | ".join(REPORT_SUBSETS) + " |", "|---|" + "---|" * len(REPORT_SUBSETS)]
        for system in REPORT_SYSTEMS[:-1]:
            md.append(f"| {system} strict / near-miss | " + " | ".join(
                f"{_pct(ref_res[s][system]['strict'])} / {_pct(ref_res[s][system]['near_miss'])}"
                if s in ref_res and system in ref_res[s] else "-" for s in REPORT_SUBSETS) + " |")
        md.append("| profile recall (size) | " + " | ".join(
            f"{_pct(ref_res[s]['profile']['rhizome'])} ({ref_res[s]['profile']['mean_size']:.1f})"
            if s in ref_res else "-" for s in REPORT_SUBSETS) + " |")
    else:
        md += ["", f"No reference benchmark at `{reference_path}`."]

    base = [it for it in items if it.get("kind") == "baseline" and it.get("benchmark")]
    if base:
        chk = base[-1].get("reference_check") or compare_to_reference(base[-1]["benchmark"], ref)
        if chk:
            largest = "-" if chk["max_abs_diff"] is None else "%.2f" % (100 * chk["max_abs_diff"])
            md += ["", f"Latest baseline ({base[-1]['id']}) against the reference: holdout "
                       f"{chk['n_test'][0]:,} vs {chk['n_test'][1]:,} texts, {chk['classes'][0]} vs "
                       f"{chk['classes'][1]} classes; largest difference {largest} points"
                       + (f" ({chk['worst']})." if chk.get("worst") else ".")]

    all_ref = ref_res.get("all classes")
    md += ["", "## Every loop: all classes, strict", "",
           "| # | loop | kind | status | train rows (labeled + pseudo) | deployed | centroid | linear head | logreg "
           "| head | head top-5 | profile recall (size) | deployed vs baseline, text by text | minutes | note |",
           "|---:|---|---|---|---|---|---|---|---|---|---|---|---|---:|---|"]
    from .evaluate import load_predictions, paired_test
    base_preds = load_predictions(Path(base[-1]["folder"])) if base and base[-1].get("folder") else None
    for it in items:
        res = (it.get("benchmark") or {}).get("results", {}).get("all classes")
        pt = None
        if base_preds and res and it["id"] != base[-1]["id"] and it.get("folder"):
            preds = load_predictions(Path(it["folder"]))
            pt = paired_test(preds, base_preds, "deployed") if preds else None
        head5 = ((it.get("metrics") or {}).get("head") or {}).get("all", {}).get("top5")
        prof = (res or {}).get("profile")
        md.append(f"| {it.get('seq', '')} | {it['id']} | {it.get('kind', '')} | {state(it)} | {_rows(it)} | "
                  + " | ".join(_cell(res, all_ref, s) for s in REPORT_SYSTEMS)
                  + f" | {_pct(head5)} | "
                  + (f"{_pct(prof['rhizome'])} ({prof['mean_size']:.1f})" if prof else "-")
                  + (f" | +{pt['gained']} / -{pt['lost']} (z {pt['z']:.1f})" if pt else " | -")
                  + f" | {it.get('seconds', 0) / 60:.1f} | {_note(it)} |")

    md += ["", "## Every loop: all classes, near-miss", "",
           "| # | loop | " + " | ".join(REPORT_SYSTEMS) + " |", "|---:|---|" + "---|" * len(REPORT_SYSTEMS)]
    for it in items:
        res = (it.get("benchmark") or {}).get("results", {}).get("all classes")
        if res:
            md.append(f"| {it.get('seq', '')} | {it['id']} | "
                      + " | ".join(_cell(res, all_ref, s, "near_miss") for s in REPORT_SYSTEMS) + " |")

    for sub in REPORT_SUBSETS[1:]:
        lines = [f"| {it.get('seq', '')} | {it['id']} | "
                 + " | ".join(_cell(res, ref_res.get(sub), s) for s in REPORT_SYSTEMS) + " |"
                 for it in items if (res := (it.get("benchmark") or {}).get("results", {}).get(sub))]
        if lines:
            md += ["", f"## Every loop: {sub}, strict", "",
                   "| # | loop | " + " | ".join(REPORT_SYSTEMS) + " |", "|---:|---|" + "---|" * len(REPORT_SYSTEMS)]
            md += lines
    md += ["", "Every number for every loop, the reference first: `report.csv`. Per-text answers: each loop's "
               "`holdout_predictions.jsonl.gz`.", ""]
    out.write_text("\n".join(md), encoding="utf-8")

    cols = ["seq", "id", "kind", "status", "batch", "started", "seconds", "peak_vram_mb", "promoted", "champion_before",
            "n_train", "n_labeled", "n_pseudo", "n_hand_added", "n_test", "classes", "git_commit", "note"]
    metric_cols = [f"{sub} | {s} | {k}" for sub in REPORT_SUBSETS for s in REPORT_SYSTEMS for k in ("strict", "near_miss")]
    metric_cols += [f"{sub} | profile | {k}" for sub in REPORT_SUBSETS for k in PROFILE_KEYS]

    def flat(results: dict) -> dict:
        row = {}
        for c in metric_cols:
            sub, s, k = c.split(" | ")
            res = results.get(sub, {})
            v = res.get(s, {}).get(k) if s != "profile" else res.get("profile", {}).get(k)
            row[c] = "" if v is None else round(v, 6)
        return row

    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols + metric_cols)
        w.writeheader()
        if ref:
            w.writerow({"id": "reference", "kind": "reference", "status": str(reference_path),
                        "n_train": ref.get("n_train"), "n_test": ref.get("n_test"),
                        "classes": ref_res.get("all classes", {}).get("classes"), **flat(ref_res)})
        for it in items:
            b = it.get("benchmark") or {}
            w.writerow({"seq": it.get("seq"), "id": it["id"], "kind": it.get("kind"), "status": it.get("status"),
                        "batch": it.get("batch") or "", "started": it.get("started"), "seconds": it.get("seconds"),
                        "peak_vram_mb": it.get("peak_vram_mb"), "promoted": it.get("promoted", ""),
                        "champion_before": it.get("champion_before") or "",
                        "n_train": b.get("n_train"), "n_labeled": b.get("n_labeled"), "n_pseudo": b.get("n_pseudo"),
                        "n_hand_added": b.get("n_hand_added"), "n_test": b.get("n_test"),
                        "classes": b.get("results", {}).get("all classes", {}).get("classes"),
                        "git_commit": (it.get("code") or {}).get("git_commit"), "note": _note(it),
                        **flat(b.get("results", {}))})
    log(f"{len(items)} loops -> {out} and {out.with_suffix('.csv')}")
    return out


# ── following a loop live ────────────────────────────────────────────────────

def _gpu_line() -> Optional[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,"
                              "power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        util, used, total, temp, power = [v.strip() for v in out.splitlines()[0].split(",")]
        return (f"  [gpu] {util}% busy, {int(float(used)):,} of {int(float(total)):,} MiB, {temp} C, "
                f"{float(power):.0f} W")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _follow(runs: Path, item: dict, gpu_every: float, log) -> None:
    folder = Path(item.get("folder") or runs / item["id"])
    path = folder / "log.txt"
    log(f"=== loop {item.get('seq')}: {item['id']} ({item.get('kind')}), folder {folder} ===")
    pos, partial = 0, b""
    if path.exists():                       # a loop already under way: start with its last 60 lines
        data = path.read_bytes()
        lines = data.split(b"\n")
        partial, pos = lines[-1], len(data)
        for line in lines[:-1][-60:]:
            log(line.decode("utf-8", "replace"))
    last_gpu = last_check = 0.0
    while True:
        if path.exists():
            with path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            if chunk:
                lines = (partial + chunk).split(b"\n")
                partial = lines[-1]
                for line in lines[:-1]:
                    log(line.decode("utf-8", "replace"))
        t = time.time()
        if t - last_check >= 3:
            last_check = t
            current = next((it for it in loops(runs) if it["id"] == item["id"]), item)
            st = state(current)
            if st not in ("running", "did_not_finish"):
                time.sleep(1.0)             # let the last lines land
                if path.exists():
                    with path.open("rb") as f:
                        f.seek(pos)
                        rest = partial + f.read()
                    for line in rest.decode("utf-8", "replace").splitlines():
                        log(line)
                log(f"=== loop {item.get('seq')} ended: {st} ===")
                return
        if gpu_every and t - last_gpu >= gpu_every:
            last_gpu = t
            line = _gpu_line()
            if line:
                log(line)
        time.sleep(1.0)


def cmd_watch(paths, loop_id: Optional[str] = None, once: bool = False, gpu_every: float = 15.0,
              log=print) -> None:
    """Follows loops live: the running loop's log.txt as it grows, a GPU line
    every gpu_every seconds, and its status when it ends; then waits for the
    next loop, unless once. Ctrl+C stops the watching, not the loop."""
    runs = paths.runs
    seen: set = set()
    waiting = False
    try:
        while True:
            items = loops(runs)
            if loop_id:
                target = next((it for it in items if it["id"] == loop_id), None)
                if target is None:
                    raise SystemExit(f"no loop {loop_id} in {runs / LEDGER}")
            else:
                running = [it for it in items if it.get("status") == "did_not_finish"
                           and it["id"] not in seen and state(it) != "killed"]
                target = running[-1] if running else None
            if target is None:
                if once and seen:
                    return
                if not waiting:
                    log(f"No loop running; waiting for one to start ({runs / LEDGER}). Ctrl+C stops watching.")
                    waiting = True
                time.sleep(2.0)
                continue
            waiting = False
            _follow(runs, target, gpu_every, log)
            seen.add(target["id"])
            if once or loop_id:
                return
    except KeyboardInterrupt:
        log("\nStopped watching. Loops keep running.")
