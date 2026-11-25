"""python -m train forever: self-training rounds with no end.

Starts one `python -m train round` after another, each in its own process so
its memory is fully released, until `python -m train stop` (or Ctrl+C, or its
window is closed). It waits for any loop that is already running, waits while
the laptop is on battery, and backs off and tries again when a loop fails.
When a round finds nothing new it searches wider.

It loads no PyTorch. What it does is in runs/forever.log; the loops themselves
are in the ledger like any other.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .record import REPO, append_event, keep_awake, loops, memory, now, state

# How widely to search, step by step, while rounds keep finding too little.
WIDTHS = [
    {"per-node": 10, "paras-per-article": 3, "langs": "es,fr,de,pt"},
    {"per-node": 20, "paras-per-article": 3, "langs": "es,fr,de,pt"},
    {"per-node": 35, "paras-per-article": 4, "langs": "es,fr,de,pt"},
    {"per-node": 60, "paras-per-article": 4, "langs": "es,fr,de,pt,it,nl"},
]
NODES = 20              # labels searched per round, and seconds between requests: Wikipedia limits
WIKI_DELAY = 2.5        # how often one address may ask, and this runs for hours
MIN_Z = 2.33            # many tries on one holdout: about a 1 in 100 chance of a false win each
MIN_FREE_GB = 20
SILENT_MINUTES = 60     # a loop that writes nothing for this long is stuck: end it and go on


def on_battery() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        class Power(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ubyte) for n in ("ac", "flag", "percent", "saver")] + [
                ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
        p = Power()
        return bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p))) and p.ac == 0
    except Exception:  # noqa: BLE001
        return False


def others_running(runs: Path) -> str:
    """A loop, or a script that starts loops, that this runner did not start."""
    for it in loops(runs):
        if it.get("kind") in ("fit", "round", "baseline") and state(it) == "running":
            return f"loop {it.get('seq')}"
    if os.name == "nt":
        try:
            out = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                                  "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
                                  "'-m train (fit|round|loop|baseline)|queue[.]sh' -and $_.Name -ne 'powershell.exe' } "
                                  "| Select-Object -First 1 -ExpandProperty Name"],
                                 capture_output=True, text=True, timeout=60).stdout.strip()
            return f"another training process ({out})" if out else ""
        except (OSError, subprocess.SubprocessError):
            return ""
    return ""


def silent_minutes(folder) -> float:
    """Minutes since the loop last wrote to its log or its live frame."""
    files = [Path(folder) / "log.txt", Path(folder) / "live" / "frame.json"]
    times = [f.stat().st_mtime for f in files if f.exists()]
    return (time.time() - max(times)) / 60 if times else 0.0


def end_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True, timeout=60)
    else:
        os.kill(int(pid), 9)


def end_stuck(runs: Path, say) -> bool:
    """Ends a running loop that has been silent too long. True if it did."""
    for it in loops(runs):
        if (it.get("kind") in ("fit", "round", "baseline") and it.get("folder") and it.get("pid")
                and state(it) == "running" and silent_minutes(it["folder"]) > SILENT_MINUTES):
            say(f"loop {it.get('seq')} has been silent for {SILENT_MINUTES} minutes: ending it")
            end_tree(it["pid"])
            return True
    return False


def last_width(journal: Path) -> int:
    """The search width the run before this one had reached, so that starting
    again does not repeat searches Wikipedia has already answered."""
    if not journal.exists():
        return 0
    width = 0
    for line in journal.read_text(encoding="utf-8").splitlines():
        found = re.search(r"search width (\d+) of", line)
        if found:
            width = int(found.group(1)) - 1
        elif re.search(r"loop \d+: promoted", line):
            width = 0
    return min(max(width, 0), len(WIDTHS) - 1)


def cmd_stop(runs: Path, log=print) -> None:
    (runs / "STOP").write_text(now(), encoding="utf-8")
    log("The endless run stops after the loop it is on.")


def cmd_forever(runs: Path, log=print, round_args=()) -> None:
    runs.mkdir(parents=True, exist_ok=True)
    stop, pid_file, journal = runs / "STOP", runs / "forever.pid", runs / "forever.log"
    if pid_file.exists():
        from .record import _alive
        old = pid_file.read_text(encoding="utf-8").strip()
        if old.isdigit() and int(old) != os.getpid() and _alive(int(old), pid_file.stat().st_mtime):
            raise SystemExit(f"the endless run is already going (process {old})")
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    stop.unlink(missing_ok=True)

    def say(message: str) -> None:
        line = f"{now()}  {message}"
        log(line)
        with journal.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def pause(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end and not stop.exists():
            time.sleep(min(5.0, max(0.1, end - time.time())))

    say("endless run started: rounds until `python -m train stop`")
    append_event(runs, {"event": "forever_start", "time": now(), "pid": os.getpid(), "min_z": MIN_Z})
    keep_awake(True)
    width, empty, failures, waiting, draw = last_width(journal), 0, 0, "", False
    try:
        while not stop.exists():
            why = others_running(runs)
            if not why and on_battery():
                why = "on battery"
            if not why and shutil.disk_usage(runs).free < MIN_FREE_GB * 2 ** 30:
                why = f"less than {MIN_FREE_GB} GB of disk left"
            if why:
                if why.startswith("loop") and end_stuck(runs, say):
                    continue
                if why != waiting:
                    say(f"waiting: {why}")
                    waiting = why
                pause(60)
                continue
            waiting = ""
            before = {it["id"] for it in loops(runs)}
            flags = [x for k, v in WIDTHS[width].items() for x in (f"--{k}", str(v))]
            cmd = [sys.executable, "-m", "train", "round", "--nodes-per-round", str(NODES),
                   "--wiki-delay", str(WIKI_DELAY), "--min-z", str(MIN_Z),
                   "--unproven", "stack", *flags, *round_args]
            kind = "round"
            if draw:                     # nothing new to read: train the same data with another seed
                kind, draw, seed = "fit", False, 1000 + len(before)
                cmd = [sys.executable, "-m", "train", "fit", "--seed", str(seed), "--min-z", str(MIN_Z)]
                say(f"starting a fit with seed {seed} (memory {memory()})")
            else:
                say(f"starting a round (search width {width + 1} of {len(WIDTHS)}, memory {memory()})")
            with (runs / "forever-rounds.out").open("a", encoding="utf-8") as out:
                child = subprocess.Popen(cmd, cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                began = time.time()
                while child.poll() is None:
                    time.sleep(10)
                    folders = [Path(it["folder"]) for it in loops(runs) if it["id"] not in before and it.get("folder")]
                    quiet = silent_minutes(folders[-1]) if folders else (time.time() - began) / 60
                    if quiet > SILENT_MINUTES:
                        say(f"nothing written for {SILENT_MINUTES} minutes: ending this round")
                        end_tree(child.pid)
                        child.wait(timeout=120)
                code = child.returncode
            new = [it for it in loops(runs) if it["id"] not in before and it.get("kind") == kind]
            it = new[-1] if new else {}
            status = it.get("status", "did_not_start")
            score = (it.get("metrics") or {}).get("deployed", {}).get("all", {}).get("strict")
            say(f"loop {it.get('seq', '?')}: {status}" + (f", deployed {score:.4f}" if score else "")
                + (f", {it.get('accepted')} new labels, {it.get('stacked_after')} unproven" if it.get("accepted") else "")
                + (f" | {it.get('reason')}" if it.get("reason") else ""))
            if status in ("promoted", "not_promoted"):
                failures = 0
                if status == "promoted":
                    width = empty = 0    # a new champion judges everything afresh
                elif kind == "round":
                    empty = 0
                    if (it.get("accepted") or 0) < 60:
                        width = min(width + 1, len(WIDTHS) - 1)
                pause(20)
            elif status in ("no_candidates", "nothing_accepted"):
                failures, empty = 0, empty + 1
                width = min(width + 1, len(WIDTHS) - 1)
                draw = empty >= 2        # twice in a row: leave Wikipedia alone for one fit
                say("nothing new to learn from; " + ("a fit with another seed is next" if draw else "searching wider"))
                pause(60)
            else:
                failures += 1
                wait = min(3600, 120 * 2 ** min(failures - 1, 5))
                say(f"that did not work (exit {code}, failure {failures} in a row); next try in {wait // 60} min. "
                    "Details: the loop's record.json and runs/forever-rounds.out")
                pause(wait)
        say("stopped: asked to")
    except KeyboardInterrupt:
        say("stopped: Ctrl+C")
    finally:
        keep_awake(False)
        append_event(runs, {"event": "forever_end", "time": now(), "pid": os.getpid()})
        stop.unlink(missing_ok=True)
        pid_file.unlink(missing_ok=True)
