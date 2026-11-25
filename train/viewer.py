"""`python -m train view`: the training loops and the classifier, in the browser.

A server on 127.0.0.1. It writes nothing but a cache, runs/stars.npz. It serves the 3D page
(viewer3d.html), the flat one (viewer.html, at /flat) and the frames that
live.LiveTap writes under runs/<loop>/live. A trained loop with no frames
(from before the capture existed, or run with --no-live) is shown as stock
LaBSE against its saved weights.

/classify is the rhizome classifier (classify.html) and /coredrill shows how
the coredrill changed from loop to loop (coredrill.html). Both use a process of
its own, started when needed and ended when left idle, so they take no memory
from a training loop in between.
"""
from __future__ import annotations

import gzip
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np

from . import coredrills, progress
from .frames import describe, pack, spread
from .record import LEDGER, REPO, loops, memory, state

PAGES = {"/": Path(__file__).with_name("viewer3d.html"), "/flat": Path(__file__).with_name("viewer.html"),
         "/classify": Path(__file__).with_name("classify.html"), "/coredrill": Path(__file__).with_name("coredrill.html"),
         "/loops": Path(__file__).with_name("loops.html")}
MAX_BODY = 64 * 1024          # bytes of a request; a text may have 8,000 characters
IDLE_MINUTES = 10             # the classifier's process ends after this long unused
CLASSIFIER_NEEDS_GB = 1.3     # it commits about 0.9
LOOP_ID = re.compile(r"^\d{8}-\d{6}-[a-z]+$")


def gpu() -> Optional[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
                              "power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        name, util, used, total, temp, power = [v.strip() for v in out.splitlines()[0].split(",")]
        return {"name": name, "util": float(util), "used": float(used), "total": float(total),
                "temp": float(temp), "power": float(power)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class Cached:
    """A value recomputed at most every `seconds`."""

    def __init__(self, fn, seconds: float):
        self.fn, self.seconds, self.at, self.value, self.lock = fn, seconds, 0.0, None, threading.Lock()

    def __call__(self):
        with self.lock:
            if time.time() - self.at > self.seconds:
                self.value, self.at = self.fn(), time.time()
            return self.value


def _stock(name: str, base: Path, files: dict):
    """The stock LaBSE tensor behind a trained parameter's name."""
    from safetensors import safe_open
    m = re.match(r"^0\.(?:auto_model|model)\.(.+)$", name)
    key, path = (m.group(1), base / "model.safetensors") if m else (name.split(".", 1)[1], base / "2_Dense" / "model.safetensors")
    if path not in files:
        files[path] = safe_open(str(path), "pt")
    return files[path].get_tensor(key)


def before_after(run_dir: Path, meta: dict) -> dict:
    """Stock LaBSE against a finished loop's saved weights, sampled like a frame."""
    import torch
    from huggingface_hub import snapshot_download
    base = Path(snapshot_download(meta["base_model"], local_files_only=True))
    w = torch.load(run_dir / "weights.pt", map_location="cpu", weights_only=True)
    files: dict = {}
    matrices, w0, total, total_rms, w_rms = [], [], [], [], []
    for name, p in w["encoder_delta"].items():
        if p.ndim != 2:
            continue
        old, new = _stock(name, base, files).float().numpy(), p.float().numpy()
        ri, ci = spread(new.shape[0]), spread(new.shape[1])
        a, b = old[np.ix_(ri, ci)], new[np.ix_(ri, ci)]
        matrices.append({"name": name, **describe(name), "shape": list(new.shape), "rows": ri.tolist(), "cols": ci.tolist()})
        w0.append(pack(a))
        total.append(pack(b - a))
        total_rms.append(float(np.sqrt(np.mean((b - a) ** 2))))
        w_rms.append(float(np.sqrt(np.mean(b ** 2))))
    return {"meta": {"labels": meta["labels"], "matrices": matrices, "w0": w0, "scale": meta["scale"],
                     "epochs": len(meta.get("history", [])), "heads": 0, "static": True},
            "frame": {"static": True, "total": total, "total_rms": total_rms, "w_rms": w_rms,
                      "series": [], "val": meta.get("history", [])}}


class Classify:
    """The classifier's process: started when needed, one request at a time,
    ended when idle."""

    def __init__(self, runs: Path, data: Path):
        self.runs, self.data = runs, data
        self.child: Optional[subprocess.Popen] = None
        self.lines: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.used = 0.0

    def _start(self) -> None:
        free = (memory() or {}).get("commit_free_gb")
        if free is not None and free < CLASSIFIER_NEEDS_GB:
            raise RuntimeError(f"Only {free} GB of memory is free and the classifier needs {CLASSIFIER_NEEDS_GB}. "
                               "Close some programs, or wait for the loop to end.")
        self.lines = queue.Queue()
        self.child = subprocess.Popen(
            [sys.executable, "-m", "train", "classify", "--serve", "--runs", str(self.runs), "--data", str(self.data)],
            cwd=str(REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8", "OPENBLAS_NUM_THREADS": "2"},
            # Short bursts someone is waiting for: ahead of a browser or a loop using every core.
            creationflags=getattr(subprocess, "ABOVE_NORMAL_PRIORITY_CLASS", 0))
        threading.Thread(target=self._read, args=(self.child, self.lines), daemon=True).start()
        threading.Thread(target=self._end_when_idle, args=(self.child,), daemon=True).start()

    @staticmethod
    def _read(child: subprocess.Popen, lines: queue.Queue) -> None:
        for line in child.stdout:
            lines.put(line)
        lines.put(None)

    def _end_when_idle(self, child: subprocess.Popen) -> None:
        while child.poll() is None:
            time.sleep(20)
            with self.lock:
                if time.time() - self.used > IDLE_MINUTES * 60:
                    self._end(child)

    @staticmethod
    def _end(child: Optional[subprocess.Popen]) -> None:
        if child is None or child.poll() is not None:
            return
        try:
            child.stdin.close()                 # it ends by itself
            child.wait(timeout=10)
        except (OSError, subprocess.SubprocessError):
            child.kill()

    def ask(self, request: dict, timeout: float = 120.0) -> dict:
        with self.lock:
            self.used = time.time()
            if self.child is None or self.child.poll() is not None:
                self._start()
            try:
                self.child.stdin.write(json.dumps(request) + chr(10))
                self.child.stdin.flush()
                line = self.lines.get(timeout=timeout)
            except (OSError, queue.Empty):
                line = None
            self.used = time.time()
            if line is None:                    # it died or hangs: a fresh one next time
                child, self.child = self.child, None
                child.kill()
                raise RuntimeError("The classifier stopped unexpectedly. Try again.")
            return json.loads(line)

    def stop(self) -> None:
        with self.lock:
            self._end(self.child)


class Handler(BaseHTTPRequestHandler):
    server_version = "coredrill-view"
    runs: Path
    classify: Classify
    loops_cache: Cached
    gpu_cache: Cached
    static_cache: dict = {}

    def log_message(self, *args) -> None:       # quiet
        pass

    def _send(self, code: int, body: bytes = b"", kind: str = "application/json", encoding: str = "") -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; img-src 'self' data:")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, code: int = 200) -> None:
        self._send(code, json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        self._answer(self._get)

    def do_POST(self) -> None:  # noqa: N802
        self._answer(self._post)

    def _answer(self, fn) -> None:
        try:
            fn()
        except (BrokenPipeError, ConnectionError):
            pass
        except Exception as e:  # noqa: BLE001 - a broken file must not take the viewer down
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            except OSError:
                pass

    def _local(self) -> bool:
        port = self.server.server_address[1]
        hosts = (f"127.0.0.1:{port}", f"localhost:{port}")
        origin = self.headers.get("Origin")
        return self.headers.get("Host", "") in hosts and (origin is None or origin in [f"http://{h}" for h in hosts])

    def _ask(self, request: dict) -> None:
        run = request.get("run")
        if run is not None and not (isinstance(run, str) and LOOP_ID.match(run)):
            return self._json({"error": "unknown loop"}, 400)
        try:
            out = self.classify.ask(request)
        except RuntimeError as e:
            return self._json({"error": str(e)}, 503)
        return self._json(out, 400 if "error" in out else 200)

    def _post(self) -> None:
        # Another website open in the browser can send a form here, but not
        # JSON with this page's origin.
        if not self._local() or self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._json({"error": "this page is only served to this machine"}, 403)
        if self.path != "/api/classify":
            return self._json({"error": "not found"}, 404)
        try:
            size = int(self.headers.get("Content-Length", ""))
        except ValueError:
            size = -1
        if not 0 <= size <= MAX_BODY:
            return self._json({"error": "the text is too long"}, 413)
        try:
            body = json.loads(self.rfile.read(size))
            text, run = body.get("text"), body.get("run")
        except (ValueError, AttributeError):
            return self._json({"error": "not valid JSON"}, 400)
        if not isinstance(text, str):
            return self._json({"error": "text is missing"}, 400)
        return self._ask({"op": "classify", "text": text, "run": run})

    def _get(self) -> None:
        if not self._local():
            return self._json({"error": "this page is only served to this machine"}, 403)
        path, _, query = self.path.partition("?")
        if path in PAGES:
            return self._send(200, PAGES[path].read_bytes(), "text/html; charset=utf-8")
        if path == "/api/loops":
            return self._json(self.loops_cache())
        if path == "/api/gpu":
            return self._json(self.gpu_cache())
        if path == "/api/classify/models":
            return self._json(list_models(self.runs))
        if path == "/api/progress":
            return self._json(self.progress_cache())
        if path == "/api/coredrills":
            return self._json(coredrills.listing(self.runs))
        if path == "/api/coredrills/live":
            loop = dict(p.partition("=")[::2] for p in query.split("&")).get("loop", "")
            if not LOOP_ID.match(loop):
                return self._json({"error": "unknown loop"}, 400)
            try:
                return self._json(coredrills.live(self.runs, loop))
            except ValueError as e:
                return self._json({"error": str(e)}, 404)
        if path == "/api/coredrills/timeline":
            return self._ask({"op": "timeline"})
        if path == "/api/coredrills/compare":
            q = dict(p.partition("=")[::2] for p in query.split("&"))
            ids = [q.get("a", ""), q.get("b", "")]
            if not all(LOOP_ID.match(i) or i == coredrills.DEPLOYED for i in ids):
                return self._json({"error": "unknown loop"}, 400)
            return self._ask({"op": "compare", "a": ids[0], "b": ids[1]})
        m = re.match(r"^/api/classify/(map|sample|stars)$", path)
        if m:
            run = re.search(r"(?:^|&)run=([^&]+)", query)
            return self._ask({"op": m.group(1), "run": run.group(1) if run else None})
        m = re.match(r"^/api/loop/([^/]+)/([a-z]+)(?:/(\d{1,9}))?$", path)
        run = coredrills.find(self.runs, m.group(1)) if m and LOOP_ID.match(m.group(1)) else None
        if run is None:
            return self._json({"error": "not found"}, 404)
        what, n = m.group(2), m.group(3)
        live = run / "live"
        if what == "meta" and (live / "meta.json").exists():
            return self._send(200, (live / "meta.json").read_bytes())
        if what == "frame" and n:
            f = live / "frames" / f"{int(n):07d}.json.gz"
            return self._send(200, f.read_bytes(), encoding="gzip") if f.exists() else self._json({}, 404)
        if what == "frame" and (live / "frame.json").exists():
            stamp = str((live / "frame.json").stat().st_mtime_ns)
            if f"since={stamp}" in query:
                return self._send(204)
            body = (live / "frame.json").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Stamp", stamp)
            self.end_headers()
            self.wfile.write(body)
            return None
        if what == "frames":
            names = sorted((live / "frames").glob("*.json.gz")) if (live / "frames").exists() else []
            return self._json([int(f.name.split(".")[0]) for f in names])
        if what == "log":
            log = run / "log.txt"
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-6:] if log.exists() else []
            return self._json(lines)
        if what == "saved" and (run / "weights.pt").exists():
            if run.name not in self.static_cache:
                meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
                self.static_cache[run.name] = json.dumps(before_after(run, meta), ensure_ascii=False).encode("utf-8")
            return self._send(200, gzip.compress(self.static_cache[run.name], 1), encoding="gzip")
        return self._json({"error": "nothing recorded"}, 404)


def list_loops(runs: Path) -> list:
    """Every loop, of every runs folder, newest first."""
    out = []
    for run, it in ((root / it["id"], it) for root in coredrills.roots(runs) for it in loops(root)):
        res = ((it.get("benchmark") or {}).get("results") or {}).get("all classes") or {}
        out.append({"id": it["id"], "seq": it.get("seq"), "kind": it.get("kind"), "state": state(it),
                    "root": coredrills.root_name(runs, run),
                    "live": (run / "live" / "frame.json").exists(), "saved": (run / "weights.pt").exists(),
                    "deployed": (res.get("deployed") or {}).get("strict")})
    return sorted(out, key=lambda it: it["id"], reverse=True)


def list_models(runs: Path) -> list:
    """The loops whose model can classify, the champion first."""
    best = [root / "champion.json" for root in coredrills.roots(runs)]
    champions = {json.loads(f.read_text(encoding="utf-8"))["run"] for f in best if f.exists()}
    out = [{**it, "champion": it["id"] in champions} for it in list_loops(runs)
           if it["saved"] and (coredrills.find(runs, it["id"]) / coredrills.FILE).exists()]
    return sorted(out, key=lambda it: not it["champion"])


def cmd_view(paths, port: int = 8765, open_browser: bool = True, log=print) -> None:
    runs = paths.runs
    classify = Classify(runs, paths.data)
    handler = type("BoundHandler", (Handler,), {
        "runs": runs, "classify": classify, "progress_cache": Cached(lambda: progress.cmd_progress(runs), 1.5), "loops_cache": Cached(lambda: list_loops(runs), 4.0), "gpu_cache": Cached(gpu, 2.0)})
    server = None
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port from {port} to {port + 19}")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    log(f"Training view: {url}  (only this machine can open it)")
    log(f"Classifier:    {url}classify")
    log(f"Loops from {runs / LEDGER}. Ctrl+C stops the view, not the training.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("View stopped. Loops keep running.")
    finally:
        classify.stop()
        server.server_close()
