"""`python -m train view`: the matrices of a training loop, in the browser.

A read-only server on 127.0.0.1. It serves the 3D page (viewer3d.html), the
flat one (viewer.html, at /flat) and the frames that live.LiveTap writes
under runs/<loop>/live. A trained loop with no frames
(from before the capture existed, or run with --no-live) is shown as stock
LaBSE against its saved weights.
"""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np

from .frames import describe, pack, spread
from .record import LEDGER, loops, state

PAGES = {"/": Path(__file__).with_name("viewer3d.html"), "/flat": Path(__file__).with_name("viewer.html")}
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


class Handler(BaseHTTPRequestHandler):
    server_version = "coredrill-view"
    runs: Path
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
        try:
            self._get()
        except (BrokenPipeError, ConnectionError):
            pass
        except Exception as e:  # noqa: BLE001 - a broken file must not take the viewer down
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            except OSError:
                pass

    def _get(self) -> None:
        port = self.server.server_address[1]
        if self.headers.get("Host", "") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return self._json({"error": "this page is only served to this machine"}, 403)
        path, _, query = self.path.partition("?")
        if path in PAGES:
            return self._send(200, PAGES[path].read_bytes(), "text/html; charset=utf-8")
        if path == "/api/loops":
            return self._json(self.loops_cache())
        if path == "/api/gpu":
            return self._json(self.gpu_cache())
        m = re.match(r"^/api/loop/([^/]+)/([a-z]+)(?:/(\d{1,9}))?$", path)
        if not m or not LOOP_ID.match(m.group(1)) or not (self.runs / m.group(1)).is_dir():
            return self._json({"error": "not found"}, 404)
        run, what, n = self.runs / m.group(1), m.group(2), m.group(3)
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
    out = []
    for it in loops(runs):
        run = runs / it["id"]
        res = ((it.get("benchmark") or {}).get("results") or {}).get("all classes") or {}
        out.append({"id": it["id"], "seq": it.get("seq"), "kind": it.get("kind"), "state": state(it),
                    "live": (run / "live" / "frame.json").exists(), "saved": (run / "weights.pt").exists(),
                    "deployed": (res.get("deployed") or {}).get("strict")})
    return out[::-1]


def cmd_view(paths, port: int = 8765, open_browser: bool = True, log=print) -> None:
    runs = paths.runs
    handler = type("BoundHandler", (Handler,), {
        "runs": runs, "loops_cache": Cached(lambda: list_loops(runs), 4.0), "gpu_cache": Cached(gpu, 2.0)})
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
    log(f"Loops from {runs / LEDGER}. Ctrl+C stops the view, not the training.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("View stopped. Loops keep running.")
    finally:
        server.server_close()
