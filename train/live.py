"""Reads real matrices out of the training step, for the live view (viewer.py).

Every couple of seconds one step is "armed". Nothing extra is computed for the
model: the capture reads the tensors that step uses anyway.

  forward   the batch's text vectors E, the label vectors P and the logits
            E x P^T x scale; for one short text, the last layer's query and
            key matrices and the attention they make
  backward  each trained matrix's gradient, after clipping
  update    each trained matrix before and after the optimizer's step

Big matrices are sampled, not averaged: the view gets every 16th row and
column of a 768 x 768 matrix, so each cell is one real number from it.

The capture draws no random numbers and writes only under <loop>/live, so a
run trains the same with or without it, and any error in here switches the
capture off instead of stopping the training.
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .frames import GRID, describe, pack, spread

DIMS = 96            # of the 768 embedding dimensions, columns drawn for E and P
MAX_TOKENS = 40
TOP_LABELS = 6

def sample(n: int, k: int = GRID) -> torch.Tensor:
    return torch.from_numpy(spread(n, k))


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:      # the viewer has the file open
            time.sleep(0.02)


class LiveTap:
    """fit() calls start(), then for every step arm(), before_step() and
    after_step() when armed, and step(); epoch() and close() at the ends."""

    def __init__(self, folder: Path, labels: List[str], every: float = 1.5, archive_every: float = 60.0,
                 log=print, paths: Optional[dict] = None):
        self.folder, self.labels, self.every, self.archive_every, self.log = folder, labels, every, archive_every, log
        self.paths = paths or {}         # every taxonomy node's path, for the Coredrill page
        self.armed = False
        self.disabled = False
        self.feats: Optional[dict] = None
        self._hooks: list = []
        self._qk: dict = {}
        self._last = self._last_archive = 0.0
        self._recent: List[float] = []
        self.series: List[list] = []
        self.val: List[dict] = []
        self._n = 0
        self._t0 = time.perf_counter()
        self._steps_seen = 0

    # every public method goes through this: the capture may fail, training may not
    def _guard(self, fn, *args):
        if self.disabled:
            return None
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            self.disabled, self.armed = True, False
            self._release()
            self.log(f"  live view switched off ({type(e).__name__}: {e}); training goes on")
            return None

    def start(self, st, head, named, ts, epochs: int, steps: int) -> None:
        self._guard(self._start, st, head, named, ts, epochs, steps)

    def arm(self) -> bool:
        if self.disabled:
            return False
        self.armed = time.perf_counter() - self._last >= self.every
        return self.armed

    def before_step(self, *args) -> None:
        self._guard(self._before, *args)

    def after_step(self, epoch: int, step: int) -> None:
        self._guard(self._after, epoch, step)
        self.armed = False

    def step(self, loss: float) -> None:
        self._steps_seen += 1
        self._recent.append(float(loss))
        if len(self._recent) >= 10:
            self.series.append([self._steps_seen, round(sum(self._recent) / len(self._recent), 5)])
            self._recent = []

    def epoch(self, history: List[dict]) -> None:
        self.val = [{"epoch": h["epoch"], "val_acc": h["val_acc"], "loss": h["loss"]} for h in history]

    def close(self) -> None:
        self._release()

    # ── inside ───────────────────────────────────────────────────────────────

    def _release(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks, self._qk, self.feats = [], {}, None
        self.mats, self._cur = [], None
        self.armed = False

    def _start(self, st, head, named, ts, epochs, steps) -> None:
        (self.folder / "frames").mkdir(parents=True, exist_ok=True)
        self.st, self.head, self.ts, self.epochs, self.steps = st, head, ts, epochs, steps
        self._cur = None
        self.mats = []
        for name, p in [(n, p) for n, p in named if p.ndim == 2] + [("protos", head.protos)]:
            self.mats.append((name, p, sample(p.shape[0]).to(p.device), sample(p.shape[1]).to(p.device)))
        self.w0 = self._subs()
        self.heads = 0
        try:
            attention = st[0].auto_model.encoder.layer[-1].attention.self
            self.heads = int(st[0].auto_model.config.num_attention_heads)
            self.att_layer = len(st[0].auto_model.encoder.layer)
            self._hooks = [attention.query.register_forward_hook(self._keeper("q")),
                           attention.key.register_forward_hook(self._keeper("k"))]
        except AttributeError:       # another architecture: no attention panel
            self.heads = 0
        meta = {"labels": self.labels, "epochs": epochs, "steps": steps, "scale": head.scale,
                "heads": self.heads, "attention_layer": getattr(self, "att_layer", None),
                "frozen_layers": (len(st[0].auto_model.encoder.layer) if self.heads else 0)
                                 - len({describe(n)["layer"] for n, *_ in self.mats} - {None}),
                "dim": int(head.protos.shape[1]), "dims_drawn": sample(head.protos.shape[1], DIMS).tolist(),
                "matrices": [{"name": n, **describe(n), "shape": list(p.shape), "rows": ri.tolist(),
                              "cols": ci.tolist()} for n, p, ri, ci in self.mats],
                "w0": [pack(w) for w in self.w0],
                # the territories' centers as training starts (each label's prototype, from its centroid)
                "taxonomy_paths": self.paths,
                "protos0": base64.b64encode(F.normalize(head.protos.detach().float(), dim=1).cpu().numpy()
                                            .astype("<f2").tobytes()).decode()}
        write_atomic(self.folder / "meta.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))

    def _keeper(self, key: str):
        def hook(module, inputs, output):
            if self.armed:
                self._qk[key] = output.detach()
        return hook

    @torch.no_grad()
    def _subs(self, grad: bool = False) -> List[np.ndarray]:
        out = []
        for _, p, ri, ci in self.mats:
            t = p.grad if grad else p
            out.append(np.zeros((len(ri), len(ci)), dtype=np.float32) if t is None
                       else t.detach().index_select(0, ri).index_select(1, ci).float().cpu().numpy())
        return out

    @torch.no_grad()
    def _before(self, emb, logits, rows, y, loss, ce, anchor, grad_norm, lrs) -> None:
        dims = sample(emb.shape[1], DIMS).to(emb.device)
        E = emb.detach().float()
        P = F.normalize(self.head.protos.detach().float(), dim=1)
        L = logits.detach().float()
        probs = L.softmax(dim=1)
        pred = probs.argmax(dim=1)
        lengths = (self.feats["attention_mask"].sum(dim=1).cpu().numpy() if self.feats is not None
                   else np.array([len(self.ts.texts[i].split()) for i in rows]))
        # The text whose attention is drawn: one short enough to read.
        good = [i for i, n in enumerate(lengths) if 8 <= n <= 28]
        pick = min(good, key=lambda i: abs(int(lengths[i]) - 20)) if good else int(np.argmin(lengths))

        cur: dict = {"loss": round(float(loss), 5), "ce": round(float(ce), 5), "anchor": round(float(anchor), 5),
                     "grad_norm": round(float(grad_norm), 5), "lr": lrs, "featured": pick}
        p_np, y_np, pred_np = probs.cpu().numpy(), y.cpu().numpy(), pred.cpu().numpy()
        cur["texts"] = [{"text": " ".join(self.ts.texts[r].split())[:220], "label": int(y_np[j]),
                         "pred": int(pred_np[j]), "p": round(float(p_np[j, pred_np[j]]), 4),
                         "p_true": round(float(p_np[j, y_np[j]]), 4), "person": bool(self.ts.labeled[r])}
                        for j, r in enumerate(rows)]
        cur["E"], cur["P"] = pack(E[:, dims].cpu()), pack(P[:, dims].cpu())
        cur["L"] = pack(L.cpu())
        top = probs[pick].argsort(descending=True)[:TOP_LABELS].tolist()
        if int(y_np[pick]) not in top:
            top.append(int(y_np[pick]))
        cur["top"] = [{"label": t, "cos": round(float(L[pick, t]) / self.head.scale, 5),
                       "p": round(float(probs[pick, t]), 5)} for t in top]
        cur["e_full"] = pack(E[pick].cpu())
        cur["p_full"] = pack(P[torch.tensor(top, device=P.device)].cpu())

        if self.heads and "q" in self._qk and "k" in self._qk and self.feats is not None:
            n = int(min(lengths[pick], MAX_TOKENS))
            size = self._qk["q"].shape[-1] // self.heads
            q = self._qk["q"][pick, :n].float().view(n, self.heads, size).permute(1, 0, 2)
            k = self._qk["k"][pick, :int(lengths[pick])].float().view(int(lengths[pick]), self.heads, size).permute(1, 0, 2)
            att = (q @ k.transpose(1, 2) / size ** 0.5).softmax(dim=-1)[:, :, :n]   # before dropout
            cur["tokens"] = self.st.tokenizer.convert_ids_to_tokens(self.feats["input_ids"][pick, :n].tolist())
            cur["att"] = {"q": pack(q.cpu()), "k": pack(k[:, :n].cpu()), "a": pack(att.cpu())}
        self._qk = {}
        self.feats = None

        self._grads = self._subs(grad=True)
        self._norms = [0.0 if p.grad is None else float(p.grad.norm()) for _, p, _, _ in self.mats]
        self._old = self._subs()
        self._cur = cur

    @torch.no_grad()
    def _centers(self) -> dict:
        """How alike every two territories' centers are now: cosines, in 1/127ths."""
        P = F.normalize(self.head.protos.detach().float(), dim=1)
        G = (P @ P.T).clamp(-1, 1).mul(127).round().to(torch.int8).cpu().numpy()
        return {"n": int(G.shape[0]), "q": base64.b64encode(G.tobytes()).decode()}

    @torch.no_grad()
    def _after(self, epoch: int, step: int) -> None:
        if self._cur is None:
            return
        new = self._subs()
        now = time.perf_counter()
        frame = dict(self._cur)
        frame.update({
            "t": time.time(), "n": self._n, "epoch": epoch, "step": step,
            "global_step": (epoch - 1) * self.steps + step,
            "rate": round(self._steps_seen / max(1e-9, now - self._t0), 2),
            "series": self.series[-2000:], "val": self.val,
            "grad": [pack(g) for g in self._grads], "grad_norms": [round(v, 7) for v in self._norms],
            "delta": [pack(a - b) for a, b in zip(new, self._old)],
            "total": [pack(a - b) for a, b in zip(new, self.w0)],
            "total_rms": [float(np.sqrt(np.mean((a - b) ** 2))) for a, b in zip(new, self.w0)],
            "w_rms": [float(np.sqrt(np.mean(a ** 2))) for a in new],
            "protos_cos": self._centers(),
        })
        data = json.dumps(frame, ensure_ascii=False).encode("utf-8")
        write_atomic(self.folder / "frame.json", data)
        if now - self._last_archive >= self.archive_every:
            self._last_archive = now
            (self.folder / "frames" / f"{frame['global_step']:07d}.json.gz").write_bytes(gzip.compress(data, 3))
        self._n += 1
        self._cur = None
        self._last = time.perf_counter()
