"""How a live frame stores matrices. No PyTorch here: the viewer uses it too."""
from __future__ import annotations

import base64
import re

import numpy as np

GRID = 48            # rows and columns sampled from a trained matrix

_PARTS = {"attention.self.query": "query", "attention.self.key": "key", "attention.self.value": "value",
          "attention.output.dense": "attention out", "intermediate.dense": "feed-forward in",
          "output.dense": "feed-forward out"}


def pack(x) -> dict:
    """An array as base64 int16 with one scale: value = int / 32767 * scale."""
    a = np.nan_to_num(np.asarray(x, dtype=np.float32))
    scale = float(np.abs(a).max()) if a.size else 0.0
    scale = scale or 1.0
    q = np.round(a / scale * 32767).astype("<i2")
    return {"shape": list(a.shape), "scale": scale, "data": base64.b64encode(q.tobytes()).decode("ascii")}


def describe(name: str) -> dict:
    """"0.model.encoder.layer.11.attention.self.query.weight" -> layer 12, query."""
    m = re.search(r"layer\.(\d+)\.(.+)\.weight$", name)
    if m:
        return {"layer": int(m.group(1)) + 1, "part": _PARTS.get(m.group(2), m.group(2))}
    return {"layer": None, "part": "label vectors" if name == "protos" else "Dense"}


def spread(n: int, k: int = GRID) -> np.ndarray:
    """k indices spread evenly over n (every n/k-th one)."""
    return np.arange(n) if n <= k else np.arange(k) * n // k
