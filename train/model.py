"""Encoder, prototype head, feature helpers and run files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_MODEL = "sentence-transformers/LaBSE"


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def free_gpu() -> None:
    """Releases GPU memory held by dropped models. A SentenceTransformer refers
    back to itself (through its model card), so `del` alone leaves it, and
    its 1.8 GB, alive until the cycle collector runs; on a 6 GB card a second
    copy pushes Windows into spilling GPU memory to system RAM, which is many
    times slower. Collect the cycles, then hand the freed blocks back."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_encoder(base: str = BASE_MODEL, device="cpu", max_seq_length: int | None = None):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(base, device=str(device))
    if max_seq_length:
        st.max_seq_length = max_seq_length
    return st


def pooling_index(st) -> int:
    for i, module in enumerate(st):
        if type(module).__name__ == "Pooling":
            return i
    raise ValueError("encoder has no Pooling module")


def set_trainable(st, unfreeze_layers: int) -> List[str]:
    """Freeze the encoder, then unfreeze its top `unfreeze_layers` transformer
    layers and every module after pooling (LaBSE: the 768x768 Dense layer).
    The word embeddings (385M of LaBSE's 471M parameters) always stay frozen.
    Returns the names of the trainable parameters, which is what a run saves."""
    for p in st.parameters():
        p.requires_grad = False
    if unfreeze_layers > 0:
        layers = st[0].auto_model.encoder.layer
        if unfreeze_layers > len(layers):
            raise ValueError(f"the encoder has {len(layers)} layers; cannot unfreeze {unfreeze_layers}")
        for layer in layers[len(layers) - unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
    for module in list(st)[pooling_index(st) + 1:]:
        for p in module.parameters():
            p.requires_grad = True
    return [n for n, p in st.named_parameters() if p.requires_grad]


class PrototypeHead(nn.Module):
    """One prototype per label. Logits are scaled cosines, so a prototype is a
    direction in embedding space, the same kind of object as a coredrill
    centroid; it starts at the label's centroid."""

    def __init__(self, init: torch.Tensor, scale: float = 20.0):
        super().__init__()
        self.protos = nn.Parameter(F.normalize(init.float(), dim=1).clone())
        self.scale = float(scale)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.scale * emb @ F.normalize(self.protos, dim=1).T


def features_to(feats: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feats.items()}


@torch.no_grad()
def pooled_features(st, texts: Sequence[str], device, batch_size: int = 128) -> torch.Tensor:
    """Transformer + pooling output (the frozen part when only the Dense layer
    trains), float32 on the CPU, one row per text in the input order."""
    st.eval()
    prefix = list(st)[:pooling_index(st) + 1]
    order = np.argsort([len(t) for t in texts], kind="stable")   # similar lengths pad less
    out = None
    for s in range(0, len(texts), batch_size):
        idx = order[s:s + batch_size]
        feats = features_to(st.preprocess([texts[i] for i in idx]), device)
        for module in prefix:
            feats = module(feats)
        pooled = feats["sentence_embedding"].float().cpu()
        if out is None:
            out = torch.empty(len(texts), pooled.shape[1])
        out[torch.as_tensor(idx)] = pooled
    return out


def apply_suffix(st, pooled: torch.Tensor) -> torch.Tensor:
    """Modules after pooling (Dense, Normalize) on pooled features; unit rows."""
    feats = {"sentence_embedding": pooled}
    for module in list(st)[pooling_index(st) + 1:]:
        feats = module(feats)
    return F.normalize(feats["sentence_embedding"].float(), dim=1)


@torch.no_grad()
def encode(st, texts: Sequence[str], device, batch_size: int = 64) -> np.ndarray:
    """Unit embeddings with the encoder's current weights."""
    if len(texts) == 0:
        return np.zeros((0, st.get_sentence_embedding_dimension()), dtype=np.float32)
    st.eval()
    emb = st.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False, device=str(device))
    return emb.astype(np.float32)


def head_probs(emb: np.ndarray, protos: np.ndarray, scale: float, temperature: float = 1.0) -> np.ndarray:
    """Softmax over labels of the scaled cosines, calibrated by temperature."""
    p = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    logits = scale * (emb @ p.T) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


# ── run files ────────────────────────────────────────────────────────────────

def save_run(run_dir: Path, st, trainable: List[str], head: PrototypeHead, meta: dict) -> None:
    """weights.pt holds only the trained parameters plus the prototypes; the
    rest comes from the base model when the run is loaded."""
    run_dir.mkdir(parents=True, exist_ok=True)
    keep = set(trainable)
    delta = {k: v.detach().cpu() for k, v in st.state_dict().items() if k in keep}
    torch.save({"encoder_delta": delta, "protos": head.protos.detach().cpu(), "scale": head.scale},
               run_dir / "weights.pt")
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def apply_delta(st, delta: dict) -> None:
    missing, unexpected = st.load_state_dict(delta, strict=False)
    if unexpected:
        raise ValueError(f"run weights do not fit this encoder: {unexpected[:5]}")


def load_run(run_dir: Path, device) -> Tuple[object, PrototypeHead, dict]:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    w = torch.load(run_dir / "weights.pt", map_location="cpu", weights_only=True)
    st = load_encoder(meta["base_model"], device, meta.get("fit", {}).get("max_seq_length"))
    apply_delta(st, w["encoder_delta"])
    head = PrototypeHead(w["protos"], w["scale"]).to(device)
    return st, head, meta
