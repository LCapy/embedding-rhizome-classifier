"""LaBSE in NumPy, for classifying one text at a time.

PyTorch commits more than 2 GB of memory to load LaBSE, which a training loop
on the same machine may not be able to spare. This runs the same network
(BERT, CLS pooling, Dense with tanh, unit length) from the files on disk. Its
1.5 GB of word embeddings are only mapped: it reads just the rows a text uses.
The layers (0.35 GB) are loaded, because Windows drops mapped pages when memory
runs short and reading them back one by one takes half a minute. A trained
loop's weights.pt is laid over them.
"""
from __future__ import annotations

import io
import json
import pickle
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BASE_MODEL = "sentence-transformers/LaBSE"


def base_dir(model: str = BASE_MODEL) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, local_files_only=True))


def map_safetensors(path: Path) -> Dict[str, np.ndarray]:
    """Every float32 tensor of a safetensors file, mapped read-only."""
    with path.open("rb") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
    out = {}
    for name, info in header.items():
        if name != "__metadata__" and info["dtype"] == "F32":
            out[name] = np.memmap(path, dtype="<f4", mode="r", offset=8 + size + info["data_offsets"][0],
                                  shape=tuple(info["shape"]))
    return out


class _Unpickler(pickle.Unpickler):
    """Reads the tensors of a torch.save() file and nothing else: any other
    class in the file is refused."""

    def __init__(self, data: bytes, storage):
        super().__init__(io.BytesIO(data))
        self.storage = storage

    def persistent_load(self, pid):
        kind, dtype, key, _, count = pid
        if kind != "storage" or dtype != "FloatStorage":
            raise pickle.UnpicklingError(f"not a float32 storage: {kind} {dtype}")
        return self.storage(key, count)

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return dict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return self.tensor
        if (module, name) == ("torch", "FloatStorage"):
            return "FloatStorage"
        raise pickle.UnpicklingError(f"{module}.{name} is not allowed in a weights file")

    @staticmethod
    def tensor(data, offset, size, stride, *_):
        count = int(np.prod(size)) if size else 1
        expected, step = [], 1
        for n in reversed(size):
            expected.insert(0, step)
            step *= n
        if any(n > 1 and s != e for n, s, e in zip(size, stride, expected)):
            raise pickle.UnpicklingError("only contiguous tensors can be read")
        return data[offset:offset + count].reshape(size)


def read_weights(path: Path) -> dict:
    """What model.save_run() saved, without PyTorch: {"encoder_delta", "protos", "scale"}."""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        root = name[:-len("data.pkl")]

        def storage(key, count):
            data = np.frombuffer(z.read(f"{root}data/{key}"), dtype="<f4")
            if len(data) != count:
                raise pickle.UnpicklingError(f"storage {key} has {len(data)} values, not {count}")
            return data

        return _Unpickler(z.read(name), storage).load()


def _norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


class Encoder:
    """encode() has the SentenceTransformer's signature, which is all the
    engine uses."""

    def __init__(self, base: Path | None = None, max_tokens: int = 256):
        from tokenizers import Tokenizer
        base = base or base_dir()
        self.config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        self.stock = {f"bert.{k}": v if "word_embeddings" in k else np.array(v)
                      for k, v in map_safetensors(base / "model.safetensors").items()}
        self.stock.update({f"dense.{k}": v for k, v in map_safetensors(base / "2_Dense" / "model.safetensors").items()})
        self.w: Dict[str, np.ndarray] = dict(self.stock)
        self.tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.set_max_tokens(max_tokens)

    def set_max_tokens(self, n: int) -> None:
        self.max_tokens = int(n)
        self.tokenizer.enable_truncation(self.max_tokens)

    def set_weights(self, delta: Dict[str, np.ndarray]) -> None:
        """Lays a loop's trained parameters over stock LaBSE, replacing any
        loop's before it."""
        w = dict(self.stock)
        for name, value in delta.items():
            for prefix, to in (("0.auto_model.", "bert."), ("0.model.", "bert."), ("2.", "dense.")):
                if name.startswith(prefix):
                    key = to + name[len(prefix):]
                    break
            else:
                raise ValueError(f"unknown trained parameter {name}")
            if key not in w or w[key].shape != value.shape:
                raise ValueError(f"trained parameter {name} does not fit LaBSE")
            w[key] = value
        self.w = w

    def count_tokens(self, text: str) -> Tuple[int, bool]:
        """(tokens read, whether the text was longer than that)."""
        e = self.tokenizer.encode(text.strip())
        return len(e.ids), bool(e.overflowing)

    def encode(self, texts: List[str], **_) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts]).astype(np.float32)

    def _linear(self, x: np.ndarray, name: str) -> np.ndarray:
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def _embed(self, text: str) -> np.ndarray:
        from scipy.special import erf
        w, eps, heads = self.w, float(self.config["layer_norm_eps"]), self.config["num_attention_heads"]
        ids = np.array(self.tokenizer.encode(text.strip()).ids)
        n = len(ids)
        x = (np.asarray(w["bert.embeddings.word_embeddings.weight"][ids])
             + w["bert.embeddings.position_embeddings.weight"][:n] + w["bert.embeddings.token_type_embeddings.weight"][0])
        x = _norm(x, w["bert.embeddings.LayerNorm.weight"], w["bert.embeddings.LayerNorm.bias"], eps)
        size = x.shape[1] // heads
        for i in range(self.config["num_hidden_layers"]):
            p = f"bert.encoder.layer.{i}."
            q, k, v = (self._linear(x, p + "attention.self." + s).reshape(n, heads, size).transpose(1, 0, 2)
                       for s in ("query", "key", "value"))
            a = q @ k.transpose(0, 2, 1) / np.float32(np.sqrt(size))
            a = np.exp(a - a.max(axis=-1, keepdims=True))
            a /= a.sum(axis=-1, keepdims=True)
            y = self._linear((a @ v).transpose(1, 0, 2).reshape(n, -1), p + "attention.output.dense")
            x = _norm(x + y, w[p + "attention.output.LayerNorm.weight"], w[p + "attention.output.LayerNorm.bias"], eps)
            y = self._linear(x, p + "intermediate.dense")
            y = y * 0.5 * (1.0 + erf(y * np.float32(1 / np.sqrt(2))))
            y = self._linear(y, p + "output.dense")
            x = _norm(x + y, w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"], eps)
        out = np.tanh(self._linear(x[0], "dense.linear"))
        return out / (np.linalg.norm(out) + 1e-12)
