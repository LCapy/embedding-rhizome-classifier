#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common.py - shared paths, data loading, splitting, and centroid computation
for the held-out evaluation (see README.md in this folder).

Data (not in the repo, see README.md for where it comes from): 33,787
records, 206 raw categories, embeddings.npz row i = LaBSE(sentences.json[i])
= LaBSE(sentences_rhizome.json[i]) (same order).

Everything here operates on a FIXED, SEEDED, STRATIFIED train/test split
saved to disk once (split.npz) so every later stage - geometric baseline,
plain trained classifier, integrated classifier - evaluates on the exact
same held-out test set. Without that, "training beats geometry" would be
an unfalsifiable claim.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


# All paths are overridable with environment variables.
DATA_DIR = _env_path("EVAL_DATA_DIR", REPO / "data")                    # sentences*.json, embeddings.npz
COREDRILL_PATH = _env_path("EVAL_COREDRILL", REPO / "coredrill" / "coredrill_hierarchical.json")
TAXONOMY_PATH = _env_path("EVAL_TAXONOMY", DATA_DIR / "TAXONOMY_fixed.txt")
TEST_TEXTS_DIR = _env_path("EVAL_TEST_TEXTS", DATA_DIR / "test_texts")
ENGINE_DIR = REPO / "scripts"   # rhizome_engine.py, same file the Space runs
OUT_DIR = _env_path("EVAL_OUT_DIR", Path(__file__).parent / "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COUNT = 5  # classes with fewer records than this are dropped (9 records total, see README)
TEST_FRAC = 0.2
SEED = 42


def load_sentences() -> List[dict]:
    return json.loads((DATA_DIR / "sentences.json").read_text(encoding="utf-8"))


def load_rhizome_sentences() -> List[dict]:
    return json.loads((DATA_DIR / "sentences_rhizome.json").read_text(encoding="utf-8"))


def load_embeddings() -> np.ndarray:
    with np.load(DATA_DIR / "embeddings.npz") as f:
        return f["embeddings"].astype(np.float32)


def load_coredrill() -> dict:
    return json.loads(COREDRILL_PATH.read_text(encoding="utf-8"))


def build_label_set(sentences: List[dict], min_count: int = MIN_COUNT) -> List[str]:
    """Classes with >= min_count records, sorted for a stable label index."""
    from collections import Counter
    counts = Counter(r.get("category", "") for r in sentences)
    labels = sorted(c for c, n in counts.items() if c and n >= min_count)
    return labels


def make_split(sentences: List[dict], labels: List[str], test_frac: float = TEST_FRAC,
                seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified per-class split. Returns (y, train_idx, test_idx).
    y[i] = class index for record i, or -1 if its category isn't in `labels`
    (dropped: either below min_count, or empty/unknown category).
    """
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y = np.array([label_to_idx.get(r.get("category", ""), -1) for r in sentences], dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_idx_parts, test_idx_parts = [], []
    for cls_idx in range(len(labels)):
        idx = np.where(y == cls_idx)[0]
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_frac))
        test_idx_parts.append(idx[:n_test])
        train_idx_parts.append(idx[n_test:])

    train_idx = np.concatenate(train_idx_parts)
    test_idx = np.concatenate(test_idx_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return y, train_idx, test_idx


def compute_centroids(embeddings: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                       n_classes: int) -> np.ndarray:
    """L2-normalized mean embedding per class, from TRAIN records only - the
    original study's rule (centroids from direct records only, never mixed
    with other classes' data)."""
    dim = embeddings.shape[1]
    centroids = np.zeros((n_classes, dim), dtype=np.float32)
    for cls_idx in range(n_classes):
        idx = train_idx[y[train_idx] == cls_idx]
        mu = embeddings[idx].mean(axis=0)
        mu /= (np.linalg.norm(mu) + 1e-12)
        centroids[cls_idx] = mu
    return centroids


def compute_sigmas(embeddings: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                    centroids: np.ndarray) -> np.ndarray:
    """Per-class sigma = 1 - mean cosine of train records to their centroid -
    same definition as the original build_coredrill()."""
    n_classes = centroids.shape[0]
    sigmas = np.zeros(n_classes, dtype=np.float32)
    for cls_idx in range(n_classes):
        idx = train_idx[y[train_idx] == cls_idx]
        if len(idx) == 0:
            sigmas[cls_idx] = 0.4  # fallback, matches original's default
            continue
        cos = embeddings[idx] @ centroids[cls_idx]
        sigmas[cls_idx] = 1.0 - float(cos.mean())
    return sigmas


def save_split(labels: List[str], y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
               centroids: np.ndarray, sigmas: np.ndarray) -> None:
    np.savez(
        OUT_DIR / "split.npz",
        y=y, train_idx=train_idx, test_idx=test_idx,
        centroids=centroids, sigmas=sigmas,
        labels=np.array(labels, dtype=object),
    )


def load_split() -> dict:
    with np.load(OUT_DIR / "split.npz", allow_pickle=True) as f:
        return {k: f[k] for k in f.files}
