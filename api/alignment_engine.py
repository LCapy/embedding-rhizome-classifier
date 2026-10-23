#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alignment_engine.py - Cross-lingual segment alignment via multilingual
sentence embeddings. Same methodology as the Phase I research (cosine and
DIEM similarity, greedy sequential alignment with segment merging) -
ported from the standalone aligned/combined_alignment.py project to reuse
the LaBSE model this Space already loads for classification, instead of
loading a second copy.

Segmentation here is intentionally lighter than the offline pipeline:
NLTK punkt for the Latin-alphabet languages it supports, punctuation-regex
fallback otherwise (including ja/zh/ko) - no spaCy/kss dependency, to keep
this Space's image lean. Treat CJK segmentation as approximate; the
offline pipeline (aligned/combined_alignment.py) uses real per-language
parsers for higher quality.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, List, Tuple

import numpy as np

try:
    import nltk

    for _resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(_resource)
        except LookupError:
            try:
                nltk.download(_resource.split("/")[-1], quiet=True)
            except Exception:
                pass
except Exception:
    nltk = None  # segmentation falls back to regex everywhere

THRESHOLDS = {
    "similarity_threshold": 0.30,
    "max_merge_segments": 4,
    "diem_alpha": 3.0,
}

_NLTK_LANG = {
    "en": "english", "de": "german", "fr": "french",
    "es": "spanish", "it": "italian", "pt": "portuguese",
}


def segment_text(text: str, lang: str) -> List[str]:
    """Split text into sentences."""
    text = text.strip()
    if not text:
        return []

    if lang in _NLTK_LANG and nltk is not None:
        try:
            sentences = nltk.sent_tokenize(text, language=_NLTK_LANG[lang])
        except Exception:
            sentences = re.split(r"(?<=[.!?;])\s+", text)
    elif lang in ("ja", "zh"):
        sentences = re.split(r"(?<=[。!?？！])\s*", text)
    elif lang == "ko":
        sentences = re.split(r"(?<=[.!?。？！])\s+", text)
    else:
        sentences = re.split(r"(?<=[.!?;])\s+", text)

    sentences = [s.strip() for s in sentences if s.strip()]

    # Further split on semicolons/newlines, matching the offline pipeline.
    out: List[str] = []
    for s in sentences:
        if any(p in s for p in (";", "；", "\n")):
            out.extend(p.strip() for p in re.split(r"[;；\n]", s) if p.strip())
        else:
            out.append(s)
    return out


def make_scorer(embed_fn: Callable[[str], np.ndarray]):
    """Build cosine/DIEM similarity functions bound to a caller-supplied
    embedding function, so this module never loads or owns a model itself."""

    @lru_cache(maxsize=4096)
    def _embed(text: str) -> np.ndarray:
        return np.asarray(embed_fn(text), dtype=np.float32)

    def cosine_sim(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        va, vb = _embed(a), _embed(b)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def diem_sim(a: str, b: str, alpha: float = THRESHOLDS["diem_alpha"]) -> float:
        if not a or not b:
            return 0.0
        va, vb = _embed(a), _embed(b)
        dist = np.linalg.norm(va - vb) / np.sqrt(2)
        return float(np.exp(-alpha * dist))

    return cosine_sim, diem_sim


def heuristic_align(
    source_segments: List[str],
    target_segments: List[str],
    sim_func: Callable[[str, str], float],
    sim_threshold: float = THRESHOLDS["similarity_threshold"],
    max_merge: int = THRESHOLDS["max_merge_segments"],
) -> List[Tuple[str, str, float, int, int]]:
    """Greedy sequential alignment: walk both segment lists together,
    merging up to `max_merge` consecutive segments on either side when the
    direct pairing scores below threshold. Same algorithm as the offline
    pipeline's heuristic_align."""
    aligned: List[Tuple[str, str, float, int, int]] = []
    i = j = 0
    while i < len(source_segments) and j < len(target_segments):
        seg_a = source_segments[i].strip()
        seg_b = target_segments[j].strip()
        sim = sim_func(seg_a, seg_b) if seg_a and seg_b else 0.0
        a_count = b_count = 1

        if sim < sim_threshold:
            best_sim, best_a, best_b = sim, seg_a, seg_b
            best_a_count, best_b_count = 1, 1

            for extra in range(1, max_merge):
                if i + extra < len(source_segments):
                    cand = " ".join(s.strip() for s in source_segments[i:i + extra + 1])
                    cand_sim = sim_func(cand, seg_b) if cand and seg_b else 0.0
                    if cand_sim > best_sim:
                        best_sim, best_a, best_a_count = cand_sim, cand, extra + 1

            for extra in range(1, max_merge):
                if j + extra < len(target_segments):
                    cand = " ".join(s.strip() for s in target_segments[j:j + extra + 1])
                    cand_sim = sim_func(seg_a, cand) if seg_a and cand else 0.0
                    if cand_sim > best_sim:
                        best_sim, best_b, best_b_count = cand_sim, cand, extra + 1

            if best_a_count > 1 or best_b_count > 1:
                seg_a, seg_b, sim = best_a, best_b, best_sim
                a_count, b_count = best_a_count, best_b_count
                if best_a_count > 1 and best_a_count >= best_b_count:
                    i += best_a_count
                    j += 1
                elif best_b_count > 1:
                    i += 1
                    j += best_b_count
                else:
                    i += 1
                    j += 1
            else:
                i += 1
                j += 1
        else:
            i += 1
            j += 1

        aligned.append((seg_a, seg_b, sim, a_count, b_count))

    while i < len(source_segments):
        aligned.append((source_segments[i].strip(), "", 0.0, 1, 0))
        i += 1
    while j < len(target_segments):
        aligned.append(("", target_segments[j].strip(), 0.0, 0, 1))
        j += 1

    return aligned
