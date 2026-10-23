#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_ui.py - Gradio demo for cross-lingual segment alignment, mounted
into this Space's FastAPI app at /align. Paste source and target text,
each is split into sentences, then aligned via LaBSE + cosine/DIEM
similarity (aligned/combined_alignment.py's methodology).

Public by design (no API key check) - this is the webpage counterpart to
the /predict JSON API, meant to be shared/opened directly in a browser.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

try:
    from api.alignment_engine import THRESHOLDS, heuristic_align, make_scorer, segment_text
except Exception:
    from alignment_engine import THRESHOLDS, heuristic_align, make_scorer, segment_text

LANG_CHOICES = [
    ("English", "en"), ("German", "de"), ("French", "fr"), ("Spanish", "es"),
    ("Italian", "it"), ("Portuguese", "pt"), ("Japanese", "ja"),
    ("Chinese", "zh"), ("Korean", "ko"), ("Other / unlisted", "other"),
]


def build_align_demo(get_model):
    """`get_model` is this Space's existing lazy model loader (_get_model
    in api/app.py), passed in so this module never loads its own copy of
    LaBSE."""

    def run(source_text, target_text, source_lang, target_lang, metric):
        if not source_text.strip() or not target_text.strip():
            return pd.DataFrame(), "Enter text on both sides, then click Align."

        model = get_model()

        def embed_fn(text: str):
            return model.encode(text, show_progress_bar=False, convert_to_numpy=True)

        cosine_sim, diem_sim = make_scorer(embed_fn)
        sim_func = diem_sim if metric == "DIEM" else cosine_sim

        src_segments = segment_text(source_text, source_lang)
        tgt_segments = segment_text(target_text, target_lang)

        if not src_segments or not tgt_segments:
            return pd.DataFrame(), "Could not segment one side into sentences - check the text/language."

        pairs = heuristic_align(src_segments, tgt_segments, sim_func)

        rows = [
            {
                "Source": a or "-",
                "Target": b or "-",
                "Similarity": round(sim, 4),
                "Merged (src/tgt)": f"{ac}/{bc}",
                "Status": "OK" if sim >= THRESHOLDS["similarity_threshold"] else "below threshold",
            }
            for a, b, sim, ac, bc in pairs
        ]
        df = pd.DataFrame(rows)
        summary = (
            f"{len(src_segments)} source segments, {len(tgt_segments)} target segments, "
            f"{len(pairs)} aligned pairs, metric = {metric}."
        )
        return df, summary

    with gr.Blocks(title="Text Alignment via Embeddings") as demo:
        gr.Markdown(
            "# Text Alignment via Embeddings\n"
            "Cross-lingual sentence alignment using "
            "[LaBSE](https://huggingface.co/sentence-transformers/LaBSE) embeddings - "
            "cosine similarity or the DIEM metric (Gaussian-decayed Euclidean distance), "
            "with greedy sequential alignment and segment merging when a direct pairing "
            "scores below threshold. No trained classifier, no parallel-corpus training - "
            "pure embedding-space geometry. Same methodology as the "
            "[Phase I geometry study](https://github.com/LCapy/embedding-rhizome-classifier/"
            "blob/main/docs/study_v5.md)."
        )
        with gr.Row():
            with gr.Column():
                source_text = gr.Textbox(
                    label="Source text", lines=10,
                    placeholder="Paste source-language text...",
                )
                source_lang = gr.Dropdown(LANG_CHOICES, value="en", label="Source language")
            with gr.Column():
                target_text = gr.Textbox(
                    label="Target text", lines=10,
                    placeholder="Paste target-language text (translation)...",
                )
                target_lang = gr.Dropdown(LANG_CHOICES, value="es", label="Target language")
        metric = gr.Radio(["cosine", "DIEM"], value="cosine", label="Similarity metric")
        run_btn = gr.Button("Align", variant="primary")
        summary_out = gr.Markdown()
        table_out = gr.Dataframe(wrap=True)

        run_btn.click(
            run,
            inputs=[source_text, target_text, source_lang, target_lang, metric],
            outputs=[table_out, summary_out],
        )

    return demo
