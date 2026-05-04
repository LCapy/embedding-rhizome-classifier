#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_folder.py — Batch prediction with optional A/B comparison.

Single mode:
  python predict_folder.py \
    --input-dir  test_texts \
    --output-dir reports \
    --coredrill  coredrill/coredrill_v4.json \
    --script     full_build_rhizome_final.py

A/B comparison mode:
  python predict_folder.py \
    --input-dir   test_texts \
    --output-dir  reports_compare \
    --script-a    full_build_rhizome_final.py \
    --coredrill-a coredrill/coredrill_v4.json \
    --script-b    full_build_rhizome_final_n.py \
    --coredrill-b coredrill/coredrill_v4.json \
    --label-a "old formula" \
    --label-b "pure cosine"
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predict_folder")


# ── helpers ──────────────────────────────────────────────────────────────────

def run_prediction(script: str, coredrill: str, text: str) -> str:
    result = subprocess.run(
        [sys.executable, script, "predict",
         "--coredrill", coredrill, "--analyze", "--text", text],
        capture_output=True, text=True, encoding="utf-8",
    )
    lines = result.stdout.splitlines()
    clean = [l for l in lines
             if not l.startswith("The following layers")
             and not l.startswith("Loading weights")]
    return "\n".join(clean)


def parse_output(raw: str) -> dict:
    p = {
        "top_node":     "?",
        "top_cos":      None,
        "primary_flow": "?",
        "flow_score":   "?",
        "cladistic":    "",
        "flow_bars":    [],
        "profile":      [],   # raw lines from profile block
        "table":        [],   # raw rows from ranked table
        "raw":          raw,
    }

    lines = raw.splitlines()
    in_profile = in_table = in_flow = False
    first_node_done = False

    for line in lines:
        s = line.strip()

        if "COREDRILL  —  RHIZOME PROFILE" in line:
            in_profile = True; in_table = in_flow = False; continue
        if "FULL RANKED TABLE" in line:
            in_table = True; in_profile = in_flow = False; continue
        if "FLOW TOPOLOGY" in line:
            in_flow = True; in_profile = False; continue
        if "CLADISTIC PATH:" in line:
            in_profile = False; continue
        if "PRIMARY FLOW:" in line:
            p["primary_flow"] = s.replace("PRIMARY FLOW:", "").strip().split()[0]
            continue

        if any(s.startswith(k) for k in ("BWO", "TERRIT", "DETERR", "RETERR", "LINE OF")):
            p["flow_bars"].append(s)
            continue

        if in_profile and s and not any(s.startswith(k) for k in ("=", "-", "[->", "NODE")):
            p["profile"].append(s)
            if not first_node_done and "%" in s:
                first_node_done = True
                cols = s.split()
                if len(cols) >= 7:
                    p["top_node"] = " ".join(cols[:-6])
                    try:
                        nums = [c for c in cols[-6:] if c.replace(".", "").replace("-", "").isdigit()]
                        if nums:
                            p["top_cos"] = float(nums[0])
                    except Exception:
                        pass

        if in_table and s and not any(s.startswith(k) for k in ("NODE", "---", "[->", "===")):
            p["table"].append(s)

    if "CLADISTIC PATH:" in raw:
        cs = raw.find("CLADISTIC PATH:") + len("CLADISTIC PATH:")
        ce = raw.find("FLOW TOPOLOGY:", cs)
        if ce > cs:
            p["cladistic"] = raw[cs:ce].strip().replace("\n", " ").replace("  ", " ")

    # flow score from bars
    for bar in p["flow_bars"]:
        if p["primary_flow"] in bar:
            for tok in reversed(bar.split()):
                try:
                    p["flow_score"] = tok
                    float(tok); break
                except Exception:
                    pass
            break

    return p


def summarize_row(row: str) -> str:
    """Extract node name + cos + prod from a ranked table row."""
    if not row:
        return ""
    cols = row.split()
    if len(cols) >= 10:
        name = " ".join(cols[:-9])
        cos  = cols[-8]
        prod = cols[-3]
        mass = cols[-2]
        return f"{name} cos={cos} prod={prod} {mass}"
    return row[:70]


# ── single-mode report ───────────────────────────────────────────────────────

def write_single_report(txt_path: Path, text: str, parsed: dict,
                        out_path: Path, elapsed: float) -> None:
    lines = [
        f"# Coredrill Report: `{txt_path.name}`",
        f"",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Time:** {elapsed:.1f}s",
        f"",
        f"## Input",
        f"",
        f"```",
        text.strip(),
        f"```",
        f"",
        f"## Rhizome Profile",
        f"",
    ]
    for l in parsed["profile"][:8]:
        lines.append(f"    {l}")
    lines += [
        f"",
        f"**Cladistic:** {parsed['cladistic']}",
        f"",
        f"## Flow Topology",
        f"",
        f"**Primary flow:** `{parsed['primary_flow']}` {parsed['flow_score']}",
        f"",
    ]
    for b in parsed["flow_bars"]:
        lines.append(f"    {b}")
    lines += [
        f"",
        f"## Ranked Table (top 10)",
        f"",
        f"| Node | sigma | cos | z | lap | gau | man | prod | mass | lvl |",
        f"|------|-------|-----|---|-----|-----|-----|------|------|-----|",
    ]
    for row in parsed["table"][:10]:
        cols = row.split()
        if len(cols) >= 10:
            name = " ".join(cols[:-9])
            rest = " | ".join(cols[-9:])
            lines.append(f"| {name} | {rest} |")
    lines += [
        f"",
        f"<details><summary>Raw output</summary>",
        f"",
        f"```",
        parsed["raw"],
        f"```",
        f"</details>",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── A/B comparison report ────────────────────────────────────────────────────

def write_compare_report(txt_path: Path, text: str,
                         pa: dict, pb: dict,
                         label_a: str, label_b: str,
                         out_path: Path,
                         elapsed_a: float, elapsed_b: float) -> None:

    def changed(a, b): return "🔴 CHANGED" if a != b else "✅ same"

    lines = [
        f"# A/B Report: `{txt_path.name}`",
        f"",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Input",
        f"",
        f"```",
        text.strip(),
        f"```",
        f"",
        f"## Summary",
        f"",
        f"| | **{label_a}** ({elapsed_a:.1f}s) | **{label_b}** ({elapsed_b:.1f}s) | Δ |",
        f"|---|---|---|---|",
        f"| Top node | `{pa['top_node']}` | `{pb['top_node']}` | {changed(pa['top_node'], pb['top_node'])} |",
        f"| Primary flow | `{pa['primary_flow']}` {pa['flow_score']} | `{pb['primary_flow']}` {pb['flow_score']} | {changed(pa['primary_flow'], pb['primary_flow'])} |",
        f"",
        f"## Rhizome Profile",
        f"",
        f"**{label_a}:**",
        f"```",
    ]
    for l in pa["profile"][:8]:
        lines.append(l)
    lines += [f"```", f"", f"**{label_b}:**", f"```"]
    for l in pb["profile"][:8]:
        lines.append(l)
    lines += [
        f"```",
        f"",
        f"## Flow Bars",
        f"",
        f"**{label_a}:**",
        f"```",
    ]
    for b in pa["flow_bars"]:
        lines.append(b)
    lines += [f"```", f"", f"**{label_b}:**", f"```"]
    for b in pb["flow_bars"]:
        lines.append(b)
    lines += [
        f"```",
        f"",
        f"## Ranked Table (top 10)",
        f"",
        f"| Rank | **{label_a}** | **{label_b}** | Δ |",
        f"|------|-------------|-------------|---|",
    ]
    rows_a = pa["table"][:10]
    rows_b = pb["table"][:10]
    for i in range(max(len(rows_a), len(rows_b))):
        sa = summarize_row(rows_a[i]) if i < len(rows_a) else ""
        sb = summarize_row(rows_b[i]) if i < len(rows_b) else ""
        node_a = rows_a[i].split()[0] if i < len(rows_a) and rows_a[i].split() else ""
        node_b = rows_b[i].split()[0] if i < len(rows_b) and rows_b[i].split() else ""
        diff = "🔴" if node_a != node_b else ""
        lines.append(f"| {i+1} | {sa} | {sb} | {diff} |")
    lines += [
        f"",
        f"<details><summary>Raw A: {label_a}</summary>",
        f"", f"```", pa["raw"], f"```", f"</details>",
        f"",
        f"<details><summary>Raw B: {label_b}</summary>",
        f"", f"```", pb["raw"], f"```", f"</details>",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Batch coredrill prediction (single or A/B)")
    ap.add_argument("--input-dir",   required=True)
    ap.add_argument("--output-dir",  required=True)
    ap.add_argument("--glob",        default="*.txt")
    # Single mode
    ap.add_argument("--script",      default=None)
    ap.add_argument("--coredrill",   default=None)
    # A/B mode
    ap.add_argument("--script-a",    default=None)
    ap.add_argument("--coredrill-a", default=None)
    ap.add_argument("--script-b",    default=None)
    ap.add_argument("--coredrill-b", default=None)
    ap.add_argument("--label-a",     default="A")
    ap.add_argument("--label-b",     default="B")
    args = ap.parse_args()

    ab_mode = bool(args.script_a and args.script_b)
    if not ab_mode and not (args.script and args.coredrill):
        ap.error("Provide either --script + --coredrill, or --script-a/b + --coredrill-a/b")

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob(args.glob))
    if not txt_files:
        log.error(f"No {args.glob} files in {input_dir}")
        sys.exit(1)

    mode_str = f"A/B [{args.label_a} vs {args.label_b}]" if ab_mode else "single"
    log.info(f"{len(txt_files)} files  |  mode: {mode_str}")

    summary_rows = []

    for i, txt_path in enumerate(txt_files, 1):
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        log.info(f"[{i}/{len(txt_files)}] {txt_path.name}")

        if ab_mode:
            t0 = time.time()
            raw_a = run_prediction(args.script_a, args.coredrill_a, text)
            elapsed_a = time.time() - t0

            t0 = time.time()
            raw_b = run_prediction(args.script_b, args.coredrill_b, text)
            elapsed_b = time.time() - t0

            pa = parse_output(raw_a)
            pb = parse_output(raw_b)

            out_path = output_dir / (txt_path.stem + "_compare.md")
            write_compare_report(txt_path, text, pa, pb,
                                  args.label_a, args.label_b,
                                  out_path, elapsed_a, elapsed_b)

            changed = pa["top_node"] != pb["top_node"] or pa["primary_flow"] != pb["primary_flow"]
            summary_rows.append({
                "file": txt_path.name,
                "top_a": pa["top_node"], "flow_a": pa["primary_flow"],
                "top_b": pb["top_node"], "flow_b": pb["primary_flow"],
                "changed": "🔴 CHANGED" if changed else "✅ same",
                "report": out_path.name,
            })
            log.info(f"   A: {pa['top_node']} / {pa['primary_flow']}")
            log.info(f"   B: {pb['top_node']} / {pb['primary_flow']}  {'← CHANGED' if changed else ''}")

        else:
            t0 = time.time()
            raw = run_prediction(args.script, args.coredrill, text)
            elapsed = time.time() - t0

            parsed   = parse_output(raw)
            out_path = output_dir / (txt_path.stem + "_report.md")
            write_single_report(txt_path, text, parsed, out_path, elapsed)

            summary_rows.append({
                "file":     txt_path.name,
                "top_node": parsed["top_node"],
                "flow":     parsed["primary_flow"],
                "time_s":   round(elapsed, 1),
                "report":   out_path.name,
            })
            log.info(f"   → {out_path.name}  ({elapsed:.1f}s)  {parsed['top_node']} / {parsed['primary_flow']}")

    # Summary file
    summary_path = output_dir / ("_summary_compare.md" if ab_mode else "_summary.md")
    sl = [f"# {'A/B Comparison' if ab_mode else 'Batch'} Summary",
          f"", f"**Run:** {time.strftime('%Y-%m-%d %H:%M:%S')}", f""]

    if ab_mode:
        sl += [
            f"**A:** `{args.label_a}` — `{args.script_a}` + `{args.coredrill_a}`  ",
            f"**B:** `{args.label_b}` — `{args.script_b}` + `{args.coredrill_b}`",
            f"",
            f"| # | File | A top | A flow | B top | B flow | Δ |",
            f"|---|------|-------|--------|-------|--------|---|",
        ]
        for i, r in enumerate(summary_rows, 1):
            sl.append(f"| {i} | `{r['file']}` | {r['top_a']} | {r['flow_a']} | {r['top_b']} | {r['flow_b']} | {r['changed']} |")
        n_changed = sum(1 for r in summary_rows if "CHANGED" in r["changed"])
        sl += [f"", f"**{n_changed}/{len(summary_rows)} results changed**"]
    else:
        sl += [
            f"**Coredrill:** `{args.coredrill}`",
            f"",
            f"| # | File | Top Node | Flow | Time | Report |",
            f"|---|------|----------|------|------|--------|",
        ]
        for i, r in enumerate(summary_rows, 1):
            sl.append(f"| {i} | `{r['file']}` | {r['top_node']} | {r['flow']} | {r['time_s']}s | [{r['report']}]({r['report']}) |")

    sl += [f"", f"---", f"*predict_folder.py*"]
    summary_path.write_text("\n".join(sl), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"DONE — {len(summary_rows)} files → {output_dir}")
    print(f"Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
