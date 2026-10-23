#!/usr/bin/env python3
"""
Coredrill evaluation suite.

Fetches real texts from Wikipedia, BBC RSS, and Reddit, runs predictions,
and grades them on two dimensions:

  Node accuracy  (0-3): did gram_winner match the expected node?
  Rhizome quality (0-3): is the expected territory properly activated,
                          with good sigma / mass / gram score?

Combined score: 0-6 per item.

Usage:
    cd ~/Studies/embeddings/hf-space
    python tests/run_eval.py --coredrill ../publish/coredrill/coredrill_hierarchical.json
    python tests/run_eval.py --coredrill ... --filter-source wikipedia
    python tests/run_eval.py --coredrill ... --items 5
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")

SUITE_PATH  = Path(__file__).parent / "eval_suite.json"
REPORTS_DIR = Path(__file__).parent / "reports"

HEADERS = {
    "User-Agent": "coredrill-eval/1.0 (academic research; HF Space LCapy/coredrill)"
}


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_wikipedia_intro(title: str) -> str:
    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(title.replace(" ", "_"))
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
            return data.get("extract", "")
    except Exception as e:
        log.warning("Wikipedia fetch failed for %r: %s", title, e)
        return ""


def fetch_rss_item(url: str, index: int = 0) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
        tree = ET.fromstring(raw)
        items = tree.findall(".//item")
        if not items:
            return ""
        item = items[min(index, len(items) - 1)]
        title = (item.findtext("title") or "").strip()
        desc  = re.sub(r"<[^>]+>", "", item.findtext("description") or "").strip()
        return f"{title}. {desc}".strip() if desc else title
    except Exception as e:
        log.warning("RSS fetch failed for %s: %s", url, e)
        return ""


def _get_with_backoff(req: urllib.request.Request, attempts: int = 5) -> bytes:
    """GET with retries on HTTP 429, honoring Retry-After when present."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == attempts - 1:
                raise
            retry_after = e.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 5 * 2 ** attempt
            log.info("HTTP 429 from %s, retrying in %ds", req.full_url, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_reddit_post(subreddit: str, sort: str = "top",
                      time_filter: str = "week", index: int = 0) -> str:
    # Reddit's JSON API now returns 403 to unauthenticated clients; the Atom
    # feed still works. Text posts carry their body in <content> ahead of
    # the "submitted by" footer; link posts only have a title.
    url = f"https://www.reddit.com/r/{subreddit}/{sort}/.rss?limit=25&t={time_filter}"
    req = urllib.request.Request(url, headers=HEADERS)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        tree = ET.fromstring(_get_with_backoff(req))
        posts = []
        for entry in tree.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            content = entry.findtext("a:content", default="", namespaces=ns) or ""
            body = html.unescape(re.sub(r"<[^>]+>", " ", content))
            body = re.sub(r"\s+", " ", body.split("submitted by")[0]).strip()
            if body in ("[removed]", "[deleted]"):
                body = ""
            posts.append((title, body))
        text_posts = [p for p in posts if p[1]]
        pool = text_posts if text_posts else posts
        if not pool:
            return ""
        title, body = pool[min(index, len(pool) - 1)]
        return (f"{title}\n\n{body}".strip() if body else title)[:2500]
    except Exception as e:
        log.warning("Reddit fetch failed for r/%s: %s", subreddit, e)
        return ""


FETCHERS = {
    "wikipedia_intro": lambda item: fetch_wikipedia_intro(item["fetch"]["title"]),
    "rss_item":        lambda item: fetch_rss_item(
        item["fetch"]["url"], item["fetch"].get("index", 0)),
    "reddit_post":     lambda item: fetch_reddit_post(
        item["fetch"]["subreddit"],
        item["fetch"].get("sort", "top"),
        item["fetch"].get("time", "week"),
        item["fetch"].get("index", 0),
    ),
}


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

NODE_GRADES = {
    "PASS":            {"score": 3, "label": "pass"},
    "PARTIAL_PARENT":  {"score": 2, "label": "partial (domain)"},
    "PARTIAL_OVERLAP": {"score": 1, "label": "partial (overlap)"},
    "FAIL":            {"score": 0, "label": "fail"},
    "FETCH_ERROR":     {"score": 0, "label": "fetch error"},
}

RHIZOME_GRADES = {
    3: "dominant",          # expected node is the top or near-top activation
    2: "present-strong",    # expected in overlap with decent mass / top-5 ranked
    1: "present-weak",      # expected appears in results but distant
    0: "absent",            # expected not found or negligible
}


def grade_node(result: dict, item: dict) -> dict:
    """Grade whether gram_winner matched the expected node."""
    accept  = set(item["expected"].get("accept_nodes", [item["expected"]["node"]]))
    partial = set(item["expected"].get("partial_nodes", []))

    gram_winner  = result.get("gram_winner") or ""
    active_nodes = [r.get("node", "") for r in result.get("active_overlap_set", [])]
    top5_nodes   = [r.get("node", "") for r in result.get("results", [])[:5]]

    if gram_winner in accept:
        return {"grade": "PASS", "score": 3, "detail": f"gram_winner '{gram_winner}' in accept set"}

    if gram_winner in partial:
        return {"grade": "PARTIAL_PARENT", "score": 2,
                "detail": f"gram_winner '{gram_winner}' is a parent/sibling of expected"}

    overlap_accept  = [n for n in active_nodes if n in accept]
    overlap_partial = [n for n in active_nodes if n in partial]
    top5_accept     = [n for n in top5_nodes  if n in accept]

    if overlap_accept:
        return {"grade": "PARTIAL_OVERLAP", "score": 1,
                "detail": f"expected found in active overlap: {overlap_accept[0]}"}
    if overlap_partial:
        return {"grade": "PARTIAL_OVERLAP", "score": 1,
                "detail": f"partial-match node in active overlap: {overlap_partial[0]}"}
    if top5_accept:
        return {"grade": "PARTIAL_OVERLAP", "score": 1,
                "detail": f"expected found in top-5 ranked: {top5_accept[0]}"}

    return {"grade": "FAIL", "score": 0,
            "detail": f"expected '{item['expected']['node']}' not found anywhere"}


def grade_rhizome(result: dict, item: dict, expected_metrics: dict,
                  gram_score_expected: float | None) -> dict:
    """
    Grade rhizome quality independently of which node won.

    Looks at:
    - Is the expected node in the active overlap set?
    - What mass / sigma_dist does it carry?
    - What gram score does it get?
    - What rank does it hold in the full results?
    """
    accept = set(item["expected"].get("accept_nodes", [item["expected"]["node"]]))

    overlap_nodes  = {r.get("node", ""): r for r in result.get("active_overlap_set", [])}
    active_accept  = {n: overlap_nodes[n] for n in overlap_nodes if n in accept}

    expected_rank  = expected_metrics.get("rank")     # 1-indexed, None if absent
    expected_z     = expected_metrics.get("sigma_dist")

    gram_pos = gram_score_expected is not None and gram_score_expected > 0

    if active_accept:
        best = max(active_accept.values(), key=lambda r: r.get("mass", 0))
        mass = best.get("mass", 0)
        z    = best.get("sigma_dist", 99)
        if mass >= 25.0 or z <= 1.0:
            return {"rq_score": 3, "rq_label": RHIZOME_GRADES[3],
                    "rq_detail": f"expected active with mass={mass:.1f}% z={z:.2f}"}
        return {"rq_score": 2, "rq_label": RHIZOME_GRADES[2],
                "rq_detail": f"expected in overlap but low mass ({mass:.1f}%)"}

    if expected_rank is not None and expected_rank <= 5 and (expected_z or 99) <= 1.5:
        return {"rq_score": 2, "rq_label": RHIZOME_GRADES[2],
                "rq_detail": f"expected rank #{expected_rank} z={expected_z:.2f} (not in overlap)"}

    if gram_pos:
        return {"rq_score": 2, "rq_label": RHIZOME_GRADES[2],
                "rq_detail": f"expected has positive gram score ({gram_score_expected:.4f})"}

    if expected_rank is not None and expected_rank <= 20:
        return {"rq_score": 1, "rq_label": RHIZOME_GRADES[1],
                "rq_detail": f"expected rank #{expected_rank} z={expected_z:.2f} (weak activation)"}

    return {"rq_score": 0, "rq_label": RHIZOME_GRADES[0],
            "rq_detail": "expected node not meaningfully activated"}


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _sf(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def extract_rhizome_data(result: dict, coredrill: dict,
                         gram_winner: str) -> dict:
    """Pull all relevant rhizome fields out of the engine result."""
    results_all  = result.get("results", [])
    active       = result.get("active_overlap_set", [])
    gram_scores  = result.get("gram_scores", {})
    gram_state   = result.get("gram_state", "unknown")
    pred         = result.get("prediction") or {}

    # Lineage of gram_winner from coredrill backbone
    backbone     = coredrill.get("taxonomy_backbone", {})
    lineage_map  = backbone.get("lineage_paths", {})
    raw_paths    = lineage_map.get(gram_winner, [])
    lineage      = raw_paths[0] if raw_paths else [gram_winner]

    # Deleuzian flow
    try:
        from rhizome_engine import _deleuzian_flow  # type: ignore
        flow = _deleuzian_flow(active, results_all, pred) or {}
    except Exception:
        flow = {}

    # Active overlap table (full metrics)
    overlap_table = []
    gram_lookup   = gram_scores  # node -> float
    for r in active:
        node = r.get("node", "")
        overlap_table.append({
            "node":             node,
            "gram_score":       gram_lookup.get(node),
            "cos":              _sf(r.get("score")),
            "sigma":            _sf(r.get("sigma_eff", r.get("sigma_dist"))),
            "z":                _sf(r.get("sigma_dist")),
            "mass":             _sf(r.get("mass")),
            "production_score": _sf(r.get("production_score")),
            "laplace":          _sf(r.get("laplace_score")),
        })

    # Gram scores table (top 10 by gram score, positive only preferred)
    gram_table = sorted(
        [{"node": n, "gram_score": v} for n, v in gram_scores.items()],
        key=lambda x: -x["gram_score"]
    )[:10]
    # Enrich with cos/z/mass from results
    result_lookup = {r.get("node", ""): r for r in results_all}
    for entry in gram_table:
        r = result_lookup.get(entry["node"], {})
        entry["cos"]   = _sf(r.get("score"))
        entry["sigma"] = _sf(r.get("sigma_eff", r.get("sigma_dist")))
        entry["z"]     = _sf(r.get("sigma_dist"))
        entry["mass"]  = _sf(r.get("mass"))

    return {
        "gram_state":    gram_state,
        "gram_table":    gram_table,
        "overlap_table": overlap_table,
        "flow_type":     flow.get("flow_type", "unknown"),
        "flow_label":    flow.get("flow_label", ""),
        "flow_scores":   flow.get("intensities", {}),
        "lineage":       lineage,
    }


def find_expected_metrics(results_all: list, gram_scores: dict,
                          accept_nodes: set) -> tuple[dict, float | None]:
    """Find the best-ranked accept node in results and its gram score."""
    # Find highest-ranked accept node
    best_rank   = None
    best_r      = {}
    for rank, r in enumerate(results_all, 1):
        if r.get("node") in accept_nodes:
            best_rank = rank
            best_r    = r
            break

    metrics = {
        "rank":             best_rank,
        "sigma_dist":       _sf(best_r.get("sigma_dist"), None) if best_r else None,
        "mass":             _sf(best_r.get("mass"), 0.0),
        "production_score": _sf(best_r.get("production_score"), 0.0),
        "conf":             _sf(best_r.get("conf"), 0.0),
        "node":             best_r.get("node"),
    }

    gram_expected = next(
        (gram_scores.get(n) for n in accept_nodes if n in gram_scores), None
    )
    return metrics, gram_expected


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _flow_bar(scores: dict) -> str:
    order = ["territorialization", "deterritorialization",
             "line_of_flight", "bwo_approach", "reterritorialization"]
    parts = []
    for k in order:
        v = scores.get(k, 0.0)
        bar = "#" * int(v * 20) + "." * (20 - int(v * 20))
        abbr = {"territorialization": "T", "deterritorialization": "D",
                "line_of_flight": "LF", "bwo_approach": "BwO",
                "reterritorialization": "Re"}.get(k, k[:2])
        parts.append(f"`{abbr}`[{bar}]{v:.2f}")
    return "  ".join(parts)


def write_report(rows: list, args_coredrill: str,
                 coredrill_version: str, report_path: Path) -> None:
    now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scored  = [r for r in rows if r["node_grade"] != "FETCH_ERROR"]
    max_pts = len(scored) * 6   # 3 node + 3 rhizome
    actual  = sum(r["node_score"] + r["rq_score"] for r in scored)
    pct     = round(100 * actual / max_pts) if max_pts else 0

    node_counts = {}
    for r in rows:
        node_counts[r["node_grade"]] = node_counts.get(r["node_grade"], 0) + 1

    rq_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in scored:
        rq_counts[r["rq_score"]] += 1

    by_source: dict[str, list] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    lines = [
        "# Coredrill Evaluation Report",
        "",
        f"**Date:** {now}  ",
        f"**Coredrill:** `{args_coredrill}`  ",
        f"**Version:** {coredrill_version}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "> Each item scores 0-6: node accuracy (0-3) + rhizome quality (0-3).",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Items tested | {len(rows)} |",
        f"| Items scored | {len(scored)} |",
        f"| Combined score | {actual} / {max_pts} ({pct}%) |",
        "",
        "**Node accuracy (gram_winner vs expected):**",
        "",
        "| Grade | Count |",
        "|-------|-------|",
        f"| PASS (3 pts) | {node_counts.get('PASS', 0)} |",
        f"| Partial - domain (2 pts) | {node_counts.get('PARTIAL_PARENT', 0)} |",
        f"| Partial - overlap (1 pt) | {node_counts.get('PARTIAL_OVERLAP', 0)} |",
        f"| FAIL (0 pts) | {node_counts.get('FAIL', 0)} |",
        f"| Fetch error | {node_counts.get('FETCH_ERROR', 0)} |",
        "",
        "**Rhizome quality (expected territory activation):**",
        "",
        "| Level | Count |",
        "|-------|-------|",
        f"| Dominant (3 pts) | {rq_counts[3]} |",
        f"| Present-strong (2 pts) | {rq_counts[2]} |",
        f"| Present-weak (1 pt) | {rq_counts[1]} |",
        f"| Absent (0 pts) | {rq_counts[0]} |",
        "",
        "## By Source",
        "",
        "| Source | Items | Node score | RQ score | Pass | Partial | Fail |",
        "|--------|-------|-----------|----------|------|---------|------|",
    ]

    for src, src_rows in sorted(by_source.items()):
        sc  = [r for r in src_rows if r["node_grade"] != "FETCH_ERROR"]
        ns  = sum(r["node_score"] for r in sc)
        rqs = sum(r["rq_score"]   for r in sc)
        mx  = len(sc) * 3
        n_pass    = sum(1 for r in src_rows if r["node_grade"] == "PASS")
        n_partial = sum(1 for r in src_rows if "PARTIAL" in r["node_grade"])
        n_fail    = sum(1 for r in src_rows if r["node_grade"] == "FAIL")
        lines.append(
            f"| {src} | {len(src_rows)} | {ns}/{mx} | {rqs}/{mx} "
            f"| {n_pass} | {n_partial} | {n_fail} |"
        )

    lines += [
        "",
        "## Results Overview",
        "",
        "| # | ID | Expected | gram_winner | Node | RQ | Total | ms |",
        "|---|----|----|-----|------|-----|-----|-----|",
    ]
    for i, r in enumerate(rows, 1):
        ng  = NODE_GRADES.get(r["node_grade"], NODE_GRADES["FAIL"])
        lat = f"{r['latency_ms']:.0f}" if r.get("latency_ms") is not None else "-"
        lines.append(
            f"| {i} | {r['id']} | `{r['expected_node']}` | `{r['gram_winner']}` "
            f"| {ng['label']} ({r['node_score']}) "
            f"| {r['rq_label']} ({r['rq_score']}) "
            f"| {r['node_score'] + r['rq_score']}/6 | {lat} |"
        )

    lines += ["", "---", "", "## Per-Item Detail", ""]

    for r in rows:
        ng    = NODE_GRADES.get(r["node_grade"], NODE_GRADES["FAIL"])
        total = r["node_score"] + r["rq_score"]
        lines += [
            f"### {r['id']}  [{total}/6]",
            "",
            f"**Label:** {r.get('label', r['id'])}  ",
            f"**Node accuracy:** {ng['label']} ({r['node_score']}/3) - {r['node_detail']}  ",
            f"**Rhizome quality:** {r['rq_label']} ({r['rq_score']}/3) - {r['rq_detail']}  ",
            f"**Latency:** {r['latency_ms']:.0f} ms" if r.get("latency_ms") else "",
            "",
            f"> {r.get('text_preview', '')}...",
            "",
        ]

        if r["node_grade"] == "FETCH_ERROR":
            lines += ["_Fetch failed - no prediction data._", ""]
            continue

        # Gram winner summary
        gw    = r["gram_winner"]
        gw_gs = r.get("gram_score_winner")
        gs_str = f"gram: {gw_gs:+.4f}" if gw_gs is not None else "gram: n/a"
        lines += [
            f"**Gram winner:** `{gw}`  ({gs_str})  ",
            f"**Gram state:** {r.get('gram_state', '?')}  ",
            f"**Flow:** `{r.get('flow_type', '?')}` - {r.get('flow_label', '')}  ",
            f"**Lineage:** {' > '.join(r.get('lineage', [gw]))}",
            "",
        ]

        # Flow intensity bar
        if r.get("flow_scores"):
            lines += [_flow_bar(r["flow_scores"]), ""]

        # Active overlap table
        ot = r.get("overlap_table", [])
        if ot:
            lines += [
                "**Active overlap set:**",
                "",
                "| Node | gram | cos | sigma | z | mass | prod |",
                "|------|------|-----|-------|---|------|------|",
            ]
            for row in ot:
                gs = f"{row['gram_score']:+.4f}" if row.get("gram_score") is not None else "-"
                lines.append(
                    f"| `{row['node']}` | {gs} | {row['cos']:.4f} "
                    f"| {row['sigma']:.4f} | {row['z']:.2f}s "
                    f"| {row['mass']:.1f}% | {row['production_score']:.4f} |"
                )
            lines.append("")

        # Gram scores table
        gt = r.get("gram_table", [])
        if gt:
            lines += [
                "**Gram scores (top 10):**",
                "",
                "| Node | gram score | cos | z | mass |",
                "|------|-----------|-----|---|------|",
            ]
            for entry in gt:
                lines.append(
                    f"| `{entry['node']}` | {entry['gram_score']:+.4f} "
                    f"| {entry['cos']:.4f} | {entry['z']:.2f}s | {entry['mass']:.1f}% |"
                )
            lines.append("")

        # Expected node in results
        em = r.get("expected_metrics", {})
        if em and em.get("rank") is not None:
            gs_exp = r.get("gram_score_expected")
            gs_str = f"{gs_exp:+.4f}" if gs_exp is not None else "not in gram"
            lines += [
                f"**Expected node** (`{em.get('node', r['expected_node'])}`) **in results:**  ",
                f"Rank #{em['rank']} | gram: {gs_str} | cos: {em.get('conf', 0):.4f} "
                f"| z: {em.get('sigma_dist', 'n/a'):.2f}s | mass: {em.get('mass', 0):.1f}%",
                "",
            ]
        else:
            lines += [
                f"**Expected node** (`{r['expected_node']}`) **not found in top results.**",
                "",
            ]

        lines.append("---")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Coredrill evaluation suite")
    parser.add_argument("--coredrill", required=True,
                        help="Path to coredrill_hierarchical.json")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--items", type=int, default=None)
    parser.add_argument("--filter-source", default=None,
                        help="wikipedia | bbc_rss | reddit")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    items = suite["items"]
    if args.filter_source:
        items = [i for i in items if i["source"] == args.filter_source]
    if args.items:
        items = items[:args.items]
    log.info("Running %d items (source: %s)", len(items), args.filter_source or "all")

    log.info("Loading coredrill ...")
    coredrill = json.loads(Path(args.coredrill).read_text(encoding="utf-8"))
    version   = coredrill.get("version", "unknown")
    log.info("Coredrill v%s - %d nodes", version, len(coredrill.get("tree", {})))

    log.info("Loading LaBSE model ...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from rhizome_engine import analyze_text_rhizome_v2  # type: ignore
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(coredrill.get("labse_model", "sentence-transformers/LaBSE"))
    log.info("Model ready.")

    tau = args.tau if args.tau is not None else coredrill.get("tau", 0.1)

    rows = []
    for idx, item in enumerate(items):
        log.info("[%d/%d] %s", idx + 1, len(items), item["id"])

        fetcher = FETCHERS.get(item["fetch"]["type"])
        text    = fetcher(item) if fetcher else ""

        if not text or len(text) < 40:
            log.warning("  -> FETCH_ERROR")
            rows.append({
                "id": item["id"], "label": item.get("label", item["id"]),
                "source": item["source"], "expected_node": item["expected"]["node"],
                "gram_winner": "", "node_grade": "FETCH_ERROR", "node_score": 0,
                "node_detail": "fetch returned empty text",
                "rq_score": 0, "rq_label": "absent", "rq_detail": "no text",
                "text_preview": "", "latency_ms": None,
                "gram_state": None, "gram_table": [], "overlap_table": [],
                "flow_type": None, "flow_label": None, "flow_scores": {},
                "lineage": [], "expected_metrics": {}, "gram_score_expected": None,
                "gram_score_winner": None,
            })
            time.sleep(1.0)
            continue

        log.info("  text: %d chars - %s...", len(text), text[:80].replace("\n", " "))

        t0     = time.perf_counter()
        result = analyze_text_rhizome_v2(coredrill, text, model,
                                          tau=tau, print_output=False)
        latency_ms = (time.perf_counter() - t0) * 1000

        gram_winner  = result.get("gram_winner") or ""
        gram_scores  = result.get("gram_scores", {})
        results_all  = result.get("results", [])
        accept_nodes = set(item["expected"].get("accept_nodes",
                                                [item["expected"]["node"]]))

        ng = grade_node(result, item)

        expected_metrics, gram_score_expected = find_expected_metrics(
            results_all, gram_scores, accept_nodes)

        rq = grade_rhizome(result, item, expected_metrics, gram_score_expected)

        rhizome_data = extract_rhizome_data(result, coredrill, gram_winner)

        rows.append({
            "id":                  item["id"],
            "label":               item.get("label", item["id"]),
            "source":              item["source"],
            "expected_node":       item["expected"]["node"],
            "gram_winner":         gram_winner,
            "node_grade":          ng["grade"],
            "node_score":          ng["score"],
            "node_detail":         ng["detail"],
            "rq_score":            rq["rq_score"],
            "rq_label":            rq["rq_label"],
            "rq_detail":           rq["rq_detail"],
            "text_preview":        text[:160].replace("\n", " "),
            "latency_ms":          latency_ms,
            "gram_state":          rhizome_data["gram_state"],
            "gram_table":          rhizome_data["gram_table"],
            "overlap_table":       rhizome_data["overlap_table"],
            "flow_type":           rhizome_data["flow_type"],
            "flow_label":          rhizome_data["flow_label"],
            "flow_scores":         rhizome_data["flow_scores"],
            "lineage":             rhizome_data["lineage"],
            "expected_metrics":    expected_metrics,
            "gram_score_expected": gram_score_expected,
            "gram_score_winner":   gram_scores.get(gram_winner),
        })

        log.info("  -> node: %s (%d/3) | rq: %s (%d/3) | gram_winner: %s",
                 ng["grade"], ng["score"], rq["rq_label"], rq["rq_score"], gram_winner)
        time.sleep(0.8)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.output) if args.output else REPORTS_DIR / f"eval_{timestamp}.md"
    write_report(rows, args.coredrill, version, report_path)

    scored  = [r for r in rows if r["node_grade"] != "FETCH_ERROR"]
    total_n = sum(r["node_score"] for r in scored)
    total_r = sum(r["rq_score"]   for r in scored)
    max_pts = len(scored) * 3

    print(f"\n{'='*60}")
    print(f"EVAL COMPLETE")
    print(f"Node accuracy:   {total_n}/{max_pts}"
          f"  (PASS:{sum(1 for r in rows if r['node_grade']=='PASS')}"
          f"  PARTIAL:{sum(1 for r in rows if 'PARTIAL' in r['node_grade'])}"
          f"  FAIL:{sum(1 for r in rows if r['node_grade']=='FAIL')})")
    print(f"Rhizome quality: {total_r}/{max_pts}"
          f"  (dom:{sum(1 for r in scored if r['rq_score']==3)}"
          f"  str:{sum(1 for r in scored if r['rq_score']==2)}"
          f"  weak:{sum(1 for r in scored if r['rq_score']==1)}"
          f"  absent:{sum(1 for r in scored if r['rq_score']==0)})")
    print(f"Combined:        {total_n+total_r}/{max_pts*2}"
          f"  ({round(100*(total_n+total_r)/(max_pts*2)) if max_pts else 0}%)")
    print(f"Report:          {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
