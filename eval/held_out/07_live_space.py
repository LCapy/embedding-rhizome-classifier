#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_live_space.py - send the same inputs as 06_try_sentences.py to the LIVE
private HF Space (https://lcapy-coredrill.hf.space) and compare its answers
with the local in-process run, the trained head, and nearest-centroid.

The Space is private: set HF_TOKEN to an HF token with read access to it
(never printed or written out). Override the URL with EVAL_SPACE_URL.
Requires artifacts/try_sentences.json from 06_try_sentences.py.
"""
import json
import os
import time
import urllib.request

from common import OUT_DIR

SPACE = os.environ.get("EVAL_SPACE_URL", "https://lcapy-coredrill.hf.space")


def call(path: str, token: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        SPACE + path, data=data, method="POST" if data else "GET",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    token = os.environ["HF_TOKEN"]
    info = call("/info", token)
    print(f"Live Space: coredrill version {info['coredrill_version']}, "
          f"{info['taxonomy_nodes']} nodes, {len(info['skip_nodes'])} skip nodes:")
    print(f"  {', '.join(info['skip_nodes'])}\n")

    local = json.loads((OUT_DIR / "try_sentences.json").read_text(encoding="utf-8"))
    rows = []
    for item in local:
        t0 = time.perf_counter()
        live = call("/predict", token, {"text": item["text"]})
        rows.append({
            "name": item["name"],
            "live": live.get("best_node"),
            "live_lineage": live.get("lineage"),
            "live_confidence": live.get("confidence"),
            "live_latency_ms": live.get("latency_ms"),
            "local_deployed": item["deployed"]["best_node"],
            "trained": item["trained_top3"][0][0],
            "centroid": item["centroid_top1"][0],
        })
        r = rows[-1]
        same = "same" if r["live"] == r["local_deployed"] else "DIFFERS"
        print(f"{r['name']:<36} live={r['live']:<26} local={r['local_deployed']:<26} [{same}]  "
              f"({time.perf_counter() - t0:.1f}s)")

    (OUT_DIR / "live_space.json").write_text(json.dumps({"info": info, "rows": rows},
                                                        ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["| Input | Live Space | Local deployed code | Trained head | Nearest centroid |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| `{r['name']}` | {r['live']} | {r['local_deployed']} | {r['trained']} | {r['centroid']} |")
    (OUT_DIR / "live_space.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUT_DIR / 'live_space.json'} and live_space.md")


if __name__ == "__main__":
    main()
