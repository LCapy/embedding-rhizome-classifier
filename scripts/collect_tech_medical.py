#!/usr/bin/env python3
"""
collect_tech_medical.py

Fetches Wikipedia article introductions for new taxonomy nodes
(Technology branch + medical depth + Jurisprudence) and writes
training records in the standard format used by the coredrill pipeline:

    {"text": "...", "category": "...", "source": "wikipedia", "lang": "en"}

Output: one JSONL file per node under --out-dir (default: data/wiki_new_nodes/).
Run this, then re-run the embed + build steps to update the coredrill.

Usage:
    python scripts/collect_tech_medical.py
    python scripts/collect_tech_medical.py --out-dir data/wiki_new_nodes --min-chars 200
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("collect_tech_medical")

HEADERS = {
    "User-Agent": "coredrill-train/1.0 (academic; HF Space LCapy/coredrill)"
}

# ---------------------------------------------------------------------------
# Node -> Wikipedia article targets
# Each tuple: (node_name, [wikipedia_titles])
# Fetch the full "extract" from REST API v1/page/summary
# ---------------------------------------------------------------------------
TARGETS: list[tuple[str, list[str]]] = [
    ("Technology", [
        "Technology",
        "Information technology",
        "Emerging technologies",
        "Engineering",
        "Innovation",
        "History of technology",
        "Digital revolution",
        "High technology",
    ]),
    ("Computing & Software", [
        "Computer science",
        "Software",
        "Algorithm",
        "Operating system",
        "Computer programming",
        "Software engineering",
        "Computer hardware",
        "Data structure",
    ]),
    ("Artificial Intelligence", [
        "Artificial intelligence",
        "Machine learning",
        "Deep learning",
        "Neural network",
        "Large language model",
        "Natural language processing",
        "Computer vision",
        "Reinforcement learning",
        "Generative artificial intelligence",
    ]),
    ("Cybersecurity", [
        "Computer security",
        "Cyberattack",
        "Encryption",
        "Data breach",
        "Malware",
        "Phishing",
        "Firewall (computing)",
        "Vulnerability (computing)",
    ]),
    ("Consumer Electronics", [
        "Consumer electronics",
        "Smartphone",
        "Laptop",
        "Personal computer",
        "Tablet computer",
        "Digital camera",
        "Video game console",
        "Home appliance",
    ]),
    ("Wearable Technology", [
        "Wearable technology",
        "Smartwatch",
        "Augmented reality",
        "Fitness tracker",
        "Virtual reality headset",
        "Google Glass",
        "Apple Watch",
    ]),
    ("Internet & Digital Media", [
        "Internet",
        "World Wide Web",
        "Digital media",
        "Web 2.0",
        "Streaming media",
        "Podcast",
        "Blog",
        "Search engine",
    ]),
    ("Social Media", [
        "Social media",
        "Twitter",
        "Facebook",
        "Instagram",
        "TikTok",
        "YouTube",
        "Reddit",
        "LinkedIn",
    ]),
    ("Online Commerce", [
        "E-commerce",
        "Amazon (company)",
        "Online shopping",
        "Digital marketing",
        "Marketplace (online)",
        "Drop shipping",
        "Subscription business model",
        "Cryptocurrency",
    ]),
    ("Epidemiology", [
        "Epidemiology",
        "Pandemic",
        "Epidemic",
        "Endemic (epidemiology)",
        "Infectious disease",
        "Influenza",
        "COVID-19",
        "Vaccination",
        "Herd immunity",
        "Transmission (medicine)",
    ]),
    ("Public Health", [
        "Public health",
        "World Health Organization",
        "Health system",
        "Preventive healthcare",
        "Disease surveillance",
        "Quarantine",
        "Health equity",
        "Sanitation",
    ]),
    ("Pharmacology", [
        "Pharmacology",
        "Drug",
        "Medication",
        "Clinical trial",
        "Antibiotics",
        "Vaccine",
        "Chemotherapy",
        "Drug resistance",
        "Pharmacokinetics",
    ]),
    ("Jurisprudence", [
        "Jurisprudence",
        "Natural law",
        "Common law",
        "Legal philosophy",
        "Roman law",
        "Constitutional law",
        "Legal positivism",
        "Rule of law",
        "Precedent",
    ]),
]


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_wikipedia_extract(title: str, min_chars: int = 150,
                            max_retries: int = 4) -> str | None:
    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(title.replace(" ", "_"))
    )
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())
            text = data.get("extract", "").strip()
            if len(text) >= min_chars:
                return text
            log.warning("  '%s': extract too short (%d chars)", title, len(text))
            return None
        except Exception as e:
            msg = str(e)
            if "429" in msg and attempt < max_retries - 1:
                wait = 2 ** (attempt + 2)   # 4s, 8s, 16s, ...
                log.warning("  '%s': rate-limited (429), retrying in %ds...", title, wait)
                time.sleep(wait)
            else:
                log.warning("  '%s': fetch failed - %s", title, e)
                return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Wikipedia training data for new taxonomy nodes")
    ap.add_argument("--out-dir", default="data/wiki_new_nodes",
                    help="Directory to write JSONL files (default: data/wiki_new_nodes)")
    ap.add_argument("--min-chars", type=int, default=150,
                    help="Minimum extract length to keep (default: 150)")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="Seconds between requests (default: 1.5)")
    ap.add_argument("--nodes", nargs="*", default=None,
                    help="Only collect for these node names (default: all)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip nodes that already have a JSONL file in --out-dir")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = TARGETS
    if args.nodes:
        targets = [(n, t) for n, t in targets if n in args.nodes]
        log.info("Filtering to %d nodes: %s", len(targets), args.nodes)

    total_written = 0

    for node, titles in targets:
        safe_name = node.lower().replace(" & ", "_").replace(" ", "_")
        out_path = out_dir / f"{safe_name}.jsonl"
        if args.skip_existing and out_path.exists():
            existing = sum(1 for _ in out_path.open(encoding="utf-8"))
            log.info("Node: %s - skipping (%d records already collected)", node, existing)
            total_written += existing
            continue

        records = []
        log.info("Node: %s (%d articles)", node, len(titles))

        for title in titles:
            text = fetch_wikipedia_extract(title, args.min_chars)
            if text:
                records.append({
                    "text":     text,
                    "category": node,
                    "source":   "wikipedia",
                    "lang":     "en",
                })
                log.info("  + '%s' (%d chars)", title, len(text))
            time.sleep(args.delay)

        if records:
            with out_path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.info("  -> %d records written to %s", len(records), out_path)
            total_written += len(records)
        else:
            log.warning("  -> no records for %s", node)

    log.info("Done. Total records: %d", total_written)
    print(f"\nDone. {total_written} records in {out_dir}/")
    print("Next steps:")
    print("  1. Merge these records into your existing training data")
    print("  2. Re-run: python scripts/rhizome_engine.py embed --data <data_dir>")
    print("  3. Re-run: python scripts/rhizome_engine.py build --data <data_dir> \\")
    print("               --taxonomy-txt api/taxonomy.py \\")
    print("               --out publish/coredrill/coredrill_hierarchical.json")


if __name__ == "__main__":
    main()
