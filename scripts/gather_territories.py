"""Gathers real Wikipedia text for territories that are short of it.

    python scripts/gather_territories.py            # check the list, and say what is missing
    python scripts/gather_territories.py --fetch    # fetch; safe to stop and start again

territory_articles.txt says which articles belong to which territory. Each
article's paragraphs go to data/sources/wikipedia.jsonl, up to --target texts
per territory counting the ones it already has to itself, spread over its
articles. Nothing here touches the corpus: the file waits to be checked and
added. Articles of the holdout are never fetched, and text the corpus already
has, under any label, is left out.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.data import record_keys, text_key, title_key  # noqa: E402
from train.harvest import Wikipedia  # noqa: E402
from train.record import now  # noqa: E402

END = re.compile(r"^=+\s*(See also|References|Notes|Footnotes|Citations|Bibliography|Further reading|External links|"
                 r"Sources|Works cited|Gallery|Explanatory notes|Notes and references)\s*=+$", re.I)


def read_articles(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, _, titles = line.partition(": ")
            if name in out:
                raise SystemExit(f"{name} is listed twice")
            out[name] = [t.strip() for t in titles.split("|") if t.strip()]
    return out


def paragraphs(text: str, min_chars: int = 180, max_chars: int = 1200) -> List[str]:
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if END.match(line):
            break
        line = re.sub(r"\s+", " ", line)
        if line.startswith("=") or len(line) < min_chars or "{\\displaystyle" in line:
            continue                             # headings, scraps and formulas
        if len(line) > max_chars:                # cut at the last full sentence
            cut = line.rfind(". ", min_chars, max_chars)
            line = line[:cut + 1] if cut > 0 else line[:max_chars]
        out.append(line)
    return out


def spread(items: list, n: int) -> list:
    """n of them, evenly spaced, in order."""
    if len(items) <= n:
        return items
    return [items[int(i * len(items) / n)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--target", type=int, default=120, help="texts of its own a territory should have")
    ap.add_argument("--per-article", type=int, default=30)
    ap.add_argument("--delay", type=float, default=2.5, help="seconds between requests")
    ap.add_argument("--data", type=Path, default=REPO / "data")
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    records = json.loads((args.data / "sentences.json").read_text(encoding="utf-8"))
    run = json.loads((args.runs / "champion.json").read_text(encoding="utf-8"))["run"]
    tree = json.loads((args.runs / run / "coredrill_hierarchical.json").read_text(encoding="utf-8"))["tree"]
    taxonomy = set(tree)
    articles = read_articles(Path(__file__).with_name("territory_articles.txt"))
    unknown = [n for n in articles if n not in taxonomy]
    count = Counter(title_key(t) for ts in articles.values() for t in ts)
    twice = [t for ts in articles.values() for t in ts if count[title_key(t)] > 1]
    if unknown or twice:
        raise SystemExit(f"not in the taxonomy: {unknown}; listed for two territories: {sorted(set(twice))}")

    holdout = json.loads((args.runs / "holdout.json").read_text(encoding="utf-8"))
    held, blocked = set(holdout["keys"]), set(holdout["blocked_titles"])
    labels = defaultdict(set)
    for r in records:
        labels[text_key(r["text"])].add(r["category"])
    own = Counter(next(iter(v)) for v in labels.values() if len(v) == 1)
    out = args.data / "sources" / "wikipedia.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(z) for z in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    done = {(r["category"], r["article"]) for r in rows}
    log = args.data / "sources" / "gather.log"
    fetched = {tuple(z.split("\t")[1:3]) for z in log.read_text(encoding="utf-8").splitlines()} if log.exists() else set()
    known = set(labels) | {text_key(r["text"]) for r in rows}
    have = Counter(own)
    have.update(r["category"] for r in rows)

    short = {n: args.target - have[n] for n in articles if have[n] < args.target}
    skipped = [t for ts in articles.values() for t in ts if title_key(t) in blocked]
    print(f"{len(articles)} territories listed, {sum(map(len, articles.values()))} articles; {len(short)} territories "
          f"short of {args.target}, {sum(short.values()):,} texts missing. {len(rows):,} gathered so far.")
    if skipped:
        print(f"never fetched, because the holdout has them: {', '.join(skipped)}")
    if not args.fetch:
        return

    wiki = Wikipedia(delay=args.delay, log=print)
    for k, (name, need) in enumerate(sorted(short.items(), key=lambda x: -x[1]), 1):
        todo = [t for t in articles[name] if title_key(t) not in blocked and (name, t) not in fetched and (name, t) not in done]
        found = 0
        for i, title in enumerate(todo):
            if need <= 0:
                break
            try:
                d = wiki._get("en", {"action": "query", "prop": "extracts|pageprops", "explaintext": 1, "exlimit": 1,
                                     "ppprop": "disambiguation", "redirects": 1, "titles": title})
                page = (d.get("query", {}).get("pages") or [{}])[0]
            except Exception as e:  # noqa: BLE001 - one failed article must not end the run
                print(f"    {title}: {e}")
                continue
            final, note = page.get("title", title), ""
            if page.get("missing"):
                note, texts = "missing", []
            elif "disambiguation" in (page.get("pageprops") or {}):
                note, texts = "disambiguation", []
            elif title_key(final) in blocked:
                note, texts = "in the holdout", []
            else:
                texts = [p for p in paragraphs(page.get("extract") or "") if text_key(p) not in known]
                share = max(1, -(-need // (len(todo) - i)))      # what is missing, over the articles left
                texts = spread(texts, min(args.per_article, max(share, 8)))[:need]
            with out.open("a", encoding="utf-8") as f:
                for p in texts:
                    known.add(text_key(p))
                    f.write(json.dumps({"text": p, "category": name, "lang": "en", "source": "wikipedia_territory",
                                        "article": title, "article_title": f"en:{final}", "source_article": final,
                                        "gathered": now()}, ensure_ascii=False) + "\n")
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{now()}\t{name}\t{title}\t{len(texts)}\t{note}\n")
            need -= len(texts)
            found += len(texts)
        print(f"{now()}  {k}/{len(short)} {name}: +{found}, {max(need, 0)} still missing ({wiki.requests} requests)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
