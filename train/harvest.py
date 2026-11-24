"""Sources of new, unlabeled text for self-training rounds.

Wikipedia: for each node, articles the search engine finds similar to the
node's own articles (CirrusSearch "morelike:"), as lead-section paragraphs.
Accepted articles can bring their other-language versions along.

Inbox: .txt files (one text each) and .jsonl files ({"text", "lang"?} per
line) dropped into data/inbox/. Processed files move to inbox/processed/.
"""
from __future__ import annotations

import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .data import Corpus, concept, eng, english_title, title_key

USER_AGENT = "coredrill-selftrain/0.1 (text classifier research; python-urllib)"


def _chunks(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def paragraphs(text: str, min_chars: int = 150, max_chars: int = 1500, limit: int = 2) -> List[str]:
    paras = [re.sub(r"\s+", " ", p).strip() for p in (text or "").split("\n")]
    return [p[:max_chars] for p in paras if len(p) >= min_chars][:limit]


class Wikipedia:
    """A small MediaWiki API client: polite delay, maxlag, retries."""

    def __init__(self, delay: float = 0.3, timeout: int = 30, retries: int = 4, log=print):
        self.delay, self.timeout, self.retries, self.log = delay, timeout, retries, log
        self.requests = 0

    def _get(self, lang: str, params: dict) -> dict:
        params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
        url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        for attempt in range(self.retries):
            time.sleep(self.delay)
            self.requests += 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                if data.get("error", {}).get("code") == "maxlag" and attempt < self.retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return data
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == self.retries - 1:
                    raise
                retry_after = e.headers.get("Retry-After")
                time.sleep(int(retry_after) if retry_after and retry_after.isdigit() else 5 * 2 ** attempt)
            except urllib.error.URLError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(5 * 2 ** attempt)
        raise RuntimeError("unreachable")

    def morelike(self, title: str, limit: int, lang: str = "en") -> List[str]:
        d = self._get(lang, {"action": "query", "list": "search", "srsearch": f"morelike:{title}",
                             "srnamespace": 0, "srlimit": limit, "srprop": ""})
        return [h["title"] for h in d.get("query", {}).get("search", [])]

    def intros(self, titles: List[str], lang: str = "en") -> Dict[str, str]:
        """Plain-text lead sections, keyed by the final (post-redirect) title.
        Disambiguation pages and missing pages are left out."""
        out: Dict[str, str] = {}
        for batch in _chunks(list(dict.fromkeys(titles)), 20):
            d = self._get(lang, {"action": "query", "prop": "extracts|pageprops", "exintro": 1,
                                 "explaintext": 1, "exlimit": "max", "ppprop": "disambiguation",
                                 "redirects": 1, "titles": "|".join(batch)})
            for page in d.get("query", {}).get("pages", []):
                if page.get("missing") or "disambiguation" in (page.get("pageprops") or {}):
                    continue
                text = (page.get("extract") or "").strip()
                if text:
                    out[page["title"]] = text
        return out

    def langlinks(self, titles: List[str], langs: Set[str], lang: str = "en") -> Dict[str, Dict[str, str]]:
        """{title: {lang: title in that language}} for the requested languages."""
        out: Dict[str, Dict[str, str]] = {}
        for batch in _chunks(list(dict.fromkeys(titles)), 50):
            params = {"action": "query", "prop": "langlinks", "lllimit": "max", "redirects": 1,
                      "titles": "|".join(batch)}
            while True:
                d = self._get(lang, params)
                for page in d.get("query", {}).get("pages", []):
                    for ll in page.get("langlinks", []) or []:
                        if ll.get("lang") in langs and ll.get("title"):
                            out.setdefault(page["title"], {})[ll["lang"]] = ll["title"]
                if "continue" not in d:
                    break
                params = {**params, **d["continue"]}
        return out


def seed_titles(corpus: Corpus) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """({node: English titles of its articles, most-used first}, {node: number of
    distinct articles}). Nodes without articles fall back to the engine's
    WIKI_ARTICLES map and the node's en_category."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    for r in corpus.records:
        t = english_title(r)
        if r.get("category") and t:
            counts[r["category"]][t] += 1
    by_node: Dict[str, List[str]] = {}
    n_articles: Dict[str, int] = {}
    en_category = {n["name"]: n.get("en_category") for n in corpus.taxonomy}
    for node in corpus.labels:
        titles = [t for t, _ in counts[node].most_common()]
        n_articles[node] = len({title_key(t) for t in titles})
        if not titles:
            titles = list(eng.WIKI_ARTICLES.get(node, []))
            if en_category.get(node):
                titles.append(en_category[node])
        by_node[node] = list(dict.fromkeys(titles))
    return by_node, n_articles


def wiki_candidates(nodes: List[str], seeds: Dict[str, List[str]], skip_titles: Set[str],
                    wiki: Wikipedia, per_node: int, paras_per_article: int, log=print) -> List[dict]:
    """Paragraphs from articles similar to each node's seeds, tagged with the
    seed node. skip_titles (title_key form) holds the holdout's articles and
    every article already in the corpus or the pool."""
    cands: List[dict] = []
    seen: Set[str] = set()
    for n, node in enumerate(nodes, 1):
        titles: List[str] = []
        for seed in seeds.get(node, [])[:3]:
            try:
                hits = wiki.morelike(seed, per_node + 5)
            except Exception as e:   # one failed search should not end the round
                log(f"    search failed for {seed!r}: {e}")
                continue
            for t in hits:
                k = title_key(t)
                if k not in skip_titles and k not in seen:
                    seen.add(k)
                    titles.append(t)
            if len(titles) >= per_node:
                break
        titles = titles[:per_node]
        if titles:
            try:
                intros = wiki.intros(titles)
            except Exception as e:
                log(f"    fetch failed for {node}: {e}")
                intros = {}
            for title, intro in intros.items():
                if title_key(title) in skip_titles:     # a redirect landed on a known article
                    continue
                for p in paragraphs(intro, limit=paras_per_article):
                    cands.append({"text": p, "source": "wikipedia_harvest", "lang": "en",
                                  "article_title": f"en:{title}", "source_article": title,
                                  "seed_node": node})
        if n % 10 == 0 or n == len(nodes):
            log(f"    {n}/{len(nodes)} nodes searched, {len(cands):,} paragraphs so far "
                f"({wiki.requests} requests)")
    return cands


def translation_candidates(wiki: Wikipedia, accepted: List[dict], langs: List[str],
                           paras_per_article: int, log=print) -> List[dict]:
    """Lead paragraphs of the other-language versions of accepted English
    articles. Each keeps the English article's label and concept (its
    source_article), so it can never cross into the holdout."""
    by_title: Dict[str, dict] = {}
    for a in accepted:
        if a.get("source") == "wikipedia_harvest" and a.get("lang") == "en":
            prev = by_title.get(a["source_article"])
            if prev is None or a["confidence"] > prev["confidence"]:
                by_title[a["source_article"]] = a
    if not by_title or not langs:
        return []
    links = wiki.langlinks(list(by_title), set(langs))
    per_lang: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
    for en_title, local in links.items():
        base = by_title.get(en_title)
        if base:
            for lang, local_title in local.items():
                per_lang[lang].append((local_title, base))
    out: List[dict] = []
    for lang, items in per_lang.items():
        try:
            intros = wiki.intros([t for t, _ in items], lang=lang)
        except Exception as e:
            log(f"    {lang} fetch failed: {e}")
            continue
        for local_title, base in items:
            for p in paragraphs(intros.get(local_title, ""), limit=paras_per_article):
                out.append({"text": p, "source": "wikipedia_harvest", "lang": lang,
                            "article_title": f"{lang}:{local_title}", "source_article": base["source_article"],
                            "seed_node": base.get("seed_node"), "category": base["category"],
                            "translated_from": "en"})
    log(f"    {len(out):,} paragraphs from {len(per_lang)} other languages")
    return out


def inbox_candidates(inbox: Path) -> Tuple[List[dict], List[Path]]:
    files = sorted(p for p in inbox.glob("*") if p.is_file() and p.suffix.lower() in (".txt", ".jsonl"))
    cands: List[dict] = []
    for f in files:
        if f.suffix.lower() == ".txt":
            texts = [(f.read_text(encoding="utf-8", errors="replace"), None)]
        else:
            texts = []
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    r = json.loads(line)
                    texts.append((r.get("text", ""), r.get("lang")))
        for text, lang in texts:
            text = re.sub(r"\s+", " ", text or "").strip()
            if len(text) >= 40:
                cands.append({"text": text[:8000], "source": f"inbox:{f.name}", "lang": lang or "und",
                              "article_title": None, "source_article": None, "seed_node": None})
    return cands, files


def archive_inbox(files: List[Path], inbox: Path, round_name: str) -> None:
    dest = inbox / "processed" / round_name
    for f in files:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest / f.name))


def known_titles(records: Iterable[dict]) -> Set[str]:
    """title_key of every Wikipedia article the records came from."""
    out: Set[str] = set()
    for r in records:
        c = concept(r)
        if c:
            out.add(title_key(c))
        t = english_title(r)
        if t:
            out.add(title_key(t))
    return out
