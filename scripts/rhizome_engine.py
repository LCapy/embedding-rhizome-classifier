#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_benchmark_full_taxonomy.py - Full 238-node taxonomy benchmark.

DATA SOURCES (7 total)
──────────────────────
  1. Wikipedia (reuse + fresh fetch)   67 nodes  L0-L2 academic
  2. Simple English Wikipedia          fills L3 scientific gaps  
  3. Bitext corpus                    ~60 nodes  conversational pairs
  4. CFPB corpus                      ~30 nodes  consumer financial
  5. DailyDialog                      ~40 nodes  daily-life dialogues
  6. MultiWOZ                         ~20 nodes  task-oriented dialogues
  7. Anchors (WordNet+Wiktionary)      all nodes  definitional fallback

CENTRALIZED EMBEDDING
─────────────────────
  sentences.json   unified record list from all sources
  embeddings.npz   single (N,768) array, row i = LaBSE(record i)
  Both files always in sync. Source field tracks provenance only.

USAGE
─────
  python 12_benchmark_full_taxonomy.py collect \\
    --taxonomy-txt TAXONOMY_fixed.txt \\
    --wiki-dir outputs/full_taxonomy \\
    --bitext-dir outputs/real_sources \\
    --cfpb-dir outputs/real_sources_cfpb \\
    --dailydialog-dir data/dailydialog \\
    --multiwoz-dir data/multiwoz \\
    --anchor-dir anchors \\
    --out outputs/taxonomy_238

  python 12_benchmark_full_taxonomy.py embed --data outputs/taxonomy_238

  python 12_benchmark_full_taxonomy.py build \\
    --taxonomy-txt TAXONOMY_fixed.txt \\
    --data outputs/taxonomy_238 --out coredrill_238

  python 12_benchmark_full_taxonomy.py predict \\
    --coredrill coredrill_238/coredrill_hierarchical.json \\
    --batch "Can I get a refund?" "Higgs boson gives particles mass."
"""
from __future__ import annotations

import argparse, json, logging, re, sys, time, urllib.parse, warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*not sharded.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

# Nodes excluded from ALL scoring and centroid updates.
# These have sigma > 0.70 - their centroids are so diffuse they sit near
# the center of S^767 and score highly against everything.
# Deleuze: the rhizome has no center. These nodes ARE the center.
# They are not territories - they are the void between territories.
# Prohibited from prediction moving forward.
SKIP_NODES: set = {
    # Original - void centroids
    "Dialogue Scene",
    "Request–Response Dialogue",
    # Collapsed (sigma > 0.70) - territory lost coherence
    "Instruction & Procedure",
    "Home Repair Request",
    "Household Maintenance",
    "Empathy & Support",
    "Emotion & Social Bonding",
    "Transactions",
    # SGD bulk - task dialogues, not semantic territories
    "Domestic Planning",
    "Task Assignment",
    "Move",
    "Visual Arts",
    "Trip Planning",
    "Restaurant & Ordering",
    "Ordering at a Restaurant",
    "Home Life",
    "Repair",
    "Jobs & Professions",
    "Commuting",
    # Non-wiki bulk - Circa/Dolly single-action nodes, no coherence
    "Report",
    "Agree",
    "Clarify",
    "Meal Preparation",
    # Anchor-only with high sigma - misleading centroids
    "Meaning Relations",
    "Agreement Formation",
    "Project Agreement",
}

# ══════════════════════════════════════════════════════════════════════════════
# § RFF - Random Fourier Features for Rhizome-Centroid (v3)
# ══════════════════════════════════════════════════════════════════════════════

class _RFF:
    """
    Reconstructed RFF object from coredrill_v3.json parameters.
    Used at inference to compute phi(x) for KME scoring.
    """
    def __init__(self, params: dict):
        self.input_dim = params["input_dim"]
        self.rff_dim   = params["rff_dim"]
        self.sigma     = params["sigma"]
        self.W         = np.array(params["W"], dtype=np.float32)
        self.b         = np.array(params["b"], dtype=np.float32)
        self.scale     = np.float32(params["scale"])

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        proj = x @ self.W.T + self.b
        return self.scale * np.cos(proj)

    def transform(self, X: np.ndarray) -> np.ndarray:
        proj = X @ self.W.T + self.b[None, :]
        return self.scale * np.cos(proj)


def _get_rff(coredrill: dict) -> Optional["_RFF"]:
    """Return RFF object if this is a v3 coredrill, else None."""
    params = coredrill.get("rff")
    if params is None:
        return None
    if not hasattr(_get_rff, "_cache"):
        _get_rff._cache = {}
    key = id(coredrill)
    if key not in _get_rff._cache:
        _get_rff._cache[key] = _RFF(params)
    return _get_rff._cache[key]


def _is_v3(coredrill: dict) -> bool:
    return coredrill.get("version", "").startswith("3")




# ══════════════════════════════════════════════════════════════════════════════
# § 1  Taxonomy
# ══════════════════════════════════════════════════════════════════════════════

def load_taxonomy(path: Path) -> List[dict]:
    src = path.read_text(encoding="utf-8").replace(": null", ": None")
    m = re.search(r"TAXONOMY:\s*List\[dict\]\s*=\s*(\[.*\])", src, re.DOTALL)
    if not m:
        sys.exit(f"[ERROR] Cannot parse taxonomy from {path}")
    try:
        raw = eval(m.group(1))
    except Exception as e:
        sys.exit(f"[ERROR] {e}")

    norm: List[dict] = []
    seen = set()
    for node in raw:
        if not isinstance(node, dict) or "name" not in node:
            continue
        n = dict(node)
        name = str(n["name"]).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        parents = n.get("parents")
        if parents is None:
            parent = n.get("parent")
            parents = [] if parent in (None, "", []) else [parent]
        elif not isinstance(parents, list):
            parents = [parents]
        parents = [str(x).strip() for x in parents if str(x).strip()]
        n["parents"] = parents
        n["parent"] = parents[0] if parents else None
        n["primary_parent"] = n["parent"]
        n["level"] = int(n.get("level", 0))
        norm.append(n)
    return norm

def taxonomy_index(taxonomy: List[dict]) -> Dict[str, dict]:
    return {n["name"]: n for n in taxonomy}


def taxonomy_parents_map(taxonomy: List[dict]) -> Dict[str, List[str]]:
    return {n["name"]: list(n.get("parents", [])) for n in taxonomy}


def taxonomy_primary_parent_map(taxonomy: List[dict]) -> Dict[str, Optional[str]]:
    return {n["name"]: (n.get("parents", [None])[0] if n.get("parents") else None) for n in taxonomy}


def taxonomy_children_map(taxonomy: List[dict]) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = defaultdict(list)
    for n in taxonomy:
        child = n["name"]
        for p in n.get("parents", []):
            children[p].append(child)
    return {k: sorted(set(v)) for k, v in children.items()}


def taxonomy_levels_map(taxonomy: List[dict]) -> Dict[int, List[str]]:
    levels: Dict[int, List[str]] = defaultdict(list)
    for n in taxonomy:
        levels[int(n["level"])].append(n["name"])
    return {lv: names for lv, names in sorted(levels.items())}


def taxonomy_root_nodes(taxonomy: List[dict]) -> List[str]:
    return [n["name"] for n in taxonomy if not n.get("parents")]


def taxonomy_lineage_paths(node: str, parents_map: Dict[str, List[str]], memo: Optional[Dict[str, List[List[str]]]] = None) -> List[List[str]]:
    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]
    parents = parents_map.get(node, [])
    if not parents:
        memo[node] = [[node]]
        return memo[node]
    out: List[List[str]] = []
    for p in parents:
        for path in taxonomy_lineage_paths(p, parents_map, memo):
            out.append(path + [node])
    uniq = []
    seen = set()
    for path in out:
        t = tuple(path)
        if t not in seen:
            seen.add(t)
            uniq.append(path)
    memo[node] = uniq
    return uniq


def taxonomy_ancestor_set(node: str, parents_map: Dict[str, List[str]]) -> Set[str]:
    seen: Set[str] = set()
    stack = list(parents_map.get(node, []))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(parents_map.get(cur, []))
    return seen


def taxonomy_top_branches(node: str, parents_map: Dict[str, List[str]]) -> List[str]:
    paths = taxonomy_lineage_paths(node, parents_map)
    return sorted(set(path[0] for path in paths if path))


def taxonomy_dag_distance(a: str, b: str, parents_map: Dict[str, List[str]], children_map: Dict[str, List[str]]) -> int:
    if a == b:
        return 0
    q = deque([(a, 0)])
    seen = {a}
    while q:
        cur, dist = q.popleft()
        neigh = list(parents_map.get(cur, [])) + list(children_map.get(cur, []))
        for nxt in neigh:
            if nxt == b:
                return dist + 1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, dist + 1))
    return 999999



# ══════════════════════════════════════════════════════════════════════════════
# § 1b  Assignment helpers
# ══════════════════════════════════════════════════════════════════════════════

def _taxonomy_level_map(taxonomy: List[dict]) -> Dict[str, int]:
    return {n["name"]: int(n["level"]) for n in taxonomy}


def _source_balance_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ps = [c / total for c in counts.values() if c > 0]
    return float(-sum(p * np.log(p + 1e-12) for p in ps))


def _resolve_category_candidate(cat: str, taxonomy: List[dict], t_index: Dict[str, dict],
                                name_lc: Dict[str, str], kw_map: Dict[str, str]) -> Optional[str]:
    cat = (cat or '').strip()
    if not cat:
        return None
    if cat in t_index:
        return cat
    if cat.lower() in name_lc:
        return name_lc[cat.lower()]
    words = set(re.findall(r"\b[a-z]{3,}\b", cat.lower()))
    if not words:
        return None
    scored = {}
    for w in words:
        cand = kw_map.get(w)
        if not cand:
            continue
        cw = set(re.findall(r"\b[a-z]{3,}\b", cand.lower()))
        sc = len(words & cw) / max(len(cw), 1)
        if sc > scored.get(cand, 0.0):
            scored[cand] = sc
    if not scored:
        return None
    best = max(scored, key=scored.get)
    return best if scored[best] >= 0.35 else None


def _pick_primary_category(candidates: List[str], taxonomy: List[dict],
                           t_index: Dict[str, dict]) -> Optional[str]:
    if not candidates:
        return None
    unique = []
    seen = set()
    for c in candidates:
        if c in t_index and c not in seen:
            seen.add(c)
            unique.append(c)
    if not unique:
        return None
    unique.sort(key=lambda c: (t_index[c]["level"], len(c)), reverse=True)
    return unique[0]


def _balanced_source_sample_indices(indices: np.ndarray, sources: np.ndarray,
                                    per_source_cap: int = 400, total_cap: int = 3000) -> np.ndarray:
    if len(indices) <= 1:
        return indices
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx in indices.tolist():
        buckets[str(sources[idx])].append(int(idx))
    sampled: List[int] = []
    for src in sorted(buckets):
        sampled.extend(buckets[src][:per_source_cap])
    if len(sampled) <= total_cap:
        return np.array(sorted(sampled), dtype=np.int32)
    # round-robin trim so one source cannot dominate
    out: List[int] = []
    pos = {s: 0 for s in buckets}
    ordered = sorted(buckets)
    while len(out) < total_cap:
        progressed = False
        for src in ordered:
            if pos[src] < min(len(buckets[src]), per_source_cap):
                out.append(buckets[src][pos[src]])
                pos[src] += 1
                progressed = True
                if len(out) >= total_cap:
                    break
        if not progressed:
            break
    return np.array(sorted(out), dtype=np.int32)




def _clip_indices(indices: np.ndarray, cap: int) -> np.ndarray:
    if indices is None:
        return np.array([], dtype=np.int32)
    arr = np.asarray(indices, dtype=np.int32)
    if cap is None or cap <= 0 or len(arr) <= cap:
        return arr
    return arr[:cap].astype(np.int32)


def _unique_concat(parts: List[np.ndarray]) -> np.ndarray:
    cleaned = []
    for p in parts:
        if p is None:
            continue
        arr = np.asarray(p, dtype=np.int32)
        if len(arr) > 0:
            cleaned.append(arr)
    if not cleaned:
        return np.array([], dtype=np.int32)
    return np.unique(np.concatenate(cleaned).astype(np.int32))

def _normalize_record(r: dict) -> dict:
    rr = dict(r)
    rr["text"] = str(rr.get("text", "")).strip()
    rr["source"] = str(rr.get("source", "unknown")).strip() or "unknown"
    rr["lang"] = str(rr.get("lang", "en")).strip() or "en"
    if "aux_categories" in rr and not isinstance(rr["aux_categories"], list):
        rr["aux_categories"] = [str(rr["aux_categories"])]
    return rr


# ══════════════════════════════════════════════════════════════════════════════
# § 2  Wikipedia article mappings (covers all 238 nodes)
# ══════════════════════════════════════════════════════════════════════════════

# Primary English Wikipedia articles per node name.
# Falls back to node["en_category"] if not listed here.
WIKI_ARTICLES: Dict[str, List[str]] = {
    # L0 - broad introductory articles for the 4 root branches
    "Natural World": ["Nature", "Natural science", "Natural environment"],
    "Human Mind & Knowledge": ["Humanities", "Cognition", "Knowledge",
                               "Human intelligence"],
    "Human Activity & Society": ["Society", "Social structure",
                                 "Human behavior", "Everyday life",
                                 "Social behavior", "Human society"],
    "Communication & Expression": ["Communication", "Language",
                                   "Human communication", "Linguistics"],
    # L1
    "Physical Reality": ["Physical science", "Matter"],
    "Life & Biology": ["Life science", "Biology"],
    "Earth & Cosmos": ["Earth science", "Astronomy"],
    "Philosophy": ["Philosophy"],
    "Formal Systems": ["Mathematics", "Logic"],
    "Social Theory": ["Social science", "Sociology"],
    "Daily Life": ["Everyday life", "Daily life", "Lifestyle (sociology)"],
    "Work & Economy": ["Labour economics", "Economy"],
    "Institutions & Governance": ["Government", "State (polity)"],
    "Politics": ["Politics", "Political science"],
    "Law & Regulation": ["Law", "Regulation"],
    "Social Relations": ["Interpersonal relationship", "Social relation",
                         "Human bonding"],
    "Language Structure": ["Linguistics", "Grammar"],
    "Creative Expression": ["Art", "Creative arts"],
    "Conflict & Cooperation": ["Conflict (process)", "Cooperation"],
    # L2
    "Physics": ["Physics"],
    "Chemistry": ["Chemistry"],
    "Biology": ["Biology", "Cell biology"],
    "Medicine": ["Medicine"],
    "Ecology": ["Ecology"],
    "Geology": ["Geology"],
    "Climate & Weather": ["Climatology", "Meteorology"],
    "Astronomy": ["Astronomy"],
    "Ethics": ["Ethics", "Moral philosophy"],
    "Epistemology": ["Epistemology"],
    "Logic": ["Logic"],
    "Mathematics": ["Mathematics"],
    "Statistics": ["Statistics"],
    "Economics": ["Economics"],
    "Sociology": ["Sociology"],
    "Government": ["Government", "Public administration"],
    "Negotiation": ["Negotiation"],
    "Collaboration": ["Collaboration"],
    "Competition": ["Competition"],
    "Syntax": ["Syntax"],
    "Semantics": ["Semantics"],
    "Pragmatics": ["Pragmatics"],
    "Literature": ["Literature"],
    "Music": ["Music"],
    # L3 scientific
    "Mechanics": ["Classical mechanics", "Mechanics"],
    "Thermodynamics": ["Thermodynamics"],
    "Quantum Mechanics": ["Quantum mechanics"],
    "Organic Chemistry": ["Organic chemistry"],
    "Inorganic Chemistry": ["Inorganic chemistry"],
    "Genetics": ["Genetics"],
    "Evolution": ["Evolution"],
    "Diagnosis": ["Medical diagnosis"],
    "Treatment": ["Therapy", "Medical treatment"],
    "Ecosystems": ["Ecosystem"],
    "Tectonics": ["Plate tectonics"],
    "Meteorology": ["Meteorology"],
    "Cosmology": ["Physical cosmology"],
    "Astrophysics": ["Astrophysics"],
    "Moral Philosophy": ["Moral philosophy", "Ethics"],
    "Knowledge & Belief": ["Epistemology", "Belief"],
    "Proof & Inference": ["Mathematical proof", "Inference"],
    "Algebra": ["Algebra"],
    "Geometry": ["Geometry"],
    "Probability & Inference": ["Probability", "Statistical inference"],
    "Macroeconomics": ["Macroeconomics"],
    "Microeconomics": ["Microeconomics"],
    "Climate Change": ["Climate change"],
    "Entropy": ["Entropy"],
    "DNA Inheritance": ["DNA", "Heredity"],
    "Natural Selection": ["Natural selection"],
    "Big Bang Model": ["Big Bang"],
    "Bargaining": ["Bargaining", "Negotiation"],
    "Debate": ["Debate"],
    "Poetry": ["Poetry"],
    "Fiction": ["Fiction"],
    "Song": ["Song", "Vocal music"],
    # L4/L5 with Wikipedia articles
    "Adaptation": ["Adaptation (biology)"],
    "Travel": ["Travel"],
    "Universe Expansion": ["Metric expansion of space"],
    "Measurement": ["Measurement"],
}

# Simple English Wikipedia for scientific L3/L4 nodes (cleaner short paras)
SIMPLE_WIKI: Dict[str, List[str]] = {
    "Mechanics": ["Mechanics"],
    "Thermodynamics": ["Thermodynamics"],
    "Entropy": ["Entropy"],
    "Quantum Mechanics": ["Quantum mechanics"],
    "Ecosystems": ["Ecosystem"],
    "Tectonics": ["Plate tectonics"],
    "Meteorology": ["Meteorology"],
    "Cosmology": ["Cosmology"],
    "Astrophysics": ["Astrophysics"],
    "Algebra": ["Algebra"],
    "Geometry": ["Geometry"],
    "Probability & Inference": ["Probability"],
    "Macroeconomics": ["Macroeconomics"],
    "Microeconomics": ["Microeconomics"],
    "Knowledge & Belief": ["Epistemology"],
    "Proof & Inference": ["Mathematical proof"],
    "Moral Philosophy": ["Ethics"],
    "DNA Inheritance": ["DNA"],
    "Natural Selection": ["Natural selection"],
    "Big Bang Model": ["Big Bang"],
    "Climate Change": ["Climate change"],
    "Genetics": ["Genetics"],
    "Evolution": ["Evolution"],
    "Bargaining": ["Negotiation"],
    "Debate": ["Debate"],
    "Poetry": ["Poetry"],
    "Fiction": ["Fiction"],
    "Song": ["Song"],
}


# ══════════════════════════════════════════════════════════════════════════════
# § 3  DailyDialog mappings
# ══════════════════════════════════════════════════════════════════════════════

DAILYDIALOG_TOPIC: Dict[int, List[str]] = {
    1:  ["Daily Life", "Home Life", "Meal Preparation", "Domestic Planning"],
    2:  ["Education & Training", "Work & Economy"],
    3:  ["Arts", "Literature", "Creative Expression"],
    4:  ["Emotion Expression", "Affection & Reassurance", "Social Relations"],
    5:  ["Social Relations", "Romance & Intimacy", "Family Life",
         "Parent\u2013Child Interaction"],
    6:  ["Trip Planning", "Transportation & Travel"],
    7:  ["Health Routine", "Appointments & Medication", "Medicine"],
    8:  ["Work & Economy", "Workplace Coordination", "Business Operations"],
    9:  ["Politics", "Policy Proposal", "Elections & Campaigns"],
    10: ["Work & Economy", "Billing & Refunds", "Purchase & Payment"],
}

DAILYDIALOG_ACT: Dict[int, List[str]] = {
    1: ["Narrative & Reportage", "Event Narrative"],
    2: ["Question\u2013Answer", "Information Seeking", "Clarifying Question"],
    3: ["Instruction & Procedure", "How-to Guidance", "Request"],
    4: ["Agreement Formation", "Confirm"],
}


# ══════════════════════════════════════════════════════════════════════════════
# § 4  MultiWOZ mappings
# ══════════════════════════════════════════════════════════════════════════════

MULTIWOZ_DOMAIN: Dict[str, List[str]] = {
    "restaurant": ["Restaurant & Ordering", "Ordering at a Restaurant",
                   "Purchase & Payment"],
    "hotel":      ["Trip Planning", "Domestic Planning"],
    "attraction": ["Trip Planning", "Transportation & Travel"],
    "taxi":       ["Commuting", "Transportation & Travel"],
    "train":      ["Commuting", "Transportation & Travel"],
    "bus":        ["Commuting", "Transportation & Travel"],
    "hospital":   ["Appointments & Medication", "Health Routine", "Medicine"],
    "police":     ["Rights & Obligations", "Public Services", "Law & Regulation"],
    "shop":       ["Purchase & Payment", "Shopping & Consumption"],
}


# ══════════════════════════════════════════════════════════════════════════════
# § 5  Wikipedia fetcher
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_wiki(title: str, simple: bool = False) -> List[str]:
    import requests
    sub = "simple" if simple else "en"
    url = f"https://{sub}.wikipedia.org/wiki/" + urllib.parse.quote(
        title.replace(" ", "_"))
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        if r.status_code != 200:
            return []
        html = r.text
    except Exception as e:
        logging.warning("Wikipedia fetch failed: %s", e)
        return []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for t in soup.find_all(["table", "sup", "span.mw-editsection"]):
            t.decompose()
        content = soup.find("div", {"id": "mw-content-text"}) or soup
        raws = [p.get_text(" ", strip=True) for p in content.find_all("p")]
    except ImportError:
        raws = [re.sub(r"<[^>]+>", " ", m.group(1))
                for m in re.finditer(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)]
    paras = []
    for text in raws:
        text = re.sub(r"\[\d+\]|\[note \d+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if 100 <= len(text) <= 1200:
            paras.append(text)
        elif len(text) > 1200:
            sents = re.split(r"(?<=[.!?])\s+", text)
            chunk = ""
            for s in sents:
                if len(chunk) + len(s) <= 1200:
                    chunk = (chunk + " " + s).strip()
                else:
                    if len(chunk) >= 100:
                        paras.append(chunk)
                    chunk = s
            if len(chunk) >= 100:
                paras.append(chunk)
    return paras


def collect_wikipedia(taxonomy: List[dict], existing_cats: Set[str],
                      existing_counts: Optional[Dict[str, int]] = None,
                      delay: float = 0.35, min_existing_before_skip: int = 80) -> List[dict]:
    t_index = taxonomy_index(taxonomy)
    records = []
    to_fetch = []
    for node in taxonomy:
        name = node["name"]
        if name in existing_cats and existing_counts and existing_counts.get(name, 0) >= min_existing_before_skip:
            continue
        arts_en = WIKI_ARTICLES.get(name, [])
        if not arts_en:
            en_cat = node.get("en_category", "")
            if en_cat and en_cat != name:
                arts_en = [en_cat]
        arts_sw = SIMPLE_WIKI.get(name, [])
        if arts_en or arts_sw:
            to_fetch.append((name, arts_en, arts_sw, node))

    print(f"  Wikipedia: {len(to_fetch)} nodes to fetch")
    for i, (name, arts_en, arts_sw, node) in enumerate(to_fetch, 1):
        print(f"  [{i:>3}/{len(to_fetch)}] {name}", end=" ", flush=True)
        n = 0
        for title in arts_en[:2]:
            for p in _fetch_wiki(title, simple=False):
                records.append({"text": p, "category": name,
                                 "level": node["level"],
                                 "parent": node.get("parent"),
                                 "lang": "en", "source": "wikipedia",
                                 "article_title": title})
                n += 1
            time.sleep(delay)
        for title in arts_sw[:1]:
            for p in _fetch_wiki(title, simple=True):
                records.append({"text": p, "category": name,
                                 "level": node["level"],
                                 "parent": node.get("parent"),
                                 "lang": "en", "source": "simple_wikipedia",
                                 "article_title": f"simple:{title}"})
                n += 1
            time.sleep(delay)
        print(f"-> {n}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# § 6  Generic source dir loader
# ══════════════════════════════════════════════════════════════════════════════

def load_source_dir(source_dir: Path, source_name: str) -> List[dict]:
    """
    Load from source dir. Handles sentences.json, records.jsonl, *.jsonl,
    subdirs, TSV/CSV.

    Field normalization applied to every record:
      - 'node' field used as 'category' if 'category' missing
      - 'snippet' used as 'text' if 'text' missing
      - template placeholders {{...}} stripped from text
      - 'Customer Service' node remapped using 'label' field
      - 'Transactions' remapped to 'Purchase & Payment'
    """
    if not source_dir or not source_dir.exists():
        return []

    # Bitext label -> taxonomy node (for records where node='Customer Service')
    LABEL_TO_NODE: Dict[str, str] = {
        "cancel_order":       "Ordering & Delivery",
        "change_order":       "Ordering & Delivery",
        "track_order":        "Ordering & Delivery",
        "place_order":        "Ordering & Delivery",
        "delivery_options":   "Ordering & Delivery",
        "delivery_period":    "Ordering & Delivery",
        "refund_request":     "Refund Processing",
        "refund":             "Refund Processing",
        "get_refund":         "Refund Processing",
        "check_refund_policy":"Refund Processing",
        "complaint":          "Complaint Handling",
        "review":             "Complaint Handling",
        "payment_issue":      "Billing & Refunds",
        "check_invoice":      "Billing & Refunds",
        "billing":            "Billing & Refunds",
        "check_payment_methods": "Purchase & Payment",
        "contact_customer_service": "Complaint Handling",
        "contact_human_agent":  "Complaint Handling",
    }

    # Node name remaps (exact node name in data → taxonomy node name)
    NODE_REMAP: Dict[str, str] = {
        "Customer Service":    None,   # use label field instead
        "Transactions":        "Purchase & Payment",
        "Conversation & Dialogue": "Conversation & Dialogue",  # keep if in taxonomy
    }

    def _norm(r: dict, default_cat: str = "") -> Optional[dict]:
        """
        Normalize a record: resolve text, category, clean placeholders.
        Returns None if the record should be skipped.
        """
        import re as _re

        # Resolve text field - try text, then snippet
        text = r.get("text", "").strip()
        if not text:
            text = r.get("snippet", "").strip()
        if not text or len(text) < 10:
            return None

        # Strip template placeholders like {{Order Number}}
        text = _re.sub(r"\{\{[^}]+\}\}", "", text).strip()
        text = _re.sub(r"\s+", " ", text).strip()
        if len(text) < 10:
            return None

        # Resolve category - try category, then node
        cat = r.get("category", "").strip() or r.get("node", default_cat).strip()

        # Apply node remaps
        if cat in NODE_REMAP:
            remapped = NODE_REMAP[cat]
            if remapped is None:
                # Use label field to get a finer-grained category
                label = r.get("label", "").lower().replace(" ", "_")
                cat = LABEL_TO_NODE.get(label, "")
                if not cat:
                    # Try partial label match
                    for k, v in LABEL_TO_NODE.items():
                        if k in label or label in k:
                            cat = v
                            break
            else:
                cat = remapped

        return {"text": text, "category": cat, "source": source_name,
                "lang": r.get("lang", "en"),
                "label": r.get("label", "")}

    def _load_jsonl(path: Path, default_cat: str = "") -> List[dict]:
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = _norm(json.loads(line), default_cat)
                if r:
                    records.append(r)
            except Exception as e:
                logging.warning("Skipped malformed Bitext record: %s", e)
        return records

    # Top-level sentences.json
    p = source_dir / "sentences.json"
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw and ("text" in raw[0] or "snippet" in raw[0]):
            data = [r for r in (_norm(x) for x in raw) if r]
            print(f"  {source_name}: {len(data):,} records (sentences.json)")
            return data

    # Top-level *.jsonl (catches records.jsonl directly)
    jsonl_top = list(source_dir.glob("*.jsonl"))
    if jsonl_top:
        records = []
        for jl in jsonl_top:
            records.extend(_load_jsonl(jl))
        if records:
            names = ", ".join(f.name for f in jsonl_top)
            print(f"  {source_name}: {len(records):,} records ({names})")
            return records

    records = []
    # Subdirectory per category
    for sub in sorted(source_dir.iterdir()):
        if not sub.is_dir():
            continue
        cat = sub.name
        sj = sub / "sentences.json"
        if sj.exists():
            raw = json.loads(sj.read_text(encoding="utf-8"))
            records.extend(r for r in (_norm(x, cat) for x in raw) if r)
            continue
        for jl in sub.glob("*.jsonl"):
            records.extend(_load_jsonl(jl, cat))
        for txt in sub.glob("*.txt"):
            for line in txt.read_text(encoding="utf-8",
                                       errors="replace").splitlines():
                line = line.strip()
                if len(line) > 30:
                    records.append({"text": line, "category": cat,
                                    "source": source_name, "lang": "en"})
    if records:
        print(f"  {source_name}: {len(records):,} records (subdirs)")
        return records

    # TSV / CSV
    for f in list(source_dir.glob("*.tsv")) + list(source_dir.glob("*.csv")):
        sep = "\t" if f.suffix == ".tsv" else ","
        rows = f.read_text(encoding="utf-8", errors="replace").splitlines()
        if not rows:
            continue
        hdr = [h.lower().strip() for h in rows[0].split(sep)]
        ti  = next((i for i, h in enumerate(hdr)
                    if h in ("text","sentence","content","utterance","snippet")), 0)
        ci  = next((i for i, h in enumerate(hdr)
                    if h in ("category","label","class","domain",
                             "intent","topic","node")), -1)
        for row in rows[1:]:
            parts = row.split(sep)
            if len(parts) <= ti:
                continue
            text = parts[ti].strip().strip('"')
            cat  = parts[ci].strip() if ci >= 0 and len(parts) > ci else ""
            if len(text) > 20:
                records.append({"text": text, "category": cat,
                                 "source": source_name, "lang": "en"})
        if records:
            print(f"  {source_name}: {len(records):,} records ({f.name})")
            return records

    print(f"  {source_name}: no data found in {source_dir}")
    return []
def load_dailydialog(dd_dir: Path) -> List[dict]:
    if not dd_dir or not dd_dir.exists():
        return []
    records = []

    # HuggingFace JSON format
    for jf in dd_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for item in data:
                utts   = item.get("dialog", item.get("utterances", []))
                topics = item.get("topic", [])
                acts   = item.get("act", item.get("dialog_act", []))
                if isinstance(topics, int):
                    topics = [topics] * len(utts)
                if isinstance(acts, int):
                    acts = [acts] * len(utts)
                for i, utt in enumerate(utts):
                    text = (utt.get("text", utt) if isinstance(utt, dict)
                            else str(utt)).strip()
                    if len(text) < 10:
                        continue
                    tid = int(topics[i]) if i < len(topics) else 0
                    aid = int(acts[i])   if i < len(acts)   else 0
                    cats = set(DAILYDIALOG_TOPIC.get(tid, []) +
                               DAILYDIALOG_ACT.get(aid, []))
                    for cat in cats:
                        records.append({"text": text, "category": cat,
                                        "source": "dailydialog", "lang": "en"})
        except Exception as e:
            logging.warning("DailyDialog parse error: %s", e)
    if records:
        print(f"  DailyDialog: {len(records):,} records (HF JSON)")
        return records

    # Original text format
    def _find(pat):
        found = list(dd_dir.rglob(pat))
        return found[0] if found else None

    for split in ("train", "validation", "test"):
        dia_f   = _find(f"dialogues_{split}.txt") or _find("dialogues_*.txt")
        act_f   = _find(f"dialogues_act_{split}.txt") or _find("dialogues_act_*.txt")
        topic_f = _find(f"dialogues_topic_{split}.txt") or _find("dialogues_topic_*.txt")
        if not dia_f:
            continue
        dialogues = dia_f.read_text(encoding="utf-8").splitlines()
        acts_l   = act_f.read_text(encoding="utf-8").splitlines()   if act_f   else []
        topics_l = topic_f.read_text(encoding="utf-8").splitlines() if topic_f else []
        for di, dline in enumerate(dialogues):
            utts    = [u.strip() for u in dline.split("__eou__") if u.strip()]
            act_ids = ([int(a) for a in acts_l[di].strip().split()]
                       if di < len(acts_l) else [])
            tid     = int(topics_l[di].strip()) if di < len(topics_l) else 0
            tcats   = DAILYDIALOG_TOPIC.get(tid, [])
            for ui, utt in enumerate(utts):
                if len(utt) < 10:
                    continue
                aid   = act_ids[ui] if ui < len(act_ids) else 0
                acats = DAILYDIALOG_ACT.get(aid, [])
                for cat in set(tcats + acats):
                    records.append({"text": utt, "category": cat,
                                    "source": "dailydialog", "lang": "en"})
        break
    print(f"  DailyDialog: {len(records):,} records (original format)")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# § 8  MultiWOZ parser
# ══════════════════════════════════════════════════════════════════════════════

def load_multiwoz(mwoz_dir: Path) -> List[dict]:
    if not mwoz_dir or not mwoz_dir.exists():
        return []
    records = []

    def _add(dial, domains):
        cats = []
        for d in domains:
            cats.extend(MULTIWOZ_DOMAIN.get(d.lower(), []))
        cats = list(set(cats))
        if not cats:
            return
        turns = dial.get("turns", dial.get("log", []))
        for turn in turns:
            text = (turn.get("utterance", turn.get("text", ""))
                    if isinstance(turn, dict) else str(turn)).strip()
            if len(text) < 10:
                continue
            for cat in cats:
                records.append({"text": text, "category": cat,
                                 "source": "multiwoz", "lang": "en"})

    data_json = mwoz_dir / "data.json"
    if data_json.exists():
        data = json.loads(data_json.read_text(encoding="utf-8"))
        for did, dial in data.items():
            domains = [k for k in dial.get("goal", {})
                       if k not in ("message", "topic") and
                       isinstance(dial["goal"].get(k), dict) and
                       dial["goal"][k]]
            _add(dial, domains)
        print(f"  MultiWOZ: {len(records):,} records (v2.0)")
        return records

    for jf in sorted(mwoz_dir.rglob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for dial in data:
                _add(dial, dial.get("services", dial.get("domains", [])))
        except Exception as e:
            logging.warning("MultiWOZ parse error: %s", e)
        if records:
            break
    print(f"  MultiWOZ: {len(records):,} records")
    return records



# ══════════════════════════════════════════════════════════════════════════════
# § 8b  Additional dataset parsers (SGD, Empathetic, Taskmaster, Circa, Dolly,
#        Persuasion) - all load from Arrow format saved by download_datasets.py
# ══════════════════════════════════════════════════════════════════════════════

# SGD domain → taxonomy node mapping
SGD_DOMAIN_MAP: Dict[str, List[str]] = {
    "Alarm":           ["Domestic Planning", "Task Assignment"],
    "Banks":           ["Billing & Refunds", "Purchase & Payment"],
    "Buses":           ["Commuting", "Transportation & Travel"],
    "Calendar":        ["Task Assignment", "Domestic Planning"],
    "Events":          ["Trip Planning", "Social Relations"],
    "Flights":         ["Trip Planning", "Transportation & Travel"],
    "Hotels":          ["Trip Planning"],
    "Homes":           ["Home Life"],
    "Media":           ["Creative Expression"],
    "Messaging":       ["Social Relations", "Task Assignment"],
    "Movies":          ["Arts", "Creative Expression"],
    "Music":           ["Music"],
    "RentalCars":      ["Commuting", "Repair"],
    "Restaurants":     ["Restaurant & Ordering", "Ordering at a Restaurant"],
    "RideSharing":     ["Commuting", "Move"],
    "Services":        ["Jobs & Professions", "Work & Economy"],
    "Trains":          ["Commuting", "Transportation & Travel"],
    "Travel":          ["Trip Planning", "Transportation & Travel"],
    "Medical":         ["Doctor Appointment", "Appointments & Medication"],
    "Weather":         ["Climate & Weather"],
    "Sport":           ["Daily Life"],
}

# Empathetic situation → taxonomy node mapping
EMPATHETIC_EMOTION_MAP: Dict[str, List[str]] = {
    "afraid":      ["Emotional Reassurance", "Reassure"],
    "anxious":     ["Emotional Reassurance", "Reassure"],
    "apprehensive":["Emotional Reassurance"],
    "sad":         ["Comfort", "Emotional Reassurance"],
    "lonely":      ["Comfort", "Social Relations"],
    "devastated":  ["Comfort", "Reassure"],
    "angry":       ["Verbal Dispute", "Emotion Expression"],
    "furious":     ["Verbal Dispute"],
    "annoyed":     ["Complaint Handling", "Verbal Dispute"],
    "disgusted":   ["Verbal Dispute"],
    "excited":     ["Social Relations"],
    "joyful":      ["Social Relations"],
    "grateful":    ["Social Relations"],
    "proud":       ["Social Relations"],
    "hopeful":     ["Emotional Reassurance"],
    "content":     ["Social Relations"],
    "embarrassed": ["Emotional Reassurance", "Comfort"],
    "ashamed":     ["Emotional Reassurance"],
    "guilty":      ["Emotional Reassurance"],
    "surprised":   ["Social Relations"],
    "trusting":    ["Social Relations"],
    "anticipating":["Planning & Scheduling"],
}

# Taskmaster domain → taxonomy node
TASKMASTER_DOMAIN_MAP: Dict[str, List[str]] = {
    "movie-ticket":  ["Ordering at a Restaurant", "Purchase & Payment"],
    "restaurant":    ["Restaurant & Ordering", "Ordering at a Restaurant"],
    "pizza":         ["Restaurant & Ordering", "Meal Preparation"],
    "coffee":        ["Restaurant & Ordering", "Purchase & Payment"],
    "uber":          ["Commuting", "Move"],
    "ride":          ["Commuting", "Move"],
    "auto-repair":   ["Repair"],
    "repair":        ["Repair"],
    "dmv":           ["Permit or Document Process", "Public Administration"],
    "sports":        ["Daily Life"],
    "hotel":         ["Trip Planning"],
    "flight":        ["Trip Planning", "Transportation & Travel"],
    "music":         ["Music"],
    "e-commerce":    ["Ordering & Delivery", "Late Delivery Problem",
                      "Double Charge Complaint", "Refund Processing"],
    "insurance":     ["Rights & Obligations", "Billing & Refunds"],
}

# Dolly category → taxonomy node
DOLLY_CATEGORY_MAP: Dict[str, List[str]] = {
    "open_qa":             ["Information Seeking", "Question–Answer"],
    "closed_qa":           ["Information Seeking", "Clarification Exchange"],
    "information_extraction": ["Report", "Information Seeking"],
    "general_information": ["Information Seeking"],
    "classification":      ["Report"],
    "summarization":       ["Report", "Narrative & Reportage"],
    "brainstorming":       ["Recommendation Dialogue", "Choice Recommendation"],
    "creative_writing":    ["Fiction", "Creative Expression"],
    "how_to":              ["How-to Guidance", "Guide", "Instruct"],
    "code_generation":     ["How-to Guidance"],
}


def _try_arrow(data_dir: Path) -> Optional[object]:
    """Try loading as Arrow/HuggingFace disk format. Returns None if not Arrow."""
    try:
        from datasets import load_from_disk
        return load_from_disk(str(data_dir))
    except Exception as e:
        logging.debug("Could not load Arrow dataset from %s: %s", data_dir, e)
        return None


def load_sgd(sgd_dir: Path) -> List[dict]:
    """
    Parse SGD dataset. Handles two formats:
    - Arrow format (HuggingFace disk cache)
    - Raw JSON from GitHub ZIP: dialogues_*.json files with
      {dialogue_id, services, turns:[{speaker, utterance}]}
    """
    if not sgd_dir or not sgd_dir.exists():
        return []

    records = []

    # Try Arrow format first
    ds = _try_arrow(sgd_dir)
    if ds is not None:
        for split_name in ds:
            for item in ds[split_name]:
                domains = item.get("domains", item.get("services", []))
                if isinstance(domains, str):
                    domains = [domains]
                cats = []
                for d in domains:
                    for k, v in SGD_DOMAIN_MAP.items():
                        if k.lower() in d.lower() or d.lower() in k.lower():
                            cats.extend(v)
                cats = list(set(cats))
                if not cats:
                    continue
                turns = item.get("dialogue", item.get("turns", []))
                for turn in (turns if isinstance(turns, list) else []):
                    text = (turn.get("content", turn.get("utterance",
                            turn.get("text", ""))) if isinstance(turn, dict)
                            else str(turn)).strip()
                    if len(text) >= 10:
                        for cat in cats:
                            records.append({"text": text, "category": cat,
                                            "source": "sgd", "lang": "en"})
        print(f"  SGD: {len(records):,} records (Arrow)")
        return records

    # Raw JSON format from GitHub ZIP
    # Walk all subdirs looking for dialogues_*.json files
    for json_file in sorted(sgd_dir.rglob("dialogues_*.json")):
        try:
            dialogues = json.loads(json_file.read_text(encoding="utf-8"))
            for dial in dialogues:
                services = dial.get("services", [])
                cats = []
                for svc in services:
                    for k, v in SGD_DOMAIN_MAP.items():
                        if k.lower() in svc.lower() or svc.lower() in k.lower():
                            cats.extend(v)
                cats = list(set(cats))
                if not cats:
                    continue
                for turn in dial.get("turns", []):
                    text = turn.get("utterance", "").strip()
                    if len(text) >= 10:
                        for cat in cats:
                            records.append({"text": text, "category": cat,
                                            "source": "sgd", "lang": "en"})
        except Exception as e:
            logging.warning("SGD parse error: %s", e)

    print(f"  SGD: {len(records):,} records (raw JSON)")
    return records


def load_empathetic(emp_dir: Path) -> List[dict]:
    """
    Parse EmpatheticDialogues. Handles:
    - Arrow format with 'context'+'utterance' fields (bdotloh mirror)
    - Arrow format with 'prompt'+'free_messages' (facebook original)
    - Raw CSV from tar.gz: train.csv with conv_id, utterance_idx, context,
                           prompt, selfeval, tags, utterance
    """
    if not emp_dir or not emp_dir.exists():
        return []

    records = []

    def _emotion_cats(emotion: str) -> List[str]:
        em = emotion.lower().strip()
        cats = EMPATHETIC_EMOTION_MAP.get(em, [])
        if not cats:
            for k, v in EMPATHETIC_EMOTION_MAP.items():
                if k in em:
                    return v
        return cats

    def _add(text: str, emotion: str) -> None:
        cats = _emotion_cats(emotion)
        for cat in cats:
            records.append({"text": text, "category": cat,
                             "source": "empathetic", "lang": "en"})

    # Try Arrow first
    ds = _try_arrow(emp_dir)
    if ds is not None:
        first_split = next(iter(ds))
        sample = ds[first_split][0] if len(ds[first_split]) > 0 else {}
        fields = set(sample.keys())

        for split_name in ds:
            for item in ds[split_name]:
                # bdotloh schema: emotion=label, situation=text
                # facebook schema: context=label, prompt=text, free_messages=list
                # Try all known emotion field names
                emotion = (item.get("emotion") or item.get("context") or
                           item.get("label") or "").strip()

                # Try all known text field names
                # bdotloh: 'situation' is the utterance text
                for text_field in ("situation", "utterance", "prompt", "text"):
                    text = item.get(text_field, "")
                    if isinstance(text, str) and len(text.strip()) >= 10:
                        _add(text.strip(), emotion)
                        break

                # facebook original also has list of response messages
                for msg in item.get("free_messages", []):
                    text = (msg if isinstance(msg, str)
                            else msg.get("text", "")).strip()
                    if len(text) >= 10:
                        _add(text, emotion)

                # bdotloh may also have 'utterances' list
                for utt in item.get("utterances", []):
                    text = (utt if isinstance(utt, str)
                            else utt.get("text", utt.get("utterance", ""))
                            ).strip()
                    if len(text) >= 10:
                        _add(text, emotion)

        print(f"  Empathetic: {len(records):,} records "
              f"(fields: {sorted(fields)})")
        return records

    # Raw CSV format from ParlAI tar.gz
    # Files: train.csv, valid.csv, test.csv
    # Columns: conv_id, utterance_idx, context, prompt, selfeval, tags, utterance
    for csv_file in emp_dir.rglob("*.csv"):
        try:
            lines = csv_file.read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            hdr = [h.strip() for h in lines[0].split(",")]
            ctx_i = next((i for i, h in enumerate(hdr) if h == "context"), -1)
            utt_i = next((i for i, h in enumerate(hdr)
                          if h in ("utterance", "text", "prompt")), -1)
            if ctx_i < 0 or utt_i < 0:
                continue
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) <= max(ctx_i, utt_i):
                    continue
                emotion = parts[ctx_i].strip().strip('"')
                text    = parts[utt_i].strip().strip('"')
                if len(text) >= 10:
                    _add(text, emotion)
        except Exception as e:
            logging.warning("Empathetic Dialogues parse error: %s", e)

    print(f"  Empathetic: {len(records):,} records (CSV)")
    return records


def load_taskmaster(tm_dir: Path) -> List[dict]:
    """
    Parse Taskmaster dataset. Handles:
    - Arrow format (HuggingFace disk cache)
    - Raw JSON from GitHub ZIP:
        Taskmaster-1/TM-1-2019/woz-dialogs.json
        Taskmaster-1/TM-1-2019/self-dialogs.json
        Format: [{conversation_id, instruction_id, utterances:[{index,speaker,text}]}]
        instruction_id prefix maps to domain
    """
    if not tm_dir or not tm_dir.exists():
        return []

    records = []

    # instruction_id prefix → domain for Taskmaster-1
    TM1_INSTRUCTION_MAP = {
        "movie":      ["Ordering at a Restaurant", "Purchase & Payment"],
        "restaurant": ["Restaurant & Ordering", "Ordering at a Restaurant"],
        "pizza":      ["Restaurant & Ordering", "Meal Preparation"],
        "coffee":     ["Restaurant & Ordering", "Purchase & Payment"],
        "auto":       ["Repair"],
        "repair":     ["Repair"],
        "dmv":        ["Permit or Document Process", "Public Administration"],
        "uber":       ["Commuting", "Move"],
        "ride":       ["Commuting", "Move"],
        "flight":     ["Trip Planning", "Transportation & Travel"],
        "hotel":      ["Trip Planning"],
        "sport":      ["Daily Life"],
        "music":      ["Music"],
        "refund":     ["Refund Processing", "Refund"],
        "cancel":     ["Ordering & Delivery"],
        "return":     ["Refund Processing", "Comply"],
        "apply":      ["Apply"],
        "register":   ["Apply", "Permit or Document Process"],
    }

    def _domain_cats(instruction_id: str) -> List[str]:
        iid = instruction_id.lower()
        cats = []
        for k, v in TM1_INSTRUCTION_MAP.items():
            if k in iid:
                cats.extend(v)
        # Also check TASKMASTER_DOMAIN_MAP
        for k, v in TASKMASTER_DOMAIN_MAP.items():
            if k in iid:
                cats.extend(v)
        return list(set(cats))

    # Try Arrow first
    ds = _try_arrow(tm_dir)
    if ds is not None:
        for split_name in ds:
            for item in ds[split_name]:
                domain = item.get("domain",
                         item.get("instruction_src",
                         item.get("instruction_id", ""))).lower()
                cats = []
                for k, v in TASKMASTER_DOMAIN_MAP.items():
                    if k in domain:
                        cats.extend(v)
                cats = list(set(cats)) or _domain_cats(domain)
                if not cats:
                    continue
                turns = item.get("dialogue", item.get("utterances", []))
                for turn in (turns if isinstance(turns, list) else []):
                    text = (turn.get("content", turn.get("text",
                            turn.get("utterance", ""))) if isinstance(turn, dict)
                            else str(turn)).strip()
                    if len(text) >= 10:
                        for cat in cats:
                            records.append({"text": text, "category": cat,
                                            "source": "taskmaster", "lang": "en"})
        print(f"  Taskmaster: {len(records):,} records (Arrow)")
        return records

    # Raw JSON format from GitHub ZIP
    for json_file in sorted(tm_dir.rglob("*.json")):
        if "schema" in json_file.name.lower():
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for conv in data:
                iid  = conv.get("instruction_id",
                       conv.get("conversation_id", "")).lower()
                cats = _domain_cats(iid)
                if not cats:
                    continue
                for utt in conv.get("utterances", []):
                    text = utt.get("text", "").strip()
                    if len(text) >= 10:
                        for cat in cats:
                            records.append({"text": text, "category": cat,
                                            "source": "taskmaster",
                                            "lang": "en"})
        except Exception as e:
            logging.warning("Taskmaster parse error: %s", e)

    print(f"  Taskmaster: {len(records):,} records (raw JSON)")
    return records


def load_circa(circa_dir: Path) -> List[dict]:
    """Parse Circa yes/no question dataset - maps to Clarify, Answer, Agree."""
    if not circa_dir:
        return []
    ds = load_from_disk_or_hf(circa_dir, "circa")
    if ds is None:
        return []

    CIRCA_LABEL_MAP = {
        0: "Agree",    # Yes
        1: "Clarify",  # Yes, subject to some conditions / middle ground
        2: "Clarify",  # No, but ...
        3: "Agree",    # No
        4: "Clarify",  # Indirect answer
        5: "Clarify",  # Middle
        6: "Answer",   # Other
        7: None,       # I am not sure how X is feeling
    }

    records = []
    for split_name in ds:
        for item in ds[split_name]:
            label = item.get("goldstandard1", item.get("label", -1))
            cat = CIRCA_LABEL_MAP.get(int(label) if label is not None else -1)
            if not cat:
                continue
            question = item.get("question-X", item.get("question", "")).strip()
            answer   = item.get("answer-Y",   item.get("answer",   "")).strip()
            if len(question) > 10:
                records.append({"text": question, "category": "Clarification Exchange",
                                 "source": "circa", "lang": "en"})
            if len(answer) > 10:
                records.append({"text": answer, "category": cat,
                                 "source": "circa", "lang": "en"})

    print(f"  Circa: {len(records):,} records")
    return records


def load_dolly(dolly_dir: Path) -> List[dict]:
    """Parse Databricks Dolly 15k instruction dataset."""
    if not dolly_dir:
        return []
    ds = load_from_disk_or_hf(dolly_dir, "databricks/databricks-dolly-15k")
    if ds is None:
        return []

    records = []
    for split_name in ds:
        for item in ds[split_name]:
            cat_key = item.get("category", "")
            cats = DOLLY_CATEGORY_MAP.get(cat_key, [])
            if not cats:
                continue
            # Use instruction as text
            instruction = item.get("instruction", "").strip()
            response    = item.get("response", "").strip()
            if len(instruction) > 10:
                for cat in cats:
                    records.append({"text": instruction, "category": cat,
                                    "source": "dolly", "lang": "en"})
            # Use response for Report/Answer categories
            if response and len(response) > 10 and len(response) < 500:
                for cat in [c for c in cats if c in ("Report", "Answer",
                                                       "Information Seeking")]:
                    records.append({"text": response, "category": cat,
                                    "source": "dolly", "lang": "en"})

    print(f"  Dolly: {len(records):,} records")
    return records


def load_persuasion(prs_dir: Path) -> List[dict]:
    """Parse PersuasionForGood dataset - maps to Election Campaign Speech."""
    if not prs_dir:
        return []
    ds = load_from_disk_or_hf(prs_dir, "Salesforce/dialogstudio",
                               {"name": "PersuasionForGood"})
    if ds is None:
        return []

    records = []
    for split_name in ds:
        for item in ds[split_name]:
            turns = item.get("dialogue", item.get("utterances", []))
            for turn in (turns if isinstance(turns, list) else []):
                text = (turn.get("content", turn.get("text",
                        turn.get("utterance", ""))) if isinstance(turn, dict)
                        else str(turn)).strip()
                if len(text) > 10:
                    records.append({"text": text,
                                    "category": "Election Campaign Speech",
                                    "source": "persuasion", "lang": "en"})
                    # Also map to broader argument/persuasion nodes
                    records.append({"text": text, "category": "Debate",
                                    "source": "persuasion", "lang": "en"})

    print(f"  Persuasion: {len(records):,} records")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# § 9  Anchor texts
# ══════════════════════════════════════════════════════════════════════════════


def load_anchors(anchor_dir: Path, taxonomy: List[dict]) -> List[dict]:
    if not anchor_dir or not anchor_dir.exists():
        return []
    p = anchor_dir / "anchors.json"
    if not p.exists():
        print(f"  Anchors: anchors.json not found in {anchor_dir}")
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    t_index = taxonomy_index(taxonomy)
    SRCS = ["simple_wikipedia", "wordnet", "wordnet_tree", "wiktionary",
            "wiktionary_category", "wikipedia_glossary", "wikipedia_lead",
            "user_sentences"]
    records = []
    for node_name, data in raw.items():
        if node_name not in t_index:
            continue
        node = t_index[node_name]
        seen: Set[str] = set()
        for src in SRCS:
            for text in data.get(src, []):
                text = text.strip()
                if text and text not in seen and len(text) > 20:
                    records.append({"text": text, "category": node_name,
                                    "level": node["level"],
                                    "parent": node.get("parent"),
                                    "lang": "en", "source": f"anchor_{src}"})
                    seen.add(text)
    print(f"  Anchors: {len(records):,} texts ({len(raw)} nodes)")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# § 10  Match records to taxonomy
# ══════════════════════════════════════════════════════════════════════════════

def match_and_enrich(records: List[dict], taxonomy: List[dict]) -> List[dict]:
    """
    Resolve records to exactly one primary taxonomy node per unique sample
    (source, lang, text), while preserving auxiliary candidate categories.

    This prevents the same utterance from being duplicated into multiple nodes,
    which previously contaminated broad parent centroids and destroyed balance.
    """
    t_index = taxonomy_index(taxonomy)
    name_lc = {n["name"].lower(): n["name"] for n in taxonomy}
    kw_map: Dict[str, str] = {}
    for node in sorted(taxonomy, key=lambda n: len(n["name"])):
        for w in re.findall(r"[a-z]{3,}", node["name"].lower()):
            kw_map[w] = node["name"]
        for w in re.findall(r"[a-z]{4,}", node.get("description", "").lower()):
            kw_map.setdefault(w, node["name"])

    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for raw in records:
        r = _normalize_record(raw)
        text = r.get("text", "")
        if len(text) < 10:
            continue
        grouped[(r["source"], r["lang"], text)].append(r)

    result: List[dict] = []
    skipped = 0
    for (source, lang, text), group in grouped.items():
        candidates: List[str] = []
        for r in group:
            cats = []
            if r.get("category"):
                cats.append(r["category"])
            cats.extend(r.get("aux_categories", []))
            for cat in cats:
                resolved = _resolve_category_candidate(cat, taxonomy, t_index, name_lc, kw_map)
                if resolved:
                    candidates.append(resolved)
        primary = _pick_primary_category(candidates, taxonomy, t_index)
        if not primary:
            skipped += 1
            continue
        node = t_index[primary]
        aux = []
        seen = set([primary])
        for c in candidates:
            if c in t_index and c not in seen:
                seen.add(c)
                aux.append(c)
        base = dict(group[0])
        base["text"] = text
        base["category"] = primary
        base["aux_categories"] = aux
        base["source"] = source
        base["lang"] = lang
        base["level"] = node["level"]
        base["parent"] = node.get("parent")
        result.append(base)

    if skipped:
        print(f"    skipped {skipped:,} unresolved grouped records")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# § 11  Centralized sentences.json
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_save(new_records: List[dict], existing: List[dict],
                   out_path: Path,
                   max_per_node: int = 3000,
                   max_per_node_per_source: int = 700) -> List[dict]:
    """
    Merge new records into existing, deduplicate by (text, category),
    cap both total-per-node and per-node-per-source, then stream-write.
    """
    node_counts: Dict[str, int] = {}
    node_source_counts: Dict[Tuple[str, str], int] = {}
    for r in existing:
        cat = r.get("category", "")
        src = r.get("source", "unknown")
        node_counts[cat] = node_counts.get(cat, 0) + 1
        node_source_counts[(cat, src)] = node_source_counts.get((cat, src), 0) + 1

    seen: Set[Tuple[str, str]] = {(r.get("text", ""), r.get("category", "")) for r in existing}

    accepted_new: List[dict] = []
    added = 0
    capped_total = 0
    capped_source = 0

    for r in new_records:
        cat = r.get("category", "")
        src = r.get("source", "unknown")
        key = (r.get("text", ""), cat)
        if key in seen:
            continue
        if node_counts.get(cat, 0) >= max_per_node:
            capped_total += 1
            continue
        if node_source_counts.get((cat, src), 0) >= max_per_node_per_source:
            capped_source += 1
            continue
        seen.add(key)
        node_counts[cat] = node_counts.get(cat, 0) + 1
        node_source_counts[(cat, src)] = node_source_counts.get((cat, src), 0) + 1
        accepted_new.append(r)
        added += 1

    if capped_total or capped_source:
        print(f"  Capped {capped_total:,} total-per-node and {capped_source:,} per-source records - accepted {added:,} new")

    tmp_path = out_path.with_suffix(".tmp.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        total = len(existing) + len(accepted_new)
        written = 0
        for r in existing:
            f.write(json.dumps(r, ensure_ascii=False))
            written += 1
            f.write(",\n" if written < total else "\n")
        for r in accepted_new:
            f.write(json.dumps(r, ensure_ascii=False))
            written += 1
            f.write(",\n" if written < total else "\n")
        f.write("]\n")


    tmp_path.replace(out_path)
    mb = out_path.stat().st_size / 1024 / 1024
    total_records = len(existing) + len(accepted_new)
    print(f"  sentences.json: {total_records:,} total (+{added:,} new)  [{mb:.1f} MB]")
    return existing + accepted_new


# ══════════════════════════════════════════════════════════════════════════════
# § 12  Centralized embed - ONE embeddings.npz
# ══════════════════════════════════════════════════════════════════════════════

def embed_all(data_dir: Path, batch_size: int = 32,
              checkpoint_every: int = 500) -> np.ndarray:
    """
    Embed sentences.json with LaBSE into a single embeddings.npz.

    INCREMENTAL: if embeddings.npz already exists with fewer rows than
    sentences.json (because new records were collected), only the new
    rows are embedded and the file is rebuilt complete. Existing
    embeddings are reused - no re-embedding.

    CHECKPOINT-SAFE: saves partial progress every ~checkpoint_every
    batches. On crash, restart with the same command. Atomic
    tmp->rename prevents corrupt final file.
    """
    from sentence_transformers import SentenceTransformer

    records_path = data_dir / "sentences.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    n     = len(records)
    texts = [r["text"] for r in records]

    final_path = data_dir / "embeddings.npz"
    ckpt_path  = data_dir / "embeddings_checkpoint.npz"

    # Incremental: load existing embeddings if partial
    existing_emb: Optional[np.ndarray] = None
    if final_path.exists():
        try:
            existing_emb = np.load(final_path)["embeddings"].astype(np.float32)
            if len(existing_emb) == n:
                print(f"  embeddings.npz complete ({n:,} rows) - nothing to do")
                return existing_emb
            elif len(existing_emb) < n:
                print(f"  embeddings.npz has {len(existing_emb):,} rows, "
                      f"sentences.json has {n:,} - embedding {n - len(existing_emb):,} new")
            else:
                print(f"  embeddings.npz has MORE rows than sentences.json - re-embedding all")
                existing_emb = None
        except Exception as e:
            logging.warning("Could not load existing embeddings.npz: %s", e)
            existing_emb = None

    # Resume from checkpoint
    start_from = len(existing_emb) if existing_emb is not None else 0
    embs_done: List[np.ndarray] = [existing_emb] if existing_emb is not None else []

    if ckpt_path.exists() and existing_emb is None:
        try:
            ckpt = np.load(ckpt_path)
            embs_done = [ckpt["embeddings"]]
            start_from = int(ckpt["n_done"])
            print(f"  Resuming from checkpoint: {start_from:,}/{n:,}")
        except Exception as e:
            logging.warning("Could not load embedding checkpoint: %s", e)

    if start_from >= n:
        emb = np.vstack(embs_done) if embs_done else np.zeros((0,768), dtype=np.float32)
        _save_final(data_dir, emb)
        return emb

    print(f"  Embedding {n - start_from:,} new texts ...")
    model = SentenceTransformer("sentence-transformers/LaBSE")
    embs_new: List[np.ndarray] = []
    t0 = time.time()
    batch_num = 0

    for start in range(start_from, n, batch_size):
        end = min(start + batch_size, n)
        e = model.encode(texts[start:end], normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
        embs_new.append(e.astype(np.float32))
        batch_num += 1
        if batch_num % 50 == 0:
            done    = start_from + batch_num * batch_size
            elapsed = time.time() - t0
            pct     = done / n
            eta     = elapsed / pct * (1 - pct) if pct > 0 else 0
            print(f"    {done:,}/{n:,}  ({pct:.0%})  "
                  f"elapsed {elapsed/60:.1f}min  ETA {eta/60:.1f}min")
        if batch_num % checkpoint_every == 0:
            n_done  = start_from + len(embs_new) * batch_size
            partial = np.vstack(embs_done + embs_new)
            _save_ckpt(ckpt_path, partial, min(n_done, n))

    embeddings = np.vstack(embs_done + embs_new)
    assert len(embeddings) == n, f"Mismatch: {len(embeddings)} != {n}"
    print(f"  Done. Shape: {embeddings.shape}  ({(time.time()-t0)/60:.1f} min)")
    _save_final(data_dir, embeddings)
    ckpt_path.unlink(missing_ok=True)
    return embeddings


def _save_ckpt(path: Path, emb: np.ndarray, n_done: int) -> None:
    tmp = path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(tmp, embeddings=emb, n_done=np.array(n_done))
        tmp.replace(path)
    except OSError as e:
        print(f"    [WARN] Checkpoint failed: {e}")
        tmp.unlink(missing_ok=True)


def _save_final(out_dir: Path, emb: np.ndarray) -> None:
    final = out_dir / "embeddings.npz"
    tmp   = out_dir / "embeddings.tmp.npz"
    print(f"  Saving {len(emb):,} embeddings ...")
    try:
        np.savez_compressed(tmp, embeddings=emb)
        tmp.replace(final)   # atomic rename
        mb = final.stat().st_size / 1024 / 1024
        print(f"  Saved -> {final}  ({mb:.0f} MB)")
    except OSError as e:
        print(f"  [ERROR] {e}")
        tmp.unlink(missing_ok=True)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# § 13  Coredrill builder
# ══════════════════════════════════════════════════════════════════════════════

def to_native(obj):
    if isinstance(obj, (bool, np.bool_)):  return bool(obj)
    if isinstance(obj, np.floating):       return float(obj)
    if isinstance(obj, np.integer):        return int(obj)
    if isinstance(obj, np.ndarray):        return obj.tolist()
    if isinstance(obj, dict):              return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):     return [to_native(x) for x in obj]
    return obj


def vmf_kappa(r_bar: float, d: int = 768) -> float:
    r = float(np.clip(r_bar, 1e-6, 1 - 1e-6))
    return r * (d - r * r) / (1 - r * r)


def build_coredrill(records: List[dict], embeddings: np.ndarray,
                    taxonomy: List[dict], tau: float = 0.1,
                    parent_direct_min: int = 250,
                    per_source_cap: int = 400,
                    parent_child_cap: int = 400,
                    rhizome_top_k: int = 6,
                    rhizome_min_sim: float = 0.45) -> dict:
    nrm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / (nrm + 1e-12)).astype(np.float32)
    d = embeddings.shape[1]
    cat_labels = np.array([r.get("category", "") for r in records], dtype=object)
    src_labels = np.array([r.get("source", "unknown") for r in records], dtype=object)

    parents_map = taxonomy_parents_map(taxonomy)
    primary_parent_map = taxonomy_primary_parent_map(taxonomy)
    children_map = taxonomy_children_map(taxonomy)
    levels = taxonomy_levels_map(taxonomy)
    roots = taxonomy_root_nodes(taxonomy)
    max_level = max(levels.keys()) if levels else 0

    def _direct_alpha(level: int) -> float:
        if level >= max_level:
            return 1.0
        table = {0: 0.04, 1: 0.08, 2: 0.14, 3: 0.24, 4: 0.40}
        return table.get(level, 0.55)

    print("  Computing centroids bottom-up (LN -> L0) with multi-parent rhizome ...")

    direct_indices: Dict[str, np.ndarray] = {}
    direct_support_indices: Dict[str, np.ndarray] = {}
    direct_centroids: Dict[str, np.ndarray] = {}
    direct_counts: Dict[str, int] = {}
    source_counts_by_node: Dict[str, Dict[str, int]] = {}

    for node in taxonomy:
        n = node["name"]
        idx = np.where(cat_labels == n)[0].astype(np.int32)
        if len(idx) == 0:
            continue
        direct_indices[n] = idx
        direct_counts[n] = int(len(idx))
        sc = defaultdict(int)
        for i in idx:
            sc[str(src_labels[i])] += 1
        source_counts_by_node[n] = dict(sc)
        balanced_idx = _balanced_source_sample_indices(idx, src_labels, per_source_cap=per_source_cap, total_cap=max(per_source_cap, 3000))
        if len(balanced_idx) == 0:
            continue
        direct_support_indices[n] = balanced_idx.astype(np.int32)
        X = embeddings[balanced_idx].astype(np.float64)
        mu = X.mean(axis=0)
        mu /= np.linalg.norm(mu) + 1e-12
        direct_centroids[n] = mu

    final_centroids: Dict[str, np.ndarray] = {}
    support_indices: Dict[str, np.ndarray] = {}
    centroid_mode: Dict[str, str] = {}
    propagated_counts: Dict[str, int] = {}
    propagated_source_counts: Dict[str, Dict[str, int]] = {}

    for level in sorted(levels.keys(), reverse=True):
        for n in levels[level]:
            children = children_map.get(n, [])
            populated_children = [c for c in children if c in final_centroids]
            direct_mu = direct_centroids.get(n)
            direct_idx = direct_support_indices.get(n, np.array([], dtype=np.int32))
            direct_n = len(direct_idx)
            is_leaf = len(children) == 0

            if is_leaf:
                if direct_mu is None:
                    continue
                final_centroids[n] = direct_mu
                support_indices[n] = direct_idx
                centroid_mode[n] = "leaf_direct"
                propagated_counts[n] = direct_counts.get(n, 0)
                propagated_source_counts[n] = dict(source_counts_by_node.get(n, {}))
                continue

            child_support_parts = []
            child_weighted = []
            child_weights = []
            merged_source_counts = defaultdict(int)
            for child in populated_children:
                child_idx = support_indices.get(child, np.array([], dtype=np.int32))
                if len(child_idx) > 0:
                    child_support_parts.append(_clip_indices(child_idx, parent_child_cap))
                for src, c in propagated_source_counts.get(child, {}).items():
                    merged_source_counts[src] += int(c)
                child_weighted.append(final_centroids[child])
                child_weights.append(max(1, len(child_idx)))

            child_support = _unique_concat(child_support_parts)
            for src, c in source_counts_by_node.get(n, {}).items():
                merged_source_counts[src] += int(c)
            propagated_source_counts[n] = dict(merged_source_counts)
            propagated_counts[n] = int(sum(merged_source_counts.values()))

            child_mu = None
            if child_weighted:
                C = np.vstack(child_weighted).astype(np.float64)
                W = np.array(child_weights, dtype=np.float64)
                child_mu = (C * W[:, None]).sum(axis=0)
                child_mu /= np.linalg.norm(child_mu) + 1e-12

            if child_mu is None and direct_mu is None:
                continue
            if child_mu is None:
                final_centroids[n] = direct_mu
                support_indices[n] = direct_idx
                centroid_mode[n] = "parent_direct_only"
                continue
            if direct_mu is None:
                final_centroids[n] = child_mu
                support_indices[n] = child_support
                centroid_mode[n] = "child_only"
                continue

            alpha = _direct_alpha(level)
            if direct_n < parent_direct_min:
                alpha *= 0.5
            mu = alpha * direct_mu + (1.0 - alpha) * child_mu
            mu /= np.linalg.norm(mu) + 1e-12
            final_centroids[n] = mu
            support_indices[n] = _unique_concat([child_support, _clip_indices(direct_idx, per_source_cap)])
            centroid_mode[n] = f"bottom_up_blend_a={alpha:.2f}"

    lineage_memo: Dict[str, List[List[str]]] = {}
    lineage_paths = {n["name"]: taxonomy_lineage_paths(n["name"], parents_map, lineage_memo) for n in taxonomy}
    ancestor_sets = {n["name"]: taxonomy_ancestor_set(n["name"], parents_map) for n in taxonomy}
    top_branch_map = {n["name"]: taxonomy_top_branches(n["name"], parents_map) for n in taxonomy}

    tree: Dict[str, dict] = {}
    for node in taxonomy:
        n = node["name"]
        lv = node["level"]
        parents = list(node.get("parents", []))
        primary_parent = primary_parent_map.get(n)
        children = children_map.get(n, [])
        mu = final_centroids.get(n)
        if mu is None:
            tree[n] = {"level": lv, "parents": parents, "primary_parent": primary_parent, "children": children,
                       "description": node.get("description", ""), "en_category": node.get("en_category", n),
                       "n": 0, "direct_n": 0, "centroid": None, "sigma": None,
                       "lineage_paths": lineage_paths.get(n, []), "top_branches": top_branch_map.get(n, []),
                       "no_data": True}
            continue

        idx = support_indices.get(n, np.array([], dtype=np.int32))
        if len(idx) > 0:
            X_sigma = embeddings[idx].astype(np.float64)
            r_bar = float((X_sigma @ mu).mean())
            sigma = float(1.0 - r_bar)
        else:
            child_vecs = [final_centroids[c] for c in children if c in final_centroids]
            if child_vecs:
                C = np.vstack(child_vecs).astype(np.float64)
                sigma = float(np.mean(1.0 - (C @ mu)))
                r_bar = 1.0 - sigma
            else:
                sigma = 0.5
                r_bar = 0.5

        sibling_set = set()
        for p in parents:
            for sib in children_map.get(p, []):
                if sib != n and sib in final_centroids:
                    sibling_set.add(sib)
        siblings = sorted(sibling_set)
        mean_oth = (np.array([final_centroids[s] for s in siblings]).mean(axis=0) if siblings else np.zeros(d))
        diag = mu - mean_oth
        order = np.argsort(-np.abs(diag))[:100]
        fisher_vs = {}
        for sib in siblings:
            gap = float(1 - mu @ final_centroids[sib])
            ss = tree.get(sib, {}).get("sigma")
            if ss is None:
                ss = 0.4
            fisher_vs[sib] = round(gap / (sigma + ss + 1e-12), 4)

        cands = [n] + siblings
        ca = np.array([final_centroids[c] for c in cands], dtype=np.float64)
        if len(idx) > 0:
            X_eval = embeddings[idx].astype(np.float64)
            pred = np.argmin(1 - X_eval @ ca.T, axis=1)
            recall = float((pred == 0).mean())
            dists = np.sort(1 - X_eval @ ca.T, axis=1)
            mean_m = float((dists[:, 1] - dists[:, 0]).mean()) if dists.shape[1] > 1 else 1.0
        else:
            recall = 1.0
            mean_m = 1.0

        langs = sorted(set(r.get("lang", "en") for r in records if r.get("category") == n))
        src_cnt = propagated_source_counts.get(n, source_counts_by_node.get(n, {}))
        tree[n] = {"level": lv, "parents": parents, "primary_parent": primary_parent, "children": children,
                   "description": node.get("description", ""), "en_category": node.get("en_category", n),
                   "n": propagated_counts.get(n, direct_counts.get(n, 0)), "direct_n": direct_counts.get(n, 0),
                   "n_languages": len(langs), "source_counts": src_cnt,
                   "source_entropy": round(_source_balance_entropy(src_cnt), 6),
                   "centroid_mode": centroid_mode.get(n, "direct"), "support_n": int(len(idx)),
                   "centroid": mu.tolist(), "sigma": round(sigma, 6), "kappa_vmf": round(vmf_kappa(r_bar), 2),
                   "recall_vs_siblings": round(recall, 4), "mean_margin": round(mean_m, 4),
                   "diagnostic_dims": order.tolist(), "diagnostic_weights": diag[order].tolist(),
                   "diagnostic_full": diag.tolist(), "fisher_vs_siblings": fisher_vs,
                   "lineage_paths": lineage_paths.get(n, []), "top_branches": top_branch_map.get(n, [])}
        print(f"    {n:<38} n={tree[n]['n']:>6,}  direct={direct_counts.get(n,0):>5,}  support={len(idx):>5,}  sigma={sigma:.4f}  recall={recall:.1%}  mode={centroid_mode.get(n,'direct')}")

    populated = [n for n in tree if tree[n].get("centroid") is not None and n not in SKIP_NODES]
    rhizome_overlay: Dict[str, List[dict]] = {n: [] for n in tree}
    if len(populated) > 1:
        C = np.array([tree[n]["centroid"] for n in populated], dtype=np.float64)
        S = C @ C.T
        idx_map = {n: i for i, n in enumerate(populated)}
        for n in populated:
            i = idx_map[n]
            scored = []
            for m in populated:
                if n == m:
                    continue
                j = idx_map[m]
                sim = float(S[i, j])
                if sim < rhizome_min_sim:
                    continue
                is_parent_child = (m in parents_map.get(n, [])) or (n in parents_map.get(m, []))
                share_parent = bool(set(parents_map.get(n, [])) & set(parents_map.get(m, [])))
                common_anc = len(ancestor_sets.get(n, set()) & ancestor_sets.get(m, set()))
                dag_dist = taxonomy_dag_distance(n, m, parents_map, children_map)
                relation = "cross_branch"
                if is_parent_child:
                    relation = "direct_lineage"
                elif share_parent:
                    relation = "sibling"
                elif common_anc > 0:
                    relation = "shared_ancestor"
                scored.append({"name": m, "sim": round(sim, 4), "relation": relation, "dag_distance": int(dag_dist),
                               "shared_top_branches": sorted(set(top_branch_map.get(n, [])) & set(top_branch_map.get(m, [])))})
            scored.sort(key=lambda x: (-x["sim"], x["dag_distance"], x["name"]))
            rhizome_overlay[n] = scored[:rhizome_top_k]
            tree[n]["related_nodes"] = rhizome_overlay[n]

    from sklearn.decomposition import PCA
    rng = np.random.default_rng(42)
    idx = rng.choice(len(embeddings), min(10000, len(embeddings)), replace=False)
    pmn = embeddings[idx].astype(np.float64).mean(axis=0)
    pca = PCA(n_components=3, random_state=42)
    pca.fit((embeddings[idx] - pmn).astype(np.float64))
    pcp = pca.components_.T
    pvar = pca.explained_variance_ratio_.tolist()
    for n, stats in tree.items():
        if stats.get("centroid"):
            stats["pca_xyz"] = ((np.array(stats["centroid"]) - pmn) @ pcp).tolist()

    nc = sum(tree[n]["recall_vs_siblings"] * tree[n]["support_n"] for n in tree if tree[n].get("support_n", 0) > 0)
    nt = sum(tree[n].get("support_n", 0) for n in tree)

    return {"version": "5.0", "type": "rhizomatic_cladistic", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "labse_model": "sentence-transformers/LaBSE", "embedding_dim": d,
            "n_nodes": len(taxonomy), "n_nodes_with_data": sum(1 for n in tree if not tree[n].get("no_data")),
            "n_levels": max(n["level"] for n in taxonomy) + 1, "n_source_records": len(records),
            "source_languages": sorted(set(r.get("lang","en") for r in records)),
            "data_sources": sorted(set(r.get("source","unknown") for r in records)),
            "tau": tau, "flat_prototype_accuracy": round(nc / nt if nt > 0 else 0, 4),
            "build_direction": "bottom_up", "levels": {str(lv): names for lv, names in levels.items()},
            "hierarchy": children_map, "tree": tree,
            "skip_nodes": sorted(SKIP_NODES),
            "taxonomy_backbone": {"roots": roots, "parents_map": parents_map, "primary_parent_map": primary_parent_map,
                                  "lineage_paths": lineage_paths, "top_branches": top_branch_map},
            "rhizome": {"overlay": rhizome_overlay, "top_k": int(rhizome_top_k), "min_sim": float(rhizome_min_sim)},
            "pca_basis": {"mean": pmn.tolist(), "components": pcp.tolist(), "explained_variance_ratio": pvar}}

# ══════════════════════════════════════════════════════════════════════════════
# § 14  Predict
# ══════════════════════════════════════════════════════════════════════════════

LEVEL_MARGIN = {0: 0.005, 1: 0.008, 2: 0.012, 3: 0.015, 4: 0.015, 5: 0.015}
LEVEL_LABELS = {0:"Branch", 1:"Domain", 2:"Discipline",
                3:"Topic",  4:"Scenario", 5:"Action"}

# Level-aware sigma thresholds.
# L0 centroids have high variance (4 very broad branches from diverse sources)
# so a short colloquial sentence will always sit far from them - don't gate hard.
# Tighten progressively as we descend into more specific nodes.
LEVEL_SIGMA  = {0: 4.0, 1: 3.5, 2: 3.0, 3: 2.5, 4: 2.5, 5: 2.5}

# Level-aware confidence thresholds.
# At L0 with 4 branches, random chance = 25% - threshold must be < 33%.
# At deeper levels with 2-3 siblings, 35% is more meaningful.
LEVEL_CONF   = {0: 0.27, 1: 0.30, 2: 0.32, 3: 0.35, 4: 0.35, 5: 0.35}


def predict(text_or_emb, coredrill: dict, tau: Optional[float] = None) -> dict:
    tau = tau or coredrill.get("tau", 0.1)
    tree = coredrill["tree"]
    levels = {int(k): v for k, v in coredrill.get("levels", {}).items()}
    backbone = coredrill.get("taxonomy_backbone", {})
    lineage_paths_map = backbone.get("lineage_paths", {})

    if isinstance(text_or_emb, str):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/LaBSE")
        x = model.encode([text_or_emb], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    else:
        x = np.asarray(text_or_emb, dtype=np.float32)
        x /= np.linalg.norm(x) + 1e-12

    _rff_p = _get_rff(coredrill)
    per_level = []
    best_overall = None
    for level in sorted(levels):
        if _rff_p is not None:
            valid = [n for n in levels[level] if n in tree and tree[n].get("rmc") is not None]
        else:
            valid = [n for n in levels[level] if n in tree and tree[n].get("centroid") is not None]
        if not valid:
            continue
        if _rff_p is not None:
            phi_x_ = _rff_p.transform_one(x)
            cents  = np.array([tree[n]["rmc"] for n in valid], dtype=np.float32)
            sims   = phi_x_ @ cents.T
        else:
            cents = np.array([tree[n]["centroid"] for n in valid], dtype=np.float32)
            sims = x @ cents.T
        dists = 1 - sims
        logits = sims / tau
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        order = np.argsort(dists)
        best_idx = int(order[0])
        best = valid[best_idx]
        conf = float(probs[best_idx])
        margin = float(dists[order[1]] - dists[order[0]]) if len(order) > 1 else 1.0
        sigma = tree[best].get("sigma") or 0.4
        sdist = float(dists[best_idx]) / (sigma + 1e-12)
        state = "confident"
        if sdist > LEVEL_SIGMA.get(level, 2.5):
            state = "deep_space"
        elif margin < LEVEL_MARGIN.get(level, 0.015):
            state = "ambiguous"
        elif conf < LEVEL_CONF.get(level, 0.35):
            state = "weak"
        step = {"level": level, "chosen": best, "confidence": round(conf, 4), "margin": round(margin, 4),
                "sigma_dist": round(sdist, 3), "inside_cloud": sdist < 1.0, "state": state,
                "candidates": [{"name": valid[i], "sim": round(float(sims[i]), 4), "conf": round(float(probs[i]), 4)} for i in order[:5]]}
        per_level.append(step)
        if best_overall is None or level >= best_overall["level"]:
            best_overall = step

    _rff_p = _get_rff(coredrill)
    if _rff_p is not None:
        # v3: rank by KME score
        populated = [n for n, st in tree.items() if st.get("rmc") is not None]
        overall_rank = []
        if populated:
            phi_x  = _rff_p.transform_one(x)
            KME_mat = np.array([tree[n]["rmc"] for n in populated], dtype=np.float32)
            sims   = phi_x @ KME_mat.T
            order  = np.argsort(-sims)
            overall_rank = [{"name": populated[i], "level": int(tree[populated[i]]["level"]),
                              "sim": round(float(sims[i]), 4)} for i in order[:10]]
    else:
        populated = [n for n, st in tree.items() if st.get("centroid") is not None]
        overall_rank = []
        if populated:
            C = np.array([tree[n]["centroid"] for n in populated], dtype=np.float32)
            sims = x @ C.T
            order = np.argsort(-sims)
            overall_rank = [{"name": populated[i], "level": int(tree[populated[i]]["level"]), "sim": round(float(sims[i]), 4)} for i in order[:10]]

    best_node = best_overall["chosen"] if best_overall else (overall_rank[0]["name"] if overall_rank else "unknown")
    best_paths = lineage_paths_map.get(best_node, [[best_node]])
    best_paths = [[n for n in p if n not in SKIP_NODES] for p in best_paths]
    best_paths = [p for p in best_paths if p]
    if not best_paths: best_paths = [[best_node]]
    related = [r for r in tree.get(best_node, {}).get("related_nodes", []) if r.get("name") not in SKIP_NODES]

    pmn = np.array(coredrill["pca_basis"]["mean"], dtype=np.float32)
    pcp = np.array(coredrill["pca_basis"]["components"], dtype=np.float32)
    pxyz = ((x - pmn) @ pcp).tolist()
    return {"levels": per_level, "overall_rank": overall_rank, "best_node": best_node,
            "best_level": int(tree.get(best_node, {}).get("level", -1)) if best_node in tree else -1,
            "best_lineage_paths": best_paths, "related_nodes": related, "pca_xyz": pxyz,
            "top_level_prediction": best_node,
            "top_level_confidence": float(best_overall.get("confidence", 0.0)) if best_overall else 0.0,
            "deepest_level": int(tree.get(best_node, {}).get("level", -1)) if best_node in tree else -1,
            "path": per_level, "stop_reason": None, "_embedding": x}

def _deleuzian_flow(active: list, results: list, pred: dict) -> dict:
    if not results or not active:
        return {"flow_type": "void", "flow_label": "VOID", "description": "No territorial signal.", "intensities": {}}
    top        = results[0]
    top_z      = top["sigma_dist"]
    top_cos    = top["score"]
    top_sigma  = top["sigma_eff"]
    n_active   = len(active)
    active_zs  = [r["sigma_dist"] for r in active]
    z_spread   = max(active_zs) - min(active_zs) if len(active_zs) > 1 else 0.0
    cross_dag  = any(rn.get("dag_distance", 0) == 999999 for rn in pred.get("related_nodes", []))
    paths      = pred.get("best_lineage_paths", [])
    branches   = set()
    for r in active:
        if paths and len(paths[0]) > 0:
            branches.add(paths[0][0])
    cross_branch = len(branches) > 1
    deltas     = [r.get("_delta_from_best", 0.0) for r in active if "_delta_from_best" in r]
    delta_spread = max(deltas) if deltas else 0.0

    t_score  = (max(0.0, 1.0 - top_z / 2.0) * 0.4 + min(top_cos, 1.0) * 0.3 +
                max(0.0, 1.0 - top_sigma / 0.5) * 0.2 + max(0.0, 1.0 - delta_spread / 0.3) * 0.1)
    d_score  = (min(top_z / 2.0, 1.0) * 0.35 + max(0.0, 1.0 - top_cos / 0.7) * 0.25 +
                min(n_active / 6.0, 1.0) * 0.2 + min(z_spread / 0.5, 1.0) * 0.2)
    lf_score = ((0.5 if cross_branch else 0.0) + (0.3 if cross_dag else 0.0) + min(top_cos / 0.7, 1.0) * 0.2)
    bwo_score= (min(top_sigma / 0.5, 1.0) * 0.4 + min(n_active / 8.0, 1.0) * 0.3 +
                max(0.0, 1.0 - (top_cos - 0.35) / 0.3) * 0.3)
    rt_score = (max(0.0, 1.0 - abs(top_z - 1.0) / 1.0) * 0.4 +
                max(0.0, 1.0 - abs(n_active - 2.5) / 2.5) * 0.3 +
                max(0.0, 1.0 - delta_spread / 0.2) * 0.3)

    scores = {
        "territorialization":   round(t_score,  3),
        "deterritorialization": round(d_score,  3),
        "line_of_flight":       round(lf_score, 3),
        "bwo_approach":         round(bwo_score,3),
        "reterritorialization": round(rt_score, 3),
    }
    primary = max(scores, key=scores.get)
    labels  = {
        "territorialization":   "TERRITORIALIZATION",
        "deterritorialization": "DETERRITORIALIZATION",
        "line_of_flight":       "LINE OF FLIGHT",
        "bwo_approach":         "BwO APPROACH",
        "reterritorialization": "RETERRITORIALIZATION",
    }
    return {"flow_type": primary, "flow_label": labels[primary],
            "intensities": scores, "cross_branch": cross_branch, "cross_dag": cross_dag}


def _print_rhizome_terminal(text: str, active: list, results: list, pred: dict,
                             top_matches: list) -> None:
    """Clean terminal display: table + ASCII rhizome flow."""
    import sys
    W = 72
    out = sys.stdout.write
    nl  = lambda: out("\n")

    nl(); out("=" * W + "\n")
    out("COREDRILL  -  RHIZOME PROFILE\n")
    out("=" * W + "\n")
    out(f'  {text[:W-4]}{"..." if len(text) > W-4 else ""}\n')
    nl()

    # ── Top territories table ────────────────────────────────────────────
    out(f"  {'NODE':<28}  {'sigma':>6}  {'z':>5}  {'lap':>7}  {'gau':>7}  {'man':>7}  {'MASS':>6}\n")
    out("  " + "-" * (W - 2) + "\n")
    for r in active[:6]:
        z    = r["sigma_dist"]
        mass = r["mass"]
        sig  = r["sigma_eff"]
        lap  = r.get("laplace_score", 0.0)
        gau  = r.get("gaussian_score", 0.0)
        man  = r.get("manifold_score", 0.0)
        bar_w   = 10
        filled  = int((mass / 100.0) * bar_w)
        bar     = "|" * filled + "." * (bar_w - filled)
        out(f"  {r['node']:<28}  {sig:6.4f}  {z:5.2f}s  {lap:7.4f}  {gau:7.4f}  {man:7.4f}  [{bar}] {mass:5.1f}%\n")
    nl()

    # ── Cladistic path ───────────────────────────────────────────────────
    paths = pred.get("best_lineage_paths", [[]])
    if paths and paths[0]:
        path = paths[0]
        out("  CLADISTIC PATH:\n")
        line = "  "
        for i, node in enumerate(path):
            seg = node + (" -> " if i < len(path) - 1 else "")
            if len(line) + len(seg) > W - 2:
                out(line + "\n")
                line = "    " + seg
            else:
                line += seg
        if line.strip():
            out(line + "\n")
        nl()

    # ── Rhizome flow ASCII ───────────────────────────────────────────────
    flow = _deleuzian_flow(active, results, pred)
    out("  FLOW TOPOLOGY:\n")
    nl()

    # Territory branches
    for r in active[:5]:
        z    = r["sigma_dist"]
        mass = r["mass"]
        if mass >= 80:   stem = "===>"
        elif mass >= 40: stem = "==- "
        elif mass >= 20: stem = "--- "
        else:            stem = "... "
        pos = "in  " if z < 1.0 else ("edge" if z < 1.5 else "out ")
        out(f"    {stem} {r['node']:<28} [{pos}] z={z:.2f}s  {mass:5.1f}%\n")

    nl()
    out(f"  PRIMARY FLOW: {flow['flow_label']}\n")
    if flow.get("cross_dag"):
        out("  (cross-branch rhizome connections detected)\n")
    nl()

    # Flow bars
    for fname, fval in sorted(flow["intensities"].items(), key=lambda x: -x[1]):
        bar_len = int(fval * 28)
        bar     = "#" * bar_len + "." * (28 - bar_len)
        marker  = " <<" if fname == flow["flow_type"] else ""
        label   = fname.replace("_", " ").upper()
        out(f"    {label:<22} [{bar}] {fval:.2f}{marker}\n")

    nl()
    out("=" * W + "\n")
    sys.stdout.flush()


def _dump_full_metrics(text: str, results: list, active: list, pred: dict,
                        winners: dict, consensus_node: str, recommended_node: str,
                        micro_chunks: list, barycenter, embedding,
                        out_path) -> None:
    """Dump complete metrics to a text file."""
    import io
    buf = io.StringIO()
    W = 110

    buf.write("=" * W + "\n")
    buf.write("FULL METRICS DUMP\n")
    buf.write("=" * W + "\n")
    buf.write(f"TEXT: {text}\n\n")

    buf.write(f"EMBEDDING PREVIEW: {_preview_embedding(embedding)}\n\n")
    buf.write(f"MICRO-CHUNKS ({len(micro_chunks)}):\n")
    for i, mt in enumerate(micro_chunks, 1):
        buf.write(f"  [{i}] {mt}\n")
    buf.write("\n")

    buf.write("WINNERS BY METRIC FAMILY\n")
    labels = [("Production", "production", "production_score", "prod"),
              ("Raw cosine", "cosine", "score", "cos"),
              ("Smallest z", "smallest_z", "sigma_dist", "z"),
              ("Laplace",    "laplace",    "laplace_score",    "lap"),
              ("Gaussian",   "gaussian",   "gaussian_score",   "gau"),
              ("Manifold",   "manifold",   "manifold_score",   "man")]
    for title, key, score_key, short in labels:
        w = winners[key]
        extra = f"{short}={w[score_key]:.4f}" if score_key != "sigma_dist" else f"z={w['sigma_dist']:.2f}s"
        buf.write(f"  {title:<20}: {w['node']}  | {extra}  | cos={w['score']:.4f}\n")
    buf.write(f"  Consensus: {consensus_node}\n")
    buf.write(f"  Recommended: {recommended_node}\n\n")

    buf.write(f"{'Category':<28} {'mass':>8} {'cos':>8} {'conf':>8} {'z':>7} {'sigma':>7} {'prod':>9} {'lap':>8} {'gau':>8} {'man':>8} {'deg':>7} {'lvl':>4} {'source':>18}\n")
    buf.write("-" * W + "\n")
    for r in results[:20]:
        buf.write(f"{r['node']:<28} {r['mass']:8.1f}% {r['score']:8.4f} {100*r['conf']:7.1f}% {r['sigma_dist']:7.2f} {r['sigma_eff']:7.4f} {r['production_score']:9.4f} {r['laplace_score']:8.4f} {r['gaussian_score']:8.4f} {r['manifold_score']:8.4f} {r['angular_deg']:7.2f} {r['level']:4d} {r['source']:>18}\n")
    buf.write("\n")

    buf.write("ACTIVE OVERLAP SET:\n")
    for r in active:
        buf.write(f"  {r['node']:<24} mass={r['mass']:.1f}%  cos={r['score']:.4f}  z={r['sigma_dist']:.2f}s  prod={r['production_score']:.4f}  delta={r.get('_delta_from_best',0):.4f}\n")
    buf.write("\n")

    buf.write("CLADISTIC PATH:\n")
    for step in pred.get("levels", []):
        lv = step["level"]
        buf.write(f"  L{lv} {LEVEL_LABELS.get(lv,'Node'):<12} -> {step['chosen']:<32} conf={step['confidence']:.1%} sigma={step['sigma_dist']:.2f} {step['state']}\n")
    paths = pred.get("best_lineage_paths", [])
    if paths:
        buf.write("  Best path: " + " -> ".join(paths[0]) + "\n")
    buf.write("\n")

    buf.write("RHIZOME NEIGHBORS:\n")
    for r in pred.get("related_nodes", [])[:8]:
        buf.write(f"  {r['name']:<28} sim={r['sim']:.4f}  relation={r['relation']}  dag={r['dag_distance']}\n")
    buf.write("\n")

    pca = pred["pca_xyz"]
    buf.write(f"PCA: ({pca[0]:+.4f}, {pca[1]:+.4f}, {pca[2]:+.4f})\n")

    content = buf.getvalue()
    out_path.write_text(content, encoding="utf-8")



# ══════════════════════════════════════════════════════════════════════════════
# § CARTOGRAPHY - Consume and reterritorialize
# ══════════════════════════════════════════════════════════════════════════════

def _welford_update(mean: list, M2: list, n: int, new_vec: list) -> tuple:
    """Online mean + variance update (Welford algorithm)."""
    import numpy as np
    n += 1
    x = np.array(new_vec, dtype=np.float64)
    m = np.array(mean, dtype=np.float64)
    delta = x - m
    m += delta / n
    delta2 = x - m
    M2_arr = np.array(M2, dtype=np.float64) + delta * delta2
    return m.tolist(), M2_arr.tolist(), n


def _sigma_from_M2(M2: list, n: int) -> float:
    """Compute sigma (std of cosine distances) from Welford M2."""
    import numpy as np
    if n < 2:
        return 0.0
    variance = np.array(M2, dtype=np.float64) / (n - 1)
    # sigma here = mean of per-dimension variance, converted to cosine std approx
    return float(np.sqrt(np.mean(variance)))


def consume(text: str, embedding: list, confirmed_node: str,
            coredrill: dict, coredrill_path: str,
            backup: bool = True) -> dict:
    """
    Add a classified sentence to the coredrill, updating the centroid
    and sigma of the confirmed node incrementally (Welford algorithm).
    Writes the updated coredrill JSON back to disk.

    Returns the updated node stats.
    """
    import numpy as np, json, shutil
    from pathlib import Path

    tree = coredrill.get("tree", {})
    if confirmed_node not in tree:
        raise ValueError(f"Node '{confirmed_node}' not in taxonomy.")

    node = tree[confirmed_node]
    old_centroid = node.get("centroid")
    old_n        = int(node.get("n", 0))
    old_sigma    = float(node.get("sigma") or 0.0)
    M2           = node.get("_M2")  # stored for Welford continuity

    x = np.array(embedding, dtype=np.float64)
    x /= np.linalg.norm(x) + 1e-12

    if old_centroid is None or old_n == 0:
        # First record for this node
        new_centroid = x.tolist()
        new_M2       = [0.0] * len(x)
        new_n        = 1
        new_sigma    = 0.0
    else:
        old_c = np.array(old_centroid, dtype=np.float64)
        old_c /= np.linalg.norm(old_c) + 1e-12
        if M2 is None:
            # Approximate M2 from existing sigma
            M2 = [old_sigma ** 2] * len(old_centroid)
        new_centroid, new_M2, new_n = _welford_update(
            old_c.tolist(), M2, old_n, x.tolist()
        )
        # Renormalize centroid to unit sphere
        nc = np.array(new_centroid, dtype=np.float64)
        nc /= np.linalg.norm(nc) + 1e-12
        new_centroid = nc.tolist()
        new_sigma = _sigma_from_M2(new_M2, new_n)

    # Update node in tree
    node["centroid"] = new_centroid
    node["n"]        = new_n
    node["sigma"]    = round(new_sigma, 6)
    node["_M2"]      = new_M2  # persist for future updates
    # Track consumed records count separately
    node["n_consumed"] = node.get("n_consumed", 0) + 1
    # Update source counts
    sc = node.get("source_counts") or {}
    sc["consumed"] = sc.get("consumed", 0) + 1
    node["source_counts"] = sc

    # v3: also update RMC (KME in RFF space) if this is a v3 coredrill
    rff_params = coredrill.get("rff")
    if rff_params is not None and node.get("rmc") is not None:
        rff_obj = _RFF(rff_params)
        phi_new = rff_obj.transform_one(x.astype(np.float32))
        old_rmc = np.array(node["rmc"], dtype=np.float64)
        # Weighted update: new_rmc = (old_rmc * old_n + phi_new) / new_n
        new_rmc = (old_rmc * (new_n - 1) + phi_new.astype(np.float64)) / new_n
        new_rmc /= (np.linalg.norm(new_rmc) + 1e-12)
        node["rmc"] = new_rmc.tolist()

    # Backup and write
    p = Path(coredrill_path)
    if backup and p.exists():
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        bak = p.parent / f"coredrill_backup_{ts}.json"
        shutil.copy2(p, bak)

    p.write_text(json.dumps(coredrill, ensure_ascii=False), encoding="utf-8")

    return {
        "node":       confirmed_node,
        "n":          new_n,
        "n_consumed": node["n_consumed"],
        "sigma":      new_sigma,
        "old_sigma":  old_sigma,
        "delta_n":    new_n - old_n,
    }


def cartography_prompt(text: str, embedding: list, active: list,
                       results: list, coredrill: dict,
                       coredrill_path: str) -> None:
    """
    Interactive terminal prompt: show classification, ask user to confirm
    or reassign to any existing node, then consume the sentence.
    """
    import sys

    W = 72
    tree = coredrill.get("tree", {})
    all_nodes = sorted(tree.keys())

    print()
    print("=" * W)
    print("CARTOGRAPHY - Consume this document?")
    print("=" * W)
    print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print()
    print("  Top territories:")
    for i, r in enumerate(active[:6], 1):
        n_consumed = tree.get(r["node"], {}).get("n_consumed", 0)
        n_total    = tree.get(r["node"], {}).get("n", 0)
        is_v3_node = tree.get(r["node"], {}).get("rmc") is not None
        flag = " [THIN]" if n_total < 20 else ""
        if is_v3_node:
            flag += " [RMC]"
        elif tree.get(r["node"], {}).get("centroid") is None:
            flag += " [NO-CENTROID]"
        print(f"  [{i}] {r['node']:<32} cos={r['score']:.4f}  z={r['sigma_dist']:.2f}s"
              f"  n={n_total}  consumed={n_consumed}{flag}")
    print()
    print("  Options:")
    print("    [1-6]  confirm one of the above nodes")
    print("    [s]    search for a different node by name")
    print("    [skip] skip without consuming")
    print()

    while True:
        sys.stdout.write("  > ")
        sys.stdout.flush()
        try:
            choice = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("  Skipped.")
            return

        if choice == "skip" or choice == "":
            print("  Skipped.")
            return

        # Numeric choice from top 6
        if choice in [str(i) for i in range(1, 7)]:
            idx = int(choice) - 1
            if idx < len(active):
                confirmed = active[idx]["node"]
                break
            else:
                print(f"  Invalid choice.")
                continue

        # Search
        if choice == "s":
            sys.stdout.write("  Search node name (partial match): ")
            sys.stdout.flush()
            try:
                query = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                continue
            matches = [n for n in all_nodes if query in n.lower()]
            if not matches:
                print(f"  No nodes matching '{query}'.")
                continue
            print(f"  Matches ({len(matches)}):")
            for i, m in enumerate(matches[:20], 1):
                n_total = tree.get(m, {}).get("n", 0)
                print(f"    [{i}] {m:<40} n={n_total}")
            sys.stdout.write(f"  Choose [1-{min(len(matches),20)}] or [s] to search again: ")
            sys.stdout.flush()
            try:
                sub = input().strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if sub.isdigit() and 1 <= int(sub) <= min(len(matches), 20):
                confirmed = matches[int(sub) - 1]
                break
            continue

        print("  Invalid input. Enter 1-6, [s], or [skip].")

    # Consume
    print(f"  Assigning to: {confirmed}")
    result = consume(text, embedding, confirmed, coredrill, coredrill_path,
                     backup=True)
    print(f"  Consumed. Node '{confirmed}':")
    print(f"    n: {result['n']-1} -> {result['n']}  "
          f"(+{result['delta_n']} consumed total: {result['n_consumed']})")
    print(f"    sigma: {result['old_sigma']:.4f} -> {result['sigma']:.4f}")
    print("=" * W)


def print_result(result: dict, text: str = "", active: list = None, results_all: list = None) -> None:
    """Legacy thin wrapper - just prints cladistic path. Full display via _print_rhizome_terminal."""
    ICONS = {"confident": "OK", "ambiguous": "AMBIGUOUS", "weak": "WEAK", "deep_space": "DEEP SPACE"}
    print("\n" + "=" * 72)
    if text:
        print(f'INPUT: "{text[:110]}{"..." if len(text)>110 else ""}"')
    print("=" * 72)
    print("Per-level winners:")
    for step in result.get("levels", []):
        lv   = step["level"]
        icon = ICONS.get(step["state"], "")
        print(f"  L{lv} {LEVEL_LABELS.get(lv,'Node'):<12} -> {step['chosen']:<32} conf={step['confidence']:.1%} margin={step['margin']:.4f} sigma={step['sigma_dist']:.2f} {icon}")
    print(f"\nBest node: {result.get('best_node','unknown')}  (L{result.get('best_level',-1)})")
    paths = result.get("best_lineage_paths", [])
    if paths:
        print("Cladistic lineage paths:")
        for path in paths[:3]:
            print("  - " + " -> ".join(path))
    rels = result.get("related_nodes", [])
    if rels:
        print("Rhizome neighbors:")
        for r in rels[:5]:
            print(f"  - {r['name']:<28} sim={r['sim']:.4f}  relation={r['relation']}  dag={r['dag_distance']}")
    if result.get("overall_rank"):
        print("Top global matches:")
        for r in [x for x in result["overall_rank"] if x.get("name") not in SKIP_NODES][:5]:
            print(f"  - {r['name']:<28} L{r['level']} sim={r['sim']:.4f}")
    pca = result["pca_xyz"]
    print(f"PCA:   ({pca[0]:+.4f}, {pca[1]:+.4f}, {pca[2]:+.4f})")
# ══════════════════════════════════════════════════════════════════════════════
# § 15  CLI
# ══════════════════════════════════════════════════════════════════════════════

def cmd_collect(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(Path(args.taxonomy_txt))
    t_index  = taxonomy_index(taxonomy)
    print("=" * 72)
    print("12_BENCHMARK_FULL_TAXONOMY -- Collect")
    print("=" * 72)
    print(f"  Taxonomy: {len(taxonomy)} nodes  |  Output: {out_dir}")
    print()
    out_json = out_dir / "sentences.json"
    existing: List[dict] = []
    existing_cats: Set[str] = set()
    existing_counts: Dict[str, int] = {}
    if out_json.exists():
        existing = json.loads(out_json.read_text(encoding="utf-8"))
        existing_cats = set(r.get("category","") for r in existing)
        for r in existing:
            c = r.get("category", "")
            existing_counts[c] = existing_counts.get(c, 0) + 1
        print(f"  Existing: {len(existing):,} records, "
              f"{len(existing_cats)} categories")
    print()
    all_new: List[dict] = []

    # SOURCE 1: reuse wiki-dir
    wiki_dir = Path(args.wiki_dir) if args.wiki_dir else None
    if wiki_dir and (wiki_dir / "sentences.json").exists():
        wr = json.loads((wiki_dir/"sentences.json").read_text(encoding="utf-8"))
        node_names = {n["name"] for n in taxonomy}
        reused = [r for r in wr
                  if r.get("category","") in node_names
                  and r.get("category","") not in existing_cats]
        for r in reused:
            r.setdefault("source", "wikipedia")
        all_new.extend(reused)
        print(f"SOURCE 1: wiki-dir reused {len(reused):,} records")

    # SOURCE 2: fresh Wikipedia
    print(f"\nSOURCE 2: Wikipedia (fresh)")
    all_new.extend(collect_wikipedia(taxonomy, existing_cats, existing_counts=existing_counts, min_existing_before_skip=args.min_wiki_existing))

    # SOURCE 3: bitext
    bitext_dir = Path(args.bitext_dir) if args.bitext_dir else None
    if bitext_dir and bitext_dir.exists():
        print(f"\nSOURCE 3: Bitext")
        raw = load_source_dir(bitext_dir, "bitext")
        m   = match_and_enrich(raw, taxonomy)
        all_new.extend(m)
        print(f"  -> {len(m):,} matched")

    # SOURCE 4: CFPB
    cfpb_dir = Path(args.cfpb_dir) if args.cfpb_dir else None
    if cfpb_dir and cfpb_dir.exists():
        print(f"\nSOURCE 4: CFPB")
        raw = load_source_dir(cfpb_dir, "cfpb")
        m   = match_and_enrich(raw, taxonomy)
        all_new.extend(m)
        print(f"  -> {len(m):,} matched")

    # SOURCE 5: DailyDialog
    dd_dir = Path(args.dailydialog_dir) if args.dailydialog_dir else None
    if dd_dir and dd_dir.exists():
        print(f"\nSOURCE 5: DailyDialog")
        raw = load_dailydialog(dd_dir)
        m   = match_and_enrich(raw, taxonomy)
        all_new.extend(m)
        print(f"  -> {len(m):,} matched")

    # SOURCE 6: MultiWOZ
    mwoz_dir = Path(args.multiwoz_dir) if args.multiwoz_dir else None
    if mwoz_dir and mwoz_dir.exists():
        print(f"\nSOURCE 6: MultiWOZ")
        raw = load_multiwoz(mwoz_dir)
        m   = match_and_enrich(raw, taxonomy)
        all_new.extend(m)
        print(f"  -> {len(m):,} matched")

    # SOURCE 7: Anchors
    anchor_dir = Path(args.anchor_dir) if args.anchor_dir else None
    if anchor_dir and anchor_dir.exists():
        print(f"\nSOURCE 7: Anchors")
        raw = load_anchors(anchor_dir, taxonomy)
        m   = match_and_enrich(raw, taxonomy)
        all_new.extend(m)
        print(f"  -> {len(m):,} matched")

    # SOURCE 8: SGD (Schema-Guided Dialogue)
    sgd_dir = Path(args.sgd_dir) if args.sgd_dir else None
    if sgd_dir:
        print(f"\nSOURCE 8: SGD")
        raw = load_sgd(sgd_dir)
        m   = match_and_enrich(raw, taxonomy)
        all_new.extend(m)
        print(f"  -> {len(m):,} matched")

    # SOURCE 9: EmpatheticDialogues
    emp_dir = Path(args.empathetic_dir) if args.empathetic_dir else None
    if emp_dir:
        print(f"\nSOURCE 9: EmpatheticDialogues")
        raw = load_empathetic(emp_dir)
        m   = match_and_enrich(raw, taxonomy)
        all_new.extend(m)
        print(f"  -> {len(m):,} matched")

    # SOURCE 10: Taskmaster
    tm_dir = Path(args.taskmaster_dir) if args.taskmaster_dir else None
    if tm_dir:
        print(f"\nSOURCE 10: Taskmaster")
        raw = load_taskmaster(tm_dir)
        m   = match_and_enrich(raw, taxonomy)
        all_new.extend(m)
        print(f"  -> {len(m):,} matched")

    # SOURCE 11: Circa
    circa_dir = Path(args.circa_dir) if args.circa_dir else None
    if circa_dir:
        print(f"\nSOURCE 11: Circa")
        raw = load_circa(circa_dir)
        m   = match_and_enrich(raw, taxonomy)
        all_new.extend(m)
        print(f"  -> {len(m):,} matched")

    # SOURCE 12: Dolly
    dolly_dir = Path(args.dolly_dir) if args.dolly_dir else None
    if dolly_dir:
        print(f"\nSOURCE 12: Dolly")
        raw = load_dolly(dolly_dir)
        m   = match_and_enrich(raw, taxonomy)
        all_new.extend(m)
        print(f"  -> {len(m):,} matched")

    # SOURCE 13: Persuasion
    prs_dir = Path(args.persuasion_dir) if args.persuasion_dir else None
    if prs_dir:
        print(f"\nSOURCE 13: Persuasion")
        raw = load_persuasion(prs_dir)
        m   = match_and_enrich(raw, taxonomy)
        all_new.extend(m)
        print(f"  -> {len(m):,} matched")

    print(f"\n{'='*72}")
    all_records = merge_and_save(all_new, existing, out_json, max_per_node=args.max_per_node, max_per_node_per_source=args.max_per_node_per_source)

    covered = set(r.get("category","") for r in all_records)
    missing = [n for n in taxonomy if n["name"] not in covered]
    print(f"\nCoverage: {len(covered)}/{len(taxonomy)} nodes")
    if missing:
        by_lv: Dict[int, List[str]] = {}
        for n in missing:
            by_lv.setdefault(n["level"], []).append(n["name"])
        print(f"Still missing ({len(missing)}):")
        for lv in sorted(by_lv):
            print(f"  L{lv}: {chr(10).join(by_lv[lv])}")
        print("  Tip: add --anchor-dir anchors/ to fill most remaining gaps")

    src_cnt: Dict[str, int] = {}
    for r in all_records:
        s = r.get("source","unknown"); src_cnt[s] = src_cnt.get(s,0) + 1
    print("\nBy source:")
    for s, n in sorted(src_cnt.items(), key=lambda x: -x[1]):
        print(f"  {s:<35} {n:>8,}")


def cmd_embed(args) -> None:
    print("=" * 72)
    print("12_BENCHMARK_FULL_TAXONOMY -- Embed (centralized)")
    print("  ONE embeddings.npz  =  LaBSE(sentences.json[i]) for all i")
    print("  Incremental: only new rows are embedded if file already exists")
    print("=" * 72)
    embed_all(Path(args.data), batch_size=args.batch_size)


def cmd_build(args) -> None:
    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(Path(args.taxonomy_txt))
    print("=" * 72)
    print("12_BENCHMARK_FULL_TAXONOMY -- Build")
    print("=" * 72)
    records = json.loads((data_dir/"sentences.json").read_text(encoding="utf-8"))
    emb     = np.load(data_dir/"embeddings.npz")["embeddings"].astype(np.float32)
    assert len(records) == len(emb), f"Record/embedding mismatch: {len(records)} vs {len(emb)}"
    print(f"  {len(records):,} records  |  shape {emb.shape}")
    cd = build_coredrill(records, emb, taxonomy, tau=args.tau,
                         parent_direct_min=args.parent_direct_min,
                         per_source_cap=args.build_per_source_cap,
                         parent_child_cap=args.parent_child_cap,
                         rhizome_top_k=args.rhizome_top_k,
                         rhizome_min_sim=args.rhizome_min_sim)
    p  = out_dir / "coredrill_hierarchical.json"
    p.write_text(json.dumps(to_native(cd), indent=2), encoding="utf-8")
    kb = p.stat().st_size / 1024
    print(f"\nSaved -> {p}  ({kb:.0f} KB)")
    print(f"Nodes with data: {cd['n_nodes_with_data']} / {cd['n_nodes']}")
    print(f"Flat accuracy:   {cd['flat_prototype_accuracy']:.1%}")
    print(f"Sources:         {chr(44).join(cd['data_sources'])}")



# Rich analysis helpers
ACTIVE_POSTERIOR_DELTA = 0.18

def _preview_embedding(vec: np.ndarray, head: int = 12, tail: int = 6, precision: int = 4) -> str:
    v = np.asarray(vec, dtype=np.float32)
    if v.size <= head + tail:
        return np.array2string(v, precision=precision, separator=", ")
    head_str = ", ".join(f"{x:.{precision}f}" for x in v[:head])
    tail_str = ", ".join(f"{x:.{precision}f}" for x in v[-tail:])
    return f"[{head_str}, ..., {tail_str}]  (dim={v.size})"

def split_segment_into_microtexts(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    clauses = re.split(r"[;:,.!?]\s+|\s+and\s+|\s+but\s+|\s+because\s+|\s+that\s+", text)
    clauses = [c.strip() for c in clauses if len(c.strip()) >= 8]
    words = text.split()
    windows = []
    if len(words) <= 6:
        windows.append(text)
    else:
        win = min(6, len(words))
        step = max(2, win // 2)
        for i in range(0, max(1, len(words) - win + 1), step):
            windows.append(" ".join(words[i:i+win]))
    out, seen = [], set()
    for p in [text] + clauses + windows:
        p = p.strip()
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out

def _node_source(tree_stats: dict) -> str:
    sc = tree_stats.get("source_counts", {}) or {}
    if sc:
        return max(sc.items(), key=lambda kv: kv[1])[0]
    return "all"

def _lineage_string(cd: dict, node: str) -> str:
    paths = cd.get("taxonomy_backbone", {}).get("lineage_paths", {}).get(node, [])
    if paths:
        return " -> ".join(paths[0])
    return node

def _score_node_analysis(node_name: str, stats: dict, x: np.ndarray,
                         rff: Optional["_RFF"] = None) -> dict:
    # Always use raw centroid for scoring (correct for cosine kernel on hypersphere).
    # RMC/KME via Cauchy RFF produces near-zero scores and is discarded.
    # The centroid built from multi-label data IS the correct Rhizome-Centroid.
    raw = stats.get("centroid")
    if raw is not None:
        mu    = np.array(raw, dtype=np.float32)
        score = float(x @ mu)
    else:
        mu    = np.zeros(x.shape[0], dtype=np.float32)
        score = 0.0
    sigma_eff = max(float(stats.get("sigma") or 0.35), 0.25)
    SIGMA_CAP = 0.45
    sigma_z   = min(sigma_eff, SIGMA_CAP)
    z         = float((1.0 - score) / sigma_z)
    source    = _node_source(stats)
    lap       = float(-z)
    gau       = float(-0.5 * z * z)
    man       = float(lap + 0.15 * score - 0.05 * np.log(max(sigma_eff, 1e-8)))
    diffuse_pen = max(0.0, sigma_eff - 0.40) * 0.60
    prod      = float(0.62 * man + 0.18 * lap + 0.10 * gau + 0.08 * score - diffuse_pen)
    cos_raw   = score
    return {
        "node": node_name,
        "level": int(stats.get("level", -1)),
        "score": score,
        "sigma_dist": z,
        "sigma_eff": sigma_eff,
        "laplace_score": lap,
        "gaussian_score": gau,
        "manifold_score": man,
        "production_score": prod,
        "source": source,
        "angular_deg": float(np.degrees(np.arccos(np.clip(score, -1.0, 1.0)))),
    }

def compute_metric_families(results: List[dict]) -> Dict[str, dict]:
    if not results:
        return {}
    return {
        "production": max(results, key=lambda r: r["production_score"]),
        "cosine": max(results, key=lambda r: r["score"]),
        "smallest_z": min(results, key=lambda r: r["sigma_dist"]),
        "laplace": max(results, key=lambda r: r["laplace_score"]),
        "gaussian": max(results, key=lambda r: r["gaussian_score"]),
        "manifold": max(results, key=lambda r: r["manifold_score"]),
    }

def _consensus_and_recommended(winners: Dict[str, dict]) -> Tuple[Tuple[str, dict], Tuple[str, dict]]:
    from collections import Counter
    order = ["production", "cosine", "smallest_z", "laplace", "gaussian", "manifold"]
    votes = Counter(winners[k]["node"] for k in order if k in winners)
    best_vote = max(votes.values())
    candidates = sorted([node for node, c in votes.items() if c == best_vote])
    consensus_node = None
    for fam in ["manifold", "laplace", "smallest_z", "gaussian", "production", "cosine"]:
        if winners[fam]["node"] in candidates:
            consensus_node = winners[fam]["node"]
            consensus_rep = {**winners[fam], "family": fam}
            break
    lap_node = winners["laplace"]["node"]
    z_node = winners["smallest_z"]["node"]
    gau_node = winners["gaussian"]["node"]
    if gau_node == lap_node or gau_node == z_node:
        recommended_node = gau_node; recommended_family = "gaussian_agreement"
    elif lap_node == z_node:
        recommended_node = lap_node; recommended_family = "laplace_smallest_z_agreement"
    else:
        recommended_node = lap_node; recommended_family = "laplace"
    rec_rep = None
    for fam in ["manifold", "laplace", "smallest_z", "gaussian", "production", "cosine"]:
        if winners[fam]["node"] == recommended_node:
            rec_rep = {**winners[fam], "family": fam, "recommended_family": recommended_family}
            break
    return (consensus_node, consensus_rep), (recommended_node, rec_rep)

def analyze_text_rhizome(coredrill: dict, text: str, model, tau: float = 0.1, print_output: bool = True) -> dict:
    x = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    microtexts = split_segment_into_microtexts(text)
    micro_embs = model.encode(microtexts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32) if microtexts else np.zeros((0, x.shape[0]), dtype=np.float32)
    barycenter = (micro_embs.mean(axis=0).astype(np.float32) if len(micro_embs) else x.copy())
    barycenter /= (np.linalg.norm(barycenter) + 1e-12)
    tree = coredrill.get("tree", {})
    _rff = _get_rff(coredrill)
    if _rff is not None:
        # v3: score nodes that have RMC; fall back to centroid nodes too
        results = [_score_node_analysis(node, stats, x, rff=_rff)
                   for node, stats in tree.items()
                   if stats.get("rmc") is not None and node not in SKIP_NODES]
    else:
        results = [_score_node_analysis(node, stats, x)
                   for node, stats in tree.items()
                   if stats.get("centroid") is not None and node not in SKIP_NODES]
    results.sort(key=lambda r: (r["production_score"], r["score"]), reverse=True)
    logits = np.array([r["production_score"] for r in results], dtype=np.float64) / max(tau, 1e-6)
    logits -= logits.max()
    probs = np.exp(logits); probs /= probs.sum()
    for r, p in zip(results, probs):
        r["conf"] = float(p)
        r["mass"] = float(100.0 * p / max(probs[0], 1e-12))
    winners = compute_metric_families(results)
    (consensus_node, consensus_rep), (recommended_node, rec_rep) = _consensus_and_recommended(winners)
    active = [r.copy() for r in results if (results[0]["production_score"] - r["production_score"]) <= ACTIVE_POSTERIOR_DELTA][:10]
    best_prod = results[0]["production_score"]
    for r in active:
        closeness = float(np.exp(-(best_prod - r["production_score"])))
        r["relative_strength"] = closeness
        r["relative_percent"] = 100.0 * closeness
        r["dominance"] = r["conf"] * 100.0
        r["lineage"] = _lineage_string(coredrill, r["node"])
        r["_delta_from_best"] = float(best_prod - r["production_score"])
    pred = predict(x, coredrill, tau=tau)
    if print_output:
        # ── clean terminal display ────────────────────────────────────────
        top_matches = [{"name": r["name"], "level": r["level"], "sim": r["sim"]}
                       for r in pred.get("overall_rank", [])
                       if r.get("name") not in SKIP_NODES][:5]
        _print_rhizome_terminal(text, active, results, pred, top_matches)

        # ── full ranked table printed to terminal ─────────────────────────
        import sys as _sys
        _sys.stdout.write("\n")
        _sys.stdout.write(f"  {'FULL RANKED TABLE'}\n")
        _sys.stdout.write(f"  {'NODE':<28} {'sigma':>6} {'cos':>7} {'z':>6} {'lap':>8} {'gau':>8} {'man':>8} {'prod':>9} {'mass':>7} {'lvl':>4}\n")
        _sys.stdout.write("  " + "-" * 96 + "\n")
        for r in results:
            _sys.stdout.write(
                f"  {r['node']:<28} {r['sigma_eff']:6.4f} {r['score']:7.4f} {r['sigma_dist']:6.2f}s "
                f"{r['laplace_score']:8.4f} {r['gaussian_score']:8.4f} {r['manifold_score']:8.4f} "
                f"{r['production_score']:9.4f} {r['mass']:6.1f}% {r['level']:4d}\n"
            )
        _sys.stdout.write("\n")
        _sys.stdout.flush()

        # ── full metrics to file ──────────────────────────────────────────
        _metrics_path = Path("coredrill_metrics_last.txt")
        try:
            _dump_full_metrics(text, results, active, pred, winners,
                               consensus_node, recommended_node,
                               microtexts, barycenter, x, _metrics_path)
        except Exception as _de:
            pass  # metrics dump is best-effort
    return {"text": text, "embedding": x.tolist(), "microtexts": microtexts, "micro_embeddings": micro_embs.tolist(),
            "barycenter": barycenter.tolist(), "results": results, "active_overlap_set": active,
            "prediction": pred, "consensus_winner": consensus_rep, "recommended_winner": rec_rep}

def cmd_predict(args) -> None:
    cd = json.loads(Path(args.coredrill).read_text(encoding="utf-8"))
    texts: List[str] = []
    if args.text:  texts.append(args.text)
    if args.batch: texts.extend(t.strip() for t in args.batch if t.strip())
    if args.file:
        texts += [l.strip() for l in
                  Path(args.file).read_text(encoding="utf-8").splitlines()
                  if l.strip() and not l.startswith("#")]
    if not texts:
        sys.exit("[ERROR] Provide --text, --batch, or --file")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE")
    qpts  = Path(args.coredrill).parent / "query_points.jsonl"

    for text in texts:
        try:
            consume_mode = getattr(args, "consume", False)
            if consume_mode or getattr(args, "analyze", False):
                analyze_text_rhizome._consume_mode = consume_mode
                analyze_text_rhizome._coredrill_path = args.coredrill
                rich = analyze_text_rhizome(cd, text, model, tau=args.tau or cd.get("tau", 0.1), print_output=True)
                result = rich["prediction"]
                if consume_mode:
                    import json as _json
                    cd = _json.loads(Path(args.coredrill).read_text(encoding="utf-8"))
            else:
                x = model.encode([text], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)
                result = predict(x, cd, tau=args.tau)
                print_result(result, text)
                sys.stdout.flush()

            pca = result["pca_xyz"]
            record = {
                "text":         text[:120],
                "prediction":   result["top_level_prediction"],
                "path":         [s["chosen"] for s in result["path"]],
                "stop_reason":  result.get("stop_reason"),
                "confidence":   float(result["top_level_confidence"]),
                "deepest_level": int(result["deepest_level"]),
                "pca_x": float(pca[0]),
                "pca_y": float(pca[1]),
                "pca_z": float(pca[2]),
            }
            with open(qpts, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[-> {qpts.name}]")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n[ERROR] Failed on: {text[:60]!r}")
            print(f"        {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()


def cmd_analyze(args) -> None:
    args.analyze = True
    cmd_predict(args)


def main():
    ap  = argparse.ArgumentParser(
        description="Full 238-node taxonomy benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_c = sub.add_parser("collect")
    p_c.add_argument("--taxonomy-txt",    required=True)
    p_c.add_argument("--wiki-dir",        default=None)
    p_c.add_argument("--bitext-dir",      default=None)
    p_c.add_argument("--cfpb-dir",        default=None)
    p_c.add_argument("--dailydialog-dir", default=None)
    p_c.add_argument("--multiwoz-dir",    default=None)
    p_c.add_argument("--anchor-dir",      default=None)
    p_c.add_argument("--sgd-dir",         default=None,
                     help="Schema-Guided Dialogue dataset dir")
    p_c.add_argument("--empathetic-dir",  default=None,
                     help="EmpatheticDialogues dataset dir")
    p_c.add_argument("--taskmaster-dir",  default=None,
                     help="Taskmaster dataset dir")
    p_c.add_argument("--circa-dir",       default=None,
                     help="Circa dataset dir")
    p_c.add_argument("--dolly-dir",       default=None,
                     help="Databricks Dolly dataset dir")
    p_c.add_argument("--persuasion-dir",  default=None,
                     help="PersuasionForGood dataset dir")
    p_c.add_argument("--out",             default="outputs/taxonomy_238")
    p_c.add_argument("--max-per-node",    type=int, default=3000)
    p_c.add_argument("--max-per-node-per-source", type=int, default=700)
    p_c.add_argument("--min-wiki-existing", type=int, default=80,
                     help="Skip fresh wiki fetch only if a node already has at least this many records")

    p_e = sub.add_parser("embed")
    p_e.add_argument("--data",       required=True)
    p_e.add_argument("--batch-size", type=int, default=32)

    p_b = sub.add_parser("build")
    p_b.add_argument("--data",         required=True)
    p_b.add_argument("--taxonomy-txt", required=True)
    p_b.add_argument("--out",          default="coredrill_238")
    p_b.add_argument("--tau",          type=float, default=0.1)
    p_b.add_argument("--parent-direct-min", type=int, default=250)
    p_b.add_argument("--build-per-source-cap", type=int, default=400)
    p_b.add_argument("--parent-child-cap", type=int, default=400)
    p_b.add_argument("--rhizome-top-k", type=int, default=6)
    p_b.add_argument("--rhizome-min-sim", type=float, default=0.45)

    p_p = sub.add_parser("predict")
    p_p.add_argument("--coredrill", required=True)
    p_p.add_argument("--text",      default=None)
    p_p.add_argument("--batch",     nargs="*", default=None)
    p_p.add_argument("--file",      default=None)
    p_p.add_argument("--tau",       type=float, default=None)
    p_p.add_argument("--consume",   "-C", action="store_true", help="After classification, prompt to assign sentence to a node and update the coredrill.")
    p_p.add_argument("--analyze",   action="store_true", help="Restore the rich analysis output with metric families, ranked table, and active overlap set")

    p_a = sub.add_parser("analyze")
    p_a.add_argument("--coredrill", required=True)
    p_a.add_argument("--text",      default=None)
    p_a.add_argument("--batch",     nargs="*", default=None)
    p_a.add_argument("--file",      default=None)
    p_a.add_argument("--tau",       type=float, default=None)
    p_a.add_argument("--analyze",   action="store_true", default=True)

    args = ap.parse_args()
    {"collect": cmd_collect, "embed": cmd_embed,
     "build":   cmd_build,   "predict": cmd_predict, "analyze": cmd_analyze}[args.command](args)



# ══════════════════════════════════════════════════════════════════════════════
# § GRAM-INVERSE RHIZOME SCORE
#
#     alpha = G^+ s
#
# G_ij = <mu_i, mu_j>  exact inter-centroid cosines from stored centroids
# s_i  = <mu_i, x>     cosine scores already computed by analyze_text_rhizome
# G^+  = Moore-Penrose pseudoinverse via SVD
# winner: argmax_i  alpha_i / sigma_i
# ══════════════════════════════════════════════════════════════════════════════

def gram_inverse_scores(results: list, tree: dict,
                        top_m: int = 20,
                        svd_threshold: float = 0.05,
                        coredrill: dict = None) -> dict:
    """
    Full topological rhizome score:

        R_i = alpha_i * s_i * NS_i * PC_i * SW_i

    alpha_i  -- frame coefficient from G^+ s (removes redundancy)
    s_i      -- cosine proximity <mu_i, x>
    NS_i     -- neighborhood support: mean cosine of centroid neighbors
    PC_i     -- path coherence: ancestors score consistently with node
    SW_i     -- specificity weight: (level+1)^0.3 / sigma^0.5

    A node survives only if its surrounding cluster agrees with it (NS),
    its lineage path is consistent (PC), and it is specific enough (SW).
    """
    candidates = [r for r in results
                  if r["node"] not in SKIP_NODES
                  and tree.get(r["node"], {}).get("centroid") is not None][:top_m]

    if len(candidates) < 2:
        return {}

    names  = [r["node"] for r in candidates]
    s      = np.array([r["score"] for r in candidates], dtype=np.float64)
    sigmas = np.array([max(float(tree[n].get("sigma") or 0.35), 0.30)
                       for n in names], dtype=np.float64)
    levels = np.array([int(tree[n].get("level", 0)) for n in names], dtype=np.float64)

    # Score lookup for all nodes (for path coherence)
    score_map = {r["node"]: r["score"] for r in results}

    M = np.array([tree[n]["centroid"] for n in names], dtype=np.float64)

    # ── Gram matrix and pseudoinverse ────────────────────────────────────
    G = M @ M.T
    U, sv, Vt = np.linalg.svd(G)
    sv_inv = np.where(sv > svd_threshold * sv[0], 1.0 / sv, 0.0)
    G_pinv = (Vt.T * sv_inv) @ U.T
    alpha = G_pinv @ s

    if np.all(np.abs(alpha) < 1e-9):
        logging.getLogger("coredrill").warning(
            "gram_inverse: rank-deficient Gram matrix - alpha collapsed to zero; returning empty")
        return {}

    # ── Neighborhood support NS_i ────────────────────────────────────────
    # NS_i = mean cosine score of semantic peers: nodes correlated with i
    # BUT at similar level (within ±1 level).
    #
    # Exclude ancestors/descendants from neighborhood: parent-child
    # correlations (G_ij > 0.5 because one is ancestor of the other)
    # are not independent evidence - they inflate NS for broad L0/L1 nodes.
    #
    # Semantic peers: same level ± 1, G_ij > neighbor_threshold.
    NS = np.zeros(len(names))
    neighbor_threshold = 0.50
    for i in range(len(names)):
        level_i = levels[i]
        peers = [j for j in range(len(names))
                 if j != i
                 and G[i, j] > neighbor_threshold
                 and abs(levels[j] - level_i) <= 1]
        if peers:
            NS[i] = np.mean(s[peers])
        else:
            # No same-level correlated peers - mild self-support
            NS[i] = s[i] * 0.8

    # Normalize NS to [0, 1] relative to max s
    s_max = s.max() if s.max() > 0 else 1.0
    NS = NS / s_max

    # ── Path coherence PC_i ──────────────────────────────────────────────
    # For each node, check if ancestors also score reasonably
    # PC_i = mean over ancestors of clamp(s_ancestor / s_i, 0, 1)
    # PC = 1 when all ancestors score >= node (fully coherent path)
    # PC < 1 when ancestors score less (orphaned node, suspicious signal)
    PC = np.ones(len(names))
    if coredrill is not None:
        backbone = coredrill.get("taxonomy_backbone", {})
        lineage_map = backbone.get("lineage_paths", {})

        for i, name in enumerate(names):
            paths = lineage_map.get(name, [])
            if not paths:
                continue
            # Use first path, strip SKIP_NODES
            path = [n for n in paths[0] if n not in SKIP_NODES and n != name]
            if not path:
                continue

            ancestor_scores = []
            for anc in path:
                anc_score = score_map.get(anc)
                if anc_score is not None and s[i] > 1e-9:
                    # How much does the ancestor support this node?
                    # If ancestor scores >= node: full support (clamped to 1)
                    # If ancestor scores less: partial support
                    ratio = min(anc_score / s[i], 1.0)
                    ancestor_scores.append(ratio)

            if len(ancestor_scores) >= 2:
                # Mean ratio - product would be too aggressive
                # Require at least 2 ancestors to compute meaningful PC
                PC[i] = float(np.mean(ancestor_scores))
            elif len(ancestor_scores) == 1:
                # Only one ancestor found in score_map - weak signal
                # Apply a mild discount to avoid rewarding shallow paths
                PC[i] = float(np.mean(ancestor_scores)) * 0.85
            else:
                # No ancestors found in score_map - penalize
                # This catches generic L4/L5 speech-act nodes (Ask, Clarify)
                # whose ancestors (Conversation & Dialogue, etc.) don't appear
                # in the top-m candidates because the text is not about dialogue
                PC[i] = 0.60

    # ── Specificity weight SW_i ──────────────────────────────────────────
    # sigma^-0.5: tighter nodes get mild bonus
    # Level bonus removed: taxonomic depth != semantic specificity.
    # A deep generic node (Ask, L5) should not outrank a shallower
    # semantically precise node (Grammar Explanation, L4).
    SW = 1.0 / (sigmas ** 0.5)

    # Normalize SW so it doesn't dominate - scale to mean=1
    SW = SW / SW.mean()

    # ── Final score ──────────────────────────────────────────────────────
    # Only nodes with positive alpha are meaningful frame contributors.
    # Negative alpha means the node actively contradicts x's position.
    # Full formula: R_i = alpha_i * s_i * NS_i * PC_i * SW_i
    R = alpha * s * NS * PC * SW

    return {names[i]: float(R[i]) for i in range(len(names))}


def analyze_text_rhizome_v2(coredrill: dict, text: str, model,
                             tau: float = 0.1,
                             print_output: bool = True,
                             top_m: int = 20,
                             svd_threshold: float = 0.05) -> dict:
    """
    Drop-in replacement for analyze_text_rhizome that adds Gram-inverse
    scores to the result.

    Adds to result dict:
        "gram_scores"  -- dict: node -> alpha_i / sigma_i
        "gram_winner"  -- node with highest positive gram_score
        "second_moment_scores" -- {} (kept for API compatibility)
        "second_moment_winner" -- None
    """
    result = analyze_text_rhizome(coredrill, text, model,
                                  tau=tau, print_output=print_output)
    try:
        tree = coredrill.get("tree", {})
        g_scores = gram_inverse_scores(
            result.get("results", []), tree,
            top_m=top_m, svd_threshold=svd_threshold,
            coredrill=coredrill)

        positive = {n: v for n, v in g_scores.items() if v > 0}
        winner = max(positive, key=positive.get) if positive else (
                 max(g_scores, key=g_scores.get) if g_scores else None)

        # L0 winner = void signal: the scoring could not localize the text.
        # Fall back to recommended_winner (best production score).
        gram_state = "ok"
        if winner and tree.get(winner, {}).get("level", -1) == 0:
            gram_state = "void"
            rec = result.get("recommended_winner")
            if rec and isinstance(rec, dict) and rec.get("node"):
                winner = rec["node"]
            elif result.get("results"):
                winner = result["results"][0]["node"]

        result["gram_scores"]  = g_scores
        result["gram_winner"]  = winner
        result["gram_state"]   = gram_state
    except Exception as e:
        logging.getLogger("coredrill.api").error(
            f"gram_inverse failed: {type(e).__name__}: {e}", exc_info=True)
        result["gram_scores"]  = {}
        result["gram_winner"]  = None
        result["gram_state"]   = "error"

    result["second_moment_scores"] = {}
    result["second_moment_winner"] = None
    return result

if __name__ == "__main__":
    main()
