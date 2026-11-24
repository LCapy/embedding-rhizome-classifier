"""Corpus, labels, the fixed holdout split and taxonomy helpers for the trainer."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import rhizome_engine as eng  # noqa: E402

MIN_COUNT = 5   # a category is a label once it has this many records (eval/held_out/common.py)
SEED = 42


def text_key(text: str) -> str:
    """Identity of a text, for deduplication and the holdout list."""
    norm = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def concept(r: dict) -> Optional[str]:
    """Same grouping as eval/held_out/12_by_article.py: every language version
    of a Wikipedia article, plus its Simple English version, is one concept.
    Records with no article return None."""
    src = r.get("source") or ""
    if src == "wikipedia_main_article":
        return "wiki:" + (r.get("source_article") or r.get("article_title") or "").lower()
    if src in ("wikipedia", "simple_wikipedia"):
        title = r.get("article_title") or ""
        if title.startswith("simple:"):
            title = title.split(":", 1)[1]
        return "wiki:" + title.lower()
    return None


_LANG_PREFIX = re.compile(r"^(?:simple|[a-z]{2,3}(?:-[a-z]+)?):", re.IGNORECASE)


def title_key(title: str) -> str:
    """A title without "wiki:" or a language prefix, lowercased, so that
    "wiki:en:Black hole", "simple:Black hole" and "Black hole" compare equal."""
    t = (title or "").strip()
    if t.lower().startswith("wiki:"):
        t = t[5:]
    return _LANG_PREFIX.sub("", t, count=1).replace("_", " ").strip().lower()


def english_title(r: dict) -> Optional[str]:
    """The English Wikipedia title a record came from, original case, if any."""
    src = r.get("source") or ""
    if src == "wikipedia_main_article":
        t = r.get("source_article") or ""
    elif src in ("wikipedia", "simple_wikipedia", "wikipedia_harvest"):
        t = r.get("source_article") or r.get("article_title") or ""
    else:
        return None
    t = _LANG_PREFIX.sub("", t.strip(), count=1).strip()
    return t or None


@dataclass
class Corpus:
    records: List[dict]
    f0: np.ndarray                 # stock LaBSE embedding per record, unit rows
    labels: List[str]              # sorted categories with at least MIN_COUNT records
    y: np.ndarray                  # label index per record, -1 when it is not a label
    taxonomy: List[dict]
    parents: Dict[str, List[str]]
    children: Dict[str, List[str]]

    @property
    def label_index(self) -> Dict[str, int]:
        return {l: i for i, l in enumerate(self.labels)}


def default_taxonomy(data_dir: Path) -> Path:
    fixed = data_dir / "TAXONOMY_fixed.txt"
    return fixed if fixed.exists() else REPO / "api" / "taxonomy.py"


def load_corpus(data_dir: Path, taxonomy_path: Optional[Path] = None) -> Corpus:
    sentences = data_dir / "sentences.json"
    if not sentences.exists():
        raise SystemExit(f"{sentences} not found. Copy data/ (sentences.json, embeddings.npz, "
                         "TAXONOMY_fixed.txt) into place first.")
    records = json.loads(sentences.read_text(encoding="utf-8"))
    with np.load(data_dir / "embeddings.npz") as f:
        f0 = f["embeddings"].astype(np.float32)
    if len(records) != len(f0):
        raise SystemExit(f"sentences.json has {len(records)} records but embeddings.npz has {len(f0)} rows; "
                         "run `python scripts/rhizome_engine.py embed --data <dir>` first")
    norms = np.linalg.norm(f0, axis=1, keepdims=True)
    if np.abs(norms - 1.0).max() > 1e-3:      # LaBSE's are unit already; leave those bit for bit
        f0 /= norms + 1e-12
    counts = Counter(r.get("category", "") for r in records)
    labels = sorted(c for c, n in counts.items() if c and n >= MIN_COUNT)
    index = {l: i for i, l in enumerate(labels)}
    y = np.array([index.get(r.get("category", ""), -1) for r in records], dtype=np.int64)
    taxonomy = eng.load_taxonomy(Path(taxonomy_path) if taxonomy_path else default_taxonomy(data_dir))
    return Corpus(records, f0, labels, y, taxonomy,
                  eng.taxonomy_parents_map(taxonomy), eng.taxonomy_children_map(taxonomy))


# ── taxonomy helpers ─────────────────────────────────────────────────────────

def ancestor_depths(node: str, parents: Dict[str, List[str]]) -> Dict[str, int]:
    """Every ancestor of node with its distance (parent = 1)."""
    out: Dict[str, int] = {}
    frontier, depth = list(parents.get(node, [])), 1
    while frontier:
        nxt = []
        for p in frontier:
            if p not in out:
                out[p] = depth
                nxt.extend(parents.get(p, []))
        frontier, depth = nxt, depth + 1
    return out


def near_miss(true: str, pred: Optional[str], parents: Dict[str, List[str]]) -> bool:
    """eval/held_out/05_full_pipeline.py's definition: the label itself, an
    ancestor or descendant of it, or a sibling."""
    if pred is None:
        return False
    if pred == true or pred in ancestor_depths(true, parents) or true in ancestor_depths(pred, parents):
        return True
    return bool(set(parents.get(true, [])) & set(parents.get(pred, [])))


def neighborhood(node: str, parents: Dict[str, List[str]], children: Dict[str, List[str]]) -> Set[str]:
    """node, its ancestors, its descendants and its siblings."""
    out = {node} | set(ancestor_depths(node, parents))
    stack = list(children.get(node, []))
    while stack:
        c = stack.pop()
        if c not in out:
            out.add(c)
            stack.extend(children.get(c, []))
    for p in parents.get(node, []):
        out |= set(children.get(p, []))
    return out


def soft_target_matrix(labels: List[str], parents: Dict[str, List[str]], eps: float) -> np.ndarray:
    """Row k is the training target for label k: 1 - eps on the label, eps
    shared by its ancestors that are labels too (halving per level up). A
    label with no labeled ancestor keeps all its mass. Parents learn to sit
    around their children, and a near miss costs less than an unrelated one."""
    index = {l: i for i, l in enumerate(labels)}
    T = np.zeros((len(labels), len(labels)), dtype=np.float32)
    for k, label in enumerate(labels):
        anc = [(index[a], d) for a, d in ancestor_depths(label, parents).items() if a in index]
        if not anc or eps <= 0:
            T[k, k] = 1.0
            continue
        T[k, k] = 1.0 - eps
        w = np.array([0.5 ** (d - 1) for _, d in anc], dtype=np.float64)
        for (a, _), share in zip(anc, w / w.sum() * eps):
            T[k, a] += share
    return T


# ── the fixed holdout ────────────────────────────────────────────────────────

def by_article_split(records: List[dict], y: np.ndarray, n_classes: int, test_frac: float = 0.2,
                     max_test_frac: float = 0.35, min_train: int = 5,
                     seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """eval/held_out/12_by_article.py's by_article_split(), line for line, so
    the holdout here is the one docs/evaluation.md reports on: whole Wikipedia
    concepts go to test, as long as every class they feed keeps min_train
    training records; records with no article are split by record."""
    rng = np.random.default_rng(seed)
    labeled = np.where(y >= 0)[0]
    total = np.bincount(y[labeled], minlength=n_classes)
    groups, loose = defaultdict(list), defaultdict(list)
    for i in labeled:
        k = concept(records[i])
        if k:
            groups[k].append(i)
        else:
            loose[y[i]].append(i)

    test, test_cnt = [], np.zeros(n_classes, dtype=int)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        idx = np.array(groups[k])
        cls = np.bincount(y[idx], minlength=n_classes)
        inv = cls > 0
        after = test_cnt + cls
        wanted = np.any(test_cnt[inv] < test_frac * total[inv])
        fits = np.all(after[inv] <= max_test_frac * total[inv]) and np.all(total[inv] - after[inv] >= min_train)
        if wanted and fits:
            test.extend(idx.tolist())
            test_cnt = after
    for c, idx in loose.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        take = int(max(0, min(round(test_frac * total[c]) - test_cnt[c], len(idx) - min_train)))
        test.extend(idx[:take].tolist())
        test_cnt[c] += take
    test_idx = np.array(sorted(test))
    train_idx = np.setdiff1d(labeled, test_idx)
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


KEY_FORMAT = "text|category|occurrence"


@dataclass
class Holdout:
    keys: Set[str]             # record_keys() of every holdout record
    blocked_titles: Set[str]   # title_key of every Wikipedia concept in the holdout

    @property
    def texts(self) -> Set[str]:
        """text_key of every holdout record."""
        return {k.split("|", 1)[0] for k in self.keys}


def _record_key(r: dict) -> str:
    return f"{text_key(r.get('text', ''))}|{r.get('category', '')}"


def record_keys(records: List[dict]) -> List[str]:
    """One key per record, "<text_key>|<category>|<n>", where n counts the
    earlier records with the same text and category. The count keeps exact
    duplicates apart, so the holdout holds the very copy that
    12_by_article.py's index split put in test: 22 records of the corpus
    duplicate a test record and sit on its training side, and they stay there,
    as in the reference numbers. Records appended later get later counts and
    go to training."""
    seen: Counter = Counter()
    out = []
    for r in records:
        k = _record_key(r)
        out.append(f"{k}|{seen[k]}")
        seen[k] += 1
    return out


def load_or_create_holdout(corpus: Corpus, path: Path, log=print) -> Holdout:
    """The holdout is drawn once and saved, so every later run, and every
    self-training round, is scored on the same records even after the corpus
    grows. Records added later always go to training. Delete the file to draw
    a new one (results before and after are then not comparable)."""
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("key_format") != KEY_FORMAT:
            raise SystemExit(f"{path} uses an older key format; delete it to draw the holdout again")
        return Holdout(set(d["keys"]), set(d["blocked_titles"]))
    _, test_idx = by_article_split(corpus.records, corpus.y, len(corpus.labels))
    all_keys = record_keys(corpus.records)
    keys = sorted(all_keys[i] for i in test_idx)
    blocked = sorted({title_key(c) for i in test_idx if (c := concept(corpus.records[i]))})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "by_article_split (eval/held_out/12_by_article.py)", "seed": SEED,
        "key_format": KEY_FORMAT, "n_records": int(len(test_idx)), "keys": keys, "blocked_titles": blocked,
    }, indent=1), encoding="utf-8")
    log(f"Drew the holdout: {len(test_idx):,} records, {len(blocked):,} Wikipedia concepts -> {path}")
    return Holdout(set(keys), set(blocked))


def split_indices(corpus: Corpus, holdout: Holdout) -> Tuple[np.ndarray, np.ndarray]:
    """(train, holdout) indices among the labeled records."""
    labeled = np.where(corpus.y >= 0)[0]
    keys = record_keys(corpus.records)
    held = np.array([keys[i] in holdout.keys for i in labeled], dtype=bool)
    return labeled[~held], labeled[held]


def load_extra_labeled(folder: Path, labels: List[str]) -> List[dict]:
    """Hand-labeled additions: *.jsonl files of {"text", "category", "lang"?}
    in data/labeled/. They always go to training, at full weight."""
    out: List[dict] = []
    if not folder.exists():
        return out
    known = set(labels)
    for f in sorted(folder.glob("*.jsonl")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("category") in known and len(r.get("text", "").strip()) >= 20:
                out.append({"text": r["text"].strip(), "category": r["category"],
                            "source": r.get("source", f"labeled:{f.name}"), "lang": r.get("lang", "en")})
            elif r.get("category") not in known:
                print(f"  [skip] {f.name}:{n}: {r.get('category')!r} is not a label")
    return out
