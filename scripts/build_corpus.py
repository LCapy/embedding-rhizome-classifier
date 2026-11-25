"""Builds the corpus again from its parts, with one label per text and an even
share for every territory.

    python scripts/build_corpus.py --out data/v3 --runs runs/v3
    python scripts/rhizome_engine.py embed --data data/v3
    python -m train baseline --data data/v3 --runs runs/v3

Reads the corpus in --data and what gather_territories.py and
gather_dialogues.py wrote to data/sources/, and changes neither. What it does:

  1. One label per text. Siblings that shared an article (Family Life, Family
     Discussion and Family Planning Talk all held "Family") keep it once: the
     territory the article is named for, else the first that lists it.
  2. Corrected labels. CFPB complaints were spread over labels by keyword; they
     are complaints (billing ones: billing). ChatDoctor messages were labeled
     by any medical word; they are labeled by who speaks and what they do.
     Bitext's lines are left out: they are generated from templates.
  3. No articles about talking. Territories that name something people do in
     conversation keep only real conversation.
  4. The gathered text is added.
  5. A new holdout: the old one's texts, with corrected labels, plus a share of
     the new articles and conversations, whole. Nothing in it was in any
     earlier model's training data, so old and new models can be compared.

Records keep where they came from (`was`, `why`). report.md in --out has the
counts per territory.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.data import KEY_FORMAT, concept, eng, record_keys, text_key, title_key  # noqa: E402

# Things people do in conversation: an encyclopedia article is not an example.
CONVERSATIONAL = {
    "Argument & Persuasion", "Persuade", "Recommendation Dialogue", "Choice Recommendation", "Recommend",
    "Dialogue Scene", "Request–Response Dialogue", "Answer", "Request", "Small Talk", "Greeting Exchange", "Greet",
    "Instruction & Procedure", "How-to Guidance", "Guide", "Step-by-Step Instruction", "Instruct",
    "Planning & Scheduling", "Calendar Coordination", "Personal Story", "Narrate", "Describe", "Question–Answer",
    "Clarification Exchange", "Clarifying Question", "Ask", "Information Seeking", "Clarify", "Explain",
    "Joint Planning", "Plan", "Disagreement & Argument", "Critique & Rebuttal", "Criticize", "Rebut",
    "Verbal Dispute", "Household Argument", "Argue", "Confirm", "Project Agreement", "Agree", "Price Bargaining",
    "Negotiate", "Ordering at a Restaurant", "Home Repair Request", "Choose", "Buy", "Move",
    "Election Campaign Speech", "Campaign", "Voting Instructions", "Apply", "Comply", "Comfort",
    "Emotional Reassurance", "Reassure", "Family Discussion", "Family Planning Talk", "Friendly Check-in",
    "Relationship Argument", "Apologize", "Troubleshoot", "Status Update", "Report", "Task Assignment", "Assign",
    "Double Charge Complaint", "Complain", "Refund", "Late Delivery Problem", "Conversation & Dialogue",
}
ENCYCLOPEDIA = ("wikipedia", "simple_wikipedia", "wikipedia_main_article", "wikipedia_new_nodes")

TWICE = re.compile(r"charged (me )?(twice|two times|double)|double[- ]charg|duplicate charge|billed (me )?twice", re.I)
BILLING = re.compile(r"\b(late fees?|overdraft|annual fee|finance charge|interest charge|billing|billed|over ?charged|"
                     r"charged me|refund|autopay|auto[- ]pay|minimum payment|statement balance)\b", re.I)
REPORTING = re.compile(r"credit (report|bureau|file)|consumer report|inquir|equifax|experian|trans ?union|fcra|15 u\.?s", re.I)
APPOINTMENT = re.compile(r"\b(appointment|reschedul|book(ed|ing)? (a|an|my) (visit|consult))", re.I)
PRESCRIBES = re.compile(r"\b(\d+ ?mg|tablets?|twice (a day|daily)|antibiotic|prescri|dose|course of|ointment|you can take)\b", re.I)
DIAGNOSES = re.compile(r"\b(diagnos|most likely|could be due to|suggestive of|consistent with|possibility of)\b", re.I)
ROUTINE = re.compile(r"\b(exercise|diet|check-?ups?|healthy|weight loss|sleep|vitamins?|workout|calories)\b", re.I)


def relabel(r: dict) -> Tuple[Optional[str], str]:
    """(label, why) for a record whose label came from a keyword; None drops it."""
    text, source, label = r["text"], r.get("source", ""), r["category"]
    if source == "bitext":
        return None, "a Bitext line (generated from templates)"
    if source == "cfpb":
        if len(text) < 60 or text.count("XX") > max(3, len(text) // 60):
            return None, "a complaint that is mostly redacted"
        if TWICE.search(text):
            return "Double Charge Complaint", "a complaint about being charged twice"
        if BILLING.search(text) and not REPORTING.search(text):
            if re.search(r"\brefund", text, re.I):
                return "Refund Processing", "a complaint about a refund"
            return "Billing & Refunds", "a billing complaint"
        return "Complaint Handling", "a complaint"
    if source == "medical_chatdoctor_patient":
        if APPOINTMENT.search(text):
            return "Doctor Appointment", "a patient on an appointment"
        if len(ROUTINE.findall(text)) >= 2:
            return "Health Routine", "a patient on diet, exercise or sleep"
        return "Symptoms & Self-Care", "a patient describing symptoms"
    if source == "medical_chatdoctor_doctor":
        if PRESCRIBES.search(text):
            return "Treatment", "a doctor prescribing"
        if DIAGNOSES.search(text):
            return "Diagnosis", "a doctor diagnosing"
        return "Medicine", "a doctor's answer"
    return label, ""


def article(r: dict) -> str:
    c = concept(r)
    return title_key(c) if c else ""


def choose(labels: List[str], title: str, listed: Dict[str, str], first: Dict[str, List[str]], level: Dict[str, int]) -> str:
    """Which of several labels keeps a text."""
    if title in listed and listed[title] in labels:
        return listed[title]

    def rank(l: str) -> tuple:
        names = [title_key(t) for t in first.get(l, [])]
        return (title_key(l) == title, bool(names) and names[0] == title, title in names,
                title in title_key(l), l not in CONVERSATIONAL, -level.get(l, 9), l)
    return max(labels, key=rank)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO / "data")
    ap.add_argument("--sources", type=Path, default=REPO / "data" / "sources")
    ap.add_argument("--holdout", type=Path, default=REPO / "runs" / "holdout.json", help="the earlier holdout")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True, help="where the new holdout goes")
    ap.add_argument("--test-share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    records = json.loads((args.data / "sentences.json").read_text(encoding="utf-8"))
    with np.load(args.data / "embeddings.npz") as z:
        emb = z["embeddings"]
    if len(emb) != len(records):
        raise SystemExit("sentences.json and embeddings.npz are out of step")
    taxonomy = eng.load_taxonomy(args.data / "TAXONOMY_fixed.txt")
    level = {n["name"]: n["level"] for n in taxonomy}
    unknown = CONVERSATIONAL - set(level)
    if unknown:
        raise SystemExit(f"not in the taxonomy: {sorted(unknown)}")
    first = {n["name"]: eng.WIKI_ARTICLES.get(n["name"]) or [n.get("en_category") or n["name"]] for n in taxonomy}
    listed: Dict[str, str] = {}
    for line in (REPO / "scripts" / "territory_articles.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, _, titles = line.partition(": ")
            listed.update({title_key(t): name for t in titles.split("|")})
    old = json.loads(args.holdout.read_text(encoding="utf-8"))
    held_before = {k.split("|", 1)[0] for k in old["keys"]}

    # 1-3: the corpus as it was, corrected
    dropped: Counter = Counter()
    by_text: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_text[text_key(r["text"])].append(i)
    kept: List[dict] = []
    rows: List[int] = []
    for key, idx in by_text.items():
        options: Dict[str, Tuple[int, str]] = {}
        for i in idx:
            label, why = relabel(records[i])
            if label is None:
                dropped[why] += 1
                continue
            if label in CONVERSATIONAL and records[i].get("source") in ENCYCLOPEDIA:
                dropped["an article about something people do in conversation"] += 1
                continue
            options.setdefault(label, (i, why))
        if not options:
            continue
        title = next((article(records[i]) for i, _ in options.values() if article(records[i])), "")
        if title in listed:                      # an article goes whole to the territory that lists it
            i, _ = options.get(listed[title]) or next(iter(options.values()))
            options = {listed[title]: (i, f"the article {title!r} belongs here")}
        label = choose(sorted(options), title, listed, first, level) if len(options) > 1 else next(iter(options))
        i, why = options[label]
        dropped["the same text under another label"] += len(idx) - 1 if len(idx) > 1 else 0
        r = {**records[i], "category": label, "level": level[label]}
        if label != records[i]["category"]:
            r.update({"was": records[i]["category"], "why": why})
        kept.append(r)
        rows.append(i)

    # 4: the gathered text
    known = {text_key(r["text"]) for r in kept}
    new: List[dict] = []
    for name in ("wikipedia.jsonl", "dialogues.jsonl"):
        f = args.sources / name
        for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
            r = json.loads(line)
            if text_key(r["text"]) in known or r["category"] not in level:
                continue
            known.add(text_key(r["text"]))
            if name == "wikipedia.jsonl" and listed.get(title_key(r["article"])) != r["category"]:
                continue                         # the list changed since it was fetched
            if name == "wikipedia.jsonl":        # the form 12_by_article.py groups by article
                r = {"text": r["text"], "category": r["category"], "lang": r["lang"], "source": "wikipedia",
                     "article_title": r["source_article"], "gathered": "territory_articles.txt"}
            new.append({**r, "level": level[r["category"]]})

    # 5: the holdout
    rng = random.Random(args.seed)
    counts = Counter(r["category"] for r in kept + new)
    in_test = {text_key(r["text"]) for r in kept if text_key(r["text"]) in held_before}
    groups: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for r in new:
        g = concept(r) or r.get("conversation") or ""
        groups[r["category"]][g].append(text_key(r["text"]))
    test_now = Counter(r["category"] for r in kept if text_key(r["text"]) in in_test)
    trained_on = {concept(r) for r in kept if text_key(r["text"]) not in in_test} - {None}
    for label in sorted(groups):
        names = sorted(g for g in groups[label] if g not in trained_on)      # new to every earlier model
        rng.shuffle(names)
        want = round(args.test_share * counts[label])
        for g in names:
            if test_now[label] >= want:
                break
            size = len(groups[label][g])
            if counts[label] - test_now[label] - size < 5 or (test_now[label] and test_now[label] + size > 1.75 * want):
                continue                         # leave enough to train on, and do not overshoot
            in_test.update(groups[label][g])
            test_now[label] += size
    left = Counter(r["category"] for r in kept + new if text_key(r["text"]) not in in_test)
    in_test -= {text_key(r["text"]) for r in kept + new if not left[r["category"]]}     # nothing to learn it from
    # a whole concept is on one side, whoever it belongs to
    tested = Counter(concept(r) for r in kept + new if text_key(r["text"]) in in_test)
    trained = Counter(concept(r) for r in kept + new if text_key(r["text"]) not in in_test)
    stray = {c for c in tested if c and trained[c] > tested[c]}       # a few shared sentences of a training article
    in_test -= {text_key(r["text"]) for r in kept + new if concept(r) in stray}
    test_concepts = {concept(r) for r in kept + new if text_key(r["text"]) in in_test} - {None}
    split = {c for c in test_concepts if trained[c]}
    moved = 0
    final: List[dict] = []
    final_rows: List[int] = []
    for r, i in list(zip(kept, rows)) + [(r, -1) for r in new]:
        if concept(r) in split and text_key(r["text"]) not in in_test:
            moved += 1                           # text of a held-out article that is not in the test: leave it out
            continue
        final.append(r)
        final_rows.append(i)
    keys = record_keys(final)
    test_keys = sorted(k for k, r in zip(keys, final) if text_key(r["text"]) in in_test)

    args.out.mkdir(parents=True, exist_ok=True)
    args.runs.mkdir(parents=True, exist_ok=True)
    (args.out / "sentences.json").write_text(
        "[\n" + ",\n".join(json.dumps(r, ensure_ascii=False) for r in final) + "\n]\n", encoding="utf-8")
    reuse = [i for i in final_rows if i >= 0]
    if final_rows[:len(reuse)] != reuse:
        raise SystemExit("records with an embedding must come first")
    np.savez_compressed(args.out / "embeddings.npz", embeddings=emb[reuse])
    shutil.copy2(args.data / "TAXONOMY_fixed.txt", args.out / "TAXONOMY_fixed.txt")
    (args.runs / "holdout.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "method": "scripts/build_corpus.py: the earlier holdout's texts with "
        "corrected labels, plus whole new articles and conversations", "seed": args.seed, "key_format": KEY_FORMAT,
        "n_records": len(test_keys), "keys": test_keys,
        "blocked_titles": sorted({title_key(c) for c in test_concepts}
                                 | set(old["blocked_titles"]))}, indent=1), encoding="utf-8")

    # report
    test_set = set(test_keys)
    tr = Counter(r["category"] for r, k in zip(final, keys) if k not in test_set)
    te = Counter(r["category"] for r, k in zip(final, keys) if k in test_set)
    before = Counter(r["category"] for r in records)
    arts = defaultdict(set)
    for r in final:
        arts[r["category"]].add(concept(r) or r.get("conversation") or r.get("source"))
    lines = ["# The corpus, rebuilt", "", f"{len(records):,} records before, {len(final):,} now "
             f"({len(reuse):,} kept, {len(final) - len(reuse):,} new). {len(test_keys):,} in the holdout, testing "
             f"{len(te)} territories. One label per text.", "", "Left out:", ""]
    lines += [f"- {n:,} {why}" for why, n in dropped.most_common()] + [f"- {moved:,} text of a held-out article", ""]
    lines += ["| Territory | Level | Before | Training | Holdout | Articles or conversations |", "|---|--:|--:|--:|--:|--:|"]
    for n in sorted(level, key=lambda n: (tr[n] + te[n], n)):
        lines.append(f"| {n} | {level[n]} | {before[n]:,} | {tr[n]:,} | {te[n]:,} | {len(arts[n]):,} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sizes = [tr[n] for n in level if tr[n]]
    print("\n".join(lines[2:9]))
    print(f"territories with training texts: {len(sizes)} of {len(level)}; training texts per territory: min {min(sizes)}, "
          f"median {int(np.median(sizes))}, max {max(sizes)}; under 40: {sum(s < 40 for s in sizes)}")
    print(f"-> {args.out} and {args.runs / 'holdout.json'}; embed the {len(final) - len(reuse):,} new texts next")


if __name__ == "__main__":
    main()
