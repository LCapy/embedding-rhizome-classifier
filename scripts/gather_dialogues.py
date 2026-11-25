"""Gathers real conversation lines for the territories that name something people
do in conversation.

    python scripts/gather_dialogues.py --sources C:/path/to/dialogue-sources

Articles about arguing are not arguments, so these territories get lines of
real dialogue instead, each chosen by what that line does (its annotated
dialogue act), not by what its whole conversation is about:

  MultiWOZ 2.2       dialogues_001.json, dialog_acts.json       (MIT)
  Taskmaster-1       woz-dialogs.json                           (CC BY 4.0)
  Schema-Guided      sgd/dialogues_*.json                       (CC BY-SA 4.0)

Writes data/sources/dialogues.jsonl, at most --target lines per territory, at
most two from one conversation, none the corpus already has. The corpus itself
is not touched.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.data import text_key  # noqa: E402

Line = Tuple[str, str, str, str, str]            # territory, text, source, conversation, why
GREETING = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening)|greetings)\b", re.I)
ORDER = re.compile(r"\b(i('d| would) like|can i (get|have)|i('ll| will) (have|take|get)|i want|could i (get|have)|order)\b", re.I)
FOOD = re.compile(r"\b(pizza|coffee|latte|cappuccino|mocha|espresso|tea|table|topping|cheese|pepperoni|crust|small|medium|"
                  r"large|milk|sauce|drink|salad|menu|dinner|lunch|seat|party of|sugar|iced)\b", re.I)
RIDE = re.compile(r"\b(ride|pick(ed)? (me )?up|drop|going to|go to|get to|from|to the|airport|station|downtown)\b", re.I)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def pair(a: str, b: str) -> str:
    return f"{clean(a)}\n{clean(b)}"


def sgd(folder: Path) -> Iterator[Line]:
    for f in sorted(folder.glob("dialogues_*.json")):
        for d in json.loads(f.read_text(encoding="utf-8")):
            turns = d["turns"]
            acts = [frozenset(a["act"] for fr in t["frames"] for a in fr["actions"]) for t in turns]
            for i, (t, a) in enumerate(zip(turns, acts)):
                say, user, key = t["utterance"], t["speaker"] == "USER", f"sgd:{f.stem}:{d['dialogue_id']}"
                services = {fr["service"].split("_")[0] for fr in t["frames"]}
                before = acts[i - 1] if i else frozenset()
                after = turns[i + 1]["utterance"] if i + 1 < len(turns) else ""
                then = acts[i + 1] if i + 1 < len(turns) else frozenset()
                if user and a in ({"AFFIRM"}, {"AFFIRM_INTENT"}) and len(say) > 14:
                    yield "Agree", say, "sgd", key, "affirms"
                if user and a == {"INFORM"} and before == {"REQUEST"}:
                    yield "Answer", say, "sgd", key, "answers a question"
                if not user and a == {"REQUEST"}:
                    yield "Clarifying Question", say, "sgd", key, "asks for a missing detail"
                    if then == {"INFORM"}:
                        yield "Clarification Exchange", pair(say, after), "sgd", key, "asks, then is told"
                if user and a == {"REQUEST"}:
                    yield "Ask", say, "sgd", key, "asks"
                    if then == {"INFORM"}:
                        yield "Question–Answer", pair(say, after), "sgd", key, "asks, then is answered"
                if user and a in ({"INFORM_INTENT"}, {"INFORM", "INFORM_INTENT"}):
                    yield "Request", say, "sgd", key, "asks for something to be done"
                    if not then & {"REQUEST"} and after:
                        yield "Request–Response Dialogue", pair(say, after), "sgd", key, "requests, then gets a reply"
                if not user and a == {"INFORM_COUNT", "OFFER"}:
                    yield "Choice Recommendation", say, "sgd", key, "offers one of several options"
                if user and a == {"REQUEST_ALTS"} and then & {"OFFER"}:
                    yield "Recommendation Dialogue", pair(say, after), "sgd", key, "asks for another option, then is offered one"
                if user and services == {"Calendar"} and a & {"INFORM", "INFORM_INTENT", "REQUEST"} and len(say) > 24:
                    yield "Calendar Coordination", say, "sgd", key, "sets or checks a calendar entry"
                if user and services == {"Services"} and "INFORM" in a and re.search(
                        r"\d ?[ap]\.?m\b|o'? ?clock|\b(morning|afternoon|evening|noon|monday|tuesday|wednesday|thursday|"
                        r"friday|saturday|sunday|tomorrow|today|next week|march)\b|\d(st|nd|rd|th)\b", say, re.I):
                    yield "Planning & Scheduling", say, "sgd", key, "sets an appointment time"
                if user and services == {"RideSharing"} and a & {"INFORM", "INFORM_INTENT"} and RIDE.search(say):
                    yield "Move", say, "sgd", key, "asks for a ride"


def multiwoz(folder: Path) -> Iterator[Line]:
    acts = json.loads((folder / "dialog_acts.json").read_text(encoding="utf-8"))
    for d in json.loads((folder / "dialogues_001.json").read_text(encoding="utf-8")):
        by_turn = acts.get(d["dialogue_id"], {})
        turns, key = d["turns"], f"multiwoz:{d['dialogue_id']}"
        kinds = [frozenset(by_turn.get(str(t["turn_id"]), {}).get("dialog_act", {})) for t in turns]
        for i, (t, a) in enumerate(zip(turns, kinds)):
            say, user = t["utterance"], t["speaker"] == "USER"
            main = {x.split("-")[1] for x in a if not x.startswith("general")}
            if not user and main == {"Recommend"}:
                yield "Recommend", say, "multiwoz22", key, "recommends"
            if user and main == {"Request"}:
                yield "Information Seeking", say, "multiwoz22", key, "asks for information"
            if i + 3 < len(turns) and i % 4 == 0 and i:
                yield ("Dialogue Scene", "\n".join(clean(x["utterance"]) for x in turns[i:i + 4]), "multiwoz22", key,
                       "four turns of a conversation")


def taskmaster(folder: Path) -> Iterator[Line]:
    for d in json.loads((folder / "woz-dialogs.json").read_text(encoding="utf-8")):
        kind, turns, key = d["instruction_id"], d["utterances"], f"taskmaster:{d['conversation_id']}"
        if len(turns) > 1 and GREETING.match(turns[0]["text"]):
            if 8 <= len(turns[0]["text"]) <= 48 and not ORDER.search(turns[0]["text"]):
                yield "Greet", turns[0]["text"], "taskmaster1_woz", key, "opens the conversation"
            if len(turns[0]["text"]) <= 90 and GREETING.match(turns[1]["text"]) and len(turns[1]["text"]) <= 110:
                yield "Greeting Exchange", pair(turns[0]["text"], turns[1]["text"]), "taskmaster1_woz", key, "greets, and is greeted"
        for t in turns:
            say = t["text"]
            if t["speaker"] != "USER":
                continue
            if kind.startswith(("pizza", "coffee", "restaurant")) and ORDER.search(say) and FOOD.search(say) and len(say) >= 25:
                yield "Ordering at a Restaurant", say, "taskmaster1_woz", key, "orders"
            if kind.startswith("uber") and ORDER.search(say) and RIDE.search(say) and len(say) >= 25:
                yield "Move", say, "taskmaster1_woz", key, "asks for a ride"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--per-conversation", type=int, default=2)
    ap.add_argument("--data", type=Path, default=REPO / "data")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    known = {text_key(r["text"]) for r in json.loads((args.data / "sentences.json").read_text(encoding="utf-8"))}
    found: Dict[str, List[Line]] = defaultdict(list)
    streams = [sgd(args.sources / "sgd"), multiwoz(args.sources), taskmaster(args.sources)]
    for stream in streams:
        for line in stream:
            found[line[0]].append(line)

    rng, rows = random.Random(42), []
    for territory in sorted(found):
        lines = found[territory]
        rng.shuffle(lines)
        seen, per, kept = set(known), Counter(), []
        for _, text, source, conversation, why in lines:
            text = text.strip()
            key = text_key(text)
            if len(text) < 8 or key in seen or per[conversation] >= args.per_conversation:
                continue
            seen.add(key)
            per[conversation] += 1
            kept.append({"text": text, "category": territory, "lang": "en", "source": source,
                         "conversation": conversation, "act": why})
            if len(kept) >= args.target:
                break
        rows += kept
        print(f"  {territory:<28} {len(kept):>4} of {len(lines):>6,} lines, {len(per):>3} conversations")
    # a line belongs to one territory: the one with fewer lines keeps it
    size, owner = Counter(r["category"] for r in rows), {}
    for r in sorted(rows, key=lambda r: size[r["category"]]):
        owner.setdefault(text_key(r["text"]), r["category"])
    rows = [r for r in rows if owner[text_key(r["text"])] == r["category"]]
    out = args.data / "sources" / "dialogues.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows):,} lines for {len(size)} territories -> {out}")


if __name__ == "__main__":
    main()
