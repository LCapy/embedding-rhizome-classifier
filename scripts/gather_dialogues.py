"""Gathers real conversation lines for the territories that name something people
do in conversation.

    python scripts/gather_dialogues.py --sources C:/path/to/dialogue-sources

Articles about arguing are not arguments, so these territories get lines of
real dialogue instead, each chosen by what that line does (its annotated
dialogue act), not by what its whole conversation is about:

  MultiWOZ 2.2       dialogues_001.json, dialog_acts.json       (MIT)
  Taskmaster-1       woz-dialogs.json                           (CC BY 4.0)
  Schema-Guided      sgd/dialogues_*.json                       (CC BY-SA 4.0)
  DailyDialog        dailydialog/files/*/dialogues_*.txt        (CC BY-NC-SA 4.0: not for commercial use)
  EmpatheticDialogues  empathetic/files/empatheticdialogues/    (CC BY-NC 4.0: not for commercial use)

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


# ── DailyDialog: an act and an emotion per line ───────────────────────────────

INFORM, QUESTION, DIRECTIVE, COMMISSIVE = 1, 2, 3, 4
ANGER, DISGUST, FEAR, HAPPY, SAD = 1, 2, 3, 4, 5


def has(pattern: str):
    return re.compile(pattern, re.I).search


WORK = has(r"\b(office|boss|project|report|meeting|deadline|client|colleague|schedule|team|company|department|proposal|contract|sales)\b")
HOME = has(r"\b(dinner|dishes|clean|laundry|kids?|cook|house|room|tv|bills?|garbage|trash|kitchen)\b")
FAMILY = has(r"\b(mom|mum|dad|mother|father|honey|darling|sweetheart|son|daughter|kids|grandma|grandpa|brother|sister)\b")
COUPLE = has(r"\b(honey|darling|sweetheart|marry|married|divorce|cheat|girlfriend|boyfriend|wife|husband|relationship|break up|love me)\b")
WHEN = has(r"\b(tomorrow|tonight|next (week|month|year|time)|this (weekend|afternoon|evening|summer)|on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)|weekend|after (work|class|dinner|school))\b")
CHECK_IN = has(r"\b(how (are|have) you( been)?|how's (everything|your|it going|life)|what's (wrong|the matter)|are you (ok|okay|all right|feeling)|how do you feel)\b")
SMALL = has(r"how('s| is) it going|how are you|nice (day|weather)|long time no see|what's new|how have you been|lovely (day|weather)|beautiful day|what a (nice|lovely|beautiful|hot|cold)")
SORRY = has(r"\b(i('m| am) (so |very |really |terribly |awfully )?sorry|i apologi[sz]e|my apologies|forgive me|it was my fault|i didn't mean)\b")
NOT_SORRY = has(r"sorry to (hear|trouble|bother|interrupt|say|disturb)|sorry ?,? (but|could|can|we|no|sir|madam|i (can't|don't|didn't (catch|hear)))|sorry (about|for) (your|that|the) loss|\bsorry \?")
ARGUE = has(r"\b(you('re| are) wrong|i don't (agree|believe)|that's not (true|fair)|how (can|could|dare) you|nonsense|ridiculous|it's (all )?your fault|you always|you never|shut up|stop (it|that|telling))\b")
DISAGREE = has(r"^(but |no ,|i don't (agree|think so)|i disagree|on the contrary|i'm afraid i (can't|don't) agree|that's not)")
REBUT = has(r"^(but |that's not (true|right|the point)|on the contrary|not (really|exactly|necessarily)|i (don't|can't) (agree|see it))")
CRITICIZE = has(r"\b(terrible|awful|horrible|poor|worst|lazy|careless|rude|stupid|selfish|irresponsible|disappoint\w*|you always|you never|should(n't| not) have)\b")
COMFORT = has(r"\b(don't worry|cheer up|it('s| is) (ok|okay|all right|not your fault)|i('m| am) sorry to hear|take it easy|things will|you('ll| will) (be|get|feel)|it'll be|don't (be|feel|take it)|don't give up)\b")
REASSURE = has(r"\b(don't worry|no need to worry|you'll be (fine|ok|okay|all right|great)|everything('ll| will) be|it('ll| will| is going to|'s going to) be (ok|okay|fine|all right|great)|you can do it|i'm sure you('ll| will| can)|there's nothing to (worry|be afraid))\b")
PERSUADE = has(r"\b(you should|you('d| had) better|why don't you|trust me|believe me|you have to (try|see|come|go)|you must (try|see|come|go)|you ought to|think about it|you won't regret)\b")
PLAN = has(r"\b(i('m| am) going to|i plan to|i'm planning|we('re| are) going to|i intend to|i('ll| will))\b")
JOINT = has(r"\b(let's|shall we|why don't we|how about (we|going|meeting)|we could|we should)\b")
HAGGLE = has(r"\b(discount|lower (the|your) price|best price|(final|last) (offer|price)|come down (a|on|in|to)|bottom price|meet (you )?"
             r"half ?way|reduce (the|your) price|special price|knock (off|\$?\d+)|make it \$?\d+|(can|could) you (make it|do it|let me have it) "
             r"(any )?(cheaper|for)|too (expensive|high|much) .{0,40}(can|could|would) you|price is too high)\b")
ASSIGN = has(r"\b(i('d| would)? (need|want|like) you to|please|could you|would you|make sure)\b")
GUIDE = has(r"\b(go (straight|along|down)|turn (left|right)|on (your|the) (left|right)|blocks?|cross the (street|road)|take (the |a )?(bus|subway|line|number))\b")
INSTRUCT = has(r"\b(press|push|pull|insert|turn (it )?(on|off)|click|fill (in|out)|sign (here|your)|hold|add|mix|stir|pour|cut|boil|fasten|type|dial)\b")
HOW_TO = has(r"\bhow (do|can|should) (i|you|we|one) (?!help|feel|like|do\b|think|look|say|know|want|wish|spend|find|get (along|home)|"
             r"call|thank|explain|believe|possibly|describe|mean|come)\w+|\bhow to \w+")
SERVICE = has(r"\b(sir|madam|ma'am|hotel|room number|customer|waiter|reservation|may i help|can i help)\b")
CLARIFY = has(r"\b(i mean|what i mean(t)?|in other words|that is to say|what i('m| am) (trying to )?say(ing)?|let me (explain|clarify)|to be (more )?(specific|exact|clear|precise))\b")
EXPLAIN = has(r"\b(because|the reason|that's why|it means|which means|is called|refers to|in order to|so that)\b")
ASK_LIKE = has(r"\bwhat (is|does|was|did|'s) .{0,40}\blike\b|describe|what kind of|tell me (something )?about|what does .{0,30}look")
FIX = has(r"\b(leak(s|ing)?|broken|clogged|plumber|landlord|repair(man)?|fix (it|the)|doesn't work|isn't working|out of order|stuck)\b")
HOUSEHOLD = has(r"\b(toilet|sink|faucet|pipe|heater|heating|air conditioner|window|door|light|roof|washing machine|fridge|refrigerator|apartment|shower|kitchen|bathroom|bathtub|stove|oven)\b")
DEVICE = has(r"\b(computer|printer|internet|software|program|screen|files?|password|virus|network|e-?mail|copier|machine|laptop|disk|keyboard|fax)\b")
FAULT = has(r"\b(doesn't work|won't|crash(ed)?|not working|something wrong|frozen|error|restart|reboot|unplug|problem with|jammed|check the|out of (paper|order))\b")
COMPLAIN = has(r"\b(complain(t|ing)?|unacceptable|(see|speak to|talk to|call) (the|your) manager|(poor|bad|terrible|slow|lousy) service|"
               r"(waited|waiting) (for )?(over |more than )?(an|one|two|half|\d+) ?(an )?hours?|overcharged|too nois[ey]|wrong (order|bill|dish))\b")
REFUND = has(r"\b(refund|money back|return (it|this|these|them)|exchange (it|this|these))\b")
APPLY = has(r"\b(appl(y|ying|ied) for|application( form)?s?|fill (in|out) (this|the|a|an) (application|form)|(renew|get) (my|a|your) "
            r"(visa|passport|driver'?s licen[sc]e|permit))\b")
LATE = has(r"\b(haven't (received|got)|hasn't (arrived|come)|still (not|hasn't|waiting)|delay(ed)?|never (came|arrived)|lost)\b")
PARCEL = has(r"\b(order|package|parcel|delivery|goods|shipment|luggage)\b")
DOCTOR = has(r"\b(doctor|dr ?\.|dentist|clinic)\b")
APPOINT = has(r"\b(appointment|see (a|the) doctor|check-?up|available|come in)\b")


def natural(text: str) -> str:
    """DailyDialog's spaced-out punctuation, put back."""
    text = clean(text).replace("’", "'").replace(" ' ", "'")
    text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
    return re.sub(r"([.!?,])(?=[A-Za-z])", r"\1 ", text)


def dailydialog(folder: Path) -> Iterator[Line]:
    src = "dailydialog"
    for split in ("train", "validation", "test"):
        d = folder / split
        lines, acts, moods = ((d / f"dialogues_{k}{split}.txt").read_text(encoding="utf-8").splitlines()
                              for k in ("", "act_", "emotion_"))
        for n, (line, act, mood) in enumerate(zip(lines, acts, moods)):
            says = [natural(u) for u in line.split("__eou__") if u.strip()]
            a, e = [int(x) for x in act.split()], [int(x) for x in mood.split()]
            if not len(says) == len(a) == len(e):
                continue
            key = f"dailydialog:{split}:{n}"
            if len(says) > 1 and (GREETING.match(says[0]) or SMALL(says[0])) and SMALL(says[0] + " " + says[1]):
                yield "Small Talk", pair(says[0], says[1]), src, key, "opens a casual conversation"
            for i in range(0, len(says) - 3, 6):
                yield "Conversation & Dialogue", "\n".join(says[i:i + 4]), src, key, "four turns of a conversation"
            for i, say in enumerate(says):
                size, angry = len(say), e[i] == ANGER
                before, after = (says[i - 1] if i else ""), (says[i + 1] if i + 1 < len(says) else "")
                two = pair(say, after) if after else ""
                both = f"{say} {after}"
                if a[i] == QUESTION and CHECK_IN(say) and size < 110:
                    yield "Friendly Check-in", say, src, key, "asks how someone is"
                if SORRY(say) and not NOT_SORRY(say):
                    yield "Apologize", say, src, key, "apologizes"
                if angry and ARGUE(say):
                    yield "Argue", say, src, key, "argues, angry"
                if e[i] in (ANGER, DISGUST) and CRITICIZE(say) and re.search(r"\b(you|your|he|his|she|her|they|their)\b", say, re.I):
                    yield "Criticize", say, src, key, "criticizes, angry or disgusted"
                if after and angry and e[i + 1] in (ANGER, DISGUST):
                    yield "Verbal Dispute", two, src, key, "both are angry"
                if after and (angry or e[i + 1] == ANGER):
                    if COUPLE(both) and not re.search(r"\b(my|his|her|your) (wife|husband|boyfriend|girlfriend)\b", both, re.I):
                        yield "Relationship Argument", two, src, key, "an angry exchange of a couple"
                    elif (HOME(both) or FAMILY(both)) and not SERVICE(" ".join(says)):
                        yield "Household Argument", two, src, key, "an angry exchange at home"
                if after and a[i] == INFORM and DISAGREE(after) and len(after) > 30:
                    yield "Disagreement & Argument", two, src, key, "a claim, and disagreement with it"
                if after and e[i] in (ANGER, DISGUST) and " you" in say.lower() and REBUT(after):
                    yield "Critique & Rebuttal", two, src, key, "a criticism, and an objection to it"
                if i and e[i - 1] == SAD and e[i] != SAD and a[i] != QUESTION and COMFORT(say):
                    yield "Comfort", say, src, key, "comforts someone sad"
                elif REASSURE(say) and a[i] != QUESTION:
                    yield "Reassure", say, src, key, "reassures"
                if a[i] == DIRECTIVE and PERSUADE(say) and size > 35:
                    yield "Persuade", say, src, key, "urges someone"
                    if after and size + len(after) < 400:
                        yield "Argument & Persuasion", two, src, key, "urges someone, and gets an answer"
                if a[i] in (INFORM, COMMISSIVE) and PLAN(say) and WHEN(say) and size > 25:
                    yield "Plan", say, src, key, "says what they will do, and when"
                if after and JOINT(say) and a[i + 1] == COMMISSIVE:
                    yield "Joint Planning", two, src, key, "proposes doing something together, and is answered"
                if HAGGLE(say):
                    yield "Negotiate", say, src, key, "negotiates a price"
                    if after and (HAGGLE(after) or re.search(r"\d", both)):
                        yield "Price Bargaining", two, src, key, "a price is haggled over"
                if (after and a[i] == DIRECTIVE and WORK(say) and ASSIGN(say) and a[i + 1] in (COMMISSIVE, INFORM)
                        and not SERVICE(both) and not re.search(
                            r"(post|box|ticket) office|(could|would) you (please )?tell me|help me find|would you mind", say, re.I)):
                    yield "Task Assignment", two, src, key, "a task is given, and answered"
                if a[i] in (INFORM, DIRECTIVE) and GUIDE(say) and size > 30:
                    yield "Guide", say, src, key, "gives directions"
                if a[i] == DIRECTIVE and INSTRUCT(say) and size > 20:
                    yield "Instruct", say, src, key, "tells someone what to do"
                if after and a[i] == QUESTION and HOW_TO(say) and a[i + 1] in (INFORM, DIRECTIVE) and len(after) > 45:
                    yield "How-to Guidance", two, src, key, "asks how to, and is told"
                if CLARIFY(say) and a[i] != QUESTION and size > 25:
                    yield "Clarify", say, src, key, "clarifies"
                if a[i] == INFORM and EXPLAIN(say) and size > 70:
                    yield "Explain", say, src, key, "explains"
                if i and a[i] == INFORM and a[i - 1] == QUESTION and ASK_LIKE(before) and size > 40:
                    yield "Describe", say, src, key, "describes what was asked about"
                if after and FAMILY(both) and HOME(both) and not angry and e[i + 1] != ANGER and not SERVICE(" ".join(says)):
                    yield "Family Discussion", two, src, key, "a family talks about home"
                if FIX(say) and HOUSEHOLD(say):
                    yield "Home Repair Request", say, src, key, "something at home needs fixing"
                if DEVICE(say) and FAULT(say):
                    yield "Troubleshoot", say, src, key, "a device has a problem"
                if COMPLAIN(say) and a[i] != QUESTION:
                    yield "Complain", say, src, key, "complains"
                if REFUND(say):
                    yield "Refund", say, src, key, "about getting money back"
                    if after:
                        yield "Refund Processing", two, src, key, "a refund or return is handled"
                if APPLY(say) and size > 25:
                    yield "Apply", say, src, key, "about an application or a document"
                if LATE(say) and PARCEL(say):
                    yield "Late Delivery Problem", say, src, key, "something has not arrived"
                if DOCTOR(both) and APPOINT(say) and size > 20:
                    yield "Doctor Appointment", say, src, key, "about seeing a doctor"


def empathetic(folder: Path) -> Iterator[Line]:
    src = "empathetic"
    worried, sad = {"anxious", "afraid", "apprehensive", "terrified"}, {"sad", "lonely", "devastated", "disappointed"}
    kind = has(r"\b(sorry|hope|hang in|there for|feel better|it('s| is) (ok|okay|not)|you('re| are)|must (be|have)|things (will|get))\b")
    calm = has(r"\b(you('ll| will)|i'm sure|don't worry|it('ll| will)|everything|be fine|good luck|no need|try not to)\b")
    for split in ("train", "valid", "test"):
        talks: Dict[str, list] = defaultdict(list)
        for line in (folder / f"{split}.csv").read_text(encoding="utf-8").splitlines()[1:]:
            c = line.split(",")
            if len(c) >= 6 and c[1].isdigit():
                talks[c[0]].append((int(c[1]), c[2], natural(c[3].replace("_comma_", ",")), natural(c[5].replace("_comma_", ","))))
        for conv, turns in talks.items():
            turns.sort()
            mood, story, key = turns[0][1], turns[0][2], f"empathetic:{conv}"
            says = [t[3] for t in turns]
            if len(story) >= 70:
                yield "Personal Story", story, src, key, "tells something that happened to them"
            if len(says[0]) >= 110:
                yield "Narrate", says[0], src, key, "tells what happened"
            if len(says) > 1 and not says[1].endswith("?"):
                if mood in sad and kind(says[1]):
                    yield "Comfort", says[1], src, key, "comforts someone sad"
                if mood in worried | {"embarrassed", "ashamed"} and (calm(says[1]) or kind(says[1])):
                    yield "Emotional Reassurance", pair(says[0], says[1]), src, key, "a worry, and a kind answer to it"
            for i, say in enumerate(says):
                if i % 2 and mood in worried and calm(say) and not say.endswith("?"):
                    yield "Reassure", say, src, key, "reassures someone worried"
                if i % 2 and CHECK_IN(say) and len(say) < 110:
                    yield "Friendly Check-in", say, src, key, "asks how someone is"
                if not i % 2 and SORRY(say) and not NOT_SORRY(say):
                    yield "Apologize", say, src, key, "apologizes"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--per-conversation", type=int, default=2)
    ap.add_argument("--least", type=int, default=20, help="leave out a territory with fewer lines than this")
    ap.add_argument("--data", type=Path, default=REPO / "data")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    known = {text_key(r["text"]) for r in json.loads((args.data / "sentences.json").read_text(encoding="utf-8"))}
    found: Dict[str, List[Line]] = defaultdict(list)
    streams = [sgd(args.sources / "sgd"), multiwoz(args.sources), taskmaster(args.sources),
               dailydialog(args.sources / "dailydialog" / "files"),
               empathetic(args.sources / "empathetic" / "files" / "empatheticdialogues")]
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
        if len(kept) >= args.least:
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
