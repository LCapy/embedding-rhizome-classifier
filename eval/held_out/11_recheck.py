#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_recheck.py - do the earlier scored results still stand?

Re-scores the two sets of texts that already have scores, on today's code:
  suite   tests/eval_suite.json, the 30 items behind the June 17
          report (122/150). Fetched again with run_eval.py's own fetchers.
          Wikipedia intros are stable, so those compare 1:1 with June (each
          is checked against the June report's excerpt); BBC and Reddit feeds
          change daily, so those are new texts.
  probes  the 18 texts from 06/07 (try_sentences.json, live_space.json).
          12 have an unambiguous topic and get an accept list below (my
          judgment, written before scoring); the mixed-topic ones are shown
          but not graded.

Systems:
  june       the engine at commit 2ca1624 (the code that wrote the June
             report) + the v5.0 coredrill (EVAL_COREDRILL), run again today
  fixed_v50  today's fixed engine + the same v5.0 coredrill
  fixed_v51  today's fixed engine + a coredrill rebuilt from all records with
             the fixed build (what would ship)
  td_trained one linear classifier per parent node, best full path
             (10_top_down.py's best variant), fitted on all labeled records
  logreg     one flat logistic regression (10_top_down.py's flat_logreg, the
             best held-out classifier), fitted on all labeled records
  trained    02's plain linear head on all labeled records
  centroid   nearest own-centroid
  live       the live Space as it runs now (--live, needs HF_TOKEN)
Engine systems are graded exactly as run_eval.py grades them (node 0-3 on
gram_winner, rhizome quality 0-3), plus the API's final answer. Classifiers
get the node grade on their top-1, with their top-5 standing in for the
active overlap. Leak check: nearest stored record to each fetched text.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from common import (COREDRILL_PATH, ENGINE_DIR, OUT_DIR, REPO, TAXONOMY_PATH, build_label_set,
                    compute_centroids, load_embeddings, load_sentences)

HERE = Path(__file__).parent
# This repo's commit with the engine that wrote the June report (identical to
# the Space's e4fb6b0), and the report itself (tests/reports/ is gitignored).
JUNE_COMMIT = os.environ.get("EVAL_JUNE_COMMIT", "2ca1624")
JUNE_REPORT = Path(os.environ.get("EVAL_JUNE_REPORT", REPO / "tests" / "reports" / "eval_20260617_100243.md"))
COREDRILL_V50 = COREDRILL_PATH
COREDRILL_V51 = OUT_DIR / "coredrill_full_v51.json"
SPACE = os.environ.get("EVAL_SPACE_URL", "https://lcapy-coredrill.hf.space")
SEEN_COS = 0.98  # a fetched text this close to a stored record is that record, give or take edits

PROBE_ACCEPT = {
    "01_riemann_en_math": ["Mathematics"],
    "02_complaint_es_customer": ["Complaint Handling", "Complain", "Customer Service",
                                 "Billing & Refunds", "Refund Processing"],
    "05_cardiology_fr_clinical": ["Medicine", "Diagnosis", "Treatment"],
    "07_amazon_ecology_en_science": ["Ecology", "Ecosystems", "Climate Change"],
    "09_inflation_vi_economics": ["Macroeconomics", "Economics"],
    "probe_en_1": ["Billing & Refunds", "Refund Processing", "Complaint Handling", "Complain"],
    "probe_en_3": ["Natural Selection", "Evolution"],
    "probe_en_4": ["Medicine", "Treatment", "Appointments & Medication"],
    "probe_en_5": ["Medicine", "Diagnosis", "Symptoms & Self-Care"],
    "probe_fr_6": ["Elections & Campaigns", "Vote", "Politics", "Political Institutions"],
    "probe_ru_7": ["Ordering & Delivery", "Purchase & Payment"],
    "probe_ja_8": ["Quantum Mechanics", "Physics"],
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def june_engine():
    path = OUT_DIR / "engine_june.py"
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{JUNE_COMMIT}:scripts/rhizome_engine.py"],
                         capture_output=True, check=True).stdout
    path.write_bytes(src)
    return load_module("engine_june", path)


def parse_june_report():
    if not JUNE_REPORT.exists():
        print(f"June report not found at {JUNE_REPORT}; skipping the June comparison")
        return {}
    text = JUNE_REPORT.read_text(encoding="utf-8")
    rows = {}
    for m in re.finditer(r"^\| \d+ \| (\S+) \| `[^`]*` \| `([^`]*)` \| [^|]*\((\d)\) \| [^|]*\((\d)\) \| (\d)/6",
                         text, re.M):
        rows[m.group(1)] = {"gram_winner": m.group(2), "node": int(m.group(3)),
                            "rq": int(m.group(4)), "total": int(m.group(5))}
    for m in re.finditer(r"^### (\S+).*?^> (.+?)$", text, re.M | re.S):
        if m.group(1) in rows:
            rows[m.group(1)]["excerpt"] = m.group(2).rstrip(".").strip()
    return rows


def fetch_texts(run_eval, suite, refetch):
    cache = OUT_DIR / "recheck_texts.json"
    if cache.exists() and not refetch:
        return json.loads(cache.read_text(encoding="utf-8"))
    texts = {}
    for item in suite:
        try:
            texts[item["id"]] = run_eval.FETCHERS[item["fetch"]["type"]](item) or ""
        except Exception as exc:  # a dead feed is a result, not a crash
            print(f"  fetch failed for {item['id']}: {exc}")
            texts[item["id"]] = ""
    cache.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")
    return texts


def coredrill_v51(eng, sents, emb):
    if not COREDRILL_V51.exists():
        print("Building the full-data v5.1 coredrill with the fixed build...")
        with contextlib.redirect_stdout(io.StringIO()):
            cd = eng.build_coredrill(sents, emb, eng.load_taxonomy(TAXONOMY_PATH))
        COREDRILL_V51.write_text(json.dumps(eng.to_native(cd)), encoding="utf-8")
    return json.loads(COREDRILL_V51.read_text(encoding="utf-8"))


def run_engine(eng, cd, text, model, default_skip):
    eng.SKIP_NODES.clear()
    eng.SKIP_NODES.update(cd.get("skip_nodes", default_skip))
    with contextlib.redirect_stdout(io.StringIO()):
        raw = eng.analyze_text_rhizome_v2(cd, text, model, tau=cd.get("tau", 0.1), print_output=False)
    gram = raw.get("gram_winner")
    if gram and gram not in eng.SKIP_NODES:
        answer = gram
    elif (raw.get("recommended_winner") or {}).get("node"):
        answer = raw["recommended_winner"]["node"]
    elif raw.get("results"):
        answer = raw["results"][0]["node"]
    else:
        answer = (raw.get("prediction") or {}).get("best_node", "unknown")
    return raw, answer


def grade_ranked(ranked, accept, partial):
    if ranked and ranked[0] in accept:
        return 3
    if ranked and ranked[0] in partial:
        return 2
    return 1 if any(n in accept or n in partial for n in ranked[:5]) else 0


def live_predict(text, token):
    req = urllib.request.Request(
        SPACE + "/predict", method="POST",
        data=json.dumps({"text": text, "full": True}).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        live = json.loads(resp.read().decode("utf-8"))
    ranked = [live.get("best_node")] + [r["node"] for r in (live.get("ranked") or [])]
    return live.get("best_node"), ranked, [r["node"] for r in live.get("active_overlap", [])]


def summarize(out):
    """Aggregate recheck.json into recheck.md and print the headline numbers."""
    ok = [r for r in out if not r.get("fetch_error")]
    systems = list(ok[0]["systems"]) if ok else []
    wiki = [r for r in ok if r["set"] == "suite" and r["source"] == "wikipedia"]
    groups = {
        "suite: Wikipedia, same text as June": [r for r in wiki if r.get("same_text_as_june")],
        f"suite: Wikipedia, text in the build data (cosine >= {SEEN_COS})":
            [r for r in wiki if r["leak"]["max_cos"] >= SEEN_COS],
        f"suite: Wikipedia, text not in the build data (cosine < {SEEN_COS})":
            [r for r in wiki if r["leak"]["max_cos"] < SEEN_COS],
        "suite: BBC/Reddit (new texts today)": [r for r in ok if r["set"] == "suite" and r["source"] != "wikipedia"],
        "probes with an accept list": [r for r in ok if r["set"] == "probes" and r["accept"]],
    }
    md = ["# Re-check of earlier scored texts", ""]
    for title, rows in groups.items():
        if not rows:
            continue
        md += [f"## {title} (n={len(rows)})", "",
               "| system | final answer in accept | accept in profile | node score | combined |",
               "|---|---|---|---|---|"]
        for s in systems:
            rs = [r["systems"][s] for r in rows if s in r["systems"]]
            node = [x["node"] for x in rs if "node" in x]
            tot = [x["total"] for x in rs if "total" in x]
            md.append(f"| {s} | {sum(x['answer_ok'] for x in rs)}/{len(rs)} | "
                      f"{sum(x['profile_ok'] for x in rs)}/{len(rs)} | "
                      f"{f'{sum(node)}/{3 * len(node)}' if node else '-'} | "
                      f"{f'{sum(tot)}/{6 * len(tot)}' if tot else '-'} |")
        if title.startswith("suite: Wiki"):
            jr = [r["june_report"] for r in rows]
            md.append(f"| June report (as published) | - | - | {sum(j['node'] for j in jr)}/{3 * len(jr)} | "
                      f"{sum(j['total'] for j in jr)}/{6 * len(jr)} |")
        md.append("")
    md += ["## Leak check (suite)", "",
           "| id | nearest stored record | cosine | source |", "|---|---|---|---|"]
    for r in ok:
        if r["set"] == "suite":
            lk = r["leak"]
            md.append(f"| {r['id']} | {lk['nearest_title'] or lk['nearest_category']} | {lk['max_cos']:.3f} | "
                      f"{lk['nearest_source']} |")
    md += ["", "## Per text", "",
           "| id | accept | june / before | " + " | ".join(systems) + " |",
           "|---|---|---|" + "---|" * len(systems)]
    for r in ok:
        before = (f"June: {r['june_report']['gram_winner']}" if "june_report" in r else
                  f"local: {r['previous']['local_deployed']}, live v4: {r['previous']['live_v4']}"
                  if "previous" in r else "-")
        cells = [f"{'**' if v['answer_ok'] else ''}{v['answer']}{'**' if v['answer_ok'] else ''}"
                 for v in r["systems"].values()]
        md.append(f"| {r['id']} | {', '.join(r['accept'][:3]) or '(ungraded)'} | {before} | " + " | ".join(cells) + " |")
    (OUT_DIR / "recheck.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:md.index("## Leak check (suite)")]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also query the live Space")
    ap.add_argument("--refetch", action="store_true", help="fetch the suite texts again")
    ap.add_argument("--summary-only", action="store_true", help="rebuild recheck.md from recheck.json")
    args = ap.parse_args()
    if args.summary_only:
        summarize(json.loads((OUT_DIR / "recheck.json").read_text(encoding="utf-8")))
        return

    sys.path.insert(0, str(ENGINE_DIR))
    import rhizome_engine as eng_new  # noqa: E402
    eng_old = june_engine()
    run_eval = load_module("run_eval", REPO / "tests" / "run_eval.py")
    td = load_module("top_down", HERE / "10_top_down.py")
    default_skip = {"june": set(eng_old.SKIP_NODES), "fixed": set(eng_new.SKIP_NODES)}

    sents, emb = load_sentences(), load_embeddings()
    cd50 = json.loads(COREDRILL_V50.read_text(encoding="utf-8"))
    cd51 = coredrill_v51(eng_new, sents, emb)
    print(f"coredrill v5.0: {cd50.get('version')}  v5.1: {cd51.get('version')}  "
          f"skip lists {len(cd50.get('skip_nodes', default_skip['june']))}/{len(cd51.get('skip_nodes', []))}")

    labels = build_label_set(sents)
    l2i = {l: i for i, l in enumerate(labels)}
    y = np.array([l2i.get(r.get("category", ""), -1) for r in sents])
    rows = np.where(y >= 0)[0]
    centroids = compute_centroids(emb, y, rows, len(labels))
    flat = td.train_flat_head(emb[rows], y[rows], len(labels))
    best_c = json.loads((OUT_DIR / "top_down.json").read_text(encoding="utf-8"))["chosen"]["C"]
    tree = td.Tree(cd51["taxonomy_backbone"]["parents_map"], labels)
    tdm = td.fit(tree, emb, y, rows, C=best_c)
    logreg = LogisticRegression(C=best_c, max_iter=2000).fit(emb[rows], y[rows])
    print(f"Classifiers fitted on {len(rows)} labeled records ({len(labels)} classes, td C={best_c})")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE")
    token = os.environ["HF_TOKEN"] if args.live else None

    suite = json.loads((REPO / "tests" / "eval_suite.json").read_text(encoding="utf-8"))["items"]
    texts = fetch_texts(run_eval, suite, args.refetch)
    june = parse_june_report()
    previous = {r["name"]: r for r in json.loads((OUT_DIR / "try_sentences.json").read_text(encoding="utf-8"))}
    live_prev = {r["name"]: r for r in
                 json.loads((OUT_DIR / "live_space.json").read_text(encoding="utf-8"))["rows"]}

    cases = [{"set": "suite", "id": it["id"], "source": it["source"], "text": texts.get(it["id"], ""),
              "accept": set(it["expected"].get("accept_nodes", [it["expected"]["node"]])),
              "partial": set(it["expected"].get("partial_nodes", [])), "item": it} for it in suite]
    cases += [{"set": "probes", "id": name, "source": "probe", "text": r["text"],
               "accept": set(PROBE_ACCEPT.get(name, [])), "partial": set(), "item": None}
              for name, r in previous.items()]
    for name, acc in PROBE_ACCEPT.items():
        missing = [n for n in acc if n not in labels]
        assert not missing, f"{name}: {missing} not labels"

    out = []
    for c in cases:
        if c["set"] == "suite" and len(c["text"]) < 40:  # same rule as run_eval.py
            out.append({"set": c["set"], "id": c["id"], "source": c["source"], "fetch_error": True})
            continue
        x = model.encode([c["text"]], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        sims = emb @ x[0]
        near = int(sims.argmax())
        row = {"set": c["set"], "id": c["id"], "source": c["source"], "accept": sorted(c["accept"]),
               "text_start": c["text"][:120],
               "leak": {"max_cos": round(float(sims[near]), 4), "nearest_source": sents[near].get("source"),
                        "nearest_category": sents[near].get("category"),
                        "nearest_title": sents[near].get("article_title")},
               "systems": {}}
        if c["id"] in june:
            j = june[c["id"]]
            row["june_report"] = {k: v for k, v in j.items() if k != "excerpt"}
            exc = re.sub(r"\s+", " ", j.get("excerpt", ""))[:150]
            row["same_text_as_june"] = bool(exc) and re.sub(r"\s+", " ", c["text"]).startswith(exc)
        if c["id"] in previous:
            row["previous"] = {"local_deployed": previous[c["id"]]["deployed"]["best_node"],
                               "live_v4": live_prev.get(c["id"], {}).get("live")}

        for name, eng, cd, skip in [("june", eng_old, cd50, default_skip["june"]),
                                    ("fixed_v50", eng_new, cd50, default_skip["fixed"]),
                                    ("fixed_v51", eng_new, cd51, default_skip["fixed"])]:
            raw, answer = run_engine(eng, cd, c["text"], model, skip)
            s = {"answer": answer, "gram_winner": raw.get("gram_winner"),
                 "active": [r["node"] for r in raw.get("active_overlap_set", [])],
                 "answer_ok": answer in c["accept"]}
            s["profile_ok"] = bool(c["accept"] & set(s["active"]))
            if c["item"] is not None:
                ng = run_eval.grade_node(raw, c["item"])
                em, gse = run_eval.find_expected_metrics(raw.get("results", []), raw.get("gram_scores", {}),
                                                         c["accept"])
                rq = run_eval.grade_rhizome(raw, c["item"], em, gse)
                s.update({"node": ng["score"], "rq": rq["rq_score"], "total": ng["score"] + rq["rq_score"]})
            row["systems"][name] = s

        with torch.no_grad():
            logits = flat(torch.from_numpy(x)).numpy()[0]
        ranked = {
            "td_trained": [labels[tdm["present"][a]] for a in np.argsort(
                -td.path_scores(tdm, x, lambda n, X_: tdm["clf"][n].predict_log_proba(X_))[0])[:5]],
            "logreg": [labels[logreg.classes_[a]] for a in np.argsort(-logreg.predict_log_proba(x)[0])[:5]],
            "trained": [labels[a] for a in np.argsort(-logits)[:5]],
            "centroid": [labels[a] for a in np.argsort(-(centroids @ x[0]))[:5]],
        }
        if args.live:
            best, live_ranked, live_active = live_predict(c["text"], token)
            ranked["live"] = live_ranked[:5]
        for name, rk in ranked.items():
            s = {"answer": rk[0], "top5": rk, "answer_ok": rk[0] in c["accept"],
                 "profile_ok": bool(c["accept"] & set(live_active if name == "live" else rk[:5]))}
            if c["item"] is not None:
                s["node"] = grade_ranked(rk, c["accept"], c["partial"])
            row["systems"][name] = s
        out.append(row)
        print(f"{c['id']:<36} " + "  ".join(f"{k}={v['answer']}{'*' if v['answer_ok'] else ''}"
                                          for k, v in row["systems"].items()))

    (OUT_DIR / "recheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_DIR / 'recheck.json'}")
    summarize(out)


if __name__ == "__main__":
    main()
