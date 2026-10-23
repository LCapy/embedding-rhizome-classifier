# Evaluation

How well the classifier generalizes to text it has not seen, and which
earlier reported numbers hold up. Scripts are in
[`eval/held_out/`](../eval/held_out/README.md); the summary JSONs behind every
table are in [`eval/held_out/results/`](../eval/held_out/results/).

## Setup

- 33,787 records; the 199 categories with at least 5 records (33,778
  records) are the classes.
- One fixed, seeded, stratified 80/20 split: 27,032 training records, 6,746
  held-out records. Every method uses the same split and the same frozen
  LaBSE embeddings.
- **Strict**: the top answer is the true label. **Near-miss**: the top answer
  is the true label, one of its ancestors or descendants, or a sibling.
- The coredrill is rebuilt from the training records only, with the real
  `build_coredrill()`, and every held-out text goes through the real
  `analyze_text_rhizome_v2()`; the final answer is chosen exactly as
  `api/app.py` chooses it.

**Limitation.** The split is by record, not by source article: 4,863 of the
4,867 Wikipedia-sourced test records come from an article that also has
training records. Every number below is therefore an upper bound for text
from unseen articles. A split by article is the next measurement.

## Single answer

| Method | Strict | Near-miss |
|---|---:|---:|
| Deployed pipeline, coredrill 5.0 build | 26.2% | 64.1% |
| **Deployed pipeline, coredrill 5.1 build** | **48.4%** | **72.8%** |
| Nearest centroid, no scoring rules | 51.2% | 75.5% |
| Top-down, geometric routing (best of 4 variants) | 39.0% | 65.5% |
| Linear head, Adam, untuned | 55.7% | 74.0% |
| Top-down, one logistic regression per parent node | 58.5% | 79.9% |
| Logistic regression, C chosen on a slice of the training split | 61.3% | 83.6% |

The logistic regression trains in about a minute on a CPU, on the same
embeddings, and is the strongest single-answer method measured.

Inside the deployed pipeline (coredrill 5.0 build → 5.1 build):

| Component | Strict | Near-miss |
|---|---|---|
| Gram-inverse winner (the deployed answer) | 26.2% → 48.4% | 64.1% → 72.8% |
| Recommended winner (smallest z) | 13.8% → 45.3% | 62.8% → 69.2% |
| Top production score | 17.3% → 47.7% | 66.1% → 71.9% |
| Top cosine | 32.2% → 50.3% | 67.3% → 75.5% |
| Void gram states | 562 → 246 | |

`predict()`'s top-down walk stops at the first level that fails its
confidence gates, so its node is usually an ancestor of the label (4.9%
strict, 29.5% near-miss). It is reported as the lineage, not used as the
answer.

## Whole profile

The rhizome returns an active set (every node within 0.18 of the top
production score, at most 10), averaging 4.45 nodes. Compared at the same
size per text:

| Profile of the same size | Contains the true label |
|---|---:|
| Rhizome active set | 84.0% |
| Linear head (untuned) top-k | 81.8% |
| Logistic regression top-k | 92.5% |

| Ranked list | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|
| Production score | 47.7% | 73.8% | 83.2% | 92.3% |
| Gram-inverse score | 48.2% | 75.1% | 84.6% | 93.5% |
| Cosine | 50.3% | 77.4% | 86.4% | 94.2% |
| Linear head (untuned) | 55.7% | 76.8% | 84.7% | 91.2% |
| Logistic regression | 61.3% | 88.1% | 94.8% | 98.2% |

## What changed in coredrill 5.1

| Change | Why | Effect |
|---|---|---|
| A node with at least 5 records of its own gets its centroid from those records; children are used only for sparse parents | Earlier builds blended every parent 60-96% toward its children, the subtree propagation `study_v5.md` reports as harmful | On parent-labeled held-out texts, the blend cut accuracy from 46.5% to 19.0% |
| No skip list (`skip_nodes` is now a build parameter, default none) | The 26 skipped nodes were diffuse centroids that won everywhere; with direct centroids they no longer do | 47.8% → 48.4% strict; 14 labels (1.8% of test texts) become reachable; the 4 most frequent answers take 10.2% of answers instead of 10.4% |
| The API copies the coredrill's skip list into the engine | The engine scored with its own copy, so skipped nodes could still win | Correctness |
| `predict()` walks down the winner's children | Each level was scored independently and the deepest winner always returned, so the gates never stopped anything | Lineage output |
| Consensus counts the three z-score transforms as one metric | They always pick the same node, so they outvoted the others | Consensus output |

## Top-down classification

Routing a text to its main class first, then down the tree, does not beat
flat classification with either decision rule. Geometric routing loses most:
a main class's centroid averages its whole branch, so it is diffuse, and the
main class is right 75-79% of the time against 86.7% for flat nearest
centroid. With a trained classifier at each parent, top-down reaches 58.5%,
below a single flat logistic regression (61.3%).

## Re-check of earlier reported scores

`11_recheck.py` re-ran the texts behind earlier scores on today's code.

**June 17 `run_eval.py` report (122/150).** Re-run with the June engine and
coredrill, the 16 Wikipedia texts whose intros are unchanged score 82/96,
exactly as published. The leak check finds 9 of the 20 Wikipedia texts in the
build data (cosine ≥ 0.98 to a stored record):

| Wikipedia texts | June engine | Coredrill 5.1 | Logistic regression (top-1) |
|---|---:|---:|---:|
| In the build data (9), combined score | 52/54 | 52/54 | 9/9 |
| Not in the build data (11), combined score | 52/66 | 58/66 | 10/11 |
| Not in the build data (11), accepted node in the profile | 9/11 | 11/11 | 11/11 in top 5 |

The BBC and Reddit items carry different stories on every fetch but are still
graded against the same accept lists, so they say little.

**Curated texts.** 12 short or document-length texts with an unambiguous
topic, in 6 languages, with accept lists written before scoring:

| System | Final answer accepted | Accepted node in profile / top 5 |
|---|---:|---:|
| June engine + coredrill 5.0 | 7/12 | 9/12 |
| Fixed engine + coredrill 5.1 | 9/12 | 12/12 |
| Logistic regression | 11/12 | 12/12 |
| Live Space before this release (coredrill 4.0) | 6/12 | 9/12 |

## Earlier claims

| Claim | Status |
|---|---|
| Flat prototype accuracy 91.3% (`study_v5.md`) | That metric is sibling recall on the build's own records (2-5 candidates per decision). Held-out top-1 over 199 classes is 48-51% for geometry |
| Subtree propagation hurts | Holds; the 5.1 build follows it |
| June suite 122/150 | Reproduces exactly, but 9 of its 20 Wikipedia texts are in the build data |
| The profile is competitive with training | Only against an untuned linear head; a tuned logistic regression's top-k covers the label more often at every size measured |

## Conclusions

- With no training step, the geometric classifier gives a usable single
  answer (48% strict, 73% near-miss over 199 classes) and an interpretable
  multi-territory profile that contains the label 84% of the time.
- A logistic regression on the same embeddings is better on every measure
  here, at the cost of a one-minute training step.
- Both numbers are upper bounds until the evaluation is repeated with a split
  by article.
