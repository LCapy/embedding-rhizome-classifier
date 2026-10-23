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

**Limitation.** This split is by record, not by source article: every one of
its 4,867 Wikipedia-sourced test records comes from an article that also has
training records, so the numbers below are upper bounds. [Split by
article](#split-by-article) measures how much: about 20 points on Wikipedia
text, for every method.

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

## Split by article

`12_by_article.py` groups every Wikipedia record by its concept (all language
versions of an article plus its Simple English version) and sends whole
concepts to train or test; no test record shares a concept with training
(0 of 3,532, against 4,867 of 4,867 in the record split). A concept goes to
test only if every class it feeds keeps at least 5 training records, so only
the 58 classes fed by several concepts, or by records with no article, can be
tested. Most of the 199 nodes are built from a single article, so how they do
on a new article cannot be measured with this data.

Both splits on the same 58 classes, strict / near-miss:

| Method | Wikipedia, record split | **Wikipedia, article split** | Other sources, record split | Other sources, article split |
|---|---:|---:|---:|---:|
| Deployed pipeline | 47.9% / 76.4% | **26.5% / 61.4%** | 55.8% / 65.5% | 56.0% / 63.8% |
| Nearest centroid | 51.0% / 79.8% | **28.7% / 65.8%** | 55.6% / 64.2% | 58.2% / 65.0% |
| Linear head (untuned) | 69.7% / 87.3% | **46.0% / 74.6%** | 63.0% / 70.8% | 64.9% / 71.0% |
| Logistic regression | 69.5% / 90.5% | **45.1% / 76.2%** | 65.2% / 72.5% | 68.6% / 74.4% |

| Profile of the same size (Wikipedia) | Record split | **Article split** |
|---|---:|---:|
| Rhizome active set | 83.5% | **62.1%** |
| Logistic regression top-k | 93.6% | **75.7%** |
| Linear head top-k | 92.2% | **77.9%** |

Records with no article (bitext, CFPB, ChatDoctor) are split by record in
both runs and serve as a control: their numbers barely move. On Wikipedia
text from unseen articles every method loses about 20 points, and the trained
models stay 17-19 points ahead of the geometric ones. The leak inflated all
methods; it did not favor training.

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
| Held-out accuracy (this report, record split) | An upper bound: on Wikipedia text from unseen articles every method loses about 20 points |

## Conclusions

- With no training step, the geometric classifier gives a usable single
  answer and an interpretable multi-territory profile: 48% strict, 73%
  near-miss and an 84% profile hit rate over 199 classes on the record split;
  on Wikipedia articles it has never seen (the 58 classes that split allows),
  27% strict, 61% near-miss and 62%.
- A logistic regression on the same embeddings is better on every measure,
  on both splits, at the cost of a one-minute training step.
- Most nodes are built from a single Wikipedia article. More articles per
  node is what would let both approaches generalize, and what would let the
  evaluation cover all 199 classes on unseen articles.
