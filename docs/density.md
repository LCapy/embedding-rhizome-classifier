# Density: what HDBSCAN finds in the trained space

`python -m train density --data data/v3 --runs runs/v3 --run <loop>` looks at
one encoder's training texts with density clustering (HDBSCAN), lists texts
for review, and scores two coredrill builds made from what it finds. It is a
recorded loop like any other (`runs/<date>-<time>-density/`), and the
**Density** page of `python -m train view` (`/density`) draws it while it
runs.

The question behind it: the coredrill treats every territory as a sphere,
one center (or a few k-means centers) and one radius. Does the data have
that shape?

## Results on the v3 champion

Measured on 2026-11-25 with the encoder of `20261125-094143-fit` (35,365
training texts, 236 territories with text; test 9,660 texts). Both runs
(`20261125-130658-density`, `20261125-132127-density`) gave the same numbers.

**Inside each territory: one cloud, not several.**

- HDBSCAN (eom) finds a single dense core in every territory large enough to
  have one (234). The core holds 58% of a territory's texts; the rest is a
  tail that thins out in some directions and not others.
- The finer reading (leaf) finds 2 to 5 small cores in 176 territories, but
  they are pockets inside the one cloud, not separate groups.
- Stock LaBSE gives the same picture, so this is not something training did.

**Across territories, labels ignored: families.**

- Density finds 58 regions of texts. 43 are shared by several territories
  (the largest territory under 80% of the region): Medicine, Treatment and
  Diagnosis; Poetry, Fiction and Literature; Vote, Voting Process and
  Campaign Messaging; Genes, Genetics and DNA Inheritance; Algebra, Geometry
  and Mathematics.
- No territory owns two regions: nothing calls for splitting a territory.
- 64% of texts are in no dense region at all.

**The errors follow the families.**

| The champion's test answers | Texts | Share |
|---|---:|---:|
| Right | 5,180 | 53.6% |
| Wrong, a territory of the same region as the right one | 1,163 | 12.0% |
| Wrong, elsewhere | 3,317 | 34.3% |

26% of the errors stay within the family. If errors fell on territories at
random, 3.3% would. Most of the rest are texts in the diffuse middle of the
space, in every territory's tail and no one's core.

**Two builds, set before scoring, against the encoder's own (radius 75%
toward the median, one center per 60 texts, at most 6; 53.6%):**

| Build | Right | Near | Gained | Lost | z |
|---|---:|---:|---:|---:|---:|
| Density centers: a territory's leaf cores as its centers | 50.6% | 74.5% | 372 | 664 | -9.07 |
| Without the review list's 2,502 texts | 53.6% | 75.9% | 251 | 257 | -0.27 |

- **Density centers lose.** Leaf cores are small dense pockets; the k-means
  centers spread over the whole cloud, tail included, and the tail is where
  half the texts are.
- **Leaving out the review list changes nothing.** Those texts are not what
  holds the coredrill back.
- Neither is adopted; `build.json` and `champion.json` are untouched.
- The test has been looked at closely before (the build search, the math
  review), so these scores are for comparing, not for choosing.

## What it means for the coredrill

- **Spheres are the wrong shape.** A territory is a core with an uneven
  tail, and one radius cannot fit both. A learned coredrill should give each
  territory a shape: a radius that depends on direction.
- **Families are the structure to learn.** A quarter of the errors are
  between territories that share one dense region. Training that weighs
  those confusions, or a level of families above the territories, aims at
  them directly. Whether some families should merge in the taxonomy is a
  decision for a person.
- **More centers will not fix it.** Density finds no hidden modes, and
  centers placed on the dense pockets do worse.

## The review list

`review.jsonl.gz` in the loop's folder: every training text outside its
territory's core and nearer another territory's mean than its own (2,502,
7.1%; most in Mathematics, Medicine, Natural World, Algebra, Economics). Each
line holds the text, its territory, the nearer one, both similarities, its
source and article. Some may be mislabeled; nothing is relabeled
automatically.

## The Density page

- Each faint point is a training text (at most 40 per territory), placed by
  t-SNE. Bright points are in their territory's core, faint ones in its
  tail; amber cores belong to a family.
- Each test text sits at the mean place of the five sampled training texts
  most like it. Its mark is the coredrill's answer: a dot when right, a ring
  when wrong within the family, a cross when wrong elsewhere. The three are
  told apart by shape as well as color.
- While a density loop runs, its answers appear as they are made, build by
  build, with a progress bar and running counts.
- Pick a territory (click a text, or the list) to see its cloud, its test
  texts and lines to where its wrong answers went. The tables list the
  territories that lose the most within their family, and the families.

## Settings

In `train/density.py`, `SETTINGS`:

| Setting | Value | What |
|---|---:|---|
| `territory_pca` | 10 | dimensions a territory's texts are reduced to (PCA fitted on them) |
| `min_cluster_share`, `min_cluster_floor` | 5%, 15 | HDBSCAN's smallest cluster in a territory |
| `min_samples` | 5 | HDBSCAN |
| `global_pca` | 32 | dimensions for the reading across territories |
| `global_min_cluster`, `global_min_samples` | 40, 10 | HDBSCAN across territories |
| `family_share` | 0.8 | a region whose largest territory has less is shared |
| `max_centers`, `min_center_size` | 6, 8 | as the champion's build and `spherical_kmeans` |

Plain DBSCAN was not used: its single density threshold does not suit
territories of very different spreads. Without `allow_single_cluster`,
HDBSCAN calls a one-cloud territory all noise.

## Files

| In the loop's folder | What |
|---|---|
| `density.json` | every reading above: per territory, regions, errors, the builds |
| `map.json` | the page's map: sampled training texts and test texts with their places |
| `answers.json` | the test's answers, build by build (written while it scores) |
| `review.jsonl.gz` | the review list |
| `predictions_*.jsonl.gz` | each build's answers to every test text |
| `holdout_predictions.jsonl.gz`, `coredrill_hierarchical.json` | the better of the two builds |
