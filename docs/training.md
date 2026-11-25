# Training loop

`python -m train` trains the coredrill's encoder on the labeled data, rebuilds
the coredrill with it, and then keeps learning from new text it labels
itself. A new model replaces the current one only when it does better on a
fixed holdout, so the loop cannot drift into a worse model.

## Why train

On Wikipedia articles neither method has seen, the deployed pipeline gets
26.5% strict and logistic regression on the same frozen LaBSE embeddings
45.1% ([evaluation.md](evaluation.md#split-by-article)). Two things hold both
back: the embedding space was never shaped for this taxonomy, and most nodes
are built from a single article. Fine-tuning works on the first, self-training
on the second.

## What gets trained

- **Encoder.** LaBSE's word embeddings (385M of its 471M parameters) stay
  frozen. Its top `--unfreeze-layers` transformer layers (default 4 with a
  GPU) and the Dense layer after pooling train.
- **Prototypes.** Each label gets one, starting at its centroid. Logits are
  scaled cosines (scale 20), so prototypes live in the same space as
  coredrill centroids.
- **Targets.** A record's target puts 0.9 on its label and 0.1 on the label's
  ancestors (halving per level), so parents settle around their children and
  a near miss costs less than an unrelated answer.
- **Anchor.** A term (weight 0.1) keeps embeddings close to stock LaBSE's.
  The labeled data is mostly English; the anchor carries LaBSE's
  cross-lingual alignment through training.
- **Fast mode.** With `--unfreeze-layers 0` only the Dense layer and the
  prototypes train, from features cached once. It runs on a CPU.

After training, every training text is embedded with the new encoder and the
coredrill is rebuilt with the engine's own `build_coredrill()`, so the API's
scoring (production score, Gram-inverse, flow) runs unchanged on the learned
geometry.

## The holdout and the gate

The first command draws the holdout with
[`12_by_article.py`](../eval/held_out/12_by_article.py)'s split: whole
Wikipedia articles, every language version, go to test, so the score is on
articles the model has never seen. It is saved in `runs/holdout.json` and
reused by every later run. It is exactly the records the reference numbers
were measured on: 5,065 test and 28,713 training records. That includes 22
training records that repeat a test record word for word, as in the
reference. Records added afterwards always go to training; hand-labeled
rows and pseudo-labels that repeat a holdout text are left out; and
harvesting never fetches a holdout article.

Each model is scored on it the way `12_by_article.py` scores, with every
system fitted on the run's own training rows and embeddings:

- the deployed pipeline (what the API answers) and its profile
- the nearest centroid
- `10_top_down.py`'s linear head
- logistic regression
- the run's prototype head

Results are also broken down into Wikipedia and other sources, English and
other languages. A model becomes the champion when three things hold:

- Its `--select-metric` (default `deployed.strict`) beats the champion's by
  `--min-delta` (default 0.002).
- The gain holds up text by text. Of the holdout texts that only one of the
  two models gets right, the new model must win clearly: a McNemar z of at
  least `--min-z`, default 1.645, which is significant at one-sided 95%.
- It loses no more than a point of near-miss accuracy.

The middle condition matters because averages move with noise. Two
baselines that differ only in floating-point rounding already disagree on
26 texts for logistic regression (11 one way, 15 the other, z −0.8).

## Self-training

Only labels the holdout tests (58 of 209) gain rows, so every change can be
measured; `--labels all` lifts that. `python -m train testset` gathers real
paragraphs for the other labels into `data/testset/to_check.csv`, for a person
to fill in `correct_label`. Their articles are never harvested for training.

A round:

1. **Harvests text.** For 40 nodes per round, fewest articles first and
   rotating, up to 8 Wikipedia articles each that the search engine finds
   similar to the node's own articles (lead paragraphs), plus whatever is in
   `data/inbox/`.
2. **Labels it with the champion.** A label is kept only if its calibrated
   probability is at least 0.8, it leads the runner-up by 0.15, and a second
   signal agrees. For a Wikipedia page, the label must be the node that led
   to it or a relative (ancestor, descendant, sibling). For an inbox text, the
   nearest centroid must be the label or a relative.
3. **Adds translations.** The Spanish, French, German and Portuguese versions
   of accepted articles come along with the English label when the champion
   places them near it.
4. **Caps** new labels per node (half its labeled count, at least 3), per
   node in all (as many self-labeled rows as labeled ones), and per round
   (1,500).
5. **Trains a challenger** from stock LaBSE on the labeled data, the pool of
   proven pseudo-labels, the unproven ones, and this round's, at half weight
   times confidence.
6. **Promotes it** if it beats the champion: the unproven pseudo-labels and
   this round's move to the pool. Otherwise this round's join the unproven
   ones (`data/selftrain/pending.jsonl`) and train again with the next
   round's. If the round is significantly worse (z at most -1.645), they go
   to `data/selftrain/rejected.jsonl` instead and are not fetched again; if
   the round before was too, so does the whole unproven stack. With
   `--unproven drop`, every round that is not promoted is set aside.

`loop` repeats rounds and stops after two in a row without a promotion.
Schedule it (Windows Task Scheduler, cron) to keep the model learning.

## Commands

Run from the repository root, in the environment that has CUDA PyTorch
(see `requirements-train.txt`).

```bash
python -m train baseline                    # stock LaBSE on the holdout; once, first
python -m train fit                         # train; promoted if it beats the champion
python -m train round                       # one self-training round
python -m train loop --rounds 10            # rounds until they stop helping
python -m train status                      # baseline, champion, pool, latest loops
python -m train report                      # every loop next to the reference benchmark
python -m train watch                       # follow the running loop live: log and GPU
python -m train view                        # the matrices in 3D and the classifier, in the browser
python -m train classify "a text"           # its rhizome profile, with the champion's model
python -m train forever                     # self-training rounds with no end ...
python -m train stop                        # ... until this; ends after the current loop
python -m train export --out exports/coredrill-trained
```

`round` and `loop` reuse the champion's training settings unless a flag
overrides one. `python -m train <command> --help` lists every flag.

### Running without end

`forever` starts one round after another, each in its own process, until
`stop`, Ctrl+C, or its window is closed. Its journal is `runs/forever.log`.

- It waits for any loop already running, while the laptop is on battery, and
  when less than 20 GB of disk is left.
- A loop that fails is tried again, after 2 minutes and then longer, up to an
  hour.
- When a round finds little, the next one searches wider. After two rounds
  that find nothing, it trains one model with another seed instead.
- It asks Wikipedia for little and slowly, and slows down further when told
  to (HTTP 429).
- Its bar for promotion is a McNemar z of 2.33 rather than 1.645, because
  many tries on one holdout make a lucky win likely. Scores chosen this way
  still drift upward on the holdout, so judge the result on text it has not
  seen.

A round that is not promoted keeps its pseudo-labels "unproven"
(`data/selftrain/pending.jsonl`): they train together with the next round's
until the stack earns a promotion. If a round is significantly worse, its
rows are set aside. Articles already judged under the current champion are
skipped.

### Following a run

Training runs on the GPU when PyTorch sees one; `baseline` needs none. In a
second terminal, `python -m train watch` follows the running loop:

- its log as it is written, with a progress line every 30 seconds while
  training (step, loss, speed, time left in the epoch)
- a GPU line every 15 seconds (load, memory, temperature, power)
- the loop's outcome when it ends

It then waits for the next loop. Ctrl+C stops the watching, not the
training. Two other views:

- **Task Manager:** in Performance, select the NVIDIA GPU and switch one
  graph to Cuda (or Compute_0). The default 3D graph stays flat during
  training.
- **nvidia-smi:** `nvidia-smi -l 5` prints the GPU's state every 5 seconds.

### Classifying with a trained model

`python -m train view` serves the classifier at
`http://127.0.0.1:8765/classify`, next to the training pages. It is only
served to this machine, and texts do not leave it.

- **The map.** Each territory with data is a point. Distances come as close
  as three dimensions allow to the model's own (the angle between centroids),
  and the text is placed by its distances to them. Lines follow the taxonomy;
  fainter ones are the engine's related nodes in other branches.
- **The answer** is the one the API would give with that model, from the same
  engine code. Next to it are the active territories, their areas, the flow,
  and the trained prototypes' probabilities as a second opinion.
- **Random holdout text** picks one of the texts the model never trained on
  and shows its true label.
- **Model** switches to any other trained loop.

The same from the command line or a script:

```bash
python -m train classify "a text"           # add --json for everything, --run <loop> for another model
curl -H "Content-Type: application/json" -d '{"text": "a text"}' http://127.0.0.1:8765/api/classify
```

It runs LaBSE in NumPy (`train/encoder_np.py`) with the weights mapped from
disk, so it commits about 0.6 GB where PyTorch takes 2.3 and peaks at 5.7,
and can run next to a training loop. On 300 random holdout texts in 14
languages it gave the same answer, profile and top ten prototypes as the
champion's GPU run recorded, every time. The page's model runs in a process
of its own that ends after 10 minutes unused.

## Records

Every loop (each `baseline`, each `fit`, each round of `round` or `loop`) is
recorded, whatever its outcome, so it can be compared later with the existing
benchmarks.

- **The ledger.** `runs/ledger.jsonl` gets one line when a loop starts and
  one when it ends. The end line holds everything below. Its status is one of:
  - `ok`: a baseline that finished
  - `promoted` or `not_promoted`: a trained model, kept either way
  - `no_candidates` or `nothing_accepted`: a round with nothing to train on
  - `failed`: an error, with its traceback
  - `interrupted`: stopped with Ctrl+C

  A start line with no end line is a loop that was killed.
- **The folder.** Each loop also gets a folder, `runs/<date>-<time>-<kind>/`,
  which the trainer never overwrites or deletes. Every folder holds:
  - `record.json`, the end line
  - `log.txt`, everything the loop printed
  - `code/`, the trainer and engine source that ran, committed or not

  Depending on what the loop did, it adds:
  - a scored loop: `holdout_predictions.jsonl.gz`, every system's answer and
    ranking for every holdout text
  - a trained loop: `weights.pt`, `run.json` and `coredrill_hierarchical.json`
  - a round: `candidates.jsonl`, every text it considered, the champion's top
    two labels, and the decision taken on it
- **What a record holds.** Each record holds:
  - the command line and every setting
  - the git commit and the uncommitted changes
  - Python, package, CUDA and GPU versions, and the exact base-model revision
  - SHA-256 fingerprints of the corpus, taxonomy, holdout and pool files
  - training history per epoch
  - time taken and peak GPU memory
  - the gate's decision and the champion before and after
  - the holdout scores in two layouts: `metrics`, which the gate reads, and
    `benchmark`, which uses the layout of
    [`by_article.json`](../eval/held_out/results/by_article.json) (the same
    subsets, systems and profile measures)
  - a leak check: how many harvested rows come from holdout articles (it
    should be 0)

**Comparing.** `python -m train report` writes two files:

- `runs/report.md`: every loop's numbers next to the reference, which is the
  by-article results in `eval/held_out/results/by_article.json` that
  [evaluation.md](evaluation.md#split-by-article) reports, with the
  difference in points for each system.
- `runs/report.csv`: every number for every loop, the reference first.

The baseline is the check that the comparison is fair. It runs on the same
holdout as the reference, so its numbers should match the reference's. Its
record says by how much they differ: `reference_check`, which holds the
largest difference and every individual one. Use another benchmark file with
`--reference`.

## First results

These were measured on 2026-11-24 on an RTX 4050 Laptop GPU (6 GB, 35 W),
using the default settings. Strict accuracy is on the by-article holdout:

| Loop | Deployed | Centroid | Linear head | Logistic regression | Prototype head (top-5) | Profile recall (size) |
|---|---|---|---|---|---|---|
| Reference (`by_article.json`, stock LaBSE) | 35.5 | 37.6 | 51.7 | 52.2 | - | 68.4 (5.0) |
| Baseline on these files | 35.6 | 37.7 | 51.6 | 52.1 | - | 69.0 (5.0) |
| First `fit` | 52.4 | 53.1 | 61.2 | 59.4 | 58.9 (84.6) | 65.3 (1.7) |
| Round 1 (+198 pseudo-labels) | 52.9 | 53.6 | 61.3 | 59.5 | 58.4 (85.4) | 65.8 (1.7) |

- **The first fit.** It gets 1,086 holdout texts right that the baseline
  misses, against 233 the other way (z 23.5). On Wikipedia articles it has
  never seen, the deployed pipeline goes from 26.8% to 44.2%.
- **Round 1.** It passed the gate narrowly: 109 texts gained, 85 lost
  (z 1.72).
- **The profile.** The rhizome profile narrows from about 5 active nodes to
  about 2. The engine's activation thresholds were tuned on stock LaBSE,
  and the trained space is tighter.
- **Coverage.** The holdout tests 58 of the 199 labels. Labels whose data
  comes from a single article can't be held out by article, and self-training
  mostly adds data to exactly those labels. Their changes don't show in
  these numbers.

**Time and memory.**

- **Baseline:** 2 to 6 minutes on the CPU.
- **Training:** 4 epochs of about 810 steps took 21 to 28 minutes (2.8 to
  4.7 steps per second). One epoch ran at half speed, probably because the
  laptop limited the GPU's power.
- **Scoring:** embedding every text with the new encoder, then scoring,
  takes another 10 to 15 minutes.
- **Peak GPU memory:** 3.5 GB for a fit. Round 1 peaked at 5.4 GB because
  the model that labeled the candidates stayed in memory. This is now fixed
  (`free_gpu()`).

## Files

| Path | What |
|---|---|
| `data/sentences.json`, `embeddings.npz`, `TAXONOMY_fixed.txt` | The labeled corpus (input) |
| `data/labeled/*.jsonl` | Hand-labeled additions, `{"text", "category"}`, always in training |
| `data/inbox/` | Text to label: `.txt` (one text each) or `.jsonl`; moved to `inbox/processed/` |
| `data/selftrain/pool.jsonl`, `pool_f0.npy` | Pseudo-labels from promoted rounds |
| `data/selftrain/pending.jsonl`, `pending_f0.npy` | Unproven pseudo-labels, training again with the next round's |
| `data/selftrain/rejected.jsonl` | Pseudo-labels set aside |
| `runs/holdout.json` | The fixed holdout |
| `runs/ledger.jsonl` | Every loop: a start line and an end line (see [Records](#records)) |
| `runs/<date>-<time>-<kind>/` | One loop: `record.json`, `log.txt`, `code/`; scored: `holdout_predictions.jsonl.gz`; trained: `weights.pt` (trained parameters and prototypes), `run.json`, `coredrill_hierarchical.json`; rounds: `candidates.jsonl` |
| `runs/baseline.json` | Points at the latest baseline (stock LaBSE on the holdout) |
| `runs/champion.json` | Points at the current best run, with its metrics |
| `runs/report.md`, `report.csv` | `python -m train report`: every loop next to the reference |

A run stores only the trained parameters (about 120 MB with four layers
unfrozen), not the whole encoder; `export` writes the full encoder. `runs/`
is not in git: back it up to keep the records.

## Deploying a trained model

`export` writes `encoder/` (sentence-transformers format) and a
`coredrill_hierarchical.json` whose `labse_model` names that encoder. The API
follows `labse_model` when `MODEL_NAME` is not set, and logs a warning when
`MODEL_NAME` names a different encoder, since the scores would then be wrong.

For the Hugging Face Space: upload `encoder/` to a private model repo, export
with `--model-id <that repo>`, upload the coredrill to the dataset repo, and
set the Space's `MODEL_NAME` to the model repo. The Dockerfile bakes in stock
LaBSE and sets `MODEL_NAME`, so it needs the same change.
