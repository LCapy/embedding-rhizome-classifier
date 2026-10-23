# Held-out evaluation

Measures how well the classifier generalizes to text it has never seen.
Everything uses one fixed, seeded, stratified 80/20 split (`01_baseline.py`
writes it), so every number below is on the same 6,746 held-out texts.
Results and conclusions: [`../../docs/evaluation.md`](../../docs/evaluation.md).

| Script | What it measures |
|---|---|
| `01_baseline.py` | Builds the split; nearest-centroid accuracy with direct-record centroids from the training split only |
| `02_train.py` | A linear head trained on the same frozen LaBSE embeddings, plus ablations: centroid-initialized weights, sigma-weighted loss, multi-label rhizome targets |
| `03_full_algorithm_baseline.py` | The original pipeline's scoring rules (sigma-cap, diffuse penalty, recommended winner) on the same centroids |
| `04_review_checks.py` | Few-shot (k examples per class) centroid vs trained head; how many errors are taxonomy near-misses |
| `05_full_pipeline.py` | The deployed engine end to end: real `build_coredrill()` on the training split, real `analyze_text_rhizome_v2()` on every held-out text, final answer chosen with `api/app.py`'s precedence |
| `06_try_sentences.py` | Curated multilingual texts through the real API in-process, the trained head, and nearest-centroid |
| `07_live_space.py` | The same texts against the live HF Space |
| `08_profile_eval.py` | The rhizome's whole profile (active set, ranked lists) vs the trained head at matched size, held-out and live |
| `09_skip_list_check.py` | The deployed answer with the 26-node skip list vs no skip list |
| `10_top_down.py` | Top-down classification (main class first, then children), geometric and trained, vs flat centroid, a flat linear head, and flat logistic regression |
| `11_recheck.py` | Re-scores the texts behind earlier reported scores (the June `tests/eval_suite.json` report and the `06`/`07` texts) with the June engine, the fixed engine, a rebuilt 5.1 coredrill, the classifiers above, and the live Space; also finds each suite text's nearest stored record (leak check) |

## Data

Not in the repo (size). Put these in `data/` at the repo root, or point the
environment variables at them:

| Variable | Default | Contents |
|---|---|---|
| `EVAL_DATA_DIR` | `data/` | `sentences.json`, `sentences_rhizome.json`, `embeddings.npz` (LaBSE, row-aligned), produced by `scripts/rhizome_engine.py collect` / `embed` |
| `EVAL_COREDRILL` | `coredrill/coredrill_hierarchical.json` | the built coredrill (`scripts/rhizome_engine.py build`) |
| `EVAL_TAXONOMY` | `data/TAXONOMY_fixed.txt` | the 238-node taxonomy the coredrill was built from |
| `EVAL_TEST_TEXTS` | `data/test_texts/` | curated `.txt` files for `06` |
| `HF_TOKEN` | (none) | an HF token with read access to the private Space, for `07`, part B of `08`, and `11 --live` |
| `EVAL_JUNE_REPORT` | `tests/reports/eval_20260617_100243.md` | the June `run_eval.py` report `11` compares against (reports are gitignored; without it `11` skips that comparison) |
| `EVAL_JUNE_COMMIT` | `2ca1624` | the commit whose `scripts/rhizome_engine.py` wrote that report |

## Run

```bash
cd eval/held_out
python 01_baseline.py && python 02_train.py && python 03_full_algorithm_baseline.py
python 04_review_checks.py
python 05_full_pipeline.py          # a few minutes on CPU
python 06_try_sentences.py          # loads LaBSE
python 07_live_space.py             # needs HF_TOKEN
python 08_profile_eval.py           # part B needs HF_TOKEN
python 09_skip_list_check.py        # after 05
python 10_top_down.py
python 11_recheck.py --live         # after 06, 07, 10; fetches the suite texts once and caches them
```

Outputs go to `artifacts/` (gitignored). The summary JSONs from the run
reported in `docs/evaluation.md` are in `results/`.
