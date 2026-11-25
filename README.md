# Embedding Rhizome Classifier

Geometry-native multilingual text classification over a hand-built rhizomatic taxonomy (251 nodes).

Coredrill maps any text, in any of the 109 languages supported by LaBSE, to a position inside a hand-built semantic taxonomy - without a trained softmax classifier and without labelled inference. Classification is a matrix-vector product against a set of stored centroids, so the same multilingual manifold serves every supported language.

Full paper: [`docs/study_v5.md`](docs/study_v5.md) - *Geometry-Native Text Classification via High-Dimensional Embedding Space Analysis*

Held-out evaluation, including which of the paper's numbers hold up: [`docs/evaluation.md`](docs/evaluation.md). In short, over 199 classes the deployed pipeline gets 48% of held-out texts exactly right and 73% right or one taxonomy step away; a logistic regression on the same embeddings gets 61% and 84%. On Wikipedia articles neither has seen, both lose about 20 points.

Diagrams of the deployment topology, the offline build pipeline, the request
flow, and the taxonomy's multi-parent structure: [`docs/architecture.md`](docs/architecture.md)

Flowcharts of the training loops: [Training](#training)

---

## What it does

Instead of predicting a single label, Coredrill measures how a piece of text activates every semantic territory in the taxonomy at once. Each taxonomy node has:

- a **centroid**, the center of a semantic territory;
- a **dispersion estimate** (sigma), the territory's radius or spread;
- **parent links**, defining a six-level taxonomy from broad domains down to specific actions.

The output is a **rhizome profile**: the territories a text activates, their relative intensities, their full taxonomy paths, and a topological flow classification (Territorialization, Deterritorialization, Line of Flight, BwO Approach, Reterritorialization).

The human-readable API output reconstructs coherent rhizome areas from the taxonomy parent map. For example, a sentence about natural selection can activate both the biological chain and a secondary family-planning neighborhood:

```text
This text is mainly distributed across 2 coherent rhizome areas:
  1) Life & Biology: Life & Biology -> Biology -> Genetics -> DNA Inheritance -> Genes, with nearby signals Adaptation, Evolution, Natural Selection.
  2) Social Relations: Social Relations -> Family Life -> Family Discussion -> Family Planning Talk.
Final recommendation: Genes.
```

---

## How it works

Winner selection uses a Gram-inverse topological score. For each candidate node *i*:

```text
R_i = alpha_i * s_i * NS_i * PC_i * SW_i * sqrt(mass_i)
```

- **alpha** - oblique projection of the query onto the centroid frame (`G⁺s`, where `G` is the Gram matrix of all centroids). This removes redundancy between correlated nodes so near-duplicate centroids don't both "win" for the same reason.
- **s** - cosine proximity between the query and the node centroid.
- **NS** - neighborhood support: do this node's semantic peers (same level, correlated) also score it well?
- **PC** - path coherence: do the node's ancestors in the taxonomy agree with it, or is this an orphaned signal?
- **SW** - specificity weight: tighter, better-calibrated centroids (lower sigma) are preferred over diffuse ones.
- **sqrt(mass)** - damps nodes the text sits far from, even if they pass the other checks.

The full derivation and the experiments that led to this formula are in [`docs/study_v5.md`](docs/study_v5.md).

The approach originated from an earlier cross-lingual alignment study - a Spanish↔Russian literary corpus used to test whether embedding-space geometry alone is enough to align meaning across languages - described in the same paper.

---

## Taxonomy structure

| Level | Name | Example nodes | Count |
|---|---|---|---:|
| L0 | Branch | Natural World, Human Activity & Society | 4 |
| L1 | Domain | Life & Biology, Work & Economy, Technology | 15 |
| L2 | Discipline / territory | Medicine, Customer Service, Music | 52 |
| L3 | Topic | Diagnosis, Complaint Handling, Cosmology | 94 |
| L4 | Scenario | Doctor Appointment, Double Charge Complaint, Natural Selection | 54 |
| L5 | Action / fine node | Request, Complain, Clarify, Genes | 32 |

The taxonomy parent map is stored in `api/taxonomy.py`: 251 nodes, each with one parent. The API uses it to reconstruct full human-readable paths such as `Life & Biology -> Biology -> Genetics -> DNA Inheritance -> Genes`. The live Space's coredrill 4.0 was built from an earlier 238-node version, before the Technology branch, Epidemiology, Public Health, Pharmacology and Jurisprudence were added.

---

## Live API

The API runs on a private Hugging Face Space. To request access, contact the author.

**Base URL:** `https://lcapy-coredrill.hf.space`
**Interactive docs:** `https://lcapy-coredrill.hf.space/docs`

Hosting stays on the Hugging Face Space for now. [`infra/`](infra/README.md)
holds a Terraform design for a single EC2 instance; it has not been applied.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check; reports loaded nodes and model. |
| GET | `/info` | Taxonomy structure, version, and skip-node list. |
| POST | `/predict` | Classify one text; returns JSON. |
| POST | `/predict/text` | Classify one text; returns human-readable terminal output. |
| POST | `/predict/batch` | Classify up to 64 texts in one request. |
| POST | `/predict/file` | Upload a `.txt` file; returns JSON. |
| POST | `/predict/file/text` | Upload a `.txt` file; returns human-readable output. |
| GET | `/taxonomy/nodes` | List all nodes, optionally filtered by level. |
| GET | `/taxonomy/node/{name}` | Node detail: sigma, n, source counts. |

---

## Quick start

### Classify a text: JSON response

```bash
curl -X POST https://lcapy-coredrill.hf.space/predict \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "The central bank raised interest rates by 50 basis points to combat inflation."}'
```

### Classify a text: human-readable response

```bash
curl -X POST https://lcapy-coredrill.hf.space/predict/text \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "Natural selection acts on heritable variation, favoring traits that increase reproductive fitness."}'
```

### Upload a text file

```bash
curl -X POST https://lcapy-coredrill.hf.space/predict/file/text \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@mytext.txt"
```

---

## Python example

```python
import json
import urllib.request

url = "https://lcapy-coredrill.hf.space/predict"
token = "YOUR_TOKEN"

text = "I was charged twice for the same order and I want a full refund."

body = json.dumps({"text": text}).encode("utf-8")

req = urllib.request.Request(
    url,
    data=body,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    },
)

with urllib.request.urlopen(req) as response:
    result = json.loads(response.read())

print(result["best_node"])
print(result["lineage"])
print(result["active_overlap"])
```

---

## Batch prediction

```python
import json
import urllib.request

token = "YOUR_TOKEN"

body = json.dumps({
    "texts": [
        "The patient presents with lower back pain radiating to the left leg.",
        "Kant argued that synthetic a priori knowledge is possible.",
        "I would like to dispute this charge on my credit card.",
    ]
}).encode("utf-8")

req = urllib.request.Request(
    "https://lcapy-coredrill.hf.space/predict/batch",
    data=body,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    },
)

with urllib.request.urlopen(req) as response:
    results = json.loads(response.read())

for item in results["results"]:
    print(item["best_node"], item["lineage"])
```

---

## Example output

For the text:

```text
Natural selection acts on heritable variation, favoring traits that increase reproductive fitness.
```

Coredrill returns a rhizome profile like:

```text
========================================================================
COREDRILL  -  RHIZOME PROFILE
========================================================================
  Natural selection acts on heritable variation, favoring traits that...

  This text is mainly distributed across 2 coherent rhizome areas:
    1) Life & Biology: Life & Biology -> Biology -> Genetics -> DNA Inheritance -> Genes, with nearby signals Adaptation, Evolution, Natural Selection.
    2) Social Relations: Social Relations -> Family Life -> Family Discussion -> Family Planning Talk.
  Final recommendation: Genes.

  GRAM-INVERSE  score = alpha * cos * NS * PC * SW
  winner: Genes
  NODE                             score     cos  sigma      z      lap      gau      man    MASS
  ----------------------------------------------------------------------------------------
  Genes                          +0.0217  0.5096 0.3188   1.54s  -1.5384  -1.1833  -1.4048   52.5% <<
  Genetics                       +0.0216  0.5034 0.3213   1.55s  -1.5457  -1.1946  -1.4134   48.3%
  DNA Inheritance                +0.0215  0.5042 0.3202   1.55s  -1.5485  -1.1989  -1.4159   47.1%
  Ecology                        +0.0213  0.5018 0.3394   1.47s  -1.4679  -1.0773  -1.3386   99.2%
  Ecosystems                     +0.0213  0.5032 0.3386   1.47s  -1.4674  -1.0766  -1.3377  100.0%

  RECOMMENDED: Genes  (level 5)

  CLADISTIC PATH:
  Natural World -> Life & Biology -> Biology -> Genetics -> DNA Inheritance -> Genes

  PRIMARY FLOW: BwO APPROACH
========================================================================
```

(Illustrative sample - exact scores shift as the taxonomy and scoring weights are tuned.)

The summary is taxonomy-backed: the API does not merely print the top-ranked isolated nodes. It reconstructs full parent chains from `api/taxonomy.py`.

---

## Multilingual support

The same taxonomy applies across all LaBSE-supported languages. Tested languages include Portuguese, Spanish, Russian, Japanese, German, French, Korean, Hebrew, Vietnamese, Arabic, Chinese, and English - with no per-language configuration.

---

## Project structure

```text
embedding-rhizome-classifier/
├── .github/
│   └── workflows/
│       ├── ci.yml              # checks on every push
│       ├── deploy_space.yml    # puts the API on the HF Space
│       └── keep_alive.yml      # pings the HF Space so it doesn't sleep
├── api/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   ├── align_ui.py             # Gradio text-alignment page at /align
│   ├── alignment_engine.py     # segmentation and LaBSE alignment for /align
│   └── taxonomy.py             # 251-node taxonomy parent map
├── docs/
│   ├── architecture.md         # deployment, build pipeline, request flow
│   ├── evaluation.md           # held-out evaluation
│   ├── free_deployment.md      # deployment guide
│   ├── releases.md             # how code and models reach Hugging Face
│   ├── study_v5.md             # full research paper
│   └── training.md             # the training loop
├── eval/
│   └── held_out/               # held-out evaluation scripts and result JSONs
├── infra/                      # Terraform for a single-EC2 option (not applied)
├── scripts/
│   ├── rhizome_engine.py       # core engine: collect / embed / build / predict
│   ├── build_corpus.py         # the corpus, rebuilt: one label per text, even sizes
│   ├── gather_territories.py   # Wikipedia text for territories (territory_articles.txt)
│   ├── gather_dialogues.py     # conversation lines for what people do in conversation
│   ├── build_sentences.py      # multi-label sentence dataset builder
│   ├── deploy_space.py         # the Space's files, and what would change there
│   ├── publish_model.py        # a trained model as a version on Hugging Face
│   ├── collect_medical.py      # targeted medical data collection
│   ├── collect_tech_medical.py # Wikipedia data for the newer nodes
│   └── predict_folder.py       # batch prediction and A/B comparison
├── space/                      # the Space's README (its Hugging Face settings)
├── tests/
│   ├── unit/                   # fast tests, no model needed
│   ├── run_eval.py             # online eval suite (Wikipedia, BBC, Reddit)
│   └── eval_suite.json
├── train/                      # training loop: python -m train
├── Dockerfile
├── README.md
├── requirements-api.txt
├── requirements-train.txt
└── start.sh
```

---

## Deployment

See [`docs/free_deployment.md`](docs/free_deployment.md) for the complete guide to deploying your own instance on Hugging Face Spaces. The coredrill JSON is distributed separately and fetched at container startup. How code and trained models get from here to Hugging Face is in [`docs/releases.md`](docs/releases.md).

---

## Training

`python -m train` fine-tunes the encoder (LaBSE's top layers) so the taxonomy's territories separate better, rebuilds the coredrill with it, and keeps learning from Wikipedia articles and dropped-in text that it labels itself. A model replaces the current one only when it does better on a fixed holdout of articles it has never seen. Every loop is recorded, whatever its outcome, and `python -m train report` lays them all next to the by-article benchmark in `eval/held_out/results/`. See [`docs/training.md`](docs/training.md).

The flowcharts below show the default settings with a GPU.

### The whole cycle

The **champion** is the best model so far. A **challenger** is any newly trained model.

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 22, "rankSpacing": 28}}}%%
flowchart LR
    Data[("Labeled<br/>data")] --> Train["Train a<br/>challenger"]
    Data -->|"drawn once"| Holdout[("Fixed<br/>holdout")]
    Train --> Score["Score it"]
    Holdout --> Score
    Score --> Gate{{"Beats the<br/>champion?"}}
    Gate -->|"no"| Keep["Champion<br/>stays"]
    Gate -->|"yes"| Champion(["New<br/>champion"])
    Champion --> New["Labels<br/>new text"]
    New --> Self[("Self-labeled<br/>texts")]
    Self --> Train

    classDef champion fill:#37483f,color:#d6d3cb,stroke:#4a5d53
    classDef store fill:#30343e,color:#d6d3cb,stroke:#474c58
    class Champion champion
    class Data,Holdout,Self store
```

| Command | What it runs |
|---|---|
| `python -m train baseline` | Scores stock LaBSE on the holdout, once, for comparison |
| `python -m train fit` | Trains one challenger on the labeled data and the pool of proven self-labeled texts |
| `python -m train round` | One self-training round: finds and labels new text, then trains a challenger with it |
| `python -m train loop` | Rounds (5 by default), stopping early after two in a row without a promotion |
| `python -m train forever` | Rounds with no end, until `python -m train stop` |
| `python -m train view` | The training loops in 3D and the classifier, in the browser |
| `python -m train classify "a text"` | The rhizome profile of a text, with the champion's model |

### Training one model

Every challenger starts again from stock LaBSE, so an earlier model's mistakes do not carry over in its weights.

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 22, "rankSpacing": 28}}}%%
flowchart LR
    Rows[("Training<br/>rows")] --> Embed["Embed a<br/>batch of 32"]
    Base["Stock<br/>LaBSE"] --> Embed
    Embed --> Loss["Loss against<br/>prototypes"]
    Loss --> Step["Update<br/>weights"]
    Step -->|"next batch,<br/>4 epochs"| Embed
    Step --> Best["Keep the<br/>best epoch"]
    Best --> Score["Rebuild coredrill,<br/>score the holdout"]

    classDef store fill:#30343e,color:#d6d3cb,stroke:#474c58
    class Rows store
```

- **Weights that train:** LaBSE's top 4 layers, its Dense layer, and one prototype per label that starts at the label's centroid.
- **Training rows:** labeled rows count fully, self-labeled ones 0.5 × their confidence. 10% of each label's labeled rows are kept for validation, which picks the best epoch and calibrates the probabilities.
- **The loss** is cross-entropy between each text's cosines with the prototypes (× 20) and soft targets (0.9 on the label, 0.1 on its ancestors), plus 0.1 × the distance from stock LaBSE's embedding, which carries its cross-lingual alignment through training.
- **Scoring** covers five systems: the deployed pipeline, the nearest centroid, the linear head, logistic regression and the prototypes.

### The gate

A challenger has to pass every test against the champion, on the same holdout texts. The first model trained becomes champion without one.

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 22, "rankSpacing": 28}}}%%
flowchart LR
    New["Challenger"] --> Strict{{"Strict accuracy<br/>up 0.2 points?"}}
    Strict -->|"yes"| Near{{"Near-miss down<br/>1 point at most?"}}
    Near -->|"yes"| Z{{"McNemar z at<br/>least 1.645?"}}
    Z -->|"yes"| Win(["New<br/>champion"])
    Strict -->|"no"| Stay["Not<br/>promoted"]
    Near -->|"no"| Stay
    Z -->|"no"| Stay

    classDef champion fill:#37483f,color:#d6d3cb,stroke:#4a5d53
    classDef ended fill:#483b37,color:#d6d3cb,stroke:#5d4d48
    class Win champion
    class Stay ended
```

Both accuracies are the deployed pipeline's. A near miss is an answer that is right or one taxonomy step away. McNemar's test compares the two models text by text: z is the number of texts only the challenger gets right, minus those only the champion gets right, over the square root of their sum. Running without end raises the bar to 2.33, because many tries on one holdout make a lucky win likely.

### A self-training round

First it finds new text and labels it:

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 22, "rankSpacing": 28}}}%%
flowchart LR
    Wiki["Similar<br/>Wikipedia<br/>articles"] --> Drop["Drop known<br/>and holdout<br/>text"]
    Inbox[("data/inbox/")] --> Drop
    Drop --> Label["Champion<br/>labels it"]
    Label --> Sure{{"Confident, and<br/>second signal<br/>agrees?"}}
    Sure -->|"yes"| Trans["Add other<br/>languages"]
    Trans --> Caps["Apply<br/>the caps"]
    Caps --> Rows[("This<br/>round's<br/>rows")]
    Sure -->|"no"| Skip["Left out,<br/>reason<br/>recorded"]

    classDef store fill:#30343e,color:#d6d3cb,stroke:#474c58
    classDef ended fill:#483b37,color:#d6d3cb,stroke:#5d4d48
    class Inbox,Rows store
    class Skip ended
```

Then it trains a challenger with those rows and decides what becomes of them:

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 22, "rankSpacing": 28}}}%%
flowchart LR
    Rows[("This round's<br/>rows")] --> Train["Train a<br/>challenger"]
    Stack[("Unproven<br/>stack")] --> Train
    Pool[("Labeled data<br/>and pool")] --> Train
    Train --> Gate{{"Passes<br/>the gate?"}}
    Gate -->|"yes"| Promote(["New champion: stack and<br/>rows move to the pool"])
    Gate -->|"no"| Worse{{"Significantly<br/>worse?"}}
    Worse -->|"no"| Join[("Rows join the<br/>unproven stack")]
    Worse -->|"yes"| Aside["Rows set aside"]

    classDef champion fill:#37483f,color:#d6d3cb,stroke:#4a5d53
    classDef store fill:#30343e,color:#d6d3cb,stroke:#474c58
    classDef ended fill:#483b37,color:#d6d3cb,stroke:#5d4d48
    class Promote champion
    class Rows,Stack,Pool,Join store
    class Aside ended
```

- **Where it looks:** 40 of the labels the holdout tests, fewest articles first and rotating. Articles this champion already judged are skipped.
- **Confident:** a probability of at least 0.8, and 0.15 ahead of the runner-up.
- **The second signal:** for a Wikipedia article, the label must be the one whose articles led to it, or a relative (ancestor, descendant or sibling). For an inbox text, the nearest centroid must be the label or a relative. The holdout must also test the label.
- **Other languages:** the article's versions in Spanish, French, German and Portuguese, where the champion places them near its label.
- **The caps:** at most half as many new rows as a label has labeled rows (at least 3), as many self-labeled rows in all as labeled ones, and 1,500 per round.
- **Significantly worse** means z is at most -1.645. If the round before was too, the whole unproven stack is set aside as well.
- **Ending early:** a round that finds no new text, or keeps none of it, ends before training.
- **No text is lost track of:** `candidates.jsonl` in the round's folder holds every text it considered, the champion's top two labels and the decision.

### Running without end

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 22, "rankSpacing": 28}}}%%
flowchart LR
    Start(["train forever"]) --> Stop{{"Stop<br/>requested?"}}
    Stop -->|"no"| Busy{{"Busy, on battery<br/>or low on disk?"}}
    Busy -->|"no"| Run["Run one<br/>round"]
    Run --> Adjust["Adjust to<br/>the outcome"]
    Adjust --> Stop
    Busy -->|"yes"| Wait["Wait a<br/>minute"]
    Wait --> Stop
    Stop -->|"yes"| End(["Ends"])

    classDef ended fill:#483b37,color:#d6d3cb,stroke:#5d4d48
    class End ended
```

| Outcome of the round | What comes next |
|---|---|
| Promoted | Searches narrow again |
| Not promoted | Searches wider if the round added fewer than 60 rows |
| Nothing new | Searches wider; after two in a row, trains one model with another seed instead |
| Failed, or silent for 60 minutes | Waits 2 minutes, then longer each time, up to an hour |

`python -m train stop` takes effect after the current loop. There are four search widths, from 10 articles per label to 60, and a restart resumes at the width it had reached. Each round searches 20 labels and asks Wikipedia slowly, slowing down further when told to. `python -m train watch` follows the running loop, and `python -m train view` shows its matrices in 3D.

### Using the trained classifier

`python -m train view` also serves the classifier at `http://127.0.0.1:8765/classify`, to this machine only. It places a text among the taxonomy's territories in 3D and shows the answer, the active territories, the areas and the flow, with the champion or any other trained loop. It can also pick a random holdout text and compare the answer with its true label. See [`docs/training.md`](docs/training.md#classifying-with-a-trained-model).

---

## License

MIT.

The LaBSE model is Apache 2.0. Training datasets are subject to their respective licenses.

---

## Citation

```bibtex
@misc{lucas2026coredrill,
  title  = {Geometry-Native Text Classification via High-Dimensional Embedding Space Analysis},
  author = {Lucas},
  year   = {2026},
  note   = {https://github.com/LCapy/embedding-rhizome-classifier}
}
```
