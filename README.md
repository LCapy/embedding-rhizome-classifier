# Embedding Rhizome Classifier

Geometry-native multilingual text classification over a 238-node rhizomatic taxonomy.

Coredrill maps any text, in any of the 109 languages supported by LaBSE, to a position inside a hand-built semantic taxonomy - without a trained softmax classifier and without labelled inference. Classification is a matrix-vector product against a set of stored centroids, so the same multilingual manifold serves every supported language.

Full paper: [`docs/study_v5.md`](docs/study_v5.md) - *Geometry-Native Text Classification via High-Dimensional Embedding Space Analysis*

Diagrams of the deployment topology, the offline build pipeline, the request
flow, and the taxonomy's multi-parent structure: [`docs/architecture.md`](docs/architecture.md)

---

## What it does

Instead of predicting a single label, Coredrill measures how a piece of text activates 238 semantic territories at once. Each taxonomy node has:

- a **centroid**, the center of a semantic territory;
- a **dispersion estimate** (sigma), the territory's radius or spread;
- **parent links**, defining a six-level taxonomy from broad domains down to specific actions.

The output is a **rhizome profile**: the territories a text activates, their relative intensities, their full taxonomy paths, and a topological flow classification (Territorialization, Deterritorialization, Line of Flight, BwO Approach, Reterritorialization).

The human-readable API output reconstructs coherent rhizome areas from the taxonomy parent map. For example, a sentence about natural selection can activate both the biological chain and a secondary daily-life/family-planning neighborhood:

```text
This text is mainly distributed across 2 coherent rhizome areas:
  1) Life & Biology: Life & Biology -> Biology -> Genetics -> DNA Inheritance -> Genes, with nearby signals Adaptation, Evolution, Natural Selection.
  2) Daily Life: Daily Life -> Family Life -> Family Discussion -> Family Planning Talk.
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
| L1 | Domain | Life & Biology, Work & Economy | ~18 |
| L2 | Discipline / territory | Medicine, Customer Service, Music | ~50 |
| L3 | Topic | Diagnosis, Complaint Handling, Cosmology | ~80 |
| L4 | Scenario | Doctor Appointment, Double Charge Complaint, Natural Selection | ~60 |
| L5 | Action / fine node | Request, Complain, Clarify, Confirm, Genes | ~26 |

The taxonomy parent map is stored in `api/taxonomy.py`. The API uses it to reconstruct full human-readable paths such as `Life & Biology -> Biology -> Genetics -> DNA Inheritance -> Genes`.

---

## Live API

The API runs on a private Hugging Face Space. To request access, contact the author.

**Base URL:** `https://lcapy-coredrill.hf.space`
**Interactive docs:** `https://lcapy-coredrill.hf.space/docs`

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
    2) Daily Life: Daily Life -> Family Life -> Family Discussion -> Family Planning Talk.
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
│       └── keep_alive.yml      # pings the HF Space so it doesn't sleep
├── api/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   └── taxonomy.py             # 238-node taxonomy parent map
├── docs/
│   ├── free_deployment.md      # deployment guide
│   └── study_v5.md             # full research paper
├── scripts/
│   ├── build_sentences.py      # sentence dataset builder
│   ├── predict_folder.py       # batch prediction and comparison
│   └── rhizome_engine.py       # core engine: collect / embed / build / predict
├── Dockerfile
├── README.md
├── requirements-api.txt
└── start.sh
```

---

## Deployment

See [`docs/free_deployment.md`](docs/free_deployment.md) for the complete guide to deploying your own instance on Hugging Face Spaces. The coredrill JSON is distributed separately and fetched at container startup.

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
