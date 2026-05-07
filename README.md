# Coredrill — Rhizome Text Classifier

**Geometry-native multilingual text classification over a 238-node rhizomatic taxonomy.**

No trained classifiers. No labelled inference. Classification by the geometric structure of the [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) embedding manifold, the same manifold across all 109 languages LaBSE supports.

Based on the paper: **Geometry-Native Text Classification via High-Dimensional Embedding Space Analysis** (Lucas, 2026).

---

## What it does

Coredrill maps any text — in any language — to a position in a hand-crafted 238-node semantic taxonomy. Instead of a softmax classifier, it uses the geometric structure of the embedding space: each taxonomy node has a centroid and a dispersion estimate (sigma) computed from 164,444 real-world training sentences. The output is not a single label. It is a **rhizome profile**: the set of semantic territories simultaneously activated by the text, their relative intensities, and a topological flow classification (Territorialization, Deterritorialization, Line of Flight, BwO Approach, Reterritorialization).

The winner selection uses the **Gram-inverse score**:

$$R_i = \alpha_i \cdot \langle \mu_i, x \rangle \cdot \sigma_i^{-0.5}$$

where $\alpha = G^+ s$ is the oblique projection of the query onto the centroid frame — removing frame redundancy between correlated nodes — $\langle \mu_i, x \rangle$ is the cosine proximity, and $\sigma_i^{-0.5}$ is a mild specificity reward. The Gram matrix $G_{ij} = \langle \mu_i, \mu_j \rangle$ is computed exactly from the stored centroids.

### Taxonomy structure

| Level | Name | Example nodes | Count |
|-------|------|---------------|-------|
| L0 | Branch | Natural World, Human Activity & Society | 4 |
| L1 | Domain | Life & Biology, Work & Economy | ~18 |
| L2 | Discipline | Medicine, Billing & Refunds, Music | ~50 |
| L3 | Topic | Diagnosis, Complaint Handling, Cosmology | ~80 |
| L4 | Scenario | Doctor Appointment, Double Charge Complaint | ~60 |
| L5 | Action | Request, Complain, Clarify, Confirm | ~26 |

### Key results

- Flat prototype accuracy: **91.3%** on held-out test sentences
- Cosine similarity to correct node centroid: **0.64–0.66** (z < 1σ)
- Cross-lingual: topology is language-agnostic — topic dominates in 69.4% of triplets across 20 language families
- Training data: 164,444 records from 16 heterogeneous sources

---

## Live API

The API runs on a private Hugging Face Space. To request access contact the author.

Base URL: `https://lcapy-coredrill.hf.space`

Interactive docs: `https://lcapy-coredrill.hf.space/docs`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check, reports loaded nodes and model |
| GET | `/info` | Taxonomy structure, version, skip-node list |
| POST | `/predict` | Classify one text — JSON response |
| POST | `/predict/text` | Classify one text — human-readable terminal output |
| POST | `/predict/batch` | Classify up to 64 texts in one request |
| POST | `/predict/file` | Upload a `.txt` file — JSON response |
| POST | `/predict/file/text` | Upload a `.txt` file — human-readable output |
| GET | `/taxonomy/nodes` | List all nodes, filter by level |
| GET | `/taxonomy/node/{name}` | Node detail: sigma, n, source counts |

---

## Quick start

### Classify a text (JSON)

```bash
curl -X POST https://lcapy-coredrill.hf.space/predict \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"text": "The central bank raised interest rates by 50 basis points to combat inflation."}'
```

### Classify a text (human-readable)

```bash
curl -X POST https://lcapy-coredrill.hf.space/predict/text \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"text": "Natural selection acts on heritable variation, favoring traits that increase reproductive fitness."}'
```

### Upload a file

```bash
curl -X POST https://lcapy-coredrill.hf.space/predict/file/text \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "file=@mytext.txt"
```

### Python

```python
import urllib.request, json

url = "https://lcapy-coredrill.hf.space/predict"
token = "YOUR_TOKEN"

text = "I was charged twice for the same order and I want a full refund."
body = json.dumps({"text": text}).encode("utf-8")
req = urllib.request.Request(url, data=body,
      headers={"Content-Type": "application/json",
               "Authorization": f"Bearer {token}"})

with urllib.request.urlopen(req) as r:
    result = json.loads(r.read())

print(result["best_node"])       # e.g. "Customer Service"
print(result["lineage"])         # e.g. ["Human Activity & Society", "Work & Economy", "Customer Service"]
print(result["active_overlap"])  # co-activated territories
```

### Batch (up to 64 texts)

```python
body = json.dumps({"texts": [
    "The patient presents with lower back pain radiating to the left leg.",
    "Kant argued that synthetic a priori knowledge is possible.",
    "I would like to dispute this charge on my credit card.",
]}).encode("utf-8")

req = urllib.request.Request(
    "https://lcapy-coredrill.hf.space/predict/batch",
    data=body,
    headers={"Content-Type": "application/json",
             "Authorization": f"Bearer {token}"}
)
with urllib.request.urlopen(req) as r:
    results = json.loads(r.read())

for item in results["results"]:
    print(item["best_node"], item["lineage"])
```

---

## Example output

For the text *"The Amazon rainforest plays a critical role in global carbon cycling..."*:

```
========================================================================
COREDRILL  —  RHIZOME PROFILE
========================================================================
  The Amazon rainforest plays a critical role in global carbon cycling...

  GRAM-INVERSE  score = alpha * cos / sigma^0.5
  winner: Climate Change
  NODE                             score     cos  sigma      z      lap      gau      man    MASS
  ----------------------------------------------------------------------------------------
  Climate Change                 +0.0424  0.7493 0.3456   0.73s  ...  100.0% <<
  Climate & Weather              +0.0387  0.6721 0.3525   0.93s  ...   14.3%
  Ecosystems                     +0.0370  0.6441 0.3386   1.05s  ...    4.6%
  Ecology                        +0.0368  0.6397 0.3394   1.06s  ...    4.2%

  RECOMMENDED: Climate Change  (level 3)
  CLADISTIC: Natural World -> Earth & Cosmos -> Climate & Weather -> Climate Change

  PRIMARY FLOW: RETERRITORIALIZATION
    RETERRITORIALIZATION   [###################.........] 0.71 <<
    TERRITORIALIZATION     [#################...........] 0.64
    LINE OF FLIGHT         [##############..............] 0.50
    BWO APPROACH           [########....................] 0.31
    DETERRITORIALIZATION   [####........................] 0.16
========================================================================
```

---

## Multilingual examples

The same taxonomy applies across all LaBSE-supported languages. Tested languages include Portuguese, Spanish, Russian, Japanese, German, French, Korean, Hebrew, Vietnamese, and English. Results on a 10-text benchmark:

| Text | Language | Domain | Gram winner |
|------|----------|--------|-------------|
| Riemann hypothesis | EN | Mathematics | Cosmology (no pure math node) |
| Customer complaint | ES | Customer service | Customer Service |
| Physicist grief | RU | Physics + emotion | Physical Reality |
| AI fairness debate | PT/EN | Philosophy + stats | Verbal Dispute |
| Cardiology note | FR | Clinical | Symptoms & Self-Care |
| Amazon ecology | EN | Science | Climate Change |
| Inflation policy | VI | Economics | Macroeconomics |
| Therapist burnout | DE | Psychology | Symptoms & Self-Care |

---

## Project structure

```
predict_embedding/
├── api/
│   ├── __init__.py
│   └── app.py                  # FastAPI application
├── scripts/
│   ├── rhizome_engine.py       # Core engine: collect / embed / build / predict
│   ├── collect_medical.py      # Medical data augmentation
│   ├── predict_folder.py       # Batch prediction and A/B comparison
│   └── build_sentences.py      # Sentence dataset builder
├── docs/
│   ├── free_deployment.md      # Deployment guide
│   └── study_v5.md             # Full research paper
├── requirements.txt
├── requirements-api.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Theoretical background

Classification is a single matrix-vector product on the sphere:

$$s = Mx, \quad M \in \mathbb{R}^{k \times 768}, \quad x \in S^{767}$$

where $M$ is the centroid matrix (one row per node, unit-normalized). The Gram-inverse winner selection solves:

$$\alpha = G^+ s, \quad G_{ij} = \langle \mu_i, \mu_j \rangle$$

to remove frame redundancy between correlated centroids before scoring. This is the oblique projection of $x$ onto the centroid frame — algebraically equivalent to asking which node most uniquely explains $x$'s position in $S^{767}$, after accounting for all inter-centroid correlations.

Five formal theorems established in Phase I:

1. **Geometric sufficiency** — 99.84% cross-lingual alignment recall; geometry alone suffices.
2. **Topic dominance** — Topic separation (0.303) is 3.24× larger than language separation (0.093).
3. **Orbit overlap** — 98.2% cross-language orbit overlap: concepts have stable geometric locations.
4. **Rhizome structure** — Semantic territories overlap and co-activate; they resist linear hierarchies.
5. **Flow topology** — The Deleuzian flow categories are thresholds in sigma-normalized distance and active-set cardinality space.

---

## Deployment

See [docs/free_deployment.md](docs/free_deployment.md) for the complete guide to deploying on Hugging Face Spaces (free tier, 16 GB RAM).

The coredrill JSON (~13 MB) is distributed via a Hugging Face Dataset repo and fetched at container startup.

---

## License

MIT. The LaBSE model is Apache 2.0. Training datasets are subject to their respective licenses.

---

## Citation

```bibtex
@misc{lucas2026coredrill,
  title  = {Geometry-Native Text Classification via High-Dimensional Embedding Space Analysis},
  author = {Lucas},
  year   = {2026},
  note   = {https://github.com/LCapy/predict_embedding}
}
```
