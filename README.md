# Embedding Rhizome Classifier:

Geometry-native multilingual text classification over a 238-node rhizomatic taxonomy.
No trained classifiers. No labelled inference. Classification is performed through the geometric structure of the LaBSE embedding manifold: the same multilingual manifold shared across all 109 languages supported by LaBSE.
Based on the paper: Geometry-Native Text Classification via High-Dimensional Embedding Space Analysis (Lucas, 2026).
---
What it does
Coredrill maps any text, in any of the 109 supported languages, to a position inside a crafted 238-node semantic taxonomy.
Instead of using a softmax classifier, it uses the geometry of the embedding space. Each taxonomy node has:
a centroid, representing the center of a semantic territory;
a dispersion estimate, sigma, representing the territory's radius or spread;
parent links, defining a 6-level rhizomatic taxonomy from broad domains to specific actions.
The output is not just a single label. It is a rhizome profile: the set of semantic territories simultaneously activated by the text, their relative intensities, their full taxonomy paths, and a topological flow classification.
The human-readable API output now also reconstructs coherent rhizome areas from the taxonomy parent map. For example, a sentence about natural selection can activate both the biological chain and a secondary daily-life/family-planning neighborhood:
```text
This text is mainly distributed across 2 coherent rhizome areas:
  1) Life & Biology: Life & Biology -> Biology -> Genetics -> DNA Inheritance -> Genes, with nearby signals Adaptation, Evolution, Natural Selection.
  2) Daily Life: Daily Life -> Family Life -> Family Discussion -> Family Planning Talk.
Final recommendation: Genes.
```
The topological flow classes are:
Territorialization
Deterritorialization
Line of Flight
BwO Approach
Reterritorialization
---
Gram-inverse winner selection
The winner selection uses the Gram-inverse score:
$$
R_i = \alpha_i \cdot \langle \mu_i, x \rangle \cdot \sigma_i^{-0.5}
$$
where:
$$
\alpha = G^+s
$$
is the oblique projection of the query onto the centroid frame, removing frame redundancy between correlated nodes.
$$
\langle \mu_i, x \rangle
$$
is the cosine proximity between the query embedding and the node centroid.
$$
\sigma_i^{-0.5}
$$
is a mild specificity reward.
The Gram matrix is computed exactly from the stored centroids:
$$
G_{ij} = \langle \mu_i, \mu_j \rangle
$$
This makes the winner selection algebraically equivalent to asking which node most uniquely explains the query's position in the 768-dimensional unit sphere, after accounting for all inter-centroid correlations.
---
Taxonomy structure
Level	Name	Example nodes	Count
L0	Branch	Natural World, Human Activity & Society	4
L1	Domain	Life & Biology, Work & Economy	~18
L2	Discipline / territory	Medicine, Customer Service, Music	~50
L3	Topic	Diagnosis, Complaint Handling, Cosmology	~80
L4	Scenario	Doctor Appointment, Double Charge Complaint, Natural Selection	~60
L5	Action / fine node	Request, Complain, Clarify, Confirm, Genes	~26
The taxonomy parent map is stored in:
```text
api/taxonomy.py
```
The API uses this file to reconstruct full human-readable paths such as:
```text
Life & Biology -> Biology -> Genetics -> DNA Inheritance -> Genes
```
and:
```text
Human Activity & Society -> Work & Economy -> Customer Service -> Complaint Handling
```
---
Key results
Flat prototype accuracy: 91.3% on held-out test sentences.
Cosine similarity to correct node centroid: 0.64–0.66.
Correct-node distance: typically z < 1σ.
Cross-lingual behavior: topology is language-agnostic.
Topic dominance: topic dominates language in 69.4% of triplets across 20 language families.
Training data: 164,444 records from 16 heterogeneous sources.
Taxonomy size: 238 nodes.
---
Live API
The API runs on a private Hugging Face Space. To request access, contact the author.
Base URL:
```text
https://lcapy-coredrill.hf.space
```
Interactive docs:
```text
https://lcapy-coredrill.hf.space/docs
```
---
Endpoints
Method	Path	Description
GET	`/health`	Liveness check, reports loaded nodes and model
GET	`/info`	Taxonomy structure, version, skip-node list
POST	`/predict`	Classify one text — JSON response
POST	`/predict/text`	Classify one text — human-readable terminal output
POST	`/predict/batch`	Classify up to 64 texts in one request
POST	`/predict/file`	Upload a `.txt` file — JSON response
POST	`/predict/file/text`	Upload a `.txt` file — human-readable output
GET	`/taxonomy/nodes`	List all nodes, optionally filtered by level
GET	`/taxonomy/node/{name}`	Node detail: sigma, n, source counts
---
Quick start
Classify a text: JSON response
```bash
curl -X POST https://lcapy-coredrill.hf.space/predict \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"text": "The central bank raised interest rates by 50 basis points to combat inflation."}'
```
Classify a text: human-readable response
```bash
curl -X POST https://lcapy-coredrill.hf.space/predict/text \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"text": "Natural selection acts on heritable variation, favoring traits that increase reproductive fitness."}'
```
Upload a text file
```bash
curl -X POST https://lcapy-coredrill.hf.space/predict/file/text \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "file=@mytext.txt"
```
---
Python example
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
Batch prediction
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
Example output
For the text:
```text
Natural selection acts on heritable variation, favoring traits that increase reproductive fitness.
```
Coredrill returns a rhizome profile like:
```text
========================================================================
COREDRILL  —  RHIZOME PROFILE
========================================================================
  Natural selection acts on heritable variation, favoring traits that...

  This text is mainly distributed across 2 coherent rhizome areas:
    1) Life & Biology: Life & Biology -> Biology -> Genetics -> DNA Inheritance -> Genes, with nearby signals Adaptation, Evolution, Natural Selection.
    2) Daily Life: Daily Life -> Family Life -> Family Discussion -> Family Planning Talk.
  Final recommendation: Genes.

  GRAM-INVERSE  score = alpha * cos / sigma^0.5
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
The summary is taxonomy-backed: the API does not merely print the top-ranked isolated nodes. It reconstructs full parent chains from `api/taxonomy.py`.
---
Multilingual examples
The same taxonomy applies across all LaBSE-supported languages. Tested languages include Portuguese, Spanish, Russian, Japanese, German, French, Korean, Hebrew, Vietnamese, Arabic, Chinese, and English.
Text	Language	Domain	Gram winner
Riemann hypothesis	EN	Mathematics	Cosmology, when no pure math node is available
Customer complaint	ES	Customer service	Customer Service
Physicist grief	RU	Physics + emotion	Physical Reality
AI fairness debate	PT/EN	Philosophy + statistics	Verbal Dispute
Cardiology note	FR	Clinical	Symptoms & Self-Care
Amazon ecology	EN	Science	Climate Change
Inflation policy	VI	Economics	Macroeconomics
Therapist burnout	DE	Psychology / health	Symptoms & Self-Care
Billing refund	KO	Customer service / billing	Customer Service
Doctor appointment	PT	Health routine	Medicine / Appointments & Medication neighborhood
---
Project structure
```text
predict_embedding/
├── .github/
│   └── workflows/
├── api/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   └── taxonomy.py             # 238-node taxonomy parent map
├── docs/
│   ├── free_deployment.md      # Deployment guide
│   └── study_v5.md             # Full research paper
├── scripts/
│   ├── build_sentences.py      # Sentence dataset builder
│   ├── predict_folder.py       # Batch prediction and comparison
│   └── rhizome_engine.py       # Core engine: collect / embed / build / predict
├── Dockerfile
├── README.md
├── requirements-api.txt
└── start.sh
```
---
Theoretical background
Classification is a single matrix-vector product on the unit sphere:
$$
s = Mx,\quad M \in \mathbb{R}^{k \times 768},\quad x \in S^{767}
$$
where `M` is the centroid matrix, one unit-normalized row per taxonomy node.
The Gram-inverse winner selection solves:
$$
\alpha = G^+s,\quad G_{ij} = \langle \mu_i, \mu_j \rangle
$$
This removes frame redundancy between correlated centroids before scoring.
Geometrically, this asks which node most uniquely explains the query's position in the embedding manifold after accounting for all inter-centroid correlations.
---
Formal claims established in Phase I
Geometric sufficiency  
Cross-lingual alignment recall reaches 99.84%; geometry alone is sufficient for semantic localization.
Topic dominance  
Topic separation is 3.24× larger than language separation.
Orbit overlap  
Concepts have stable cross-language geometric locations.
Rhizome structure  
Semantic territories overlap and co-activate; they resist a purely linear hierarchy.
Flow topology  
The Deleuzian flow categories are thresholds in sigma-normalized distance and active-set cardinality space.
---
Deployment
See:
```text
docs/free_deployment.md
```
for the complete guide to deploying on Hugging Face Spaces.
The coredrill JSON is distributed separately and fetched at container startup.
---
License
MIT.
The LaBSE model is Apache 2.0.
Training datasets are subject to their respective licenses.
---
Citation
```bibtex
@misc{lucas2026coredrill,
  title  = {Geometry-Native Text Classification via High-Dimensional Embedding Space Analysis},
  author = {Lucas},
  year   = {2026},
  note   = {https://github.com/LCapy/predict_embedding}
}
```
