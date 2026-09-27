# Architecture

Five views of the same system: how the current deployment fits together, the
target self-hosted deployment on EKS, how the offline build pipeline turns
raw text into `coredrill_hierarchical.json`, what happens on a single
`/predict` call, and what the taxonomy graph itself looks like. See
[`docs/study_v5.md`](study_v5.md) for the math behind the scoring formulas
referenced here, [`docs/free_deployment.md`](free_deployment.md) for the
Hugging Face deploy guide, and [`infra/README.md`](../infra/README.md) for
the EKS deploy guide.

---

## 1. Deployment topology (current: Hugging Face)

Three repos, one running service. Code and docs live on GitHub; the
running API lives on a Hugging Face Space; the large coredrill file
lives on a separate HF Dataset repo so it never has to go through git.

```mermaid
flowchart LR
    Dev[Developer] -->|git push| GH[("GitHub\nembedding-rhizome-classifier")]
    Dev -->|git push| HFR[("HF Space repo\nLCapy/coredrill")]

    HFR -->|docker build\non push| Space["HF Space container\nFastAPI + LaBSE, port 7860"]
    DS[("HF Dataset repo\ncoredrill-data")] -->|start.sh downloads\ncoredrill_hierarchical.json| Space

    GH -->|scheduled workflow| Action["GitHub Action\nkeep_alive.yml, every 24h"]
    Action -->|GET /health| Space

    Client["API client\ncurl / Python / app"] -->|Bearer token| Space
    Space -->|JSON / text| Client

    style Space fill:#2d6a4f,color:#fff
    style GH fill:#1f2937,color:#fff
    style HFR fill:#1f2937,color:#fff
    style DS fill:#1f2937,color:#fff
```

Notes:
- The Docker image bakes in the LaBSE model weights at build time (`RUN python -c
  "...SentenceTransformer('sentence-transformers/LaBSE')"`) so cold starts don't
  re-download ~1.5 GB of weights.
- `coredrill_hierarchical.json` (the taxonomy centroids, tens of MB) stays out of
  git entirely and is pulled at container start from an HF Dataset repo, keyed by
  the `HF_DATASET_REPO` / `HF_DATASET_FILE` / `HF_TOKEN` Space secrets.
- The free HF Space tier sleeps after 48h idle; `keep_alive.yml` pings `/health`
  every 24h so it never does.

---

## 2. Deployment topology (target: self-hosted on EKS)

Same container image and API, no external hosting dependency. All AWS
resources are provisioned by [`infra/terraform`](../infra/terraform); the
Deployment/Service/HPA live in [`infra/k8s`](../infra/k8s). Details and exact
commands: [`infra/README.md`](../infra/README.md).

```mermaid
flowchart LR
    Dev[Developer] -->|docker push| ECR[("ECR\ncoredrill-api")]
    Dev -->|terraform apply| TF["Terraform\nVPC + EKS + ECR + S3 + IRSA"]
    TF -.provisions.-> Cluster

    S3[("S3 bucket\ncoredrill-data")] -->|IRSA role, no static keys\nboto3 s3.download_file| Pod

    subgraph Cluster["EKS cluster (namespace: coredrill)"]
        Pod["coredrill-api pods\nFastAPI + LaBSE, port 7860"]
        HPA["HPA\nscale 1-3 on CPU"]
        HPA -.-> Pod
    end

    ECR -->|image pull| Pod

    NLB["Network Load Balancer\ninternet-facing, port 80"] --> Pod
    Client["API client"] --> NLB

    style Pod fill:#2d6a4f,color:#fff
    style TF fill:#1f2937,color:#fff
```

Notes:
- No Hugging Face dependency at runtime: the image is pulled from ECR, and
  `coredrill_hierarchical.json` is pulled from S3 using the pod's IRSA role
  (`COREDRILL_S3_URI`, checked before `HF_DATASET_REPO` in `start.sh`) - so
  the same image serves both the HF Space and EKS.
- Nodes are CPU-only (`t3.large`), matching the current HF Space's resource
  profile; the HPA scales pods 1-3 on CPU utilization once `metrics-server`
  is installed.
- No TLS/custom domain or CI/CD wiring yet - the NLB is plain HTTP on port
  80. See "Next steps" in `infra/README.md`.

---

## 3. Offline build pipeline

This runs locally, not on the Space. Its only output that matters at
serve time is `coredrill_hierarchical.json`.

```mermaid
flowchart TD
    subgraph Sources["Raw data sources"]
        Wiki["Wikipedia extracts"]
        Dialog["DailyDialog / MultiWOZ / SGD /\nEmpatheticDialogues / Taskmaster / Circa"]
        Dolly["Dolly / Persuasion / anchors"]
    end

    Sources --> Match["match_and_enrich()\nassign taxonomy category per record"]
    Match --> Merge["merge_and_save()\ndedupe, merge with existing corpus"]
    Merge --> Embed["embed_all()\nLaBSE encode, batched, checkpointed"]
    Embed --> Emb[("embeddings.npz")]

    Tax[("taxonomy_legal.yaml /\ntaxonomy definition, 238 nodes, 6 levels")] --> Build
    Emb --> Build["build_coredrill()\nbottom-up centroids, L5 -> L0"]

    Build --> Leaf["Leaf nodes L5:\ncentroid = mean of direct-labeled embeddings"]
    Leaf --> Parent["Parent nodes L4 -> L0:\ncentroid = weighted blend of\nchild centroids + own direct data\n(multi-parent: a leaf can feed >1 branch)"]
    Parent --> Sigma["Per-node sigma (dispersion),\nvMF kappa, PCA basis for 3D projection"]

    Sigma --> Out[("coredrill_hierarchical.json")]
    Out -->|huggingface-cli upload| DS[("HF Dataset repo")]
```

Notes:
- `build_coredrill()` walks the taxonomy from the deepest level (L5) up to the
  root (L0). Leaf centroids come directly from labeled embeddings; every
  internal node's centroid is a source-balanced blend of its children plus
  whatever direct data it has.
- "Rhizome" refers to the multi-parent step: a single sentence's embedding can
  contribute to more than one branch of the taxonomy (e.g. a sentence can
  support both a biology node and a daily-life node), so the graph is a DAG,
  not a strict tree.

---

## 4. Request flow (`POST /predict`)

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI (api/app.py)
    participant Eng as rhizome_engine.py
    participant M as LaBSE model
    participant CD as coredrill (in-memory, lazy-loaded once)

    C->>API: POST /predict {text, tau?, full?}
    API->>API: _check_key() - bearer token if API_KEY set
    API->>Eng: analyze_text_rhizome_v2(coredrill, text, model)
    Eng->>M: encode(text) -> 768-dim unit vector
    Eng->>CD: cosine score vs every node centroid, per level
    Eng->>Eng: predict() - best node per level,\nsigma-distance, confidence, margin
    Eng->>Eng: gram_inverse_scores()\nR_i = alpha_i * s_i * NS_i * PC_i * SW_i * sqrt(mass_i)
    Eng->>Eng: _deleuzian_flow()\nclassify flow topology from active set
    Eng-->>API: raw result dict
    API->>API: _build_response()\nresolve lineage path, format NodeResult rows
    API-->>C: PredictionResponse (JSON) or plain-text report
```

Notes:
- The model and the coredrill JSON are both loaded lazily on first request and
  cached as process globals (`_get_model()`, `_get_coredrill()`) - the first
  request after a cold start is slow, every one after is fast.
- `gram_inverse_scores()` is the actual winner-selection step: cosine
  similarity alone (`s_i`) is necessary but not sufficient. A node also needs
  neighborhood support (`NS`, do semantically nearby nodes agree), path
  coherence (`PC`, do its taxonomy ancestors agree), and enough specificity
  (`SW`, prefer tight low-sigma territories over diffuse ones) to win.
- `full=true` additionally returns the top-40 ranked table instead of only the
  active overlap set.

---

## 5. Taxonomy structure (illustrative subgraph)

The real taxonomy has 238 nodes across 6 levels (L0-L5); this is a small
slice showing the multi-parent shape described above - `Natural Selection`
feeds both the biology branch and, indirectly, the daily-life branch through
shared source sentences.

```mermaid
flowchart TD
    L0["L0 Natural World"] --> L1a["L1 Life & Biology"]
    L1a --> L2a["L2 Biology"]
    L2a --> L3a["L3 Genetics"]
    L3a --> L4a["L4 DNA Inheritance"]
    L4a --> L5a["L5 Genes"]
    L4a --> L5b["L5 Natural Selection"]

    L0b["L0 Human Activity & Society"] --> L1b["L1 Daily Life"]
    L1b --> L2b["L2 Family Life"]
    L2b --> L3b["L3 Family Discussion"]
    L3b --> L4b["L4 Family Planning Talk"]

    L5b -. shared source data .-> L4b

    style L5a fill:#2d6a4f,color:#fff
    style L5b fill:#2d6a4f,color:#fff
```
