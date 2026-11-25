# Free Deployment Guide

Everything here is open-source.

---

## What you will end up with

| Resource | Provider | Purpose | Cost |
|----------|----------|---------|------|
| Code repository | GitHub | Source of truth for all code | Free |
| `coredrill_hierarchical.json` | HF Dataset repo | Stores the coredrill file (about 9 MB for the 5.1 build) | Free |
| Live API | HF Spaces (Docker) | `https://your-username-coredrill.hf.space` | Free |
| Keep-alive pinger | GitHub Actions (cron) | Prevents the Space from sleeping | Free |

---

## Step 1 - Push the code to GitHub

1. Create a new repository at https://github.com/new
   - Name: `coredrill`
   - Visibility: Public
   - Do not initialize with README (you have one already)

2. In your local project folder:

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/your-username/coredrill.git
git push -u origin main
```

---

## Step 2 - Upload the coredrill JSON to a HF Dataset repo

The `coredrill_hierarchical.json` is about 9 MB (5.1 build) and changes on every rebuild, which makes it a poor fit for git history. HF Dataset repos are designed for exactly this.

1. Go to https://huggingface.co/new-dataset
   - Name: `coredrill-data`
   - Visibility: Public (so the Space can fetch it without a token)
   - License: MIT

2. Upload `coredrill_hierarchical.json` to the dataset repo. You can do this via the web UI (drag and drop) or via the CLI:

```bash
pip install huggingface_hub
huggingface-cli login          # enter your HF token

python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="coredrill/coredrill_hierarchical.json",
    path_in_repo="coredrill_hierarchical.json",
    repo_id="your-username/coredrill-data",
    repo_type="dataset",
)
print("Done.")
EOF
```

Note the dataset repo ID - you will need it in Step 3.

---

## Step 3 - Create the HF Space

1. Go to https://huggingface.co/new-space
   - Owner: your username
   - Space name: `coredrill`
   - SDK: **Docker**
   - Hardware: **CPU basic** (free, 16 GB RAM, 2 cores)
   - Visibility: Public

2. This creates a git repo at `https://huggingface.co/spaces/your-username/coredrill`.
   Clone it:

```bash
git clone https://huggingface.co/spaces/your-username/coredrill hf-space
cd hf-space
```

3. Copy the deployment files into it (they live at the root of this repo):

```bash
# From your main project directory:
cp Dockerfile                       hf-space/Dockerfile
cp start.sh                         hf-space/start.sh
cp requirements-api.txt             hf-space/requirements-api.txt
cp -r api/                          hf-space/api/
cp -r scripts/                      hf-space/scripts/
```

4. Give `hf-space/README.md` the Space's front matter, so Hugging Face builds it with Docker on port 7860:

```yaml
---
title: Coredrill
sdk: docker
app_port: 7860
---
```

5. Commit and push to HF:

```bash
cd hf-space
git add .
git commit -m "initial deployment"
git push
```

The Space will now build. This takes 5–10 minutes on first build because it downloads
and caches LaBSE inside the Docker image.

---

## Step 4 - Set the Space secrets

In your Space settings (https://huggingface.co/spaces/your-username/coredrill/settings),
add these secrets:

| Secret name | Value | Required |
|-------------|-------|---------|
| `HF_DATASET_REPO` | `your-username/coredrill-data` | Yes |
| `HF_DATASET_FILE` | `coredrill_hierarchical.json` | No (this is the default) |
| `HF_TOKEN` | your HF read token | Only if the dataset repo is private |

Secrets are injected as environment variables at runtime. They are never visible in logs.

After setting secrets, restart the Space (Settings → Factory reboot).

The startup script (`start.sh`) will download the JSON from your dataset repo on first boot,
then start the API.

---

## Step 5 - Verify it works

```bash
curl https://your-username-coredrill.hf.space/health
# Expected: {"status":"ok","coredrill_nodes":238,...}

curl -X POST https://your-username-coredrill.hf.space/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Algebra is a branch of mathematics."}'
```

The interactive Swagger UI is at:
```
https://your-username-coredrill.hf.space/docs
```

---

## Step 6 - Set up the keep-alive GitHub Action

The free Space sleeps after 48 hours of no requests. The keep-alive workflow pings `/health`
every 24 hours so it never sleeps.

1. In your GitHub repo, go to **Settings → Secrets and variables → Actions → New repository secret**:
   - Name: `HF_SPACE_URL`
   - Value: `https://your-username-coredrill.hf.space`

2. The workflow is already in `.github/workflows/keep_alive.yml`. It runs daily at 12:00 UTC.

3. To test it immediately: go to **Actions → Keep HF Space Awake → Run workflow**.

---

## Updating the coredrill JSON

Publish it as a version, together with the encoder it was built with, and then
switch the Space to it: see [releases.md](releases.md).

---

## Updating the API code

The Space runs what this repository holds: GitHub Actions deploys it (see
[releases.md](releases.md)). Do not edit the Space's files by hand; the next
deploy replaces them.

---

## Cold start behaviour

When someone hits a sleeping Space, HF wakes it automatically. The cold start takes
roughly 60–90 seconds because uvicorn starts, the startup script runs (the JSON is already
cached in the container filesystem after the first boot), and LaBSE loads into memory.

The keep-alive ping prevents this from ever happening in practice.

---

## Limits of the free tier

| Limit | Value | Impact |
|-------|-------|--------|
| RAM | 16 GB | Fine - LaBSE + coredrill needs ~1.5 GB |
| CPU | 2 cores | Inference is single-threaded; fine for moderate traffic |
| Disk | 50 GB ephemeral | Fine - model + code + JSON < 2 GB |
| Inactivity sleep | 48 hours | Solved by the keep-alive workflow |
| Concurrent requests | No hard limit, but single worker | Each request takes ~50–200 ms on CPU; queue builds under heavy load |

If you ever need to handle more concurrent requests, the only free path is to add more workers
(`--workers 2` in the uvicorn command) - but 2 CPU cores is the ceiling on the free tier, so
2 workers is also the ceiling without paying.
