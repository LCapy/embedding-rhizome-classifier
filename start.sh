#!/bin/bash
set -e

COREDRILL_DIR="/app/coredrill"
COREDRILL_FILE="$COREDRILL_DIR/coredrill_hierarchical.json"

mkdir -p "$COREDRILL_DIR"

# Fetch coredrill JSON if not already present. Two backends:
#   - COREDRILL_S3_URI  (s3://bucket/key.json)          - used on EKS, no
#     static credentials needed: boto3 picks up the pod's IRSA role.
#   - HF_DATASET_REPO   (e.g. your-username/coredrill-data) - used on
#     Hugging Face Spaces. HF_TOKEN only needed if the dataset is private.
# S3 takes priority if both are set.
if [ ! -f "$COREDRILL_FILE" ]; then
    if [ -n "$COREDRILL_S3_URI" ]; then
        echo "[start.sh] Downloading coredrill JSON from S3: $COREDRILL_S3_URI ..."
        python - <<'PYEOF'
import os, sys
import boto3

uri = os.environ["COREDRILL_S3_URI"]
if not uri.startswith("s3://"):
    print(f"[start.sh] ERROR: COREDRILL_S3_URI must start with s3://, got: {uri}", file=sys.stderr)
    sys.exit(1)

bucket, key = uri[len("s3://"):].split("/", 1)
dest = "/app/coredrill/coredrill_hierarchical.json"
print(f"[start.sh] Fetching s3://{bucket}/{key} -> {dest}")
boto3.client("s3").download_file(bucket, key, dest)
print("[start.sh] Downloaded.")
PYEOF
    elif [ -n "$HF_DATASET_REPO" ]; then
        echo "[start.sh] Downloading coredrill JSON from HF Dataset..."
        python - <<'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download

repo_id  = os.environ.get("HF_DATASET_REPO")
filename = os.environ.get("HF_DATASET_FILE", "coredrill_hierarchical.json")
token    = os.environ.get("HF_TOKEN")           # only needed if the dataset is private

print(f"[start.sh] Fetching {filename} from {repo_id} ...")
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    token=token or None,
    local_dir="/app/coredrill",
)
print(f"[start.sh] Downloaded to {path}")
PYEOF
    else
        echo "[start.sh] ERROR: neither COREDRILL_S3_URI nor HF_DATASET_REPO is set." >&2
        echo "[start.sh] Set COREDRILL_S3_URI=s3://bucket/coredrill_hierarchical.json (EKS)" >&2
        echo "[start.sh] or HF_DATASET_REPO=your-username/coredrill-data (HF Spaces)." >&2
        exit 1
    fi
fi

echo "[start.sh] Starting Coredrill API on port $PORT ..."
exec uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-7860}" --workers 1
