#!/bin/bash
set -e

COREDRILL_DIR="/app/coredrill"
COREDRILL_FILE="$COREDRILL_DIR/coredrill_hierarchical.json"

mkdir -p "$COREDRILL_DIR"

# Download coredrill JSON from Hugging Face Dataset repo if not already present.
# Set HF_DATASET_REPO as a Space Secret, e.g.:  your-username/coredrill-data
if [ ! -f "$COREDRILL_FILE" ]; then
    echo "[start.sh] Downloading coredrill JSON from HF Dataset..."
    python - <<'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download

repo_id  = os.environ.get("HF_DATASET_REPO")
filename = os.environ.get("HF_DATASET_FILE", "coredrill_hierarchical.json")
token    = os.environ.get("HF_TOKEN")           # only needed if the dataset is private

if not repo_id:
    print("[start.sh] ERROR: HF_DATASET_REPO environment variable is not set.", file=sys.stderr)
    print("[start.sh] Set it to your HF Dataset repo, e.g. your-username/coredrill-data", file=sys.stderr)
    sys.exit(1)

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
fi

echo "[start.sh] Starting Coredrill API on port $PORT ..."
exec uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-7860}" --workers 1
