"""Publishes a trained model to Hugging Face, as a version.

    python -m train export --out exports/v6.0 --model-id LCapy/coredrill-encoder
    python scripts/publish_model.py exports/v6.0 --version v6.0            # what would be uploaded
    python scripts/publish_model.py exports/v6.0 --version v6.0 --publish  # upload it

Two private repositories, each tagged with the version:

  LCapy/coredrill-encoder   model: the trained encoder, with a model card
  LCapy/coredrill-data      dataset: coredrill_hierarchical.json, which the Space downloads

Nothing the Space uses changes by itself. To switch it, set its variables
MODEL_NAME (LCapy/coredrill-encoder) and HF_DATASET_REVISION (v6.0) and restart
it; to go back, set the earlier version. Sign in first with `hf auth login`
and a token that may write to these two repositories.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def card(version: str, meta: dict, report: dict) -> str:
    m = report.get("deployed", {}).get("all", {})
    return f"""---
license: apache-2.0
base_model: sentence-transformers/LaBSE
library_name: sentence-transformers
tags: [sentence-similarity, text-classification, taxonomy]
---

# Coredrill encoder {version}

LaBSE with its top layers fine-tuned so that the 251 territories of the Coredrill taxonomy separate better. Trained with
[embedding-rhizome-classifier](https://github.com/LCapy/embedding-rhizome-classifier) (`docs/training.md`), loop `{meta.get('loop', '?')}`.

Use it with the coredrill of the same version: a coredrill built with another encoder gives wrong answers.

| On the holdout of {report.get('n', 0):,} texts from unseen articles and conversations | |
|---|---:|
| Right territory | {100 * m.get('strict', 0):.1f}% |
| Right, or one taxonomy step away | {100 * m.get('near_miss', 0):.1f}% |
| Training texts | {meta.get('data', {}).get('labeled', 0):,} labeled, {meta.get('data', {}).get('pseudo', 0):,} self-labeled |
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path, help="a folder written by `python -m train export`")
    ap.add_argument("--version", required=True, help="e.g. v6.0")
    ap.add_argument("--run", type=Path, default=None, help="the loop's folder, for the model card (runs/<id>)")
    ap.add_argument("--encoder-repo", default="LCapy/coredrill-encoder")
    ap.add_argument("--data-repo", default="LCapy/coredrill-data")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    encoder, coredrill = args.export / "encoder", args.export / "coredrill_hierarchical.json"
    for p in (encoder / "modules.json", coredrill):
        if not p.exists():
            raise SystemExit(f"{p} is missing: run `python -m train export` first")
    named = json.loads(coredrill.read_text(encoding="utf-8")).get("labse_model")
    if named != args.encoder_repo:
        raise SystemExit(f"the coredrill names {named!r} as its encoder, not {args.encoder_repo!r}: "
                         f"export again with --model-id {args.encoder_repo}")
    meta = json.loads((args.run / "run.json").read_text(encoding="utf-8")) if args.run else {}
    text = card(args.version, meta, meta.get("metrics", {}))
    files = [f for f in encoder.rglob("*") if f.is_file()]
    print(f"{args.encoder_repo} {args.version}: {len(files)} files, {sum(f.stat().st_size for f in files) / 2**30:.2f} GB")
    print(f"{args.data_repo} {args.version}: coredrill_hierarchical.json, {coredrill.stat().st_size / 2**20:.1f} MB")
    if not args.publish:
        print("Nothing was uploaded. Add --publish to do it.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    for repo, kind in ((args.encoder_repo, "model"), (args.data_repo, "dataset")):
        api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
        if args.version in {t.name for t in api.list_repo_refs(repo, repo_type=kind).tags}:
            raise SystemExit(f"{repo} already has {args.version}; a version is never replaced")
    (encoder / "README.md").write_text(text, encoding="utf-8")
    api.upload_folder(repo_id=args.encoder_repo, repo_type="model", folder_path=str(encoder), commit_message=args.version)
    api.create_tag(args.encoder_repo, repo_type="model", tag=args.version)
    api.upload_file(path_or_fileobj=str(coredrill), path_in_repo="coredrill_hierarchical.json", repo_id=args.data_repo,
                    repo_type="dataset", commit_message=args.version)
    api.create_tag(args.data_repo, repo_type="dataset", tag=args.version)
    print(f"published {args.version}. The Space still runs what it ran; see the top of this file to switch it.")


if __name__ == "__main__":
    sys.exit(main())
