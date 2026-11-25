"""Puts this repository's API on the Hugging Face Space.

    python scripts/deploy_space.py --dry-run                          # what the bundle holds
    python scripts/deploy_space.py --space LCapy/coredrill --diff     # what would change there
    python scripts/deploy_space.py --space LCapy/coredrill --deploy   # change it

The Space holds only what it runs: the API, the engine, its Dockerfile and its
README (space/README.md, with the settings Hugging Face reads). Files of those
folders that are gone here are removed there too. The token comes from
HF_DEPLOY_TOKEN, or from `hf auth login`; it is never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parent.parent

# path in the Space: path here
FILES = {
    "README.md": "space/README.md",
    "Dockerfile": "Dockerfile",
    "start.sh": "start.sh",
    "requirements-api.txt": "requirements-api.txt",
    "scripts/rhizome_engine.py": "scripts/rhizome_engine.py",
}
FOLDERS = ["api"]
OWNED = ("api/", "scripts/")        # in these, the Space holds nothing else


def bundle(out: Path) -> Dict[str, Path]:
    """Copies the Space's files to `out`, with Unix line ends."""
    files = dict(FILES)
    for folder in FOLDERS:
        for f in sorted((REPO / folder).rglob("*.py")):
            rel = f.relative_to(REPO).as_posix()
            files[rel] = rel
    made = {}
    for there, here in files.items():
        source = REPO / here
        if not source.exists():
            raise SystemExit(f"{here} is missing")
        target = out / there
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        made[there] = target
    return made


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", help="owner/name")
    ap.add_argument("--message", default="Deploy from GitHub")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--dry-run", action="store_true")
    what.add_argument("--diff", action="store_true")
    what.add_argument("--deploy", action="store_true")
    args = ap.parse_args()

    out = Path(tempfile.mkdtemp(prefix="space-"))
    try:
        made = bundle(out)
        if args.dry_run:
            for there, f in sorted(made.items()):
                print(f"{f.stat().st_size:>9,}  {there}")
            print(f"{len(made)} files")
            return
        if not args.space:
            ap.error("--space is needed")
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_DEPLOY_TOKEN") or None)
        there = {s.rfilename: s.blob_id for s in api.repo_info(args.space, repo_type="space", files_metadata=True).siblings}
        changed = [n for n, f in made.items() if there.get(n) != git_blob(f.read_bytes())]
        gone = [n for n in there if n.startswith(OWNED) and n not in made]
        for n in sorted(changed):
            print(f"  {'change' if n in there else 'add   '}  {n}")
        for n in sorted(gone):
            print(f"  remove  {n}")
        if not changed and not gone:
            print("The Space already runs this.")
            return
        if args.diff:
            print(f"{len(changed)} to change or add, {len(gone)} to remove. Nothing was changed.")
            return
        info = api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(out), commit_message=args.message,
                                 delete_patterns=gone or None)
        print(f"deployed: {info.oid[:7]}; Hugging Face rebuilds the Space now (5 to 10 minutes)")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
