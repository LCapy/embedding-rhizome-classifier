#!/usr/bin/env python3
"""
collect_medical.py - Targeted medical data collection for coredrill.

Fetches clinical text from HuggingFace datasets and injects it
directly into the existing sentences.json, then triggers a rebuild
of affected nodes in the coredrill JSON.

Target nodes:
  - Medicine (L2)          - needs clinical text, not encyclopedic
  - Treatment (L3)         - only 16 records
  - Diagnosis (L3)         - only 30 records
  - Doctor Appointment (L4)- NO CENTROID
  - Health Routine (L2)    - 54 records, needs more
  - Symptoms & Self-Care (L3) - 54 records
  - Appointments & Medication (L3) - 66 records

Usage:
  python collect_medical.py \
    --sentences data/sentences.json \
    --embeddings data/embeddings.npz \
    --coredrill coredrill/coredrill_hierarchical.json \
    --max-per-node 500
"""

from __future__ import annotations
import argparse, json, logging, sys, warnings
from pathlib import Path
from typing import List, Dict

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("collect_medical")

# ── Target nodes and their keywords for routing ──────────────────────────────
MEDICAL_NODES = {
    "Doctor Appointment": {
        "level": 4,
        "keywords": ["appointment", "schedule", "doctor", "clinic", "visit",
                     "see a doctor", "book", "cancel appointment", "reschedule"],
        "description": "Scheduling and conducting medical appointments"
    },
    "Diagnosis": {
        "level": 3,
        "keywords": ["diagnos", "presents with", "clinical picture", "consistent with",
                     "patient has", "examination reveals", "assessment", "impression"],
        "description": "Clinical diagnosis of medical conditions"
    },
    "Treatment": {
        "level": 3,
        "keywords": ["prescri", "treatment", "therapy", "medication", "dose",
                     "mg", "administer", "recommend", "course of"],
        "description": "Medical treatment plans and prescriptions"
    },
    "Medicine": {
        "level": 2,
        "keywords": ["patient", "clinical", "medical", "symptom", "disease",
                     "condition", "health", "physician", "hospital", "nurse"],
        "description": "Clinical medicine - doctor-patient interactions, clinical notes"
    },
    "Symptoms & Self-Care": {
        "level": 3,
        "keywords": ["symptom", "pain", "fever", "cough", "headache", "nausea",
                     "feel", "hurts", "ache", "sore", "tired", "fatigue"],
        "description": "Patient-reported symptoms and self-care"
    },
    "Appointments & Medication": {
        "level": 3,
        "keywords": ["take", "pill", "tablet", "twice daily", "once a day",
                     "follow up", "follow-up", "return", "refill", "pharmacy"],
        "description": "Medication instructions and follow-up appointments"
    },
    "Health Routine": {
        "level": 2,
        "keywords": ["routine", "checkup", "check-up", "annual", "blood pressure",
                     "weight", "exercise", "diet", "healthy", "wellness"],
        "description": "Health maintenance and routine care"
    },
}

def route_text(text: str) -> str:
    """Route a medical text to the most specific matching node."""
    text_l = text.lower()
    # Specificity order: most specific first
    order = ["Doctor Appointment", "Diagnosis", "Treatment",
             "Appointments & Medication", "Symptoms & Self-Care",
             "Health Routine", "Medicine"]
    for node in order:
        kws = MEDICAL_NODES[node]["keywords"]
        if any(kw in text_l for kw in kws):
            return node
    return "Medicine"  # fallback


def load_hf_medical(max_per_node: int = 500) -> List[dict]:
    """Load medical text from HuggingFace datasets."""
    records = []
    counts: Dict[str, int] = {n: 0 for n in MEDICAL_NODES}

    def add(text: str, source: str):
        text = text.strip()
        if len(text) < 30 or len(text) > 1500:
            return
        node = route_text(text)
        if counts[node] >= max_per_node:
            return
        records.append({
            "text": text,
            "category": node,
            "level": MEDICAL_NODES[node]["level"],
            "parent": None,
            "lang": "en",
            "source": f"medical_{source}",
        })
        counts[node] += 1

    # ── Dataset 1: ChatDoctor-HealthCareMagic ────────────────────────────
    try:
        log.info("Loading lavita/ChatDoctor-HealthCareMagic-100k ...")
        from datasets import load_dataset
        ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k",
                          split="train", trust_remote_code=True)
        for row in ds:
            if all(counts[n] >= max_per_node for n in MEDICAL_NODES):
                break
            # input = patient question, output = doctor response
            if row.get("input"):
                add(row["input"], "chatdoctor_patient")
            if row.get("output"):
                add(row["output"], "chatdoctor_doctor")
        log.info(f"  ChatDoctor: {sum(1 for r in records if 'chatdoctor' in r['source'])} records")
    except Exception as e:
        log.warning(f"  ChatDoctor failed: {e}")

    # ── Dataset 2: medical_meadow_mediqa ────────────────────────────────
    try:
        log.info("Loading medalpaca/medical_meadow_mediqa ...")
        from datasets import load_dataset
        ds = load_dataset("medalpaca/medical_meadow_mediqa",
                          split="train", trust_remote_code=True)
        n_before = len(records)
        for row in ds:
            if all(counts[n] >= max_per_node for n in MEDICAL_NODES):
                break
            if row.get("input"):
                add(row["input"], "mediqa_q")
            if row.get("output"):
                add(row["output"], "mediqa_a")
        log.info(f"  MEDIQA: {len(records)-n_before} records")
    except Exception as e:
        log.warning(f"  MEDIQA failed: {e}")

    # ── Dataset 3: medical_meadow_health_advice ──────────────────────────
    try:
        log.info("Loading medalpaca/medical_meadow_health_advice ...")
        from datasets import load_dataset
        ds = load_dataset("medalpaca/medical_meadow_health_advice",
                          split="train", trust_remote_code=True)
        n_before = len(records)
        for row in ds:
            if all(counts[n] >= max_per_node for n in MEDICAL_NODES):
                break
            if row.get("input"):
                add(row["input"], "health_advice_q")
            if row.get("output"):
                add(row["output"], "health_advice_a")
        log.info(f"  Health advice: {len(records)-n_before} records")
    except Exception as e:
        log.warning(f"  Health advice failed: {e}")

    # ── Dataset 4: medical_questions_pairs ──────────────────────────────
    try:
        log.info("Loading medical_questions_pairs ...")
        from datasets import load_dataset
        ds = load_dataset("medical_questions_pairs",
                          split="train", trust_remote_code=True)
        n_before = len(records)
        for row in ds:
            if all(counts[n] >= max_per_node for n in MEDICAL_NODES):
                break
            for field in ["question_1", "question_2"]:
                if row.get(field):
                    add(row[field], "med_questions")
        log.info(f"  Medical QA pairs: {len(records)-n_before} records")
    except Exception as e:
        log.warning(f"  Medical QA pairs failed: {e}")

    log.info(f"\nCollection complete: {len(records)} total records")
    log.info("Per node:")
    for node, cnt in counts.items():
        log.info(f"  {node:<35} {cnt:>4}")

    return records


def embed_records(records: List[dict], model) -> np.ndarray:
    """Embed a list of records using LaBSE."""
    texts = [r["text"] for r in records]
    log.info(f"Embedding {len(texts)} texts ...")
    batch_size = 64
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, normalize_embeddings=True,
                             convert_to_numpy=True).astype(np.float32)
        all_embs.append(embs)
        if (i // batch_size) % 10 == 0:
            log.info(f"  {i+len(batch)}/{len(texts)}")
    return np.vstack(all_embs)


def update_coredrill(coredrill_path: str, new_records: List[dict],
                     new_embs: np.ndarray) -> None:
    """Incrementally update centroid and sigma for affected nodes."""
    import shutil, time

    p = Path(coredrill_path)
    # Backup
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = p.parent / f"coredrill_backup_medical_{ts}.json"
    shutil.copy2(p, bak)
    log.info(f"Backup: {bak}")

    cd = json.loads(p.read_text(encoding="utf-8"))
    tree = cd["tree"]

    # Group new records by node
    from collections import defaultdict
    by_node: Dict[str, List[np.ndarray]] = defaultdict(list)
    for rec, emb in zip(new_records, new_embs):
        node = rec["category"]
        by_node[node].append(emb)

    for node, embs in by_node.items():
        if node not in tree:
            log.warning(f"  Node '{node}' not in tree - skipping")
            continue

        stats = tree[node]
        old_centroid = stats.get("centroid")
        old_n = int(stats.get("n", 0) or 0)
        old_sigma = float(stats.get("sigma") or 0.0)

        new_vecs = np.array(embs, dtype=np.float64)

        if old_centroid is None or old_n == 0:
            # First data for this node
            new_centroid = new_vecs.mean(axis=0)
            new_centroid /= np.linalg.norm(new_centroid) + 1e-12
            # Sigma = std of cosine distances from centroid
            cosines = new_vecs @ new_centroid
            new_sigma = float(np.std(1 - cosines)) if len(cosines) > 1 else 0.0
            new_n = len(embs)
        else:
            # Combine old centroid (weighted by n) with new embeddings
            old_c = np.array(old_centroid, dtype=np.float64)
            old_c /= np.linalg.norm(old_c) + 1e-12

            # Weighted mean: old_n * old_c + sum(new_vecs)
            combined_sum = old_c * old_n + new_vecs.sum(axis=0)
            new_n = old_n + len(embs)
            new_centroid = combined_sum / new_n
            new_centroid /= np.linalg.norm(new_centroid) + 1e-12

            # Approximate sigma: weighted std
            all_vecs = np.vstack([old_c.reshape(1,-1)] * old_n + [new_vecs])
            cosines = all_vecs @ new_centroid
            new_sigma = float(np.std(1 - cosines))

        # Update tree
        stats["centroid"] = new_centroid.tolist()
        stats["n"] = new_n
        stats["sigma"] = round(new_sigma, 6)

        # Update source counts
        sc = stats.get("source_counts") or {}
        for rec in new_records:
            if rec["category"] == node:
                src = rec["source"]
                sc[src] = sc.get(src, 0) + 1
        stats["source_counts"] = sc

        log.info(f"  {node}: n {old_n} -> {new_n}  sigma {old_sigma:.4f} -> {new_sigma:.4f}")

    # Write back
    p.write_text(json.dumps(cd, ensure_ascii=False), encoding="utf-8")
    log.info(f"Coredrill updated: {p}")


def main():
    ap = argparse.ArgumentParser(description="Targeted medical data collection")
    ap.add_argument("--sentences",   required=True, help="path to sentences.json")
    ap.add_argument("--coredrill",   required=True, help="path to coredrill JSON")
    ap.add_argument("--max-per-node",type=int, default=500)
    ap.add_argument("--dry-run",     action="store_true",
                    help="collect and show stats without writing")
    args = ap.parse_args()

    # Collect
    records = load_hf_medical(max_per_node=args.max_per_node)
    if not records:
        log.error("No records collected. Check dataset availability.")
        sys.exit(1)

    if args.dry_run:
        log.info("DRY RUN - not writing anything")
        from collections import Counter
        c = Counter(r["category"] for r in records)
        for node, cnt in sorted(c.items(), key=lambda x: -x[1]):
            log.info(f"  {node:<35} {cnt:>4}")
        return

    # Embed
    log.info("Loading LaBSE ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE")
    embs = embed_records(records, model)

    # Append to sentences.json
    sentences_path = Path(args.sentences)
    existing = json.loads(sentences_path.read_text(encoding="utf-8"))
    combined = existing + records
    sentences_path.write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")
    log.info(f"sentences.json: {len(existing)} -> {len(combined)} records")

    # Update coredrill centroids
    update_coredrill(args.coredrill, records, embs)

    log.info("\nDone. Run rebuild if you want to recompute the full coredrill:")
    log.info("  python scripts\\full_build_rhizome.py build \\")
    log.info("    --taxonomy-txt taxonomy\\TAXONOMY_fixed.txt \\")
    log.info("    --data data\\ --out coredrill\\")


if __name__ == "__main__":
    main()
