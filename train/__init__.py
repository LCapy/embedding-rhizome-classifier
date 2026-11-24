"""Training loop for the coredrill.

Fine-tunes the encoder (LaBSE's top layers plus its Dense layer) so the
taxonomy's territories separate better in embedding space, rebuilds the
coredrill with it, and keeps learning from new text it labels itself, keeping
only rounds that improve a fixed holdout. Every loop is recorded in
runs/ledger.jsonl and its own folder, scored the way the existing benchmark
is (record.py, evaluate.py). Run `python -m train --help`; design and usage
notes are in docs/training.md.
"""
