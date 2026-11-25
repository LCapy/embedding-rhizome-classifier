"""python -m train <command>: train the coredrill's encoder and keep it learning.

  baseline   stock LaBSE on the fixed holdout (run once, first)
  fit        train a model on the labeled data plus the pseudo-label pool
  round      one self-training round: harvest, pseudo-label, train, gate
  loop       rounds until they stop helping
  status     baseline, champion, pool and the latest loops
  report     every recorded loop next to the reference benchmark (runs/report.md, .csv)
  watch      follow the running loop live: its log and the GPU (Ctrl+C stops watching)
  view       see the matrices being multiplied and trained, in the browser
  testset    gather paragraphs to hand-check for the labels the holdout cannot test
  export     the champion (or any run) as an encoder + coredrill bundle

Every baseline, fit and round is recorded in runs/ledger.jsonl and its own
folder under runs/, whatever its outcome. Run from the repository root.
Details: docs/training.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

from .fit import FitConfig
from .model import pick_device
from .record import REFERENCE, cmd_report
from .selftrain import (Gate, Paths, RoundConfig, cmd_baseline, cmd_export, cmd_fit, cmd_loop, cmd_round, cmd_score,
                        cmd_status, cmd_testset, read_champion)

FIT_HELP = {
    "unfreeze_layers": "top transformer layers to train; 0 trains only the Dense layer and the "
                       "prototypes (default: 4 with a GPU, else 0)",
    "epochs": "epochs (default: 40 with --unfreeze-layers 0, else 4)",
    "batch_size": "batch size (at least 256 with --unfreeze-layers 0)",
    "max_seq_length": "tokens per text",
    "lr_encoder": "learning rate of the unfrozen transformer layers",
    "lr_dense": "learning rate of the Dense layer",
    "lr_protos": "learning rate of the label prototypes",
    "anchor_weight": "pull toward stock LaBSE embeddings (keeps the space multilingual)",
    "ancestor_eps": "target mass a label shares with its ancestors",
    "pseudo_weight": "loss weight of a pseudo-label, times its confidence",
    "balance": "1: every label weighs the same in all, whatever its size; 0: by its rows",
    "node_cap": "the coredrill uses at most this many rows per node, evenly over its articles (0: no cap)",
    "radius_pull": "the coredrill: 0 to 1, how far each node's radius moves toward the median one",
    "centers_per": "the coredrill: one center per this many rows of a node (0: one center)",
    "centers_max": "the coredrill: centers per node, at most",
    "patience": "epochs without validation gain before stopping",
    "seed": "random seed",
}
ROUND_HELP = {
    "sources": "comma-separated: wiki, inbox",
    "labels": "labels that may gain rows: tested (by the holdout) or all",
    "nodes_per_round": "Wikipedia: nodes searched per round, fewest articles first",
    "per_node": "Wikipedia: new articles per node",
    "paras_per_article": "lead paragraphs kept per article",
    "langs": "other-language versions of accepted articles, e.g. es,fr,de,pt ('' for none)",
    "conf": "calibrated probability a pseudo-label needs",
    "margin": "lead over the runner-up a pseudo-label needs",
    "per_class_frac": "new pseudo-labels per label per round, as a share of its labeled rows",
    "min_per_class": "per-label cap floor",
    "max_new": "pseudo-labels per round, at most",
    "total_frac": "self-labeled rows a label may have in all, as a share of its labeled rows",
    "unproven": "what a round that is not promoted does with its rows: stack (with the next round's) or drop",
    "wiki_delay": "seconds between Wikipedia requests",
}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _add_fit_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("training")
    g.add_argument("--no-live", action="store_true", help="do not capture matrices for `view`")
    for f in fields(FitConfig):
        if f.name in FIT_HELP:
            g.add_argument(_flag(f.name), dest=f.name, type=type(f.default) if f.default is not None else int,
                           default=None, help=FIT_HELP[f.name])


def _add_round_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("self-training round")
    for f in fields(RoundConfig):
        g.add_argument(_flag(f.name), dest=f.name, type=type(f.default), default=None,
                       help=f"{ROUND_HELP[f.name]} (default: {f.default!r})")


def _add_gate_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("promotion")
    g.add_argument("--select-metric", default="deployed.strict",
                   choices=["deployed.strict", "head.strict", "centroid.strict", "logreg.strict",
                            "linear_head.strict"],
                   help="holdout metric a new model must improve to replace the champion")
    g.add_argument("--min-delta", type=float, default=0.002, help="required improvement")
    g.add_argument("--min-z", type=float, default=1.645,
                   help="required McNemar z of the gain over the champion, text by text (1.645: one-sided 95%%)")


def _fit_config(args, base: dict | None, device, runs: Path | None = None) -> FitConfig:
    cfg = FitConfig(**base) if base else FitConfig()
    if runs is not None:                         # the adopted way of building the coredrill (python -m train builds)
        from .builds import default_build
        for k, v in default_build(runs).items():
            setattr(cfg, k, v)
    for f in fields(FitConfig):
        value = getattr(args, f.name, None)
        if value is not None:
            setattr(cfg, f.name, value)
    if base is None and args.unfreeze_layers is None:
        cfg.unfreeze_layers = 4 if device.type == "cuda" else 0
    return cfg


def _round_config(args) -> RoundConfig:
    cfg = RoundConfig()
    for f in fields(RoundConfig):
        value = getattr(args, f.name, None)
        if value is not None:
            setattr(cfg, f.name, value)
    return cfg


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", type=Path, default=Path("data"),
                        help="sentences.json, embeddings.npz, TAXONOMY_fixed.txt (default: data)")
    common.add_argument("--runs", type=Path, default=Path("runs"), help="runs, holdout, champion (default: runs)")
    common.add_argument("--taxonomy", type=Path, default=None,
                        help="default: <data>/TAXONOMY_fixed.txt, else api/taxonomy.py")
    common.add_argument("--device", default="auto", help="auto, cuda or cpu")

    ap = argparse.ArgumentParser(prog="python -m train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("baseline", parents=[common], help="stock LaBSE on the holdout")
    p.add_argument("--node-cap", type=int, default=0)
    p = sub.add_parser("score", parents=[common], help="an earlier model on this corpus and holdout")
    p.add_argument("--run", type=Path, required=True, help="its folder")
    p.add_argument("--node-cap", type=int, default=0)
    p = sub.add_parser("fit", parents=[common], help="train on the labeled data (+ pool)")
    _add_fit_flags(p)
    _add_gate_flags(p)
    p.add_argument("--no-pool", action="store_true", help="leave the pseudo-label pool out")
    for name, helptext in (("round", "one self-training round"), ("loop", "rounds until they stop helping")):
        p = sub.add_parser(name, parents=[common], help=helptext,
                           description="Training settings default to the champion's own.")
        _add_fit_flags(p)
        _add_round_flags(p)
        _add_gate_flags(p)
        if name == "loop":
            p.add_argument("--rounds", type=int, default=5)
            p.add_argument("--stop-after", type=int, default=2,
                           help="stop after this many rounds in a row without a promotion")
    sub.add_parser("status", parents=[common], help="baseline, champion, pool, latest loops")
    p = sub.add_parser("report", parents=[common], help="every loop next to the reference benchmark")
    p.add_argument("--reference", type=Path, default=REFERENCE,
                   help="a by_article.json from eval/held_out/12_by_article.py (default: the committed one)")
    p.add_argument("--out", type=Path, default=None, help="markdown report; a .csv is written next to it "
                                                          "(default: <runs>/report.md)")
    p = sub.add_parser("view", parents=[common], help="the matrices being trained, in the browser")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open the page")
    p = sub.add_parser("watch", parents=[common], help="follow the running loop live: log and GPU")
    p.add_argument("--loop", default=None, help="a loop id (folder name) to follow instead of the running one")
    p.add_argument("--once", action="store_true", help="stop when the loop ends instead of waiting for the next")
    p.add_argument("--gpu-every", type=float, default=15.0, help="seconds between GPU lines (0: none)")
    p = sub.add_parser("testset", parents=[common], help="gather paragraphs to hand-check for untested labels")
    p.add_argument("--per-label", type=int, default=4)
    p = sub.add_parser("export", parents=[common], help="encoder + coredrill bundle for the API")
    p.add_argument("--run", default="champion", help="run name, or champion")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-id", default=None,
                   help="where the encoder will be served from, e.g. an HF model repo (default: the local path)")
    args = ap.parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    paths = Paths(args.data.resolve(), args.runs.resolve(), args.taxonomy,
                  live=not getattr(args, "no_live", False))
    device = pick_device(args.device)
    if args.cmd not in ("status", "report", "watch", "testset", "view"):
        import torch
        log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if args.cmd == "baseline":
        cmd_baseline(paths, device, args.node_cap)
    elif args.cmd == "score":
        cmd_score(paths, args.run, device, args.node_cap)
    elif args.cmd == "fit":
        cmd_fit(paths, _fit_config(args, None, device, paths.runs),
                Gate(args.select_metric, args.min_delta, min_z=args.min_z), device,
                use_pool=not args.no_pool)
    elif args.cmd in ("round", "loop"):
        champ = read_champion(paths)
        base = json.loads((paths.runs / champ["run"] / "run.json").read_text(encoding="utf-8"))["fit"] if champ else None
        cfg, rcfg = _fit_config(args, base, device, paths.runs), _round_config(args)
        gate = Gate(args.select_metric, args.min_delta, min_z=args.min_z)
        if args.cmd == "round":
            cmd_round(paths, rcfg, cfg, gate, device)
        else:
            cmd_loop(paths, rcfg, cfg, gate, device, args.rounds, args.stop_after, log)
    elif args.cmd == "status":
        cmd_status(paths, log)
    elif args.cmd == "report":
        cmd_report(paths, args.reference, args.out, log)
    elif args.cmd == "testset":
        cmd_testset(paths, args.per_label, log)
    elif args.cmd == "export":
        cmd_export(paths, args.run, args.out, args.model_id, device, log)

