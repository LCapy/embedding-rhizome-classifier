"""python -m train <command>: train the coredrill's encoder and keep it learning.

  baseline   stock LaBSE on the fixed holdout (run once, first)
  score      an earlier model on this corpus and holdout, to compare with a new one
  builds     better ways of building the coredrill for an encoder; the best becomes the default
  fit        train a model on the labeled data plus the pseudo-label pool
  round      one self-training round: harvest, pseudo-label, train, gate
  loop       rounds until they stop helping
  status     baseline, champion, pool and the latest loops
  report     every recorded loop next to the reference benchmark (runs/report.md, .csv)
  watch      follow the running loop live: its log and the GPU (Ctrl+C stops watching)
  view       see the matrices being trained, and use the classifier, in the browser
  classify   the rhizome profile of a text, with the champion's model
  forever    self-training rounds with no end, until `stop`
  stop       end the endless run after the loop it is on
  testset    gather paragraphs to hand-check for the labels the holdout cannot test
  export     the champion (or any run) as an encoder + coredrill bundle

Every baseline, fit and round is recorded in runs/ledger.jsonl and its own
folder under runs/, whatever its outcome. Run from the repository root.
Details: docs/training.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if not argv or argv[0] not in ("watch", "view", "classify", "builds", "math", "forever", "stop"):
        from .cli import main as full
        return full(argv)

    # These run next to a training run, so they must not load PyTorch: it
    # commits over a gigabyte of memory per process.
    ap = argparse.ArgumentParser(prog=f"python -m train {argv[0]}")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    if argv[0] == "math":
        ap.add_argument("--run", required=True, help="a loop id that saved its embeddings")
    if argv[0] in ("view", "classify", "builds", "math"):
        ap.add_argument("--data", type=Path, default=Path("data"))
    if argv[0] == "builds":
        ap.add_argument("--run", default="stock", help="a loop id whose encoder to use, or stock (default)")
        ap.add_argument("--quick", action="store_true", help="fewer ways")
        ap.add_argument("--taxonomy", type=Path, default=None)
    if argv[0] in ("forever", "stop"):
        pass
    elif argv[0] == "classify":
        ap.add_argument("text", nargs="?", help="the text (in quotes), or - to read it from standard input")
        ap.add_argument("--run", default=None, help="a loop id (folder name) instead of the champion")
        ap.add_argument("--json", action="store_true", help="print everything, as JSON")
        ap.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    elif argv[0] == "watch":
        ap.add_argument("--loop", default=None, help="a loop id (folder name) instead of the running one")
        ap.add_argument("--once", action="store_true", help="stop when the loop ends")
        ap.add_argument("--gpu-every", type=float, default=15.0, help="seconds between GPU lines (0: none)")
    elif argv[0] == "view":
        ap.add_argument("--port", type=int, default=8765)
        ap.add_argument("--no-browser", action="store_true", help="do not open the page")
    args, rest = ap.parse_known_args(argv[1:]) if argv[0] == "forever" else (ap.parse_args(argv[1:]), [])
    paths = SimpleNamespace(runs=args.runs.resolve())

    def log(message: str) -> None:
        print(message, flush=True)

    if argv[0] == "forever":
        from .forever import cmd_forever
        cmd_forever(paths.runs, log, rest)
    elif argv[0] == "stop":
        from .forever import cmd_stop
        cmd_stop(paths.runs, log)
    elif argv[0] == "watch":
        from .record import cmd_watch
        cmd_watch(paths, args.loop, args.once, args.gpu_every, log)
    elif argv[0] == "math":
        from .mathsteps import measure
        measure(paths.runs, args.run, args.data.resolve(), log)
    elif argv[0] == "builds":
        from .builds import Paths, cmd_builds
        cmd_builds(Paths(args.data.resolve(), paths.runs, args.taxonomy), None if args.run == "stock" else args.run,
                   args.quick, log)
    elif argv[0] == "classify":
        from .classify import cmd_classify, serve
        if args.serve:
            for stream in (sys.stdin, sys.stdout):
                stream.reconfigure(encoding="utf-8")
            return serve(paths.runs, args.data.resolve())
        text = sys.stdin.read() if args.text == "-" else args.text
        if not text:
            ap.error("give a text in quotes, or - to read it from standard input")
        cmd_classify(paths.runs, args.data.resolve(), text, args.run, args.json, log)
    else:
        from .viewer import cmd_view
        paths.data = args.data.resolve()
        cmd_view(paths, args.port, not args.no_browser, log)


if __name__ == "__main__":
    main()
