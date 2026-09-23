"""Evaluate a checkpoint on one dataset, driving the PRISTINE TraceGen repo.

Runs the repo's own `test_benchmark.py` under runpy with the loading shims
applied from outside, so the checkout stays byte-identical to upstream.

Metrics (trainer.py:756-838) are trajectory MSE / MAE / endpoint MSE, computed
in normalized space over masked points after cumsum back to absolute positions.

Run in the trace_gen env:
  python -m msgen.run_eval --dataset data/ds/peg_test --ckpt <path> --tag peg_zeroshot
"""
from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import sys

from msgen.paths import GENERALIST, TRACEGEN_DIR, add_tracegen_to_path, patch_torch_load

METRIC_RE = re.compile(r"(Trajectory MSE|Trajectory MAE|Endpoint MSE):\s*([0-9.eE+-]+)")


def patch_split_all_to_val():
    """Route EVERY episode into the val split, which is what trainer.test() reads.

    `val_split=1.0` cannot do this: the split computes train as the remainder,
    so an empty train split makes DataLoader raise `num_samples=0`. The trainer
    builds both loaders in __init__ regardless of which one test() uses, so the
    train split is given a single episode purely to keep it constructible. It is
    never iterated -- test() evaluates the val loader only.
    """
    from dataio.datasets import EpisodePointDataset

    def _split(self, all_episodes):
        return list(all_episodes[:1]), list(all_episodes)

    EpisodePointDataset._split_episodes = _split


class Tee:
    """Mirror stdout/stderr to a log file so metrics can be parsed afterwards."""

    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)
        self.f.flush()
        return len(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()

    def isatty(self):
        # trainer.py:33 probes this to decide whether to draw progress bars.
        return False

    def fileno(self):
        return sys.__stdout__.fileno()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", default=GENERALIST)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/metrics")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_path = f"{args.out}/{args.tag}.log"

    add_tracegen_to_path()
    patch_torch_load()
    patch_split_all_to_val()
    # MEASURED BUG (fixed): this was missing, so `MSGEN_VISPOS=1 run_eval` built a
    # STOCK model and `load_state_dict(strict=False)` dropped the trained `vis_pos_scale`
    # (see `results/metrics/peg_n50_vispos.log`, unexpected_keys). Layer-1 numbers from
    # before this date are void for any patched arm. `apply_all` also installs the guard
    # that turns that silent drop into a RuntimeError.
    from msgen.patch_all import apply_all
    apply_all()

    overrides = [
        f"data.dataset_dirs=['{os.path.abspath(args.dataset)}']",
        # trainer.test() evaluates the VAL split and ignores cfg.test_path, so
        # the whole dataset must be routed into val.
        "data.val_split=1.0",   # with the split patch above, val = every episode
        f"data.num_workers={args.num_workers}",
        f"data.cache_dir='{os.path.abspath('data/cache')}'",
        f"train.batch_size={args.batch_size}",
        # eval.yaml omits this key, and the metric block at trainer.py:961 is
        # gated on it -- without the override the run reports losses only.
        "train.visualize_during_validation=true",
        "hardware.mixed_precision=true",
        # Checkpoint keys are prefixed `_orig_mod.`; only a compiled model's
        # state_dict matches. With compile=false the decoder silently fails to
        # load under strict=False and the metrics are garbage.
        "hardware.compile_model=true",
        "logging.use_wandb=false",
    ]

    sys.argv = [
        f"{TRACEGEN_DIR}/test_benchmark.py",
        "--config", f"{TRACEGEN_DIR}/cfg/eval.yaml",
        "--resume", os.path.abspath(args.ckpt),
        "--no-wandb",
        "--override", *overrides,
    ]

    tee = Tee(log_path)
    sys.stdout = sys.stderr = tee
    try:
        runpy.run_path(f"{TRACEGEN_DIR}/test_benchmark.py", run_name="__main__")
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        tee.flush()

    metrics = {}
    for line in open(log_path):
        for name, val in METRIC_RE.findall(line):
            metrics[name.lower().replace(" ", "_")] = float(val)

    rec = dict(tag=args.tag, dataset=args.dataset, ckpt=os.path.abspath(args.ckpt), **metrics)
    json.dump(rec, open(f"{args.out}/{args.tag}.json", "w"), indent=2)
    print(json.dumps(rec, indent=2))
    if not metrics:
        raise SystemExit(f"no metrics parsed from {log_path}")


if __name__ == "__main__":
    main()
