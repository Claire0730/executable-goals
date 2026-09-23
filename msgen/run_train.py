"""Warm-up fine-tune the Generalist on one task's 10 demos, driving the
PRISTINE TraceGen repo via runpy.

Run in the trace_gen env:
  python -m msgen.run_train --dataset data/ds/peg_train --tag peg_n10 --epochs 150
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import runpy
import sys

from msgen.paths import GENERALIST, TRACEGEN_DIR, add_tracegen_to_path, patch_torch_load
from msgen.run_eval import Tee


def patch_for_warmup():
    """Two shims train.py needs to warm-start rather than resume.

    1. train.py calls `load_checkpoint` (the DDP variant) unconditionally. On a
       single GPU the checkpoint's `_orig_mod.` keys need
       `load_checkpoint_singlegpu` instead.
    2. Both loaders set `start_epoch = checkpoint['epoch'] + 1`. The Generalist
       is epoch 19, so training would start at 20 and inherit the pretraining
       schedule. We want a fresh 0..N warm-up, so reset it.

    Also routes every episode into train (val gets one episode purely so the
    loader is constructible); held-out evaluation is a separate run on the
    disjoint test set, so the in-run val split carries no weight.
    """
    from dataio.datasets import EpisodePointDataset
    from trainer.trainer import TrajectoryDiffusionTrainer

    def load_warmstart(self, path):
        TrajectoryDiffusionTrainer.load_checkpoint_singlegpu(self, path)
        self.start_epoch = 0
        self.best_metric = float("-inf")

    TrajectoryDiffusionTrainer.load_checkpoint = load_warmstart
    EpisodePointDataset._split_episodes = lambda self, eps: (list(eps), list(eps[:1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default=GENERALIST, help="checkpoint to warm-start from")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-decoder", type=float, default=1.5e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--extra-override", nargs="*", default=[],
                    help="appended to the override list, e.g. num_kps=1600 "
                         "model.decoder.frame_size=40 for a denser query grid. Both "
                         "must move together: num_kps sets the dataset contract and "
                         "frame_size sets the decoder's latent side, and "
                         "cogvideox_flow.py:189 makes the token count "
                         "(frame_size/patch_size)^2 * num_frames -- 3200 at 20, 12800 "
                         "at 40. sample_height/width=frame_size also sizes CogVideoX's "
                         "positional embeddings, so a changed frame_size cannot warm-start "
                         "from the Generalist; pass --scratch with it.")
    ap.add_argument("--scratch", action="store_true",
                    help="train from random init instead of warm-starting from a "
                         "checkpoint -- the 'finetuned from scratch' control in the "
                         "TraceGen paper's human-to-robot transfer figure")
    args = ap.parse_args()

    # The loader enumerates episodes with dataset_dir.iterdir() (TraceGen/dataio/datasets.py:277)
    # and never reads dataset.json, so the two can disagree silently -- data/ds/realcam_n2400 held
    # 2400 clip dirs while its json claimed 800, and the run trained on all 2400. Report the
    # disagreement before a 5-hour training run commits to it. MSGEN_DS_AUDIT=strict aborts;
    # 'off' skips the check entirely; the default warns.
    _audit = os.environ.get("MSGEN_DS_AUDIT", "warn").lower()
    if _audit != "off":
        from msgen.ds_audit import check as _ds_check
        _ds_check(args.dataset, strict=(_audit == "strict"))

    run_dir = os.path.abspath(f"runs/{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = f"{run_dir}/train.log"

    add_tracegen_to_path()
    patch_torch_load()
    patch_for_warmup()
    from msgen.patch_all import apply_all
    apply_all()
    from msgen.patch_ckpt import maybe_patch as _ckpt
    _ckpt()   # training only: drop epoch-0 and best-model writes (3.3 GB each)
    # MSGEN_BEST_CKPT=1 re-admits best_model.pth AND replaces the validation split with a
    # task-balanced one -- without that second half, "best" is chosen by 22 samples from a single
    # LiftPeg clip, which is what the 800-clip run actually validated on.
    from msgen.patch_bestckpt import maybe_patch as _best
    _best()

    overrides = [
        f"data.dataset_dirs=['{os.path.abspath(args.dataset)}']",
        "data.val_split=0.1",
        f"data.num_workers={args.num_workers}",
        f"data.cache_dir='{os.path.abspath('data/cache')}'",
        f"train.epochs={args.epochs}",
        f"train.batch_size={args.batch_size}",
        f"train.lr_decoder={args.lr_decoder}",
        "train.save_every=1000",             # only the final model is wanted
        # eval.yaml omits this key and trainer.py:132 DEFAULTS IT TO 10, which
        # writes ten 3.3 GB intermediate checkpoints per epoch (59 GB in three
        # minutes before this was caught). train.yaml sets it to 0; eval.yaml,
        # which we need for the real 6/12/768 arch, does not.
        "train.num_log_steps_per_epoch=0",
        # every 5 epochs: often enough that best_model tracks the run, rare enough that
        # validation is not a meaningful share of the wall clock
        "train.eval_every=5",
        "train.visualize_every=1000",
        "train.visualize_during_validation=false",   # sampling every val is slow
        "hardware.mixed_precision=true",
        "hardware.compile_model=true",       # required for `_orig_mod.` key match
        "logging.use_wandb=false",
        f"logging.checkpoint_dir='{run_dir}'",
        "logging.save_dir='ckpt'",
    ] + list(args.extra_override)

    sys.argv = [
        f"{TRACEGEN_DIR}/train.py",
        "--config", f"{TRACEGEN_DIR}/cfg/eval.yaml",   # 6/12/768 = the real arch
        "--no-wandb",
        "--override", *overrides,
    ]
    if not args.scratch:
        # train.py computes action statistics itself when --resume is absent,
        # which is exactly what a from-scratch control needs.
        sys.argv[4:4] = ["--resume", os.path.abspath(args.ckpt)]

    tee = Tee(log_path)
    sys.stdout = sys.stderr = tee
    try:
        runpy.run_path(f"{TRACEGEN_DIR}/train.py", run_name="__main__")
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        tee.flush()

    finals = sorted(glob.glob(f"{run_dir}/ckpt/*/final_model.pth"))
    if not finals:
        raise SystemExit(f"no final_model.pth under {run_dir}/ckpt")
    rec = dict(tag=args.tag, dataset=args.dataset, epochs=args.epochs,
               batch_size=args.batch_size, lr_decoder=args.lr_decoder,
               warm_start=None if args.scratch else os.path.abspath(args.ckpt),
               from_scratch=bool(args.scratch), final_ckpt=finals[-1])
    json.dump(rec, open(f"{run_dir}/run.json", "w"), indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
