"""Stop writing checkpoints nothing ever loads. 16.5 GB per run -> 6.6 GB.

`msgen/run_train.py:89` already sets `train.save_every=1000` so that only the final model is
kept, but `trainer.py:1101` gates on `epoch % save_every == 0 or is_best`, and **0 % 1000 == 0**
-- so epoch 0 is written every single time. Together with the best-model pair that is three
files per run nobody reads:

    checkpoint_epoch_0.pth    a byte-for-byte copy of the warm start
    checkpoint_epoch_25.pth   written because that epoch happened to be `is_best`
    best_model.pth            selected on a 28-sample validation set

`best_model` is not merely redundant, it is unusable as a selection: the val split is 28
samples, which is why the evaluation protocol here has always scored `checkpoint_epoch_29`
(final) and why mixing "A's best vs B's final" once produced a false "B loses 2 of 3".

At 3.3 GB a file this is the difference between 47 runs costing 643 GB and costing 260 GB --
the reason a prune was needed in the first place (<private-repo>/experiments/20260817_backup/KEEP_MANIFEST.md).

Kept: `checkpoint_epoch_29.pth` and `final_model.pth`, both written by the same is_final call.

    MSGEN_KEEP_ALL_CKPT=1        opt out, write everything as before
    MSGEN_CKPT_ROLL_PROGRESS=N   with the opt-out on: keep only the newest N mid-epoch `*_progress_*` checkpoints
                                 (end-of-epoch and final are never pruned). An 11 h run on a host that
                                 hard-reset twice in one day must not depend on reaching the last epoch.
"""
from __future__ import annotations

import os

_APPLIED = False


def maybe_patch() -> bool:
    global _APPLIED
    if os.environ.get("MSGEN_KEEP_ALL_CKPT", "").strip() == "1":
        keep = int(os.environ.get("MSGEN_CKPT_ROLL_PROGRESS", "0") or 0)
        if keep > 0 and not _APPLIED:
            import glob
            from trainer.trainer import TrajectoryDiffusionTrainer as _T
            _orig = _T.save_checkpoint

            def save_checkpoint(self, epoch, is_best=False, is_final=False, progress_fraction=None):
                r = _orig(self, epoch, is_best=is_best, is_final=is_final, progress_fraction=progress_fraction)
                if progress_fraction is not None:
                    old = sorted(glob.glob(str(self.checkpoint_dir / "*_progress_*.pth")), key=os.path.getmtime)[:-keep]
                    for f in old:
                        try:
                            os.remove(f)
                        except OSError:
                            pass
                    if old:
                        print(f"[patch_ckpt] pruned {len(old)} old mid-epoch checkpoint(s)", flush=True)
                return r

            _T.save_checkpoint = save_checkpoint
            _APPLIED = True
            print(f"[patch_ckpt] keeping every end-of-epoch checkpoint; rolling the mid-epoch ones at {keep}", flush=True)
        return False
    if _APPLIED:
        return True
    from trainer.trainer import TrajectoryDiffusionTrainer

    orig = TrajectoryDiffusionTrainer.save_checkpoint

    def save_checkpoint(self, epoch, is_best=False, is_final=False, progress_fraction=None):
        # epoch 0 is the warm start; `is_best` is chosen on 28 samples and never loaded.
        if not is_final and (epoch == 0 or is_best):
            return
        return orig(self, epoch, is_best=is_best, is_final=is_final,
                    progress_fraction=progress_fraction)

    TrajectoryDiffusionTrainer.save_checkpoint = save_checkpoint
    _APPLIED = True
    print("[patch_ckpt] writing only the final checkpoint (set MSGEN_KEEP_ALL_CKPT=1 to opt out)")
    return True
