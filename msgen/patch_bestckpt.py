"""Make `best_model.pth` mean something, and let it be written. Enable with MSGEN_BEST_CKPT=1.

WHY THIS EXISTS. `msgen/patch_ckpt.py` drops best-model writes, and its reasoning was correct at
the time: selecting on the validation set this pipeline builds is worse than not selecting at all.
Measured on an 800-clip run:

    - Train: 16278 samples, 2034 batches
    - Val:      22 samples,    3 batches

Twenty-two samples, from ONE episode, and that episode is `liftpeg_clip_000` -- so "best model"
would have been chosen by a single LiftPeg clip's diffusion loss while three other tasks went
unrepresented. `data.val_split=0.1` over 800 episodes should hold out 80; it held out 1.

Rather than reverse-engineer the upstream split (TraceGen/ is a read-only reference and is not to
be edited), this patch replaces `_split_episodes` with one that is correct by construction:

  * the hold-out is TASK-BALANCED. Episodes are grouped by the task prefix the mix farm writes
    (`<task>_clip_NNN`), and the same fraction is taken from each group, so every task is
    represented in proportion. A validation loss that is really one task's loss cannot select a
    model for four.
  * a FLOOR of `MSGEN_VAL_MIN` episodes per task (default 5) applies, because a fraction of a
    small group rounds to zero and silently drops that task from validation.
  * the split stays deterministic under the caller's `random_seed`, so a rerun of the same config
    validates on the same episodes.

WHAT IT DOES NOT CHANGE. The epoch-0 checkpoint stays dropped -- it is a byte-for-byte copy of the
warm start and costs 3.3 GB. Only `best_model.pth` is let through, and it is overwritten in place,
so the disk cost of keeping a best model is one file per run, not one per improvement.

STILL TRUE, AND WORTH REPEATING WHEN READING RESULTS: an arm's best and another arm's final are
not comparable. Mixing them once produced a false "B loses 2 of 3". The evaluation protocol scores
`checkpoint_epoch_29` / `final_model`; `best_model` is for recovering a run that was interrupted
or that diverged late, not for cross-arm tables.

    MSGEN_BEST_CKPT=1     write best_model.pth and use the task-balanced split
    MSGEN_VAL_MIN=5       minimum validation episodes per task
    MSGEN_VAL_FRAC        override the fraction per task (default: the cfg's val_split)
"""
from __future__ import annotations

import os

_APPLIED = False


def maybe_patch() -> bool:
    global _APPLIED
    if os.environ.get("MSGEN_BEST_CKPT", "0").strip() != "1":
        return False
    if _APPLIED:
        return True

    import logging
    import random
    from dataio.datasets import EpisodePointDataset
    from trainer.trainer import TrajectoryDiffusionTrainer

    log = logging.getLogger(__name__)
    vmin = int(os.environ.get("MSGEN_VAL_MIN", "5"))
    vfrac = os.environ.get("MSGEN_VAL_FRAC", "").strip()

    orig_split = EpisodePointDataset._split_episodes

    def _split_episodes(self, all_episodes):
        frac = float(vfrac) if vfrac else float(getattr(self, "val_split", 0.1) or 0.1)
        groups = {}
        for p in all_episodes:
            # the mix farm names clips `<task>_clip_NNN`; anything else groups under ""
            name = getattr(p, "name", str(p))
            groups.setdefault(name.rsplit("_clip_", 1)[0] if "_clip_" in name else "", []).append(p)
        rng = random.Random(getattr(self, "random_seed", 0))
        train, val = [], []
        for task in sorted(groups):
            eps = sorted(groups[task], key=lambda q: str(q))
            rng.shuffle(eps)
            k = min(len(eps) - 1, max(vmin, int(round(len(eps) * frac)))) if len(eps) > 1 else 0
            val += eps[:k]
            train += eps[k:]
        log.info("[bestckpt] task-balanced split: %d train / %d val over %d task groups (%s)",
                 len(train), len(val), len(groups),
                 ", ".join(f"{t or '-'}:{sum(1 for e in val if str(e).rsplit('_clip_', 1)[0].endswith(t))}"
                           for t in sorted(groups)))
        if not val:
            log.warning("[bestckpt] empty validation split; falling back to upstream")
            return orig_split(self, all_episodes)
        return train, val

    EpisodePointDataset._split_episodes = _split_episodes

    # patch_ckpt drops best-model writes; re-admit them while keeping epoch-0 dropped
    orig_save = TrajectoryDiffusionTrainer.save_checkpoint

    def save_checkpoint(self, epoch, is_best=False, is_final=False, progress_fraction=None):
        if not (is_best or is_final):
            return None
        return orig_save(self, epoch, is_best=is_best, is_final=is_final,
                         progress_fraction=progress_fraction)

    TrajectoryDiffusionTrainer.save_checkpoint = save_checkpoint
    _APPLIED = True
    print(f"[bestckpt] task-balanced validation (min {vmin} ep/task) + best_model writes enabled",
          flush=True)
    return True
