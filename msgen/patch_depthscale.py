"""Output-side depth fix (SPEC_DEPTH_CHANNEL, option 2): rebalance the depth channel
of the learning target. Enable with MSGEN_DEPTH_SCALE=<s>.

THE IMBALANCE, measured on data/ds/pickcube_wallsel200 (moving points, per-step deltas):
    pixel channels (normalised by 384)   0.00987
    depth channel  (metres)              0.00231          ratio 4.27
The flow-matching target is the per-step delta of (x/384, y/384, z_m), and the loss is a
plain MSE over the three channels (losses/trajectory_loss.py:71). A channel whose deltas are
4.3x smaller contributes ~1/18 of the squared error, so the depth channel is under-supervised
by construction. This matches the error budget: the depth axis carries 36.5 of PickCube's
40.5 mm and neither the solver (wdisp) nor the K=4 ensemble moves it (k12_axes.txt).

THE FIX. Multiply the target's depth channel by s in the dataset (training AND inference, so
the model always sees the same units) and divide the model's depth output by s where the
prediction is read back (msgen/predict.py, after to_absolute). s=4 equalises the per-step
magnitudes; the flag is a float so an arm can sweep it. Nothing in TraceGen/ is edited.

Single-variable retrain only: a checkpoint trained without the flag must not be inferred
with it (its depth output would be divided by s). `install_load_guard` cannot see this, so
the run tag carries `_dz<s>` and the predict driver sets the same flag.

    MSGEN_DEPTH_SCALE=4 $PG -m msgen.run_train ...
    MSGEN_DEPTH_SCALE=4 $PG -m msgen.predict ...
"""
from __future__ import annotations

import os

_APPLIED = False


def scale() -> float:
    v = os.environ.get("MSGEN_DEPTH_SCALE", "")
    return float(v) if v not in ("", "0", "1", "1.0") else 1.0


def maybe_patch() -> bool:
    global _APPLIED
    s = scale()
    if _APPLIED or s == 1.0:
        return _APPLIED
    from dataio.datasets import EpisodePointDataset

    orig = EpisodePointDataset.__getitem__

    def __getitem__(self, idx):
        out = orig(self, idx)
        t = out.get("trajectory")
        if t is not None and t.shape[-1] == 3:
            t = t.clone()
            t[..., 2] = t[..., 2] * s          # frame-0 absolute z AND every per-step delta
            out["trajectory"] = t
        return out

    EpisodePointDataset.__getitem__ = __getitem__
    _APPLIED = True
    print(f"[patch_depthscale] active: trajectory depth channel x{s} in the target "
          f"(predict.py divides it back)")
    return True
