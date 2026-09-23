"""Input-side depth fix (SPEC_DEPTH_CHANNEL, option 1): let the depth tower learn.
Enable with MSGEN_DEPTH_UNFREEZE=<N> (last N transformer blocks of the DEPTH encoder only).

WHY. Depth reaches the model as raw metres through a 6-parameter Conv2d(1,3,1) stem into a
SigLIP tower that is FROZEN and was pretrained on RGB in [-1, 1] (patch_depthnorm docstring;
siglip_vision.py:44). Re-normalising the input alone (patch_depthnorm, log1p+affine) kept the
per-point MSE and collapsed the solved |dt| to 0.30-0.44 of truth: the decoder had adapted to
the frozen tower's raw-metre features, and changing the input without letting the tower
adapt broke the scale it was carrying. This patch keeps the input as-is and instead lets the
last N blocks of the depth tower train, so the depth features can become informative for
this data without a distribution jump at the input.

SCOPE. `TrajectoryFlow.unfreeze_backbones` would unfreeze ALL four encoders; here only
`depth_encoder.unfreeze_last_blocks(N)` runs, at the end of `TrajectoryFlow.__init__`, i.e.
before the trainer builds its optimizer (trainer.py:230) so the new parameters are picked up.
MSGEN_LRGROUPS=1 MUST be on with it: without the routing fix the unfrozen blocks would train
at lr_decoder (1.5e-4) instead of lr_backbone (2.0e-5), a 7.5x error that would look like
"unfreezing does not work" (patch_lrgroups docstring).

Inference needs no flag: the trained weights are in the checkpoint. The load guard sees no
new parameter names, so the run tag carries `_dt<N>`.

    MSGEN_DEPTH_UNFREEZE=2 MSGEN_LRGROUPS=1 $PG -m msgen.run_train ...
"""
from __future__ import annotations

import os

_APPLIED = False


def blocks() -> int:
    v = os.environ.get("MSGEN_DEPTH_UNFREEZE", "")
    return int(v) if v.strip() else 0


def maybe_patch() -> bool:
    global _APPLIED
    n = blocks()
    if _APPLIED or n <= 0:
        return _APPLIED
    if os.environ.get("MSGEN_LRGROUPS", "0") in ("0", "", "false", "False"):
        raise SystemExit("MSGEN_DEPTH_UNFREEZE needs MSGEN_LRGROUPS=1 (see patch_lrgroups): "
                         "refusing to train unfrozen blocks at the decoder learning rate")
    from msgen.paths import add_tracegen_to_path
    add_tracegen_to_path()
    from models.model_flow import TrajectoryFlow

    orig_init = TrajectoryFlow.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        before = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.depth_encoder.unfreeze_last_blocks(n)
        after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[patch_depthtower] depth encoder: last {n} blocks unfrozen, trainable "
              f"{before / 1e6:.1f}M -> {after / 1e6:.1f}M parameters", flush=True)

    TrajectoryFlow.__init__ = __init__
    _APPLIED = True
    print(f"[patch_depthtower] active: depth encoder last {n} blocks trainable")
    return True
