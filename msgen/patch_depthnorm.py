"""Make `preprocess_depth` do what its docstring says. Enable with MSGEN_DEPTHNORM.

THE DEFECT. `TraceGen/dataio/transforms.py` declares:

    def preprocess_depth(self, depth_tensor):
        \"\"\"Preprocess depth using logarithmic transform and normalize to [0, 1].\"\"\"
        return depth_tensor

The body is a no-op. Depth therefore reaches the depth tower as RAW METRES
(0.05-3.0 in this project's scenes) through a 6-parameter `Conv2d(1,3,1)` stem
adapter (`siglip_vision.py:44`) into a SigLIP tower that is FROZEN and was
pretrained on RGB in [-1,1]. Six parameters are the only thing standing between
a metric depth map and a distribution the tower has never seen.

⚠️ THIS IS NOT A FREE BUG FIX, AND MUST NOT BE APPLIED SILENTLY TO AN EXISTING
CHECKPOINT. The released Generalist was itself trained through this same no-op,
so its decoder is adapted to whatever the frozen tower emits for raw-metre
input. Turning the transform on changes the depth token distribution at the
input of a warm start that never saw it. The correct use is a SINGLE-VARIABLE
RETRAIN from the Generalist with the flag on, judged against the identical arm
with it off -- never a patched inference pass over weights trained without it.

WHY IT IS WORTH A RETRAIN AT ALL. The planner's error is dominated by a common
translation (67-84% of per-point error energy, measured), and depth is the only
channel carrying scale. If the depth pathway is not functioning, that is a
mechanism for a systematic, not random, displacement. Whether the error actually
lives along the line of sight is a separate measurement, and one that is a
property of the pre-bugfix model, since fixing the input can change which axis
dominates.

THE TRANSFORM. log1p compresses the 0.05-3.0 m range, then a fixed affine maps
it to [0,1] using constants, NOT per-sample statistics: a per-sample min/max
would make the mapping scene-dependent and destroy metric scale across scenes,
which is the one thing depth is here to provide. Bounds come from the project's
own render range (`msgen/tasks.py` clips valid depth to 0.05-3.0 m) and are
exposed so an arm can sweep them.

    MSGEN_DEPTHNORM=1 $PG -m msgen.run_train ...
"""
from __future__ import annotations

import os

D_MIN = float(os.environ.get("MSGEN_DEPTH_MIN", "0.05"))
D_MAX = float(os.environ.get("MSGEN_DEPTH_MAX", "3.0"))
_APPLIED = False


def _enabled() -> bool:
    return os.environ.get("MSGEN_DEPTHNORM", "0") not in ("0", "", "false", "False")


def maybe_patch() -> bool:
    """Apply when MSGEN_DEPTHNORM is set. Returns whether it engaged."""
    global _APPLIED
    if _APPLIED or not _enabled():
        return _APPLIED
    import math
    import torch
    from dataio.transforms import NormalizeTransform

    lo, hi = math.log1p(D_MIN), math.log1p(D_MAX)
    span = hi - lo

    def preprocess_depth(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        d = depth_tensor.clamp(min=0.0)
        out = (torch.log1p(d) - lo) / span
        # invalid/zero depth maps to 0 after the clamp below, which is the same
        # value the stock path would have produced for a 0 m reading -- so the
        # `is_depth_valid` mask keeps its meaning and the learnable mask token
        # still substitutes for whole invalid frames upstream.
        return out.clamp(0.0, 1.0)

    NormalizeTransform.preprocess_depth = preprocess_depth
    _APPLIED = True
    print(f"[patch_depthnorm] active: log1p + fixed affine over "
          f"[{D_MIN}, {D_MAX}] m -> [0, 1]  (stock path is a no-op)")
    return True
