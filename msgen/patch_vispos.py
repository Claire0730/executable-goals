"""Give the visual conditioning tokens a position, from outside the pristine checkout.

VERIFIED DEFECT. `model_flow.py:220-256` builds the conditioning as
`cat([vision_features (576), text_features (128)], dim=1)` = 704 tokens, and those enter
CogVideoX through the TEXT slot of its joint attention. diffusers'
`CogVideoXPatchEmbed._get_positional_embeddings` fills the joint positional embedding
only from `max_text_seq_length` onward:

    joint_pos_embedding = pos_embedding.new_zeros(1, max_text_seq_length + num_patches, D)
    joint_pos_embedding.data[:, max_text_seq_length:].copy_(pos_embedding)

so every conditioning token receives EXACTLY ZERO positional embedding, while the trace
latents get 3D sincos. `use_rotary_positional_embeddings: false` removes the only
fallback. The trace side therefore knows where it is in the 20x20 grid; the visual side
does not know which image region it came from, and has to infer that from content alone.

That asymmetry matches what the peg predictions actually do wrong, measured three ways:

  * 15.1% of truly-moving object points are predicted near-static while the movers'
    amplitude is correct (median ratio 0.986) -- a per-location classification failure,
    not a regression-scale one
  * the largest predicted displacements land on `panda_link6` (79.7%) and on the peg
    only 0.8% -- a global "the arm moves" prior rather than a per-query answer
  * 75% of the endpoint error is in the image plane, 25% along the viewing ray -- the
    pixel trajectory is what is wrong

TWO CHOICES HERE ARE ABOUT OUR DATA BUDGET, NOT ELEGANCE:

  * ZERO-INIT, so the patched model is bit-identical to the original at step 0 and the
    Generalist warm-start survives. Measured: fine-tuning beats from-scratch 3.8x on our
    50 clips (endpoint_mse 0.000333 vs 0.001272), so warm-start is not optional.
  * A FIXED 2D sincos basis times a learnable PER-CHANNEL scale (768 new parameters),
    not a free [576, 768] table (442k). With 1399 training samples a free table would be
    fitting 316 parameters per sample.

Additive zero-init still trains: the gradient w.r.t. an additive term is the upstream
gradient, unlike a multiplicative gate at zero which would cut the branch off.

    from msgen.patch_vispos import patch_vis_pos
    patch_vis_pos()          # before the trainer builds the model
"""
from __future__ import annotations

import os

_APPLIED = False


def _sincos_2d(dim: int, grid: int):
    import numpy as np
    import torch

    def _1d(d, pos):
        omega = np.arange(d // 2, dtype=np.float64) / (d / 2.0)
        omega = 1.0 / 10000 ** omega
        out = pos.reshape(-1)[:, None] * omega[None]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    g = np.arange(grid, dtype=np.float64)
    gy, gx = np.meshgrid(g, g, indexing="ij")          # row-major, matching the tokens
    emb = np.concatenate([_1d(dim // 2, gx), _1d(dim // 2, gy)], axis=1)
    if emb.shape[1] < dim:                              # odd dim -> pad
        emb = np.concatenate([emb, np.zeros((emb.shape[0], dim - emb.shape[1]))], 1)
    return torch.from_numpy(emb).float()[None]          # [1, grid*grid, dim]


def patch_vis_pos(scale_init: float = 0.0):
    """Wrap `TrajectoryFlow` so its visual conditioning tokens carry a 2D position.

    The scale parameter is created in `__init__`, NOT lazily on first forward, because
    `create_optimizer` (`utils/misc.py:176`) walks `model.named_parameters()` once during
    `_setup_optimization`, which runs before any forward. A parameter added later would
    be absent from the optimizer and would silently never train -- the failure would look
    like "the idea does not work".

    Its name contains neither `vision_encoder` nor `text_encoder`, so it lands in the
    decoder param group at `lr_decoder`, which is the group being fine-tuned.

    `load_checkpoint_singlegpu` loads with `strict=False` (`trainer.py:1230`), so the new
    key appears under missing_keys and keeps its zero init instead of failing the load.
    """
    global _APPLIED
    if _APPLIED:
        return
    import torch
    import torch.nn as nn
    from models.model_flow import TrajectoryFlow

    orig_init = TrajectoryFlow.__init__
    orig_enc = TrajectoryFlow.encode_images

    def _d_model(cfg, default=768):
        if isinstance(cfg, dict):
            return int(cfg.get("d_model", default))
        return int(getattr(cfg, "d_model", default))

    def __init__(self, cfg):
        orig_init(self, cfg)
        D = _d_model(cfg)
        self.vis_pos_scale = nn.Parameter(torch.full((D,), float(scale_init)))
        self._vis_pos_n = None

    def encode_images(self, images, depth, is_depth_valid):
        vis = orig_enc(self, images, depth, is_depth_valid)          # [B, N, D]
        N, D = vis.shape[1], vis.shape[2]
        if getattr(self, "_vis_pos_n", None) != N:
            grid = int(round(N ** 0.5))
            self._vis_pos_n = N
            if grid * grid != N or D != self.vis_pos_scale.numel():
                # no square 2D layout, or a width mismatch: encode nothing rather than
                # encode something wrong
                self._vis_pos_basis = None
                print(f"[patch_vispos] SKIPPED: N={N} grid^2={grid*grid} D={D} "
                      f"scale={self.vis_pos_scale.numel()}")
            else:
                self.register_buffer("_vis_pos_basis", _sincos_2d(D, grid), persistent=False)
                print(f"[patch_vispos] active: {grid}x{grid}={N} visual tokens, D={D}")
        b = getattr(self, "_vis_pos_basis", None)
        if b is None:
            return vis
        return vis + b.to(vis.device, vis.dtype) * self.vis_pos_scale.to(vis.dtype)

    TrajectoryFlow.__init__ = __init__
    TrajectoryFlow.encode_images = encode_images
    _APPLIED = True
    print(f"[patch_vispos] patched TrajectoryFlow (scale_init={scale_init}); "
          f"zero init keeps the warm-start bit-identical at step 0")


def maybe_patch():
    """Apply only when MSGEN_VISPOS is set, so every existing run is unaffected."""
    v = os.environ.get("MSGEN_VISPOS", "")
    if v and v != "0":
        patch_vis_pos(scale_init=0.0 if v in ("1", "on", "true") else float(v))
        return True
    return False
