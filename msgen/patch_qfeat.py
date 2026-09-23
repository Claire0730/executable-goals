"""Give every trace query the visual content at its own pixel, from outside the checkout.

WHAT THIS FIXES. `patch_vispos.py` gave the 576 visual conditioning tokens a POSITION,
because they enter CogVideoX through the text slot and diffusers fills the joint positional
embedding only from `max_text_seq_length` onward, so they were receiving exactly zero.
That says "you are looking at image region (u,v)". It still does not say, to the query that
lives at grid cell (h,w), "here is what YOUR location looks like".

The remaining asymmetry is a correspondence, not a coordinate: the 20x20 query grid and the
24x24 visual grid cover the same image, and nothing in the architecture ties cell to patch.
Measured consequence -- the model answers a global question instead of a per-location one:

  * 15.1% of truly-moving object points are predicted near-static while the movers'
    amplitude is correct (median ratio 0.986) -- per-location classification, not scale
  * the largest predicted displacements land on `panda_link6` 79.7%, on the peg 0.8%
  * 75% of the endpoint error is in the image plane, 25% along the viewing ray

DO NOT confuse this with adding Fourier(u,v) on the query side, which is REFUTED for our
data: all 1399 `samples/*.npz` carry bit-identical keypoints (a fixed uniform 20x20 grid,
(9.6,9.6)...(374.4,374.4) = (k+0.5)*19.2, `msgen/labels.py:45-55`), so query k maps to
latent cell (k//20, k%20) deterministically and CogVideoX's patch embed already supplies
that position as 3D sincos. Position on the query side is a bijection of something the model
has. CONTENT is not.

WHERE IT HOOKS, and why here (verified against a live instance, diffusers 0.35.1):

    forward(text_embeds[B,704,768], image_embeds[B,16,6,20,20]) -> [B,2304,768]
                                              704 conditioning + 1600 image = 16 x 10x10

`patch_size_t=1` is not None, so the else-branch runs and image tokens come out T-MAJOR,
row-major within a frame: index = t*(h*w) + h_i*w + w_i. That matches the row-major query
order, which `msgen/labels.py:46-50` warns is mandatory and fails SILENTLY if wrong.
`patch_size=2`, so one latent token covers a 2x2 block of queries and the 24x24 visual grid
is resampled to 10x10.

NO SIDE CHANNEL IS NEEDED, which is the reason this hook was chosen over adding an argument
somewhere upstream. `trunk_conditioning = cat([vision(576), text(128)])` reaches this
function as `text_embeds` after `trunk_to_text_proj`, and that is a PER-TOKEN linear, so
token identity and order survive it: `text_embeds[:, :576, :]` IS the 24x24 visual grid.
A module-attribute side channel would graph-break under `torch.compile(self.model)`, which
wraps the whole model (`trainer/trainer.py:191-193`).

TWO CHOICES ARE ABOUT OUR DATA BUDGET, NOT ELEGANCE, same as patch_vispos:

  * ZERO-INIT gate, so the patched model is bit-identical at step 0 and the Generalist
    warm-start survives. Measured: fine-tuning beats from-scratch 3.8x on our 50 clips
    (endpoint_mse 0.000333 vs 0.001272).
  * A PER-CHANNEL gate (768 parameters), not a [768,768] projection (590k). With 1399
    training samples the projection would fit 422 parameters per sample. The gate is
    additive, so at zero it still trains: d(out)/d(gate) is the feature itself.

    from msgen.patch_qfeat import patch_qfeat
    patch_qfeat()            # before the trainer builds the model
"""
from __future__ import annotations

import os

_APPLIED = False


def patch_qfeat(scale_init: float = 0.0, n_vis: int = 576):
    """Wrap `CogVideoXPatchEmbed` so image tokens carry the visual feature at their pixel.

    The parameter is created in `__init__`, NOT lazily, because `create_optimizer`
    (`utils/misc.py:176`) walks `model.named_parameters()` once during `_setup_optimization`,
    which runs before any forward. A parameter added later would be absent from the
    optimizer and would silently never train -- the failure would look like "the idea does
    not work".

    Its name (`diffusion_decoder.cogvideox.patch_embed.q_feat_scale`) contains neither
    `vision_encoder` nor `text_encoder`, so it lands in the decoder param group at
    `lr_decoder`, the group being fine-tuned.

    `load_checkpoint_singlegpu` loads with `strict=False` (`trainer.py:1230`), so on a
    checkpoint that predates the patch the new key appears under missing_keys and keeps its
    zero init instead of failing the load. The reverse is the landmine: loading a
    patch-trained checkpoint WITHOUT the patch makes the key "unexpected" and it is dropped
    in silence, so every inference process must set the env flag too.
    """
    global _APPLIED
    if _APPLIED:
        return
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from diffusers.models.embeddings import CogVideoXPatchEmbed

    orig_init = CogVideoXPatchEmbed.__init__
    orig_fwd = CogVideoXPatchEmbed.forward

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        self.q_feat_scale = nn.Parameter(torch.full((self.embed_dim,), float(scale_init)))
        self._qfeat_n_vis = int(n_vis)
        self._qfeat_state = None

    def forward(self, text_embeds, image_embeds):
        out = orig_fwd(self, text_embeds, image_embeds)
        if getattr(self, "_qfeat_state", "off") is None:
            self._qfeat_state = _describe(self, text_embeds, image_embeds, out)
            print(f"[patch_qfeat] {self._qfeat_state}")
        if self._qfeat_state != "active":
            return out

        B, T, C, H, W = image_embeds.shape
        p = self.patch_size
        h, w = H // p, W // p
        n_text = text_embeds.shape[1]
        n_img = out.shape[1] - n_text
        t_lat = n_img // (h * w)
        g = int(round(self._qfeat_n_vis ** 0.5))
        D = self.embed_dim

        vis = text_embeds[:, : self._qfeat_n_vis, :]              # [B, 576, D]
        vis = vis.transpose(1, 2).reshape(B, D, g, g)             # [B, D, 24, 24]
        # AREA pooling, not bilinear. Bilinear was the first choice, for consistency with
        # model_flow.py:166-170 -- and u1_localization.py measured why it is wrong here:
        # bilinear POINT-SAMPLES at 10 centres, which only ever read input indices adjacent
        # to those centres ({0,1},{3,4},... -> indices 2, 9, 14, 21 are never read), so
        # 30.6% of visual patches reach no latent cell at all. That directly defeats the
        # purpose: those queries would still be blind to their own location. Area pooling
        # covers every input cell, and matches the semantics -- one latent token spans a
        # 2x2 block of queries, i.e. a 2.4x2.4 block of visual patches.
        vis = F.adaptive_avg_pool2d(vis, (h, w))
        vis = vis.flatten(2).transpose(1, 2)                      # [B, h*w, D]
        inj = (vis * self.q_feat_scale).to(out.dtype)
        inj = inj.repeat(1, t_lat, 1)                             # t-major tiling

        # cat rather than in-place, so autograd and compile never see a mutated view
        return torch.cat([out[:, :n_text, :], out[:, n_text:, :] + inj], dim=1)

    def _describe(self, text_embeds, image_embeds, out):
        """Return "active", or a reason string -- encode nothing rather than wrong."""
        n_vis = self._qfeat_n_vis
        g = int(round(n_vis ** 0.5))
        B, T, C, H, W = image_embeds.shape
        p = self.patch_size
        h, w = H // p, W // p
        n_text = text_embeds.shape[1]
        n_img = out.shape[1] - n_text
        if g * g != n_vis:
            return f"SKIPPED: n_vis={n_vis} is not a square"
        if n_text < n_vis:
            return f"SKIPPED: conditioning seq {n_text} < n_vis {n_vis}"
        if text_embeds.shape[2] != self.embed_dim:
            return (f"SKIPPED: text width {text_embeds.shape[2]} != embed_dim "
                    f"{self.embed_dim}; would need a projection")
        if h * w == 0 or n_img % (h * w) != 0:
            return f"SKIPPED: {n_img} image tokens is not a multiple of {h}x{w}"
        return "active"

    CogVideoXPatchEmbed.__init__ = __init__
    CogVideoXPatchEmbed.forward = forward
    _APPLIED = True
    print(f"[patch_qfeat] patched CogVideoXPatchEmbed (scale_init={scale_init}, "
          f"n_vis={n_vis}); zero init keeps the warm-start bit-identical at step 0")


def maybe_patch():
    """Apply only when MSGEN_QFEAT is set, so every existing run is unaffected."""
    v = os.environ.get("MSGEN_QFEAT", "")
    if v and v != "0":
        patch_qfeat(scale_init=0.0 if v in ("1", "on", "true") else float(v))
        return True
    return False
