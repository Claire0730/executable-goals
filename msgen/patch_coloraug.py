"""Training-time colour augmentation: the deploy sensitivity sweep measured the gmap
planner's dependence on object colour (cubeA blue: goal readout 8.9 -> 14.0 mm, P(<25) .94 -> .77,
tracked-perception success .754 -> .613; cubeB yellow .629). Real cubes will not match the sim's red/green, so the head is
fine-tuned with per-sample photometric jitter on the RGB the vision towers consume.

    MSGEN_COLOR_AUG=<prob> python -m msgen.run_train_logged ...     (via msgen.patch_all)

Per sample, with probability <prob>: hue rotation U(-0.5, 0.5) (full circle, so any object colour is
covered), saturation U(0.5, 1.5), brightness U(0.8, 1.2), contrast U(0.8, 1.2), applied to the WHOLE
image (table and wall shift too -- invariance rather than object-specific recolouring). d['image'] is a [0,1]
float tensor at this point (a later transform converts it to PIL for the SigLIP/DINO processors), so the
jitter runs in [0,1] space and clamps; a normalised tensor is detected by range and round-tripped. Depth, segmentation-derived queries and targets are untouched.
Applied AFTER patch_t2k / patch_depthaug (wraps the composed __getitem__)."""
from __future__ import annotations

import os

import numpy as np
import torch

_APPLIED = False
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def jitter(img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Works in [0,1] space. The dataset hands the model a [0,1] float tensor (a later transform converts it to
    PIL for the SigLIP/DINO processors -- the first run with de-normalisation crashed there); an ImageNet-
    normalised tensor is detected by its range and round-tripped."""
    import torchvision.transforms.functional as TF
    normed = bool(img.min() < -0.01 or img.max() > 1.01)
    x = (img * STD + MEAN) if normed else img
    x = x.clamp(0, 1)
    x = TF.adjust_hue(x, float(rng.uniform(-0.5, 0.5)))
    x = TF.adjust_saturation(x, float(rng.uniform(0.5, 1.5)))
    x = TF.adjust_brightness(x, float(rng.uniform(0.8, 1.2)))
    x = TF.adjust_contrast(x, float(rng.uniform(0.8, 1.2))).clamp(0, 1)
    return (((x - MEAN) / STD) if normed else x).to(img.dtype)


def maybe_patch() -> bool:
    global _APPLIED
    v = os.environ.get("MSGEN_COLOR_AUG", "")
    if _APPLIED or v in ("", "0"):
        return _APPLIED
    prob = float(v)
    from dataio.datasets import EpisodePointDataset
    prev_getitem = EpisodePointDataset.__getitem__

    def __getitem__(self, idx):
        d = prev_getitem(self, idx)
        rng = np.random.default_rng()
        if rng.random() >= prob:
            return d
        im = d.get("image")
        if isinstance(im, torch.Tensor) and im.ndim == 3 and im.shape[0] == 3:
            d["image"] = jitter(im, rng)
        return d

    EpisodePointDataset.__getitem__ = __getitem__
    _APPLIED = True
    print(f"[patch_coloraug] active: photometric augmentation p={prob} (hue U(-.5,.5), sat U(.5,1.5), bri/con U(.8,1.2))", flush=True)
    return True
