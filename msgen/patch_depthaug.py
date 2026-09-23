"""Training-time depth augmentation: the D435 degradation model from the
depth-noise probe (<private-repo>/experiments/20260901_dnoise/patch_depth_noise.py), applied to the
dataloader's depth channels as domain randomisation.

    MSGEN_DEPTH_AUG=<prob> python -m msgen.run_train_logged ...     (via msgen.patch_all)

Per sample, with probability <prob>: draw a severity scale s ~ U(0.5, 2.0) and corrupt
BOTH depth channels the model consumes -- the vision-tower map (d['depth'], metres,
post-transform) and the T2K geometry map (d['t2k_depth_m'], metres, 384x384). The two
draws are independent noise realisations of the same model; for augmentation this is at
least as strong as a shared realisation, and it keeps the patch a pure __getitem__ wrapper.
Clean samples are preserved with probability 1-<prob> so the clean-depth eval regime stays
in-distribution (mirrors the walled-pushcube cure recipe, which kept the original clips).

Must be applied AFTER patch_t2k (it wraps the composed __getitem__, which includes
t2k_depth_m); patch_all orders it accordingly.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import torch

_APPLIED = False


def _load_noise():
    spec = importlib.util.spec_from_file_location(
        "dn_noise",
        os.environ.get("MSGEN_DEPTH_NOISE_PATCH",
                       "<private-repo>/experiments/20260901_dnoise/patch_depth_noise.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def maybe_patch() -> bool:
    global _APPLIED
    v = os.environ.get("MSGEN_DEPTH_AUG", "")
    if _APPLIED or v in ("", "0"):
        return _APPLIED
    prob = float(v)
    dn = _load_noise()
    from dataio.datasets import EpisodePointDataset

    prev_getitem = EpisodePointDataset.__getitem__      # composed: stock + patch_t2k keys

    def _corrupt2d(arr_m, rng):
        dn._SCALE = float(rng.uniform(0.5, 2.0))
        return dn.corrupt_depth(arr_m, int(rng.integers(0, 2**31)))

    def __getitem__(self, idx):
        d = prev_getitem(self, idx)
        rng = np.random.default_rng()                   # worker-seeded; fresh noise each epoch
        if rng.random() >= prob:
            return d
        dep = d.get("depth")                            # [1,H,W] metres (vision tower)
        if isinstance(dep, torch.Tensor) and dep.ndim == 3 and dep.numel() > 1:
            a = dep[0].numpy().astype(np.float32)
            d["depth"] = torch.from_numpy(_corrupt2d(a, rng)).unsqueeze(0).to(dep.dtype)
        t2k = d.get("t2k_depth_m")                      # [384,384] metres (T2K geometry)
        if isinstance(t2k, torch.Tensor) and t2k.ndim == 2:
            a = t2k.numpy().astype(np.float32)
            d["t2k_depth_m"] = torch.from_numpy(_corrupt2d(a, rng)).to(t2k.dtype)
        return d

    EpisodePointDataset.__getitem__ = __getitem__
    _APPLIED = True
    print(f"[patch_depthaug] active: D435 depth augmentation p={prob}, scale U(0.5,2.0)", flush=True)
    return True
