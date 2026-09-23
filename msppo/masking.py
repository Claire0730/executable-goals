"""Dex4D's keypoint occlusion model, batched.

Ported from Dex4D-Simulation:
  utils/util.py:291-320  mask_keypoints_oneside              (per EPISODE, at reset)
  utils/util.py:363-390  ..._random_height_test_time         (per STEP)

Both return a boolean KEEP mask rather than zeroed points, so a caller can choose
between "what a pose solver could use" (weighted Kabsch over the kept points) and
"what a mean pool sees" (zeros dragging the centroid). Upstream only ever does
the latter; the distinction is what `msppo/maskprobe.py` needed to measure that
this masking removes no information on a known cube.

One deliberate implementation difference in `mask_oneside`: upstream thresholds
at a plane and then randomly adds or removes points to hit the target count,
looping one env at a time. Taking the `keep_n` smallest signed distances is the
same intent -- a spatially contiguous half-space cut of exactly the right size --
without the per-env loop, which would run 1024 times on every reset.
"""
from __future__ import annotations

import torch


def mask_oneside(kp: torch.Tensor, ratio: int, gen: torch.Generator) -> torch.Tensor:
    """[B,K,3] -> keep mask [B,K] with exactly K//ratio points on one side of a
    random plane through a random one of the points."""
    B, K, _ = kp.shape
    keep_n = max(K // ratio, 3)          # a pose needs 3 points; never go below
    n = torch.randn((B, 3), generator=gen, device=kp.device)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    anchor = torch.randint(0, K, (B,), generator=gen, device=kp.device)
    p0 = kp[torch.arange(B, device=kp.device), anchor]
    d = torch.einsum("bkj,bj->bk", kp - p0[:, None, :], n)
    idx = d.argsort(dim=1)[:, :keep_n]
    keep = torch.zeros((B, K), dtype=torch.bool, device=kp.device)
    keep.scatter_(1, idx, True)
    return keep


def mask_height(kp: torch.Tensor, keep: torch.Tensor, gen: torch.Generator,
                max_mask_height: float = 0.8, above_p: float = 0.9,
                below_p: float = 0.05) -> torch.Tensor:
    """Per-step occlusion: a height plane at a random quantile of the points'
    z, everything above it dropped with p=0.9 and below with p=0.05. Models a
    table-top camera whose view of the object's top is blocked by the gripper."""
    B, K, _ = kp.shape
    frac = torch.rand((B,), generator=gen, device=kp.device) * max_mask_height
    z = kp[..., 2]
    z_sorted, _ = z.sort(dim=1)
    k = ((1 - frac) * (K - 1)).long()
    plane = z_sorted[torch.arange(B, device=kp.device), k]
    above = z > plane[:, None]
    r = torch.rand((B, K), generator=gen, device=kp.device)
    drop = torch.where(above, r < above_p, r < below_p)
    out = keep & ~drop
    # Never hand the policy a fully blank object: if a draw wipes an env out,
    # restore its three highest-priority points. Upstream has no such guard
    # because a zeroed point cloud is merely a bad input there; here the same
    # buffer also feeds the Kabsch solve in `recon`, which would return garbage.
    dead = out.sum(1) < 3
    if dead.any():
        out[dead] = keep[dead]
    return out
