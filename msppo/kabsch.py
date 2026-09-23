"""Paired-point SE(3) fit, used to turn a handful of traced points into a dense
keypoint set.

TraceGen only puts ~4-6 grid points on a 4 cm cube, which is too thin to feed a
mean-pooled PointNet directly. Fitting a rigid transform from those correspondences
and carrying the dense canonical keypoints through it keeps the project rule
"predict points, solve pose, hand points to the policy" while restoring K=128.

The residual is returned so degenerate fits (near-collinear points, or a bad
trace) can be caught rather than silently producing a wrong goal.
"""
from __future__ import annotations

import torch


def kabsch(P: torch.Tensor, Q: torch.Tensor, weights: torch.Tensor | None = None):
    """Least-squares rigid transform mapping P onto Q.

    P, Q: [B,N,3] corresponding points. Returns (R [B,3,3], t [B,3], rmse [B]).
    """
    if weights is None:
        weights = torch.ones(P.shape[:2], device=P.device, dtype=P.dtype)
    w = (weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9))[..., None]

    Pc, Qc = (P * w).sum(1, keepdim=True), (Q * w).sum(1, keepdim=True)
    X, Y = P - Pc, Q - Qc
    H = torch.einsum("bni,bnj->bij", X * w, Y)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(torch.einsum("bij,bjk->bik", Vt.transpose(1, 2), U.transpose(1, 2))))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    R = torch.einsum("bij,bjk,bkl->bil", Vt.transpose(1, 2), D, U.transpose(1, 2))
    t = Qc.squeeze(1) - torch.einsum("bij,bj->bi", R, Pc.squeeze(1))

    err = torch.einsum("bij,bnj->bni", R, P) + t[:, None, :] - Q
    rmse = err.pow(2).sum(-1).mean(1).sqrt()
    return R, t, rmse


def apply(R: torch.Tensor, t: torch.Tensor, kp: torch.Tensor) -> torch.Tensor:
    """R [B,3,3], t [B,3], kp [B,K,3] or [K,3] -> [B,K,3]."""
    if kp.dim() == 2:
        kp = kp[None].expand(R.shape[0], -1, -1)
    return torch.einsum("bij,bkj->bki", R, kp) + t[:, None, :]


def fit_goal_keypoints(src_pts, dst_pts, canonical_kp, object_p, object_q,
                       max_rmse: float = 0.02):
    """Carry the dense canonical keypoints from the object to the traced goal.

    src_pts/dst_pts: [B,N,3] the traced points at t0 and at the trace endpoint.
    Falls back to a translation-only transform where the rigid fit is poor, which
    is the honest degradation: with 4-6 noisy points the rotation is the first
    thing to become unreliable, while the centroid shift stays informative.
    """
    from msppo.kp import to_world

    R, t, rmse = kabsch(src_pts, dst_pts)
    obj_kp = to_world(canonical_kp, object_p, object_q)
    goal_kp = apply(R, t, obj_kp)

    shift = dst_pts.mean(1) - src_pts.mean(1)
    goal_kp_translate = obj_kp + shift[:, None, :]

    bad = (rmse > max_rmse) | ~torch.isfinite(rmse)
    goal_kp = torch.where(bad[:, None, None], goal_kp_translate, goal_kp)
    return goal_kp, rmse, bad
