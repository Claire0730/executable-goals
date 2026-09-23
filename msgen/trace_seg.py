"""Trace-derived segmentation and phase structure — planner output only, no simulator segmentation.

Measured basis (2026-08-30, production planner, K=1 preds, 4 tasks x 256 scenes): points the trace moves
>8 px overlap (object ∪ arm) segmentation at IoU 0.79-0.87 with ~1% background false positives; the arm
starts moving before the object in 100% of scenes, so thresholding each point's motion ONSET splits
arm from object at 0.98-1.00 accuracy. Hence the trace alone yields: a t=0 object mask, the grasp step
(object onset), the release step (object stops while the arm continues), the arm approach vector before
grasp, and the object carry-height profile. Nothing here reads seg.npz.
"""
from __future__ import annotations
import numpy as np

MOVE_PX = 8.0      # endpoint displacement that counts as "moving" (384x384 pixels)
ONSET_PX = 4.0     # displacement that defines motion onset
SPLIT_MARGIN = 2   # arm/object split: onset > median(arm onset) + margin -> object


def unproject(px, depth_m, K, ext):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (px[..., 0] - cx) / fx * depth_m
    y = (px[..., 1] - cy) / fy * depth_m
    p_cam = np.stack([x, y, depth_m], axis=-1)
    R, t = ext[:, :3], ext[:, 3]
    return (p_cam - t) @ R


def segment(pred):
    """pred [N,T,3] (x,y in px, z in metres) -> dict with masks and phase steps. Planner output only."""
    ok = np.isfinite(pred).all(axis=(1, 2))
    disp = np.linalg.norm(pred[:, :, :2] - pred[:, :1, :2], axis=-1)          # [N,T]
    moving = (disp[:, -1] > MOVE_PX) & ok
    onset = np.where((disp > ONSET_PX).any(1), np.argmax(disp > ONSET_PX, 1), 99)
    out = dict(ok=ok, moving=moving, onset=onset,
               obj_mask=np.zeros_like(moving), arm_mask=np.zeros_like(moving),
               grasp_step=-1, release_step=-1)
    if moving.sum() < 6:
        return out
    thr = np.median(onset[moving]) + SPLIT_MARGIN
    # early-onset movers = arm; late-onset movers = object (arm reaches first, then the object follows)
    arm = moving & (onset <= thr)
    obj = moving & (onset > thr)
    if obj.sum() >= 3 and arm.sum() >= 3:
        out["obj_mask"], out["arm_mask"] = obj, arm
        out["grasp_step"] = int(np.median(onset[obj]))
        # release: object centroid per-step speed falls below eps while the arm still moves
        c = np.nanmedian(pred[obj][:, :, :2], axis=0)                         # [T,2] px
        v = np.linalg.norm(np.diff(c, axis=0), axis=-1)                       # [T-1]
        g = out["grasp_step"]
        rel = -1
        for s in range(max(g + 2, 1), len(v)):
            if v[s] < 0.5 and v[max(s - 1, 0)] < 0.5:
                rel = s; break
        out["release_step"] = rel
    return out


def world_profiles(pred, seg_out, K, ext):
    """Carry-height profile of the object and the arm approach vector before grasp, in world metres."""
    res = dict(carry_mm=np.nan, approach=None, vert_frac=np.nan)
    obj, arm, g = seg_out["obj_mask"], seg_out["arm_mask"], seg_out["grasp_step"]
    if obj.sum() < 3 or g < 0:
        return res
    pw = unproject(pred[obj][:, :, :2], pred[obj][:, :, 2].astype(np.float64), K, ext)  # [n,T,3]
    z = np.nanmedian(pw[:, :, 2], axis=0)
    res["carry_mm"] = float((np.nanmax(z) - z[0]) * 1000.0)
    if arm.sum() >= 3:
        aw = unproject(pred[arm][:, :, :2], pred[arm][:, :, 2].astype(np.float64), K, ext)
        s0, s1 = max(g - 4, 0), max(g, 1)
        d = np.nanmedian(aw[:, s1] - aw[:, s0], axis=0)
        n = np.linalg.norm(d)
        if n > 1e-6:
            res["approach"] = d / n
            res["vert_frac"] = float(abs(d[2]) / n)
    return res
