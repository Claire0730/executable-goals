"""patch_t2k_couple.py -- make the keypose head REACH the flow decoder (a stronger head-decoder coupling).

The current head (patch_t2k) consumes the fused vision tokens, so its gradient stops at `vision_fusion`; the CogVideoX flow
decoder never receives it. Two couplings, both losses only (inference path untouched; all flags off == bit-identical):

  A1  MSGEN_T2K_DEC=<L>  MSGEN_T2K_WDEC=<w>   a second entity head on the DECODER's block-L hidden states. The decoder
      sees the 20x20 query grid as 10x10 patches x T/2 frame groups (patch_size 2, put_frames_in_channels 2); tokens
      are mean-pooled over time -> 100 units, each with the camera xyz of its 4 queries at t0 and the fraction of
      those queries on the object. Same keyframe / structure targets as the vision head. Gradient reaches blocks 0..L.
  A3  MSGEN_T2K_KC=<w>                         keypose-consistency on the flow's own clean estimate. Stochastic
      interpolant x_t=(1-t)x0+t*eps, v=eps-x0  =>  x0_hat = x0 + t*(v_true - v_pred). The object's query points of
      x0_hat at the labelled keyframe step are compared with where the label puts them (dT @ P_t0), the per-point error
      soft-capped at 0.5 m (slope 0.1 beyond, so the gradient stays bounded without vanishing),
      weighted (1-t)^2. Gradient reaches every decoder weight through v_pred. See keypose_consistency for why the
      first version (Kabsch fit) had to go.

Requires patch_t2k to be active (labels, batch stash). Applied after it by patch_all.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_MAX, GRID_Q, GRID_D = 4, 20, 10
_APPLIED = False


def _cfg():
    e = os.environ.get
    return dict(dec=int(e("MSGEN_T2K_DEC", "-1") or -1), wdec=float(e("MSGEN_T2K_WDEC", "1.0") or 1),
                kc=float(e("MSGEN_T2K_KC", "0") or 0), hidden=int(e("MSGEN_T2K_HIDDEN", "256") or 256))


def active() -> bool:
    c = _cfg()
    return c["dec"] >= 0 or c["kc"] > 0


# ---- small geometry helpers (self-contained; patch_t2k keeps its own inside a closure) -------------------------------
def rot6d_to_R(x):
    a, b = x[..., :3], x[..., 3:6]
    a = F.normalize(a, dim=-1); b = b - (a * b).sum(-1, keepdim=True) * a; b = F.normalize(b, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], dim=-1)


def geodesic(R1, R2):
    c = ((R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))


def unproject(uvd, K):
    """uvd [B,N,3] (px, px, m), K [B,3,3] -> camera xyz [B,N,3]."""
    fx, fy, cx, cy = K[:, 0, 0, None], K[:, 1, 1, None], K[:, 0, 2, None], K[:, 1, 2, None]
    d = uvd[..., 2]
    return torch.stack([(uvd[..., 0] - cx) * d / fx, (uvd[..., 1] - cy) * d / fy, d], dim=-1)


def rigid_inv(T):
    """[B,4,4] rigid inverse without linalg.inv (which is a NaN source on degenerate input)."""
    R, t = T[:, :3, :3], T[:, :3, 3]
    Ti = torch.zeros_like(T); Ti[:, 3, 3] = 1.0
    Ti[:, :3, :3] = R.transpose(1, 2); Ti[:, :3, 3] = -(R.transpose(1, 2) @ t[..., None]).squeeze(-1)
    return Ti


class DecHead(nn.Module):
    """Entity readout on decoder features: attention-pool over object units, cross-attend to all units with their
    camera xyz, predict keyframes (valid, step, t, 6d) and the structure / relative-end poses."""
    def __init__(self, D, hidden):
        super().__init__()
        self.pool_q = nn.Linear(D, 1)
        self.pos = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, D))
        self.xattn = nn.MultiheadAttention(D, 8, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(2 * D, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.kf = nn.Linear(hidden, K_MAX * 11)
        self.st = nn.Linear(hidden, 18)

    def forward(self, feat, obj_mask, xyz, dvalid):
        w = self.pool_q(feat).squeeze(-1).masked_fill(obj_mask < 0.25, -1e4)
        w = torch.softmax(w, dim=1)
        ent = (w[..., None] * feat).sum(1, keepdim=True)                              # [B,1,D]
        kv = feat + self.pos(torch.cat([xyz, dvalid[..., None]], -1))
        ctx, _ = self.xattn(ent, kv, kv)
        h = self.mlp(torch.cat([ent, ctx], -1).squeeze(1))
        kf = self.kf(h).view(-1, K_MAX, 11); st = self.st(h)
        return dict(kf_valid=kf[..., 0], kf_step=kf[..., 1] * 32.0, kf_t=kf[..., 2:5], kf_R=rot6d_to_R(kf[..., 5:11]),
                    struct_t=st[:, :3], struct_R=rot6d_to_R(st[:, 3:9]), rel_t=st[:, 9:12], rel_R=rot6d_to_R(st[:, 12:18]))


def head_loss(o, b):
    has = b["t2k_has"].float()
    def m(x, w): return (x * w).sum() / w.sum().clamp_min(1.0)
    kv = b["t2k_kf_valid"] * has[:, None]; parts = {}
    parts["kf_valid"] = m(F.binary_cross_entropy_with_logits(o["kf_valid"], b["t2k_kf_valid"], reduction="none"), has[:, None].expand_as(kv))
    parts["kf_step"] = m((o["kf_step"] - b["t2k_kf_step"]).abs() / 32.0, kv)
    parts["kf_t"] = m((o["kf_t"] - b["t2k_kf_T"][:, :, :3, 3]).abs().sum(-1), kv)
    parts["kf_R"] = m(geodesic(o["kf_R"], b["t2k_kf_T"][:, :, :3, :3]), kv)
    parts["st_t"] = m((o["struct_t"] - b["t2k_struct_T"][:, :3, 3]).abs().sum(-1), has)
    parts["st_R"] = m(geodesic(o["struct_R"], b["t2k_struct_T"][:, :3, :3]), has)
    parts["rel_t"] = m((o["rel_t"] - b["t2k_rel_T"][:, :3, 3]).abs().sum(-1), has)
    parts["rel_R"] = m(geodesic(o["rel_R"], b["t2k_rel_T"][:, :3, :3]), has)
    W = dict(kf_valid=0.5, kf_step=0.5, kf_t=2.0, kf_R=0.5, st_t=2.0, st_R=0.5, rel_t=2.0, rel_R=0.5)
    return sum(W[k] * v for k, v in parts.items()), {k: v.detach() for k, v in parts.items()}


def keypose_consistency(x0hat_uvd, traj, b, t_float, cap=0.5, dmin=0.1, dmax=3.0):
    """Object points of the flow's clean estimate must land where the keypose label says.

    v1 (09-10 12:50) fitted a Kabsch pose to the predicted points and compared SE(3); it drove the whole run to NaN
    within ~250 steps. Two independent defects, both intrinsic to that formulation: `torch.linalg.svd` has a
    1/(s_i^2 - s_j^2) backward that blows up on the near-planar, near-symmetric point sets a cube face produces; and
    rows masked out AFTER the fit still receive a 0 * NaN = NaN gradient through the batched decomposition.

    v2 needs no fit. The label already says where each point must go -- the motion is rigid, so
        P_target = dR @ P_t0 + dt,     dT = kf_T_cam @ inv(obj_T_t0_cam)
    and the loss is the weighted mean of |P_pred - P_target| over the object's query points, capped at `cap` metres so
    a wild early prediction cannot produce an unbounded gradient. Only multiplies and adds: an invalid row contributes
    0 * finite = 0. Rotation is still constrained (points, not just the centroid), and translation directly."""
    B, N, T, _ = x0hat_uvd.shape
    K = b["t2k_K"].to(x0hat_uvd.dtype)
    uvd0 = traj[:, :, 0, :]
    P0 = unproject(uvd0, K)                                                            # [B,N,3] object at t0
    w = b["t2k_obj_mask"].float() * (uvd0[..., 2] > 0.05).float()                       # object queries with depth
    has = b["t2k_has"].float() * (w.sum(1) >= 6).float()
    kf_v = b["t2k_kf_valid"] * has[:, None]                                            # [B,K_MAX]
    dT = b["t2k_kf_T"] @ rigid_inv(b["t2k_obj_T_t0"])[:, None]                          # [B,K,4,4] label motion
    wt = (1.0 - t_float.view(-1)) ** 2                                                  # trust x0_hat at low noise
    steps = b["t2k_kf_step"].round().clamp(1, T).long() - 1                             # [B,K]
    tot = x0hat_uvd.sum() * 0.0; n = 0
    for k in range(K_MAX):
        sel = kf_v[:, k]
        if float(sel.sum()) == 0:
            continue
        uvd = x0hat_uvd[torch.arange(B, device=x0hat_uvd.device), :, steps[:, k], :]
        uvd = torch.stack([uvd[..., 0], uvd[..., 1], uvd[..., 2].clamp(dmin, dmax)], -1)
        Ps = unproject(uvd, K)
        Pt = (dT[:, k, :3, :3][:, None] @ P0[..., None]).squeeze(-1) + dT[:, k, :3, 3][:, None]
        e = (Ps - Pt).abs().sum(-1)                                                     # [B,N] metres
        err = torch.where(e <= cap, e, cap + 0.1 * (e - cap))                           # gradient bounded by 1, never 0
        per = (err * w).sum(1) / w.sum(1).clamp_min(1.0)                                # [B]
        tot = tot + (per * wt * sel).sum(); n += int(sel.sum())
    if n == 0:
        return x0hat_uvd.sum() * 0.0, 0
    return tot / n, n


def maybe_patch() -> bool:
    global _APPLIED
    if _APPLIED or not active():
        return _APPLIED
    cfg = _cfg()
    from msgen.paths import add_tracegen_to_path
    add_tracegen_to_path()
    from models.model_flow import TrajectoryFlow
    from losses.trajectory_loss import TrajectoryLoss
    from einops import rearrange
    orig_init, orig_fwd, orig_loss = TrajectoryFlow.__init__, TrajectoryFlow.forward_diffusion_training, TrajectoryLoss.forward

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        self._t2k_dec_feat = None
        if cfg["dec"] >= 0:
            blocks = self.diffusion_decoder.cogvideox.transformer_blocks
            L = min(cfg["dec"], len(blocks) - 1)
            D = blocks[L].norm1.norm.normalized_shape[0] if hasattr(blocks[L], "norm1") else self.diffusion_decoder.cogvideox.config.num_attention_heads * self.diffusion_decoder.cogvideox.config.attention_head_dim
            self.t2k_dec_head = DecHead(D, cfg["hidden"])
            def hook(mod, inp, out):
                self._t2k_dec_feat = out[0] if isinstance(out, (tuple, list)) else out
            blocks[L].register_forward_hook(hook)
            print(f"[patch_t2k_couple] A1: DecHead on decoder block {L}/{len(blocks)} (D={D}), weight {cfg['wdec']}", flush=True)
        if cfg["kc"] > 0:
            print(f"[patch_t2k_couple] A3: keypose-consistency on x0_hat, weight {cfg['kc']}", flush=True)

    def forward_diffusion_training(self, images, texts, depth, is_depth_valid, target_trajectory, first_keypoint, diffusion_loss, attractor=None):
        self._t2k_dec_feat = None
        out = orig_fwd(self, images, texts, depth, is_depth_valid, target_trajectory, first_keypoint, diffusion_loss, attractor)
        b = getattr(self, "_t2k_batch", None)
        if b is None or "t2k_has" not in b or "t2k_obj_T_t0" not in b:
            return out
        dev = target_trajectory.device
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        traj = target_trajectory                                                       # [B,400,T+1,3] (px,px,m)
        Bn, N, T1, _ = traj.shape; T = T1 - 1
        parts = {}
        # ---- A1: decoder-feature entity head ---------------------------------------------------------------------
        if cfg["dec"] >= 0 and self._t2k_dec_feat is not None:
            feat = self._t2k_dec_feat.float()                                          # [B, T'*100, D]
            S = feat.shape[1]; Tg = S // (GRID_D * GRID_D)
            feat = feat.view(Bn, Tg, GRID_D, GRID_D, -1).mean(1).reshape(Bn, GRID_D * GRID_D, -1)
            K = b["t2k_K"].float()
            xyz_q = unproject(traj[:, :, 0, :].float(), K).view(Bn, GRID_Q, GRID_Q, 3)
            dv_q = (traj[:, :, 0, 2] > 0.05).float().view(Bn, GRID_Q, GRID_Q)
            om_q = b["t2k_obj_mask"].float().view(Bn, GRID_Q, GRID_Q)
            pool = lambda x: F.avg_pool2d(x.permute(0, 3, 1, 2) if x.dim() == 4 else x[:, None], 2).flatten(2).transpose(1, 2)
            xyz = pool(xyz_q); dv = pool(dv_q).squeeze(-1); om = pool(om_q).squeeze(-1)
            o = self.t2k_dec_head(feat, om, xyz, dv)
            l_dec, p = head_loss(o, b)
            if torch.isfinite(l_dec):
                out["t2k_dec_loss"] = l_dec; parts.update({f"dec_{k}": v for k, v in p.items()})
            else:
                parts["dec_SKIPPED"] = torch.tensor(1.0)
        # ---- A3: keypose consistency on the clean estimate -----------------------------------------------------------
        if cfg["kc"] > 0 and "t_float" in out:
            dec = self.diffusion_decoder
            vp, vt = out["noise_pred"].float(), out["noise_target"].float()             # [B,3,T,20,20]
            tgt = rearrange(traj[:, :, 1:, :].float().view(Bn, GRID_Q, GRID_Q, T, 3), "b h w t c -> b t c h w")
            x0 = rearrange(dec.normalize_act_data(tgt), "b t c h w -> b c t h w")
            tf = out["t_float"].view(Bn, 1, 1, 1, 1).float()
            x0hat = x0 + tf * (vt - vp)
            uvd = rearrange(dec.unnormalize_act_data(rearrange(x0hat, "b c t h w -> b t c h w")), "b t c h w -> b (h w) t c")
            l_kc, n = keypose_consistency(uvd, traj.float(), b, out["t_float"].float())
            if torch.isfinite(l_kc):
                out["t2k_kc_loss"] = l_kc; parts["kc"] = l_kc.detach(); parts["kc_n"] = torch.tensor(float(n))
            else:
                parts["kc_SKIPPED"] = torch.tensor(1.0)
        out["t2k_couple_parts"] = parts
        return out

    def loss_forward(self, predictions, targets, trajectory_mask=None):
        d = orig_loss(self, predictions, targets, trajectory_mask)
        for key, w in (("t2k_dec_loss", cfg["wdec"]), ("t2k_kc_loss", cfg["kc"])):
            t = predictions.get(key)
            if t is not None and w != 0.0 and torch.isfinite(t):
                d[key] = t.detach(); d["total_loss"] = d["total_loss"] + w * t
        return d

    # SAFETY NET (09-10, after gen1_A_w1 went NaN in ~250 steps): `clip_grad_norm_` is an AMPLIFIER, not a guard --
    # a non-finite total norm makes the clip coefficient non-finite, which multiplies EVERY gradient by NaN, so one bad
    # tensor anywhere poisons all weights in a single step. Zero the gradients instead, i.e. skip that step.
    _orig_clip = torch.nn.utils.clip_grad_norm_
    skipped = {"n": 0}

    def guarded_clip(parameters, max_norm, *a, **k):
        ps = [parameters] if torch.is_tensor(parameters) else list(parameters)
        n = _orig_clip(ps, max_norm, *a, **k)
        if not torch.isfinite(n):
            for p in ps:
                if p.grad is not None:
                    p.grad.zero_()
            skipped["n"] += 1
            if skipped["n"] <= 5 or skipped["n"] % 200 == 0:
                print(f"[patch_t2k_couple] non-finite gradient norm -> optimizer step skipped ({skipped['n']} so far)", flush=True)
        return n

    torch.nn.utils.clip_grad_norm_ = guarded_clip
    TrajectoryFlow.__init__, TrajectoryFlow.forward_diffusion_training, TrajectoryLoss.forward = __init__, forward_diffusion_training, loss_forward
    _APPLIED = True
    print(f"[patch_t2k_couple] active: {cfg}", flush=True)
    return True
