"""Trace-to-Keyframe head (T2K) -- an entity-level decoding head on top of TraceGen.

    MSGEN_T2K=1 [MSGEN_T2K_W=1.0] python -m msgen.run_train ...      (via msgen.patch_all)

What it adds (the dense trace prediction is untouched):
  * OBJECT ENTITY TOKEN   the 24x24 fused vision tokens (DINOv3 + SigLIP + depth, `vision_fusion`) pooled over the
                          patches covered by the object's queries (attention pooling). Membership: sim segmentation in
                          training (`obj_mask` of the _t2k labels), randomly swapped for the rule-based motion
                          segmentation of the GT trace (msgen.trace_seg) so the head sees what inference gives it.
  * STRUCTURE ATTENTION   the entity token cross-attends to ALL vision patches, each carrying its camera-frame xyz
                          (depth average-pooled to the patch grid, unprojected with K) and a depth-validity bit.
  * OUTPUTS (camera frame) keyframes: K_MAX=4 x (validity logit, step, translation, 6-D rotation)
                          twists: 3 segments x (type logits [line/revolute/screw/still], axis, theta, d)
                          contact: point c (3), normal n (3) in the OBJECT frame, mode logits (none/grasp/push), precision
                          structure frame: translation + 6-D rotation;  terminal pose relative to it: translation + 6-D
  * LOSSES                masked L1 / geodesic on poses, CE on discrete outputs, cosine on axes; folded into total_loss
                          with weight MSGEN_T2K_W (0 -> bit-identical to the unpatched model, the identity test).
Labels come from samples/<stem>_t2k.npz (msgen.labels_t2k); samples without one contribute no head loss.
The batch reaches the model through a thin loader proxy that stashes it on the model (the trainer's call signature is
untouched). At inference, `predict_trajectory` runs the head with the object mask segmented from the PREDICTED trace
and stashes `model._t2k_out` (numpy per sample) for msgen.predict to save.
"""
from __future__ import annotations
import fnmatch
import os
import numpy as np

_APPLIED = False
_W_PRINTED = False
TRAJ_SEL = (8, 16, 24, 32)   # variant B: per-patch trace displacements at these steps + terminal magnitude
K_MAX, N_SEG, GRID_Q, GRID_P = 4, 3, 20, 24
# Production-protocol intrinsics (384x384, FOV 0.754) -- identical across every label in realcam_t2k_n3000 (verified).
# Eval banks carry no _t2k.npz label, so the missing-label fallback MUST still supply the true K: with an
# identity K the patch-grid unprojection collapses and the head sees garbage geometry at inference.
K_PROD = np.array([[484.92407, 0.0, 192.0], [0.0, 484.92407, 192.0], [0.0, 0.0, 1.0]], np.float32)
T2K_KEYS = ("kf_T_cam", "kf_step", "kf_kind", "tw_type", "tw_axis_cam", "tw_mag", "contact_c", "contact_n", "contact_mode",
            "contact_precise", "contact_step", "struct_T_cam", "rel_end_T", "obj_mask", "struct_mask", "K")


def _load_t2k(npz_path):
    """samples/<stem>.npz -> fixed-size numpy dict (zeros + has=0 when the label file is missing)."""
    p = str(npz_path)[:-4] + "_t2k.npz"
    out = dict(has=np.int64(0), kf_T=np.zeros((K_MAX, 4, 4), np.float32), kf_valid=np.zeros(K_MAX, np.float32), kf_step=np.zeros(K_MAX, np.float32),
               kf_kind=np.zeros(K_MAX, np.int64), tw_type=np.full(N_SEG, 3, np.int64), tw_axis=np.zeros((N_SEG, 3), np.float32), tw_mag=np.zeros((N_SEG, 2), np.float32),
               tw_valid=np.zeros(N_SEG, np.float32), contact_c=np.zeros(3, np.float32), contact_n=np.zeros(3, np.float32), contact_mode=np.int64(0),
               contact_precise=np.float32(0), contact_valid=np.float32(0), struct_T=np.eye(4, dtype=np.float32), rel_T=np.eye(4, dtype=np.float32),
               obj_mask=np.zeros(GRID_Q * GRID_Q, np.float32), struct_mask=np.zeros(GRID_Q * GRID_Q, np.float32), K=K_PROD.copy(),
               obj_T_t0=np.eye(4, dtype=np.float32),            # camera-frame object pose at t0 (patch_t2k_couple A3)
               gmap=np.zeros(576, np.float32), gmap_valid=np.float32(0),
               gmap48=np.zeros(2304, np.float32), gmap48_valid=np.float32(0),
               amap=np.zeros(576, np.float32), amap_valid=np.float32(0),
               smask24=np.zeros(576, np.float32), omask24=np.zeros(576, np.float32), ent_valid=np.float32(0))
    if not os.path.exists(p):
        return out
    z = np.load(p, allow_pickle=True)
    if "kf_T_cam" not in z.files:                      # first-pass labels (world frame only): treat as missing
        return out
    k = min(len(z["kf_step"]), K_MAX)
    out["has"] = np.int64(1)
    if k:
        out["kf_T"][:k] = z["kf_T_cam"][:k]; out["kf_valid"][:k] = 1.0; out["kf_step"][:k] = z["kf_step"][:k]; out["kf_kind"][:k] = z["kf_kind"][:k]
    s = min(len(z["tw_type"]), N_SEG)
    if s:
        out["tw_type"][:s] = z["tw_type"][:s]; out["tw_axis"][:s] = z["tw_axis_cam"][:s]; out["tw_mag"][:s] = z["tw_mag"][:s]; out["tw_valid"][:s] = 1.0
    if "obj_T_t0_cam" in z.files: out["obj_T_t0"] = z["obj_T_t0_cam"].astype(np.float32)
    out["contact_c"] = z["contact_c"].astype(np.float32); out["contact_n"] = z["contact_n"].astype(np.float32)
    out["contact_mode"] = np.int64(z["contact_mode"]); out["contact_precise"] = np.float32(z["contact_precise"])
    out["contact_valid"] = np.float32(1.0 if float(z["contact_step"]) >= 0 else 0.0)
    out["struct_T"] = z["struct_T_cam"].astype(np.float32); out["rel_T"] = z["rel_end_T"].astype(np.float32)
    out["obj_mask"] = z["obj_mask"].astype(np.float32); out["struct_mask"] = z["struct_mask"].astype(np.float32); out["K"] = z["K"].astype(np.float32)
    if "gmap" in z.files:
        gm = z["gmap"].astype(np.float32).ravel()
        sm = float(gm.sum())
        if sm > 1e-6:
            out["gmap"] = gm / sm; out["gmap_valid"] = np.float32(1.0)
    if "gmap48" in z.files:
        gm = z["gmap48"].astype(np.float32).ravel()
        sm = float(gm.sum())
        if sm > 1e-6:
            out["gmap48"] = gm / sm; out["gmap48_valid"] = np.float32(1.0)
    if "amap" in z.files:
        am = z["amap"].astype(np.float32).ravel()
        sa = float(am.sum())
        if sa > 1e-6:
            out["amap"] = am / sa; out["amap_valid"] = np.float32(1.0)
    if "smask24" in z.files:
        out["smask24"] = z["smask24"].astype(np.float32).ravel()
        out["omask24"] = z["omask24"].astype(np.float32).ravel()
        out["ent_valid"] = np.float32(1.0)
    return out


def patch_t2k(weight: float = 1.0, hidden: int = 256, swap_prob: float = 0.5, zoom: bool = False, traj: bool = False,
              traj_noise: float = 1.0, gmap: int = 0):
    global _APPLIED
    traj_on = bool(traj)
    gmap_on = bool(gmap)   # goal map: per-patch heatmap head
    gmap2 = int(gmap) == 2  # v2: 4 logits/token -> row-major 48x48 (8px cells)
    gbank = int(gmap) == 3  # v3 bank: gmap + amap(appearance disc) + smask/omask(entity)
    if _APPLIED:
        return
    import torch, torch.nn as nn, torch.nn.functional as F
    from losses.trajectory_loss import TrajectoryLoss
    from models.model_flow import TrajectoryFlow
    from dataio.datasets import EpisodePointDataset
    from trainer.trainer import TrajectoryDiffusionTrainer
    from dataio.datasets import PointDatasetCollator
    orig_collate = PointDatasetCollator.__call__
    orig_init, orig_fwd, orig_pred, orig_loss = TrajectoryFlow.__init__, TrajectoryFlow.forward_diffusion_training, TrajectoryFlow.predict_trajectory, TrajectoryLoss.forward
    orig_getitem, orig_trainer_init = EpisodePointDataset.__getitem__, TrajectoryDiffusionTrainer.__init__

    # ---------------------------------------------------------------- dataset: add the labels as fixed-size keys
    def __getitem__(self, idx):
        d = orig_getitem(self, idx)
        try:
            episode_idx, sample_idx = self.sample_indices[idx]
            _, npz_path, depth_path = self.episode_metadata[episode_idx]["valid_pairs"][sample_idx]
            t = _load_t2k(npz_path)
            depth_m = np.load(depth_path)["depth"].astype(np.float32) if depth_path is not None else np.zeros((384, 384), np.float32)
        except Exception:
            t = _load_t2k("__missing__.npz"); depth_m = np.zeros((384, 384), np.float32)
        for k, v in t.items():
            d[f"t2k_{k}"] = torch.as_tensor(v)
        d["t2k_depth_m"] = torch.as_tensor(depth_m)
        return d

    # ---------------------------------------------------------------- collator: the stock one builds a FIXED key set; pass the labels through
    def collate(self, batch):
        out = orig_collate(self, batch)
        for k in batch[0]:
            if k.startswith("t2k_") and all(k in b for b in batch):
                out[k] = torch.stack([torch.as_tensor(b[k]) for b in batch])
        return out

    # ---------------------------------------------------------------- model head
    def rot6d_to_R(x):
        a, b = x[..., :3], x[..., 3:6]
        a = F.normalize(a, dim=-1); b = b - (a * b).sum(-1, keepdim=True) * a; b = F.normalize(b, dim=-1); c = torch.cross(a, b, dim=-1)
        return torch.stack([a, b, c], dim=-1)

    def geodesic(R1, R2):
        c = ((R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))

    class T2KHead(nn.Module):
        def __init__(self, D, hidden, pos_in=4):
            super().__init__()
            self.pool_q = nn.Linear(D, 1)                                    # attention pooling over object patches
            self.pos = nn.Sequential(nn.Linear(pos_in, hidden), nn.GELU(), nn.Linear(hidden, D))   # xyz_cam + validity [+ traj skeleton] -> D
            self.xattn = nn.MultiheadAttention(D, 8, batch_first=True)
            self.norm1, self.norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
            self.mlp = nn.Sequential(nn.Linear(2 * D, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
            self.kf = nn.Linear(hidden, K_MAX * (1 + 1 + 3 + 6))            # valid, step, t, 6d
            self.tw = nn.Linear(hidden, N_SEG * (4 + 3 + 2))                  # type logits, axis, (theta, d)
            self.ct = nn.Linear(hidden, 3 + 3 + 3 + 1)                        # c, n, mode logits, precise
            self.st = nn.Linear(hidden, 9 + 9)                                # struct (t, 6d), rel (t, 6d)
            if gmap_on:
                # learned grounding heatmap over the vision patches (separate from xattn:
                # feature pooling and spatial localisation must not share one distribution
                # -- the unsupervised-attention probe failed exactly there)
                # v2: 4 logits/token = 2x2 sub-cells -> 48x48 map (8px), for sub-16px readouts
                self.gmap = nn.Linear(D, 4 if gmap2 else 1)
            if gbank:
                # v3 grounding bank: appearance-target heat (RGB-defined/aerial destinations),
                # structure & object entity masks (learned evidence gate + pooling decoupled
                # from the trace-motion rule mask)
                self.amap = nn.Linear(D, 1)
                self.smask = nn.Linear(D, 1)
                self.omask = nn.Linear(D, 1)

        def forward(self, vis, obj_patch_mask, xyz_cam, dvalid, traj_feat=None):
            """vis [B,P,D]; obj_patch_mask [B,P] float; xyz_cam [B,P,3]; dvalid [B,P]; traj_feat [B,P,13] (variant B)."""
            w = self.pool_q(vis).squeeze(-1).masked_fill(obj_patch_mask < 0.5, -1e4)
            w = torch.softmax(w, dim=1)
            ent = (w.unsqueeze(-1) * vis).sum(1, keepdim=True)              # [B,1,D]
            pin = [xyz_cam, dvalid.unsqueeze(-1)]
            if traj_feat is not None:
                pin.append(traj_feat)
            kv = self.norm1(vis + self.pos(torch.cat(pin, -1)))
            att, attw = self.xattn(ent, kv, kv, need_weights=True)           # [B,1,D], [B,1,P]
            h = self.mlp(torch.cat([self.norm2(ent + att), ent], -1)).squeeze(1)
            B = h.shape[0]
            kf = self.kf(h).view(B, K_MAX, 11); tw = self.tw(h).view(B, N_SEG, 9); ct = self.ct(h); st = self.st(h)
            d = dict(kf_valid=kf[..., 0], kf_step=kf[..., 1] * 32.0, kf_t=kf[..., 2:5], kf_R=rot6d_to_R(kf[..., 5:11]),
                     tw_type=tw[..., :4], tw_axis=F.normalize(tw[..., 4:7], dim=-1), tw_mag=tw[..., 7:9],
                     contact_c=ct[:, :3], contact_n=F.normalize(ct[:, 3:6], dim=-1), contact_mode=ct[:, 6:9], contact_precise=ct[:, 9],
                     struct_t=st[:, :3], struct_R=rot6d_to_R(st[:, 3:9]), rel_t=st[:, 9:12], rel_R=rot6d_to_R(st[:, 12:18]), attn=attw.squeeze(1))
            if gmap_on:
                g = self.gmap(kv)                                            # [B,P,1|4]
                if gmap2:
                    B_ = g.shape[0]
                    # [B,576,4] -> [B,24,24,2,2] -> row-major [B,48,48] -> flat [B,2304]
                    g = g.view(B_, 24, 24, 2, 2).permute(0, 1, 3, 2, 4).reshape(B_, 48 * 48)
                    d["gmap"] = g
                else:
                    d["gmap"] = g.squeeze(-1)                                # [B,576] logits
            if gbank:
                d["amap"] = self.amap(kv).squeeze(-1)
                d["smask"] = self.smask(kv).squeeze(-1)
                d["omask"] = self.omask(kv).squeeze(-1)
            return d

    class T2KZoom(nn.Module):
        """Coarse-to-fine metric refinement (option 2). The 24x24 avg-pooled depth dilutes small
        targets (the pickcube marker disc loses 50-90% of its metric signal before reaching the head), so the
        head regresses range from priors -- measured 84%-along-ray error. This module re-reads the FULL-RES
        depth in a 64x64 crop at the coarse struct_t's projection and predicts a residual correction.
        refined = coarse.detach() + delta: the zoom trains on its own loss part without disturbing the head."""
        CROP = 64

        def __init__(self, hidden=128):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(1, 16, 5, 2, 2), nn.GELU(),
                                      nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
                                      nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(4))
            self.fc = nn.Sequential(nn.Linear(64 * 16 + 3, hidden), nn.GELU(), nn.Linear(hidden, 3))

        def forward(self, depth_m, K, struct_t):
            B, H, Wd = depth_m.shape
            h = self.CROP // 2
            with torch.no_grad():
                z = struct_t[:, 2].clamp(0.15, 3.0)
                u = (K[:, 0, 0] * struct_t[:, 0] / z + K[:, 0, 2]).round().long().clamp(h, Wd - h)
                v = (K[:, 1, 1] * struct_t[:, 1] / z + K[:, 1, 2]).round().long().clamp(h, H - h)
            crops = torch.stack([depth_m[i, v[i]-h:v[i]+h, u[i]-h:u[i]+h] for i in range(B)]).clamp(0, 3.0)
            f = self.conv(crops.unsqueeze(1)).flatten(1)
            delta = self.fc(torch.cat([f, struct_t.detach()], -1))
            return struct_t.detach() + delta

    def _patch_geometry(depth_m, K, dvalid_img):
        """depth [B,H,W] metres, K [B,3,3] -> per-patch camera xyz [B,P,3] and validity [B,P] on the 24x24 grid."""
        B, H, W = depth_m.shape
        d = F.adaptive_avg_pool2d(depth_m.unsqueeze(1), (GRID_P, GRID_P)).squeeze(1)          # [B,24,24]
        v = F.adaptive_avg_pool2d(dvalid_img.unsqueeze(1).float(), (GRID_P, GRID_P)).squeeze(1)
        ys, xs = torch.meshgrid(torch.arange(GRID_P, device=depth_m.device), torch.arange(GRID_P, device=depth_m.device), indexing="ij")
        u = (xs.float() + 0.5) * (W / GRID_P); vv = (ys.float() + 0.5) * (H / GRID_P)
        fx, fy, cx, cy = K[:, 0, 0, None, None], K[:, 1, 1, None, None], K[:, 0, 2, None, None], K[:, 1, 2, None, None]
        x = (u[None] - cx) / fx * d; y = (vv[None] - cy) / fy * d
        xyz = torch.stack([x, y, d], -1).reshape(B, -1, 3); return xyz, v.reshape(B, -1)

    def _q_to_patch(mask_q):
        """[B,400] query mask on the 20x20 lattice -> [B,576] patch mask on the 24x24 grid (nearest)."""
        B = mask_q.shape[0]
        m = mask_q.view(B, 1, GRID_Q, GRID_Q)
        return (F.interpolate(m, size=(GRID_P, GRID_P), mode="nearest") > 0.5).float().view(B, -1)

    def _traj_patch_feat(traj_like, is_target, noisy):
        """trace in the MODEL's convention -> per-patch skeleton features [B,576,13]:
        displacements from start at steps TRAJ_SEL (4x3, 20x20 lattice nearest-mapped to the 24x24 patch grid)
        + terminal displacement magnitude (1). is_target: [:, :, 0] is absolute (GT), deltas are [:, :, 1:];
        else every step is a delta. noisy: add per-sample step-scaled gaussian noise (train/test-gap treatment,
        sigma at the terminal = (0.06, 0.06, 0.03) in the model's units, amplitude U(0, traj_noise))."""
        with torch.no_grad():
            t = torch.nan_to_num(traj_like.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            d = t[:, :, 1:, :] if is_target else t[:, :, -32:, :]                       # [B,400,32,3] per-step deltas
            B = d.shape[0]
            feats = []
            cum = d.cumsum(2)                                                           # displacement from start
            for s in TRAJ_SEL:
                feats.append(cum[:, :, min(s, cum.shape[2]) - 1, :])
            f = torch.stack(feats, 2)                                                   # [B,400,4,3]
            if noisy > 0:
                amp = torch.rand(B, 1, 1, 1, device=f.device) * noisy
                sig = torch.tensor([0.06, 0.06, 0.03], device=f.device).view(1, 1, 1, 3)
                scale = (torch.tensor([float(s) for s in TRAJ_SEL], device=f.device) / 32.0).view(1, 1, -1, 1)
                f = f + torch.randn_like(f) * sig * scale * amp
            mag = f[:, :, -1, :].norm(dim=-1, keepdim=True)                             # [B,400,1]
            f = torch.cat([f.reshape(B, 400, -1), mag], -1)                             # [B,400,13]
            g = f.view(B, GRID_Q, GRID_Q, 13).permute(0, 3, 1, 2)
            g = F.interpolate(g, size=(GRID_P, GRID_P), mode="nearest")
            return g.permute(0, 2, 3, 1).reshape(B, GRID_P * GRID_P, 13)

    def _rule_mask_from_traj(traj, first_xy=None, is_target=False):
        """Object mask by the motion rule (late-onset movers, msgen.trace_seg) from a trace in the MODEL's convention:
        per-step DELTAS in normalised units (x, y) and metres (z). `is_target`: step 0 is included and absolute (GT
        trajectory); otherwise step 0 comes from `first_xy` ([B,N,2+] normalised) as msgen.predict.to_absolute does.
        Samples whose rule mask is empty fall back to the 16 largest movers, so the entity pooling never sees an empty set."""
        try:
            from msgen.trace_seg import segment
            with torch.no_grad():
                t = torch.nan_to_num(traj.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                if is_target:
                    ab = t.cumsum(2)
                else:
                    f0 = first_xy.detach().float()[:, :, :3] if first_xy is not None else torch.zeros_like(t[:, :, 0, :])
                    if f0.shape[-1] < 3: f0 = torch.cat([f0, torch.zeros_like(f0[..., :1])], -1)
                    ab = torch.cat([f0[:, :, None, :], t], 2).cumsum(2)
                ab = ab.cpu().numpy(); ab[..., :2] *= 384.0
            out = []
            for b in range(ab.shape[0]):
                p = ab[b]
                sf = segment(p); m = sf["obj_mask"].astype(np.float32)
                if m.sum() < 3:
                    disp = np.linalg.norm(p[:, -1, :2] - p[:, 0, :2], axis=-1); m = np.zeros(p.shape[0], np.float32); m[np.argsort(-disp)[:16]] = 1.0
                out.append(m)
            return torch.as_tensor(np.stack(out), device=traj.device)
        except Exception as ex:
            print(f"[patch_t2k] rule mask failed: {ex!r}", flush=True)
            return None

    def _head_loss(o, b, valid_img):
        """head outputs vs stashed batch labels (camera frame). Returns (loss, dict of detached parts)."""
        has = b["t2k_has"].float(); n = has.sum().clamp_min(1.0); parts = {}
        def m(x, w): return (x * w).sum() / w.sum().clamp_min(1.0)
        # keyframes
        kv = b["t2k_kf_valid"] * has[:, None]
        l_kfv = F.binary_cross_entropy_with_logits(o["kf_valid"], b["t2k_kf_valid"], reduction="none"); parts["kf_valid"] = m(l_kfv, has[:, None].expand_as(l_kfv))
        parts["kf_step"] = m((o["kf_step"] - b["t2k_kf_step"]).abs() / 32.0, kv)
        parts["kf_t"] = m((o["kf_t"] - b["t2k_kf_T"][:, :, :3, 3]).abs().sum(-1), kv)
        parts["kf_R"] = m(geodesic(o["kf_R"], b["t2k_kf_T"][:, :, :3, :3]), kv)
        # twists
        tv = b["t2k_tw_valid"] * has[:, None]
        l_tt = F.cross_entropy(o["tw_type"].reshape(-1, 4), b["t2k_tw_type"].reshape(-1), reduction="none").view_as(tv); parts["tw_type"] = m(l_tt, tv)
        parts["tw_axis"] = m(1 - (o["tw_axis"] * b["t2k_tw_axis"]).sum(-1).abs(), tv * (b["t2k_tw_type"] != 3).float())
        parts["tw_mag"] = m((o["tw_mag"] - b["t2k_tw_mag"]).abs().sum(-1), tv)
        # contact
        cv = b["t2k_contact_valid"] * has
        parts["ct_c"] = m((o["contact_c"] - b["t2k_contact_c"]).abs().sum(-1), cv); parts["ct_n"] = m(1 - (o["contact_n"] * b["t2k_contact_n"]).sum(-1), cv)
        parts["ct_mode"] = m(F.cross_entropy(o["contact_mode"], b["t2k_contact_mode"], reduction="none"), cv)
        parts["ct_prec"] = m(F.binary_cross_entropy_with_logits(o["contact_precise"], b["t2k_contact_precise"], reduction="none"), cv)
        # structure + relative terminal pose (the goal)
        parts["st_t"] = m((o["struct_t"] - b["t2k_struct_T"][:, :3, 3]).abs().sum(-1), has); parts["st_R"] = m(geodesic(o["struct_R"], b["t2k_struct_T"][:, :3, :3]), has)
        parts["rel_t"] = m((o["rel_t"] - b["t2k_rel_T"][:, :3, 3]).abs().sum(-1), has); parts["rel_R"] = m(geodesic(o["rel_R"], b["t2k_rel_T"][:, :3, :3]), has)
        if gmap_on:
            lk = "t2k_gmap48" if gmap2 else "t2k_gmap"
            gv = b[lk + "_valid"].float() * has
            ce = -(b[lk] * F.log_softmax(o["gmap"], dim=-1)).sum(-1)
            parts["gmap"] = m(ce, gv)
        if gbank:
            av = b["t2k_amap_valid"].float() * has
            parts["amap"] = m(-(b["t2k_amap"] * F.log_softmax(o["amap"], dim=-1)).sum(-1), av)
            ev = b["t2k_ent_valid"].float() * has
            parts["smask"] = m(F.binary_cross_entropy_with_logits(o["smask"], b["t2k_smask24"].clamp(0, 1), reduction="none").mean(-1), ev)
            parts["omask"] = m(F.binary_cross_entropy_with_logits(o["omask"], b["t2k_omask24"].clamp(0, 1), reduction="none").mean(-1), ev)
        # translations in metres (L1 sum over xyz), rotations in radians: weights chosen so each part starts O(0.1-1); the
        # overall scale vs the diffusion loss (~0.01) is MSGEN_T2K_W (default 0.3 -- the head must shape vision_fusion, not own it)
        W = dict(kf_valid=0.5, kf_step=0.5, kf_t=2.0, kf_R=0.5, tw_type=0.5, tw_axis=1.0, tw_mag=2.0, ct_c=2.0, ct_n=0.5, ct_mode=0.5, ct_prec=0.2, st_t=2.0, st_R=0.5, rel_t=3.0, rel_R=0.5)
        if gmap_on:
            W["gmap"] = 0.3   # CE vs a 576-way distribution starts O(6); 0.3 puts it in the family band
        if gbank:
            W["amap"] = 0.3; W["smask"] = 0.2; W["omask"] = 0.2
        # per-part ablation override, e.g. MSGEN_T2K_WPARTS="kf_*=0,tw_*=0,ct_c=0" (fnmatch on part names);
        # a zeroed part is still computed and printed but contributes no gradient.
        for _spec in filter(None, os.environ.get("MSGEN_T2K_WPARTS", "").split(",")):
            _pat, _val = _spec.split("=")
            _hit = [k for k in W if fnmatch.fnmatch(k, _pat)]
            assert _hit, f"MSGEN_T2K_WPARTS pattern {_pat!r} matches no loss part"
            for _k in _hit:
                W[_k] = float(_val)
        global _W_PRINTED
        if not _W_PRINTED:
            _W_PRINTED = True
            print(f"[patch_t2k] loss weights: {W}", flush=True)
        loss = sum(W[k] * v for k, v in parts.items())
        return loss, {k: v.detach() for k, v in parts.items()}

    def __init__(self, cfg):
        orig_init(self, cfg)
        D = int(cfg.get("d_model", 768)) if isinstance(cfg, dict) else int(getattr(cfg, "d_model", 768))
        self.t2k_head = T2KHead(D, hidden, pos_in=17 if traj_on else 4); self._t2k_batch = None; self._t2k_out = None
        if zoom:
            self.t2k_zoom = T2KZoom()

    def forward_diffusion_training(self, images, texts, depth, is_depth_valid, target_trajectory, first_keypoint, diffusion_loss, attractor=None):
        device = images.device
        vision_features = self.encode_images(images, depth, is_depth_valid)
        text_features = self.encode_t5_texts(texts, device)
        combined = torch.cat([vision_features, text_features], dim=1)
        out = self.diffusion_decoder.forward_diffusion_training(trunk_conditioning=combined, target_video=target_trajectory, diffusion_loss=diffusion_loss, attractor=attractor)
        b = self._t2k_batch
        if not getattr(self, "_t2k_checked", False):
            self._t2k_checked = True
            print(f"[patch_t2k] first forward: batch stashed={b is not None}, labels in batch={b is not None and 't2k_has' in b}, "
                  f"labelled samples={int(b['t2k_has'].sum()) if (b is not None and 't2k_has' in b) else 0}/{len(b['t2k_has']) if (b is not None and 't2k_has' in b) else 0}", flush=True)
        if b is not None and weight != 0.0 and "t2k_has" in b:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items() if k.startswith("t2k_")}
            mask_q = b["t2k_obj_mask"]
            if swap_prob > 0 and self.training:
                rm = _rule_mask_from_traj(target_trajectory, is_target=True)
                if rm is not None:
                    sw = (torch.rand(mask_q.shape[0], device=device) < swap_prob).float()[:, None]
                    mask_q = sw * rm + (1 - sw) * mask_q
            valid_img = b["t2k_depth_m"] > 0.05
            xyz, dv = _patch_geometry(b["t2k_depth_m"], b["t2k_K"], valid_img)
            tf = _traj_patch_feat(target_trajectory, is_target=True, noisy=traj_noise if self.training else 0.0) if traj_on else None
            if traj_on and not getattr(self, "_t2k_traj_checked", False):
                self._t2k_traj_checked = True
                print(f"[patch_t2k] traj channel (train, GT+noise): mean|f|={float(tf.abs().mean()):.4f}", flush=True)
            o = self.t2k_head(vision_features, _q_to_patch(mask_q), xyz, dv, tf)
            loss, parts = _head_loss(o, b, valid_img)
            if zoom:
                refined = self.t2k_zoom(b["t2k_depth_m"], b["t2k_K"], o["struct_t"])
                has = b["t2k_has"].float()
                l_z = (((refined - b["t2k_struct_T"][:, :3, 3]).abs().sum(-1)) * has).sum() / has.sum().clamp_min(1.0)
                parts["st_zoom"] = l_z.detach(); loss = loss + 2.0 * l_z
            out["t2k_loss"] = loss; out["t2k_parts"] = parts
            self._t2k_n = getattr(self, "_t2k_n", 0) + 1
            if self._t2k_n % 200 == 1:
                print("[patch_t2k] parts " + " ".join(f"{k}={float(v):.3f}" for k, v in parts.items()), flush=True)
        return out

    def predict_trajectory(self, images, texts, depth, is_depth_valid, first_keypoint, noise_scheduler, num_inference_steps=100, guidance_scale=2.0, attractor=None):
        import inspect
        _kw = dict(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        if "attractor" in inspect.signature(orig_pred).parameters: _kw["attractor"] = attractor
        traj = orig_pred(self, images, texts, depth, is_depth_valid, first_keypoint, noise_scheduler, **_kw)
        self._t2k_out = None
        b = self._t2k_batch
        if b is not None and "t2k_depth_m" in b:
            with torch.no_grad():
                device = images.device
                vis = self.encode_images(images, depth, is_depth_valid)
                mask_q = _rule_mask_from_traj(traj, first_xy=first_keypoint)                  # from the PREDICTED trace (deltas)
                if mask_q is None:
                    mask_q = b["t2k_obj_mask"].to(device)
                dm = b["t2k_depth_m"].to(device); Kc = b["t2k_K"].to(device)
                xyz, dv = _patch_geometry(dm, Kc, dm > 0.05)
                if os.environ.get("MSGEN_T2K_NODEPTH", "") not in ("", "0"):   # probe: blind the cross-attention geometry channel
                    xyz, dv = torch.zeros_like(xyz), torch.zeros_like(dv)
                tf = _traj_patch_feat(traj, is_target=False, noisy=0.0) if traj_on else None
                if traj_on and not getattr(self, "_t2k_traj_pred_checked", False):
                    self._t2k_traj_pred_checked = True
                    print(f"[patch_t2k] traj channel (predict, sampled): mean|f|={float(tf.abs().mean()):.4f}", flush=True)
                o = self.t2k_head(vis, _q_to_patch(mask_q.to(device)), xyz, dv, tf)
                if zoom and hasattr(self, "t2k_zoom"):
                    o["struct_t_zoom"] = self.t2k_zoom(dm.to(device), Kc.to(device), o["struct_t"])
                self._t2k_out = {k: v.detach().cpu().numpy() for k, v in o.items()}
                self._t2k_out["obj_mask_used"] = mask_q.cpu().numpy()
        return traj

    def loss_forward(self, predictions, targets, trajectory_mask=None):
        d = orig_loss(self, predictions, targets, trajectory_mask)
        t = predictions.get("t2k_loss")
        if t is not None and weight != 0.0:
            d["t2k_loss"] = t.detach(); d["total_loss"] = d["total_loss"] + weight * t
            for k, v in predictions.get("t2k_parts", {}).items():
                d[f"t2k_{k}"] = v
        return d

    # ---------------------------------------------------------------- trainer: loader proxy that stashes the batch on the model
    class _Proxy:
        def __init__(self, loader, model_getter): self.loader, self.model_getter = loader, model_getter
        def __len__(self): return len(self.loader)
        def __getattr__(self, k): return getattr(self.loader, k)
        def __iter__(self):
            for batch in self.loader:
                m = self.model_getter(); m._t2k_batch = batch
                yield batch

    def trainer_init(self, *a, **k):
        orig_trainer_init(self, *a, **k)
        def get():
            m = self.model.module if hasattr(self.model, "module") else self.model
            return getattr(m, "_orig_mod", m)          # torch.compile wraps the module; the head lives on the original
        self.train_loader = _Proxy(self.train_loader, get)
        if getattr(self, "val_loader", None) is not None:
            self.val_loader = _Proxy(self.val_loader, get)
        print("[patch_t2k] loader proxy installed (batch stashed on the model for the head)", flush=True)

    EpisodePointDataset.__getitem__ = __getitem__
    PointDatasetCollator.__call__ = collate
    TrajectoryFlow.__init__ = __init__
    TrajectoryFlow.forward_diffusion_training = forward_diffusion_training
    TrajectoryFlow.predict_trajectory = predict_trajectory
    TrajectoryLoss.forward = loss_forward
    TrajectoryDiffusionTrainer.__init__ = trainer_init
    _APPLIED = True
    print(f"[patch_t2k] Trace-to-Keyframe head added (weight {weight}, hidden {hidden}, mask swap p={swap_prob}); outputs in the camera frame", flush=True)


def maybe_patch():
    v = os.environ.get("MSGEN_T2K", "")
    if v and v != "0":
        patch_t2k(weight=float(os.environ.get("MSGEN_T2K_W", "0.3")), hidden=int(os.environ.get("MSGEN_T2K_HIDDEN", "256")),
                  zoom=os.environ.get("MSGEN_T2K_ZOOM", "0") not in ("0", "", None),
                  swap_prob=float(os.environ.get("MSGEN_T2K_SWAP", "0.5")),
                  traj=os.environ.get("MSGEN_T2K_TRAJ", "0") not in ("0", "", None),
                  traj_noise=float(os.environ.get("MSGEN_T2K_TRAJ_NOISE", "1.0")),
                  gmap=int(os.environ.get("MSGEN_T2K_GMAP", "0") or 0))
        return True
    return False
