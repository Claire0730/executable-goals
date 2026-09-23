"""Centred-residual auxiliary loss: penalise the SHAPE of the object's motion, not its offset.

WHAT IT OPTIMISES, and why that is not what the base loss optimises.
For object points p_i taken relative to their own centroid, a rigid motion gives a displacement
    d_i = (R - I) p_i + t
whose mean over those points is exactly t. Subtracting the mean therefore removes the common
translation EXACTLY -- algebraically, for any common vector, whether or not it is the true one:
    d_i - mean(d) = (R - I) p_i
what remains depends only on rotation and shape. This loss compares the two sides after that
subtraction:
    L_shape = mean_i || (d_hat_i - mean d_hat) - (d*_i - mean d*) ||^2
so it is completely blind to the goal's translation error and spends all of its gradient on the
component the base MSE barely sees. Measured on this project's own numbers: the common
translation is 82-92% of the per-point error by magnitude (67-84% by energy), and the object
occupies only ~1.6% of the 400 grid points, so the shape term carries roughly 0.06-0.4% of the
base objective's gradient energy. That is the defect, and the reason lambda has to be large.

It is the project's OWN diagnostic turned into a training target. the 'common translation' in the planner
report is |mean of the per-point error| (s1_se3viz.py:115) and the 'shape residual' is what is left after
removing it; the report's 'translation energy share' column is exactly (common/total)^2, which reproduces to
two decimals on all four tasks. So the metric being optimised and the metric being trusted are
the same object -- no risk of moving A while B stays put.

IT IS NOT KABSCH. Centring is Kabsch's first step, but this stops there: it never solves for R,
so it never differentiates an SVD. That matters because 34.4% of PegInsertionSide samples have
sigma2/sigma1 < 0.05, exactly where SVD gradients blow up. Skipping the SVD removes the
NUMERICAL hazard; it does not remove the identifiability one -- with 3-4 object points the
centred pattern spans a degenerate subspace and rotation about the unspanned axis has no
gradient. Samples with fewer than MIN_PTS object points are dropped from the term.

RECOVERING THE PREDICTED TRAJECTORY WITHOUT xt.
The interpolant is linear: xt = (1-t) x_clean + t eps, and the target velocity is
v_true = -x_clean + eps, so xt = x_clean + t v_true and the one-step estimate is
x_hat = xt - t v_hat = x_clean + t (v_true - v_hat). Both velocities are already in the loss --
TrajectoryLoss.forward rearranges them to [B, 400, T, 3] and the decoder has already undone the
put_frames_in_channels grouping -- so the only extra tensors needed are t and the normalisation
scale/bias. No layout surgery, no second forward pass.

WORLD COORDINATES ARE MANDATORY. The trace's three channels are (pixel_x, pixel_y, camera depth
in metres), not xyz. A world translation is NOT a constant pixel offset -- the pixel shift of a
point depends on its own (u, v, z) through the perspective division -- so centring in the native
channel space would not cancel the common translation and the whole mechanism would fail. K and
the extrinsic therefore have to reach the loss, which is why msgen/labels.py now writes them.

NO NEW PARAMETERS. Unlike vispos/qfeat/movehead/cfg this patch adds nothing to the state dict,
so there is no trained-with/inferred-with contract, no SENTINELS entry, and inference is
byte-identical to the baseline. The observable proof that it was active is that the training log
shows Loss != Diffusion.

    MSGEN_ROTLOSS=300 python -m msgen.run_train --dataset ... --tag ...

lambda: the shape term carries 0.06-0.4% of the base gradient energy, so lambda in the low
hundreds to low thousands is what makes the two comparable. 0 (the default) leaves the patch
inert, so importing it is always safe.
"""
from __future__ import annotations

import os

_APPLIED = False
MIN_PTS = 3


def _lam() -> float:
    try:
        return float(os.environ.get("MSGEN_ROTLOSS", "0") or 0)
    except ValueError:
        return 0.0


def _patch_decoder():
    """Return t and the normalisation constants alongside the velocities.

    Line-based surgery, anchored on a unique token and taking the indentation FROM the matched
    line: `inspect.getsource` on a method plus `textwrap.dedent` shifts every line left by the
    class indent, so any anchor written with the file's own indentation misses.
    """
    import inspect
    import textwrap

    from models.decoder.cogvideox_flow import CogVideoXDecoder_flow

    src = textwrap.dedent(inspect.getsource(
        CogVideoXDecoder_flow.forward_diffusion_training)).splitlines()
    hits = [i for i, l in enumerate(src) if "'actions_norm': latent_video" in l]
    if len(hits) != 1:
        raise SystemExit(f"[patch_rotloss] decoder anchor matched {len(hits)} lines -- upstream "
                         f"changed; refusing to patch rather than silently training without "
                         f"the term")
    i = hits[0]
    ind = src[i][:len(src[i]) - len(src[i].lstrip())]
    if not src[i].rstrip().endswith(","):
        src[i] = src[i].rstrip() + ","
    src[i + 1:i + 1] = [f"{ind}'t_float': t_float,",
                        f"{ind}'act_scale': self.data_act_scale,",
                        f"{ind}'act_bias': self.data_act_bias,"]
    ns = {}
    exec(compile("\n".join(src), "<patch_rotloss:decoder>", "exec"),
         CogVideoXDecoder_flow.forward_diffusion_training.__globals__, ns)
    CogVideoXDecoder_flow.forward_diffusion_training = ns["forward_diffusion_training"]


def _patch_dataset():
    """Attach obj_mask / K / ext to every sample, from the npz labels.py now writes."""
    import numpy as np
    import torch

    from dataio.datasets import EpisodePointDataset

    orig = EpisodePointDataset.__getitem__
    missing = {"n": 0}

    def __getitem__(self, idx):
        s = orig(self, idx)
        if not isinstance(s, dict):
            return s
        try:
            e, k = self.sample_indices[idx]
            npz = self.episode_metadata[e]["valid_pairs"][k][1]
            z = np.load(npz)
            if "obj_mask" in z.files:
                s["obj_mask"] = torch.as_tensor(z["obj_mask"], dtype=torch.bool)
                s["K"] = torch.as_tensor(np.asarray(z["K"]), dtype=torch.float32)
                s["ext"] = torch.as_tensor(np.asarray(z["ext"]), dtype=torch.float32)
            else:
                missing["n"] += 1
                if missing["n"] == 1:
                    print(f"[patch_rotloss] {npz} has no obj_mask -- augment this dataset "
                          f"with object masks first; "
                          f"those samples contribute nothing to the shape term")
        except Exception as exc:                       # a bad row must not kill training
            if missing["n"] == 0:
                print(f"[patch_rotloss] sample {idx}: {exc}")
            missing["n"] += 1
        return s

    EpisodePointDataset.__getitem__ = __getitem__


def _patch_collator():
    """Keep the three new keys through collation.

    `PointDatasetCollator.__call__` (datasets.py:775-789) does not forward the sample dict --
    it builds an explicit return dict key by key, so anything the dataset adds is DROPPED
    silently and the loss sees nothing. Measured: without this the shape term never fires and
    the training log shows Loss == Diffusion with no error anywhere.
    """
    import torch

    from dataio.datasets import PointDatasetCollator

    orig = PointDatasetCollator.__call__

    def __call__(self, batch):
        out = orig(self, batch)
        for k in ("obj_mask", "K", "ext"):
            if batch and k in batch[0]:
                out[k] = torch.stack([b[k] for b in batch])
        return out

    PointDatasetCollator.__call__ = __call__


def _patch_trainer():
    """Forward the three new batch keys into `targets`, which is where the loss can see them.

    Same line-based, indentation-derived anchoring as _patch_decoder, for the same reason.
    """
    import inspect
    import textwrap

    from trainer.trainer import TrajectoryDiffusionTrainer

    src = textwrap.dedent(inspect.getsource(
        TrajectoryDiffusionTrainer.train_epoch)).splitlines()
    hits = [i for i, l in enumerate(src) if "'trajectory_mask': trajectory_mask" in l]
    if len(hits) != 1:
        raise SystemExit(f"[patch_rotloss] trainer anchor matched {len(hits)} lines -- "
                         f"upstream changed")
    i = hits[0]
    while i < len(src) and src[i].strip() != "}":
        i += 1
    if i >= len(src):
        raise SystemExit("[patch_rotloss] trainer targets dict has no closing brace")
    ind = src[i][:len(src[i]) - len(src[i].lstrip())]
    src[i + 1:i + 1] = [f"{ind}for _k in ('obj_mask', 'K', 'ext'):",
                        f"{ind}    if _k in batch:",
                        f"{ind}        targets[_k] = batch[_k].to(self.device)"]
    ns = {}
    exec(compile("\n".join(src), "<patch_rotloss:trainer>", "exec"),
         TrajectoryDiffusionTrainer.train_epoch.__globals__, ns)
    TrajectoryDiffusionTrainer.train_epoch = ns["train_epoch"]


def _unproject(uvz, K, E):
    """(u, v, z_cam) [B,N,T,3] -> world xyz, inverting peg_perception._project."""
    import torch
    u, v, z = uvz[..., 0], uvz[..., 1], uvz[..., 2]
    fx = K[:, 0, 0].view(-1, 1, 1)
    fy = K[:, 1, 1].view(-1, 1, 1)
    cx = K[:, 0, 2].view(-1, 1, 1)
    cy = K[:, 1, 2].view(-1, 1, 1)
    pc = torch.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], dim=-1)
    R = E[:, :, :3]                       # world -> camera
    t = E[:, :, 3]
    # p_cam = R p_world + t  =>  p_world = R^T (p_cam - t)
    return torch.einsum("bji,bntj->bnti", R, pc - t[:, None, None, :])


def _patch_loss(lam: float):
    import torch
    from einops import rearrange

    from losses.trajectory_loss import TrajectoryLoss

    orig = TrajectoryLoss.forward

    def forward(self, predictions, targets, trajectory_mask=None):
        out = orig(self, predictions, targets, trajectory_mask)
        need = ("noise_pred", "noise_target", "t_float", "act_scale", "act_bias")
        gone = [k for k in need if k not in predictions]
        if "obj_mask" not in targets:
            gone.append("targets.obj_mask")
        if gone:
            # Loud, once: a silently-inactive auxiliary loss looks exactly like a null result.
            if not getattr(self, "_rl_warned", False):
                self._rl_warned = True
                print(f"[rotloss] INACTIVE -- missing {gone}. The shape term is contributing "
                      f"NOTHING and this run is a plain baseline. Do not report it as an "
                      f"arm.", flush=True)
            return out
        m = targets["obj_mask"]                                   # [B, N]
        if m.sum() == 0:
            return out

        vp = rearrange(predictions["noise_pred"], "b c t h w -> b (h w) t c")
        vt = rearrange(predictions["noise_target"], "b c t h w -> b (h w) t c")
        t = predictions["t_float"].reshape(-1, 1, 1, 1).to(vp.dtype)
        sc = predictions["act_scale"].reshape(1, 1, 1, -1).to(vp.dtype)
        bi = predictions["act_bias"].reshape(1, 1, 1, -1).to(vp.dtype)

        gt = targets["trajectory"][:, :, 1:].to(vp.dtype)          # [B, N, T, 3] raw (u,v,z)
        # the model's target is the CLAMPED normalisation, so clamp before comparing
        xc = torch.clamp((gt - bi) / sc, -1.0, 1.0)
        xh = xc + t * (vt - vp)
        gt_raw = xc * sc + bi
        pr_raw = xh * sc + bi

        K = targets["K"].to(vp.dtype)
        E = targets["ext"].to(vp.dtype)
        p0 = _unproject(targets["trajectory"][:, :, :1].to(vp.dtype), K, E)   # [B,N,1,3]
        pg = _unproject(gt_raw, K, E)
        pp = _unproject(pr_raw, K, E)
        dg, dp = pg - p0, pp - p0

        w = m.to(vp.dtype)[:, :, None, None]                      # [B,N,1,1]
        if trajectory_mask is not None:
            tm = trajectory_mask[:, :, 1:]
            if tm.dim() == 4:
                tm = tm.all(dim=-1)
            w = w * tm.to(vp.dtype)[:, :, :, None]
        n = w.sum(dim=1, keepdim=True).clamp_min(1e-6)            # points per (sample, step)
        cg = dg - (dg * w).sum(dim=1, keepdim=True) / n
        cp = dp - (dp * w).sum(dim=1, keepdim=True) / n
        se = ((cp - cg) ** 2).sum(-1) * w[..., 0]

        ok = (m.sum(dim=1) >= MIN_PTS).to(vp.dtype)[:, None]      # degenerate sets contribute 0
        denom = (w[..., 0] * ok[:, :, None]).sum().clamp_min(1e-6)
        shape = (se * ok[:, :, None]).sum() / denom

        out["shape_loss"] = shape.detach()
        out["total_loss"] = out["total_loss"] + lam * shape
        # The trainer logs only total and diffusion, so print the split periodically:
        # lambda has to be set from the MEASURED ratio, and this is the only place it is visible.
        _n = getattr(self, "_rl_n", 0) + 1
        self._rl_n = _n
        if _n <= 3 or _n % 200 == 0:
            print(f"[rotloss] step {_n:6d}  diffusion {float(out['diffusion_loss']):.6f}  "
                  f"shape {float(shape):.3e}  lam*shape {lam * float(shape):.6f}  "
                  f"ratio {lam * float(shape) / max(float(out['diffusion_loss']), 1e-12):.2f}",
                  flush=True)
        return out

    TrajectoryLoss.forward = forward


def maybe_patch() -> bool:
    """Apply when MSGEN_ROTLOSS is a non-zero float. Returns whether it engaged."""
    global _APPLIED
    lam = _lam()
    if _APPLIED or lam == 0.0:
        return _APPLIED
    _patch_decoder()
    _patch_dataset()
    _patch_collator()
    _patch_trainer()
    _patch_loss(lam)
    _APPLIED = True
    print(f"[patch_rotloss] active: lambda={lam:g}, min object points={MIN_PTS}; "
          f"the training log will show Loss != Diffusion")
    return True
