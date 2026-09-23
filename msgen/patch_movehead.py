"""Give the model somewhere to say "this point does not move".

THE DIAGNOSIS THIS FOLLOWS. Across every measurement the failure has the same shape: the
amplitude of the points that DO move is right (median predicted/true ratio 0.986) while a
minority of genuinely-moving points is predicted near-static (12-16% on peg). The distribution
is bimodal, so this is a per-location CLASSIFICATION error, not a regression-scale one.

But the decoder's only output is a displacement field. There is nowhere for the model to
express "I am not sure this pixel belongs to the thing that moves" -- it must emit a number,
and the safe number, with 88% of the 400 queries static, is zero.

WHY THIS IS NOT MOTION RE-WEIGHTING, WHICH FAILED. Re-weighting scales the regression loss on
points whose amplitude is already correct (measured worse: alpha=1 +1.7%, alpha=3 +13%/+18%).
This adds a second TASK: a per-query binary head trained with BCE against `movement_bool`.
Its gradient reaches `vision_fusion`, which the flow decoder also consumes, so the auxiliary
label shapes the shared representation -- that is the mechanism, not the extra output.

The label already exists and has never been used in any loss: `datasets.py:424` computes
`movement_bool`, and `trainer.py:297` uses it only to drop samples that contain no motion at
all. It is recomputed here from `target_trajectory`, which `forward_diffusion_training` already
receives, so the trainer needs no modification:

    movement = |dx|.sum(steps) + |dy|.sum(steps) > 0.1        (normalised units, as in datasets.py)

WARM START. The head is new, so there is nothing to preserve inside it, but its gradient does
reach the shared trunk -- deliberately. The loss weight defaults to 0.1 so the auxiliary task
shapes the representation without dominating it, and at weight 0 the flow output is bit-identical
to the unpatched model, which is the identity test.

    MSGEN_MOVEHEAD=0.1 python -m msgen.run_train ...
"""
from __future__ import annotations

import os

_APPLIED = False
MOVE_THRESHOLD = 0.1          # datasets.py:424 uses this in normalised units


def patch_movehead(weight: float = 0.1, hidden: int = 256):
    """Add a per-query movement classifier and fold its BCE into the total loss."""
    global _APPLIED
    if _APPLIED:
        return
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from losses.trajectory_loss import TrajectoryLoss
    from models.model_flow import TrajectoryFlow

    orig_init = TrajectoryFlow.__init__
    orig_fwd = TrajectoryFlow.forward_diffusion_training
    orig_pred = TrajectoryFlow.predict_trajectory
    orig_loss = TrajectoryLoss.forward

    def _d_model(cfg, default=768):
        if isinstance(cfg, dict):
            return int(cfg.get("d_model", default))
        return int(getattr(cfg, "d_model", default))

    def _grid(cfg, default=20):
        n = cfg.get("num_kps", 400) if isinstance(cfg, dict) else getattr(cfg, "num_kps", 400)
        return int(round(int(n) ** 0.5)) or default

    def __init__(self, cfg):
        orig_init(self, cfg)
        D = _d_model(cfg)
        self._move_grid = _grid(cfg)
        # Named so `create_optimizer` (utils/misc.py:184-191) puts it in the DECODER group at
        # lr_decoder: the substring test there is for 'vision_encoder' / 'text_encoder'.
        self.move_head = nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self._move_logit = None

    def _logits(self, vision_features):
        """[B, 576, D] visual tokens -> [B, num_kps] per-query logits.

        Area pooling from the 24x24 SigLIP grid onto the query lattice, the same resampling
        `patch_qfeat` settled on: bilinear point-samples and leaves 30.6% of visual patches
        unread, which for a per-location decision is exactly the wrong failure.
        """
        B, N, D = vision_features.shape
        g = int(round(N ** 0.5))
        k = self._move_grid
        x = vision_features.transpose(1, 2).reshape(B, D, g, g)
        x = F.adaptive_avg_pool2d(x, (k, k))
        x = x.flatten(2).transpose(1, 2)                       # [B, k*k, D]
        return self.move_head(x).squeeze(-1)                   # [B, k*k]

    def _target(traj):
        """Reproduce datasets.py:424 from the per-step deltas the model already receives."""
        d = torch.nan_to_num(traj[:, :, 1:, :2], nan=0.0, posinf=0.0, neginf=0.0)
        return (d.abs().sum(dim=(2, 3)) > MOVE_THRESHOLD).float()   # [B, N]

    def forward_diffusion_training(self, images, texts, depth, is_depth_valid,
                                   target_trajectory, first_keypoint, diffusion_loss):
        device = images.device
        vision_features = self.encode_images(images, depth, is_depth_valid)
        text_features = self.encode_t5_texts(texts, device)
        combined = torch.cat([vision_features, text_features], dim=1)
        out = self.diffusion_decoder.forward_diffusion_training(
            trunk_conditioning=combined, target_video=target_trajectory,
            diffusion_loss=diffusion_loss)
        logit = _logits(self, vision_features)
        tgt = _target(target_trajectory)
        if logit.shape[1] == tgt.shape[1]:
            out["aux_move_loss"] = F.binary_cross_entropy_with_logits(logit, tgt)
            out["aux_move_acc"] = ((logit > 0).float() == tgt).float().mean().detach()
        return out

    def predict_trajectory(self, images, texts, depth, is_depth_valid, first_keypoint,
                           noise_scheduler, num_inference_steps: int = 100,
                           guidance_scale: float = 2.0):
        # stash the per-query movement probability so a consumer can use it as a confidence
        with torch.no_grad():
            self._move_logit = _logits(self, self.encode_images(images, depth, is_depth_valid))
        return orig_pred(self, images, texts, depth, is_depth_valid, first_keypoint,
                         noise_scheduler, num_inference_steps, guidance_scale)

    def loss_forward(self, predictions, targets, trajectory_mask=None):
        d = orig_loss(self, predictions, targets, trajectory_mask)
        aux = predictions.get("aux_move_loss")
        if aux is not None and weight != 0.0:
            d["aux_move_loss"] = aux.detach()
            d["total_loss"] = d["total_loss"] + weight * aux
        if "aux_move_acc" in predictions:
            d["aux_move_acc"] = predictions["aux_move_acc"]
        return d

    TrajectoryFlow.__init__ = __init__
    TrajectoryFlow.forward_diffusion_training = forward_diffusion_training
    TrajectoryFlow.predict_trajectory = predict_trajectory
    TrajectoryLoss.forward = loss_forward
    _APPLIED = True
    print(f"[patch_movehead] per-query movement classifier added (BCE weight {weight}); "
          f"gradient reaches vision_fusion, which the flow decoder shares")


def maybe_patch():
    v = os.environ.get("MSGEN_MOVEHEAD", "")
    if v and v != "0":
        patch_movehead(weight=0.1 if v in ("1", "on", "true") else float(v))
        return True
    return False
