"""Make classifier-free guidance actually available, by training a null condition for it.

WHY. `models/decoder/cogvideox_flow.py:495` builds the unconditional branch as

    negative_prompt_embeds = torch.randn(batch_size, seq_len, embed_dim, ...) * 0.1

and there is NO condition dropout anywhere in training -- grepped across `model_flow.py`,
`trainer/trainer.py`, `dataio/datasets.py` and `cfg/train.yaml`. So `guidance_scale > 1`
extrapolates away from the model's response to a condition it has never seen, which is not
guidance, just noise amplification. TraceGen's own `test_example.py:335` sweeps 1.0/2.0/3.0 by
default; those samples are not trustworthy. Our metric path uses 1.0 and is therefore safe --
but it also means the guidance knob does not exist for us.

WHAT THIS ADDS. A learned null condition, trained by dropping the real condition on a fraction
of samples, so CFG becomes a knob that is tuned AFTER training with no retraining per setting.
It aims at the measured defect directly: the bottleneck is a per-location move/static decision
(`<private-repo>/experiments/20260814_ab8/a2_task.py`), and guidance sharpens exactly that kind of conditional
decision.

    MSGEN_CFG_DROP=0.1     training: replace the condition with the null on 10% of samples
    MSGEN_CFG=2.0          inference: run real CFG at this scale

WARM START. `runs/peg_n50_scratch` measured from-scratch at 3.8x worse, so the pretrained
weights must survive. Hence:
  * one new parameter, created in `__init__` because `create_optimizer` (`utils/misc.py:176`)
    walks `named_parameters()` exactly once
  * named `cfg_null`, avoiding the `vision_encoder` / `text_encoder` substrings that
    `utils/misc.py:184-191` uses to route parameters to `lr_backbone` 2e-5
  * zero-initialised, and no existing tensor changes shape, so at MSGEN_CFG_DROP unset and
    guidance_scale 1.0 the model is bit-identical to stock -- which `m0`-style identity tests
    must confirm before any training is launched
  * `cfg_null` is registered in `patch_all.SENTINELS`, so a checkpoint carrying it that is
    loaded without this patch raises instead of being silently dropped

The null is a single broadcast token `[1, 1, D]` rather than a full `[1, 704, D]` block: the
parameter has to exist before the conditioning length is known, and a per-position null would
have to hardcode 704. A broadcast token needs no such assumption and is a standard null
embedding.

THE SAMPLING LOOP IS REIMPLEMENTED HERE. The stock decoder builds its negative internally with
no hook (line 495), so there is no way to substitute the null without owning the loop. The copy
follows `cogvideox_flow.py:483-603` line for line, including the hand-rolled Euler update at
585-586 -- which means `msgen/patch_steps.py` still works, since that shims `set_timesteps` on
the scheduler instance and this loop calls it the same way. The risk of a divergent copy is
covered by an identity test: at `guidance_scale=1.0` the stock path never touches its negative
branch, so our output must match stock EXACTLY, which exercises noise init, scheduler stepping,
dt, the put_frames_in_channels round trip and the unnormalise.
"""
from __future__ import annotations

import os

_APPLIED = False


def _drop() -> float:
    return float(os.environ.get("MSGEN_CFG_DROP", "0") or 0)


def _scale() -> float:
    return float(os.environ.get("MSGEN_CFG", "0") or 0)


def maybe_patch() -> bool:
    """Install the patch if either flag is set at build time.

    The VALUES are read per call, not captured here, for two reasons. One process can then
    sweep guidance scales without rebuilding a 3.3 GB checkpoint per setting -- which is what
    makes the sweep cheap enough to be worth having. And the identity tests can compare the
    stock path against the guided path on ONE model instance, instead of holding several
    trainers on a GPU that is usually busy with something else.
    """
    global _APPLIED
    if _drop() <= 0 and _scale() <= 0:
        return False
    if _APPLIED:
        return True

    import torch
    import torch.nn as nn
    from einops import rearrange
    from models.model_flow import TrajectoryFlow

    orig_init = TrajectoryFlow.__init__
    orig_train = TrajectoryFlow.forward_diffusion_training
    orig_pred = TrajectoryFlow.predict_trajectory

    def _d_model(cfg, default=768):
        if isinstance(cfg, dict):
            return int(cfg.get("d_model", default))
        return int(getattr(cfg, "d_model", default))

    def __init__(self, cfg):
        orig_init(self, cfg)
        self.cfg_null = nn.Parameter(torch.zeros(1, 1, _d_model(cfg)))

    def _null_like(self, cond):
        return self.cfg_null.to(cond.dtype).expand(cond.shape[0], cond.shape[1], -1)

    def forward_diffusion_training(self, images, texts, depth, is_depth_valid,
                                   target_trajectory, first_keypoint, diffusion_loss):
        drop = _drop()
        if drop <= 0:
            return orig_train(self, images, texts, depth, is_depth_valid,
                              target_trajectory, first_keypoint, diffusion_loss)
        device = images.device
        vision_features = self.encode_images(images, depth, is_depth_valid)
        text_features = self.encode_t5_texts(texts, device)
        combined = torch.cat([vision_features, text_features], dim=1)

        # Drop the WHOLE condition per sample, not per token: CFG's unconditional branch is
        # "no condition at all", and dropping tokens independently would teach a partially
        # conditioned model that guidance cannot then extrapolate away from.
        keep = (torch.rand(combined.shape[0], 1, 1, device=device) >= drop).to(combined.dtype)
        combined = keep * combined + (1.0 - keep) * _null_like(self, combined)

        return self.diffusion_decoder.forward_diffusion_training(
            trunk_conditioning=combined, target_video=target_trajectory,
            diffusion_loss=diffusion_loss)

    def predict_trajectory(self, images, texts, depth, is_depth_valid, first_keypoint,
                           noise_scheduler, num_inference_steps: int = 100,
                           guidance_scale: float = 2.0):
        scale = _scale()
        if scale <= 1.0:
            # nothing to guide with -- stay on the stock path so the default stays untouched
            return orig_pred(self, images, texts, depth, is_depth_valid, first_keypoint,
                             noise_scheduler, num_inference_steps, guidance_scale)
        self.eval()
        with torch.no_grad():
            v = self.encode_images(images, depth, is_depth_valid)
            t = self.encode_t5_texts(texts, images.device)
            cond = torch.cat([v, t], dim=1)
            return _guided_sample(self.diffusion_decoder, cond, _null_like(self, cond),
                                  noise_scheduler, num_inference_steps, scale)

    def _guided_sample(dec, cond, null, scheduler, num_inference_steps, guidance_scale):
        """Copy of cogvideox_flow.py:483-603 with the learned null replacing `randn * 0.1`."""
        pos = dec.project_latents_to_cogvideox_format(cond).to(dec.cogvideox.dtype)
        neg = dec.project_latents_to_cogvideox_format(null).to(dec.cogvideox.dtype)
        dtype = next(dec.cogvideox.parameters()).dtype
        ehs = torch.cat([neg, pos], dim=0).to(dtype)

        # Stock draws its throwaway negative at cogvideox_flow.py:495 UNCONDITIONALLY, before
        # the latents at :517 -- so it consumes RNG that our copy would otherwise skip, and the
        # initial noise would differ. Caught by identity test 5 (max abs diff 8.7e-2). Burn the
        # same draw so guidance 1.0 reproduces the stock trace exactly.
        torch.randn(pos.shape, device=pos.device, dtype=pos.dtype)

        b = pos.shape[0]
        latents = torch.randn((b, dec.num_frames, dec.input_channels, dec.frame_size,
                               dec.frame_size), device=cond.device, dtype=dtype)
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(num_inference_steps, device=cond.device)
        for t_step in scheduler.timesteps:
            inp = torch.cat([latents] * 2)
            if hasattr(scheduler, "scale_model_input"):
                inp = scheduler.scale_model_input(inp, t_step)
            ts = t_step.expand(inp.shape[0]).to(dtype)
            pred = dec.cogvideox(hidden_states=inp, encoder_hidden_states=ehs, timestep=ts,
                                 image_rotary_emb=None, return_dict=False)[0].to(dtype)
            uncond, condp = pred.chunk(2)
            pred = uncond + guidance_scale * (condp - uncond)
            dt = 1 / scheduler.config.num_train_timesteps
            latents = (latents - pred * dt).to(dtype)

        pf = getattr(dec, "put_frames_in_channels", 1)
        if pf > 1:
            B, T, C, H, W = latents.shape
            latents = latents.view(B, T, C // pf, pf, H, W)
            latents = latents.permute(0, 1, 3, 2, 4, 5).contiguous().view(
                B, T * pf, C // pf, H, W)
        latents = dec.unnormalize_act_data(latents)
        return rearrange(latents, "b t c h w -> b (h w) t c")

    TrajectoryFlow.__init__ = __init__
    TrajectoryFlow.forward_diffusion_training = forward_diffusion_training
    TrajectoryFlow.predict_trajectory = predict_trajectory
    _APPLIED = True
    print(f"[patch_cfg] installed; read per call -- dropout={_drop()} guidance={_scale()}")
    return True
