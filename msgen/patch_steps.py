"""Make `num_inference_steps` actually mean something.

THE DEFECT, at `TraceGen/models/decoder/cogvideox_flow.py:585-586`:

    dt = 1 / scheduler.config.num_train_timesteps
    latents = latents - noise_pred * dt

The Euler step size is pinned to 1/`num_train_timesteps` (= 1/100, set at
`losses/trajectory_loss.py:34`) while the loop runs `num_inference_steps` times.  The
integration therefore covers

    total_t = num_inference_steps / 100

and only reaches the t=1 -> t=0 span the model was trained on when
`num_inference_steps == 100`.  Ask for 20 steps and the sampler stops at t=0.8: the
returned "trajectory" is 80% noise, silently, with no warning and no shape error.

The metric path is correct only by coincidence -- `models/model_flow.py:275`
defaults to 100 and `msgen/predict.py:95` does not override it.  The knob is a trap
rather than a knob, so "run the planner with fewer steps for closed-loop replanning"
is currently not available.

THE FIX, without touching the pristine checkout: the scheduler's own timestep grid is
already correct.  `CogVideoXDDIMScheduler.set_timesteps(N)` lays N points on [0, 99]
with gap `step_ratio = 100 // N`, so the right step size is

    dt = step_ratio / 100 = 1 / N

which is what `dt = 1 / scheduler.config.num_train_timesteps` computes if that attribute
reads N at line 585.  So we shim `set_timesteps` on the scheduler INSTANCE: restore the
true 100 on entry (the real implementation needs it to build the grid), then leave N
behind for the dt line to read.  No reimplementation of the sampling loop, and no edit
inside TraceGen/.

    MSGEN_STEPS=20        # ask for 20 steps
    MSGEN_DT=fix          # and integrate the full t=1 -> t=0 span

`MSGEN_STEPS` without `MSGEN_DT=fix` reproduces the trap on purpose -- see
`<private-repo>/experiments/20260815_sampler`, the control that shows the bug is real rather than
theoretical.
"""
from __future__ import annotations

import os

_TRUE_NUM_TRAIN = None


def maybe_patch() -> bool:
    steps = os.environ.get("MSGEN_STEPS", "").strip()
    fix = os.environ.get("MSGEN_DT", "").strip().lower() == "fix"
    if not steps and not fix:
        return False
    n_steps = int(steps) if steps else None

    from models.model_flow import TrajectoryFlow

    orig_pred = TrajectoryFlow.predict_trajectory

    def _install_dt_shim(scheduler, n):
        """Leave `config.num_train_timesteps == n` behind for the dt line to read."""
        global _TRUE_NUM_TRAIN
        if getattr(scheduler, "_msgen_dt_shimmed", False):
            return
        _TRUE_NUM_TRAIN = int(scheduler.config.num_train_timesteps)
        real = scheduler.set_timesteps

        def set_timesteps(num_inference_steps, device=None, **kw):
            # The real implementation divides by num_train_timesteps to build the grid,
            # so it must see the true 100 -- restore it every call.
            scheduler.config.num_train_timesteps = _TRUE_NUM_TRAIN
            out = real(num_inference_steps, device=device, **kw)
            if _TRUE_NUM_TRAIN % num_inference_steps != 0:
                raise ValueError(
                    f"num_inference_steps={num_inference_steps} does not divide "
                    f"num_train_timesteps={_TRUE_NUM_TRAIN}; the timestep grid would be "
                    f"non-uniform and a single dt would be wrong. Use a divisor of "
                    f"{_TRUE_NUM_TRAIN} (100, 50, 25, 20, 10, 5)."
                )
            scheduler.config.num_train_timesteps = num_inference_steps
            return out

        scheduler.set_timesteps = set_timesteps
        scheduler._msgen_dt_shimmed = True

    def predict_trajectory(self, images, texts, depth, is_depth_valid, first_keypoint,
                           noise_scheduler, num_inference_steps: int = 100,
                           guidance_scale: float = 2.0):
        n = n_steps if n_steps is not None else num_inference_steps
        if fix:
            _install_dt_shim(noise_scheduler, n)
        return orig_pred(self, images, texts, depth, is_depth_valid, first_keypoint,
                         noise_scheduler, n, guidance_scale)

    TrajectoryFlow.predict_trajectory = predict_trajectory
    print(f"[patch_steps] num_inference_steps={n_steps or 'unchanged'} dt={'fixed' if fix else 'STOCK (under-integrates)'}")
    return True
