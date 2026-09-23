"""Record the sampler's OWN integration path as a confidence signal (spec §8.1).

`CogVideoXDecoder_flow.predict_trajectory` (cogvideox_flow.py:533-587) integrates the
flow ODE with 100 Euler steps: at every step the transformer is called with the current
latent `x_t` and returns the velocity `v`, and `x <- x - v * dt`. Nothing outside the loop
ever sees those velocities. This patch registers a forward hook on the transformer
(`decoder.cogvideox`) so that every call contributes to per-sample running statistics,
and wraps `TrajectoryFlow.predict_trajectory` to reset them before the loop and
finalise them after it -- the model gets a `_ode_stats` dict the way `patch_movehead`
leaves `_move_logit`, and `msgen.predict` saves it as optional `ode_*` keys.

Per sample (scene) the statistics are, with N = number of steps taken:
    ode_pathlen    sum_i ||v_i|| dt          length of the integration path (latent units)
    ode_chord      ||x_N - x_0||             straight-line distance start -> end
    ode_curv       pathlen / chord           1.0 = a straight path; larger = it wandered
    ode_vnorm_cv   std_i ||v_i|| / mean_i    speed variability along the path
    ode_tailturn   mean over the last 20% of steps of (1 - cos(v_i, v_{i-1}))
                                             direction change while "arriving"; 0 = still
    ode_vnorm      [N] the per-step ||v_i|| itself, for anything not anticipated here

Norms are over the whole latent tensor of a sample (all frames, channels, patches), so
these are scene-level scalars that can be correlated with the scene-level goal error.
Zero extra inference: the hook reads what the loop already computes.

    MSGEN_ODESTATS=1   # on; off by default so every existing run is unaffected
"""
from __future__ import annotations

import os

import torch


class _OdeRecorder:
    """Running per-sample statistics over the transformer calls of ONE sampling loop."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.x0 = None
        self.x_last = None
        self.v_last = None
        self.v_prev = None
        self.vnorm = []          # list of [B] tensors
        self.turn = []           # list of [B] tensors, 1 - cos(v_i, v_{i-1})
        self.x_prev = None
        self.stepratio = []      # ||x_i - x_{i-1}|| / ||v_{i-1}||: the loop's EFFECTIVE dt

    def hook(self, module, args, kwargs, output):
        x = kwargs.get("hidden_states", args[0] if args else None)
        v = output[0] if isinstance(output, (tuple, list)) else output
        if x is None or v is None:
            return
        x = x.detach().float().flatten(1)
        v = v.detach().float().flatten(1)
        if self.x0 is None:
            self.x0 = x.clone()
        if self.x_prev is not None:
            self.stepratio.append((x - self.x_prev).norm(dim=1) / self.v_prev.norm(dim=1).clamp_min(1e-12))
        self.x_prev = x
        self.x_last, self.v_last = x, v
        self.vnorm.append(v.norm(dim=1))
        if self.v_prev is not None:
            cos = (v * self.v_prev).sum(1) / (v.norm(dim=1) * self.v_prev.norm(dim=1)).clamp_min(1e-12)
            self.turn.append(1.0 - cos)
        self.v_prev = v

    def finalise(self, dt):
        if self.x0 is None:
            return None
        vn = torch.stack(self.vnorm, dim=1)                      # [B,N]
        n = vn.shape[1]
        # dt is MEASURED from the loop (||x_i - x_{i-1}|| / ||v_{i-1}||), not read from the
        # scheduler: patch_steps leaves config.num_train_timesteps back at 100 by the time
        # this runs, and reading it gave pathlen 5x too small on the 08-26 smoke test.
        eff = torch.stack(self.stepratio, dim=1) if self.stepratio else torch.full_like(vn[:, :1], dt)
        dt_eff = eff.median(dim=1).values                        # [B]
        pathlen = vn.sum(dim=1) * dt_eff
        x_end = self.x_last - self.v_last * dt_eff[:, None]      # the loop's final Euler update
        chord = (x_end - self.x0).norm(dim=1)
        turn = torch.stack(self.turn, dim=1) if self.turn else torch.zeros_like(vn[:, :1])
        tail = max(1, int(round(0.2 * turn.shape[1])))
        return dict(ode_pathlen=pathlen, ode_chord=chord, ode_effdt=dt_eff,
                    ode_curv=pathlen / chord.clamp_min(1e-12),
                    ode_vnorm_cv=vn.std(dim=1) / vn.mean(dim=1).clamp_min(1e-12),
                    ode_tailturn=turn[:, -tail:].mean(dim=1),
                    ode_vnorm=vn, ode_steps=torch.full_like(chord, float(n)))


def maybe_patch():
    """Install iff MSGEN_ODESTATS is set. Returns True when active."""
    if os.environ.get("MSGEN_ODESTATS", "") not in ("1", "true", "on"):
        return False
    from msgen.paths import add_tracegen_to_path
    add_tracegen_to_path()
    from models.model_flow import TrajectoryFlow

    rec = _OdeRecorder()
    orig = TrajectoryFlow.predict_trajectory

    def predict_trajectory(self, *a, **kw):
        dec = self.diffusion_decoder
        if getattr(dec, "_ode_hook", None) is None:
            dec._ode_hook = dec.cogvideox.register_forward_hook(rec.hook, with_kwargs=True)
        rec.reset()
        out = orig(self, *a, **kw)
        # Same dt the loop used (cogvideox_flow.py:585, possibly rewritten by patch_steps
        # to 1/N): read it off the scheduler the way the loop does.
        sched = kw.get("noise_scheduler")
        dt = 1.0 / float(sched.config.num_train_timesteps) if sched is not None else 0.01
        self._ode_stats = rec.finalise(dt)
        return out

    TrajectoryFlow.predict_trajectory = predict_trajectory
    TrajectoryFlow._ode_stats = None
    print("[patch_odestats] recording per-sample ODE path statistics")
    return True
