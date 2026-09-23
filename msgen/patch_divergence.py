"""Divergence of the learned velocity field as a post-hoc confidence signal (spec §8.1).

Reference: "The Divergence is the Uncertainty" -- for a flow-matching interpolant the posterior
covariance of the clean sample given the current state has the closed form
    Cov(x_1 | x_t) = ((1-t)^2 / t) [ I + (1-t) J_v(x_t, t) ],   J_v = d v / d x_t,
so its trace is  U(x_t,t) = ((1-t)^2 / t) [ d + (1-t) div v ].  div v is estimated with
Hutchinson probes,  div v ~ E_u[ u^T J u ],  and here each probe is a FINITE-DIFFERENCE jvp,
    J u ~ ( v(x_t + eps u, t) - v(x_t, t) ) / eps,
one extra transformer forward per probe (bf16 autograd jvp through the CogVideoX blocks is
not worth fighting for; eps = 1e-2 on a latent whose norm is ~197 is far above bf16 noise).
The same probes give the DIAGONAL estimator  diag(J) ~ E_u[ u ⊙ J u ]  per latent element,
which, because the trajectory latent's 20x20 spatial grid IS the 20x20 query grid, is a
per-query-point divergence map for free.

What it can and cannot see, stated before running: this is the model's OWN posterior spread
(the paper's sampling-seed baseline is what it approximates). We measured that baseline at
K=16: Spearman +0.088 against the goal error. The pre-registered expectations are therefore
  (1) U (scene level) vs the 16-seed sampling spread: >= +0.5  (the estimator works)
  (2) U vs goal error: ~ +0.1 (inherits the null), unless the per-step / per-point structure
      carries something the endpoint spread does not -- that is the only thing being tested.
Recorded at a few timesteps only (MSGEN_DIV_STEPS, default 5 evenly spaced over the 100),
with MSGEN_DIV_PROBES probes each (default 4): 20 extra forwards per batch, ~+20% time.

    MSGEN_DIVSTATS=1 [MSGEN_DIV_STEPS=5 MSGEN_DIV_PROBES=4] $PG -m msgen.predict ...
Saved by msgen.predict as  div_t [S] (paper t, 1 = data), div_trace [B,S], div_U [B,S],
div_diag [B,S,H,W] (spatial map, channels/frames summed).
"""
from __future__ import annotations

import os

import torch

_APPLIED = False


def _on() -> bool:
    return os.environ.get("MSGEN_DIVSTATS", "") in ("1", "true", "on")


class _DivRecorder:
    def __init__(self, n_steps, n_probes, eps):
        self.n_steps, self.n_probes, self.eps = n_steps, n_probes, eps
        self.busy = False
        self.reset()

    def reset(self):
        self.seen = 0
        self.steps_total = None
        self.trace, self.diag, self.t_paper = [], [], []
        self.d = None

    def want(self, i):
        # evenly spaced step indices over the loop; steps_total known after set_timesteps
        n = self.steps_total or 100
        picks = set(int(round(k * (n - 1) / max(self.n_steps - 1, 1))) for k in range(self.n_steps))
        return i in picks

    def hook(self, module, args, kwargs, output):
        if self.busy:
            return
        i = self.seen
        self.seen += 1
        if not self.want(i):
            return
        x = kwargs["hidden_states"]
        self.d = float(x[0].numel())
        v = output[0] if isinstance(output, (tuple, list)) else output
        self.busy = True
        try:
            tr = torch.zeros(x.shape[0], device=x.device)
            dg = torch.zeros(x.shape[0], x.shape[-2], x.shape[-1], device=x.device)
            kw = dict(kwargs); kw.pop("hidden_states")
            for _ in range(self.n_probes):
                u = torch.randint_like(x, 0, 2).float().mul_(2).sub_(1).to(x.dtype)   # Rademacher
                vp = module(hidden_states=x + self.eps * u, **kw)
                vp = vp[0] if isinstance(vp, (tuple, list)) else vp
                ju = (vp.float() - v.float()) / self.eps
                prod = (u.float() * ju)                                     # u ⊙ J u
                tr += prod.flatten(1).sum(1)
                dg += prod.sum(dim=tuple(range(1, prod.dim() - 2)))        # sum frames+channels
        finally:
            self.busy = False
        self.trace.append(tr / self.n_probes)
        self.diag.append(dg / self.n_probes)
        n = self.steps_total or 100
        self.t_paper.append(1.0 - i / max(n - 1, 1))                       # 0 = noise ... 1 = data (approx.)

    def finalise(self, d):
        if not self.trace:
            return None
        tr = torch.stack(self.trace, 1)                                     # [B,S]
        t = torch.tensor(self.t_paper, device=tr.device).clamp(1e-3, 1 - 1e-3)
        U = ((1 - t) ** 2 / t) * (d + (1 - t) * tr)
        return dict(div_t=t, div_trace=tr, div_U=U, div_diag=torch.stack(self.diag, 1))


def maybe_patch() -> bool:
    global _APPLIED
    if _APPLIED or not _on():
        return _APPLIED
    from msgen.paths import add_tracegen_to_path
    add_tracegen_to_path()
    from models.model_flow import TrajectoryFlow

    rec = _DivRecorder(int(os.environ.get("MSGEN_DIV_STEPS", "5")),
                       int(os.environ.get("MSGEN_DIV_PROBES", "4")),
                       float(os.environ.get("MSGEN_DIV_EPS", "1e-2")))
    orig = TrajectoryFlow.predict_trajectory

    def predict_trajectory(self, *a, **kw):
        dec = self.diffusion_decoder
        if getattr(dec, "_div_hook", None) is None:
            dec._div_hook = dec.cogvideox.register_forward_hook(rec.hook, with_kwargs=True)
        rec.reset()
        sched = kw.get("noise_scheduler")
        rec.steps_total = int(len(sched.timesteps)) if sched is not None and hasattr(sched, "timesteps") and len(sched.timesteps) else 100
        out = orig(self, *a, **kw)
        self._div_stats = rec.finalise(rec.d or 1.0)
        return out

    TrajectoryFlow.predict_trajectory = predict_trajectory
    TrajectoryFlow._div_stats = None
    print(f"[patch_divergence] active: {rec.n_steps} timesteps x {rec.n_probes} Hutchinson probes (finite-difference jvp)")
    return True
