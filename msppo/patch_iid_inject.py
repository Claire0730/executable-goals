"""patch_iid_inject.py -- MSPPO_IID_INJECT=1: the goal error is redrawn i.i.d. at every control step during
distillation, with the SAME per-episode marginal as the episode-fixed variant (msppo/patch_frozen_inject.py).
The released Pose-Native Executor (mt5_rciid_gmpc_s0) was distilled with this patch.

Episode-fixed variant (msppo/patch_frozen_inject.py; executor mt5_rcfz_gmpc_s0): one relbank draw (row index + severity a) per episode,
expanded ONCE at the first observation after the env's reset (object at its initial pose) and held all episode.

This file: at that same first observation, M independent draws are expanded in exactly the same way --
same relbank, same severity curriculum (_a_max), same object pose, i.e. the same anchor -- and at every control
step each env picks one of its M uniformly at random. The per-step goal error is therefore i.i.d. across steps and
its per-episode marginal equals Arm B's; only the temporal process differs. M -> inf recovers exact i.i.d. sampling
from p_plan; M = 64 by default (MSPPO_IID_POOL). This is the conservative choice: re-expanding each fresh draw
around the LIVE object (the pre-08-28 "moving" semantics) would change the anchor as well as the timing.

The per-episode draw state (_rel_idx, _rel_a) is restored after the pool is built, so nothing else that reads it
sees a different value. `_apply_rel_error` has exactly one call site, once per control step
(msppo/task_student_env.py:945).

USE
    MSPPO_IID_INJECT=1 python - <<'PY'
    from msppo.patch_iid_inject import maybe_patch; assert maybe_patch()
    import runpy; sys.argv = ["multi_distill", ...]; runpy.run_module("msppo.multi_distill", run_name="__main__")
    PY
Refuses to run together with MSPPO_FROZEN_INJECT. msppo/ is not edited.
MSPPO_IID_DEBUG=1 prints, for env 0 of each task env, the injected goal and its change since the previous call
(first 12 calls) -- the smoke test reads these lines.
"""
from __future__ import annotations

import os

_APPLIED = False


def maybe_patch() -> bool:
    global _APPLIED
    if _APPLIED or os.environ.get("MSPPO_IID_INJECT", "0") in ("0", ""):
        return _APPLIED
    if os.environ.get("MSPPO_FROZEN_INJECT", "0") not in ("0", ""):
        raise SystemExit("patch_iid_inject: MSPPO_FROZEN_INJECT is set -- the two arms are mutually exclusive")
    import torch
    from msppo.task_student_env import TaskStudentEnv

    M = int(os.environ.get("MSPPO_IID_POOL", "64"))
    debug = os.environ.get("MSPPO_IID_DEBUG", "0") not in ("0", "")
    orig_apply = TaskStudentEnv._apply_rel_error
    orig_resample = TaskStudentEnv._resample_goal_error

    def resample(self, idx=None):
        orig_resample(self, idx)
        if getattr(self, "_iid_pool", None) is None or idx is None:
            self._iid_pool = None                      # full reset: rebuild every env's pool
            return
        m = idx.to(self._iid_stale.device)
        if m.dtype != torch.bool:                      # integer index tensor -> bool mask
            mm = torch.zeros_like(self._iid_stale)
            mm[m.long()] = True
            m = mm
        self._iid_stale = self._iid_stale | m

    def _build(self, gp, gq, rows):
        saved = (None if self._rel_idx is None else self._rel_idx.clone(),
                 None if self._rel_a is None else self._rel_a.clone())
        draws = []
        for _ in range(M):
            orig_resample(self, None)                  # fresh (row, a) for every env; rows selected below
            draws.append(orig_apply(self, gp, gq))
        new = torch.stack(draws, 0)                    # [M, B, K, 7]
        self._rel_idx, self._rel_a = saved
        if getattr(self, "_iid_pool", None) is None:
            self._iid_pool = new
        else:
            self._iid_pool = torch.where(rows[None, :, None, None], new, self._iid_pool)

    def apply(self, gp, gq):
        B = gp.shape[0]
        if getattr(self, "_iid_pool", None) is None:
            self._iid_stale = torch.zeros(B, dtype=torch.bool, device=gp.device)
            _build(self, gp, gq, torch.ones(B, dtype=torch.bool, device=gp.device))
        elif bool(self._iid_stale.any()):
            _build(self, gp, gq, self._iid_stale.clone())
            self._iid_stale.zero_()
        pool = self._iid_pool
        pick = torch.randint(0, M, (B,), generator=self._gen, device=self._gen.device)
        out = pool[pick.to(pool.device), torch.arange(B, device=pool.device)]      # [B, K, 7]
        if debug:
            n = getattr(self, "_iid_dbg_n", 0)
            if n < 12:
                prev = getattr(self, "_iid_dbg_prev", None)
                d = float((out[0, 0, :3] - prev).norm()) * 1000 if prev is not None else float("nan")
                print(f"[iid_inject] {getattr(self, 'task', '?')} call {n}: env0 goal "
                      f"{[round(float(x), 4) for x in out[0, 0, :3]]} |delta| vs previous call {d:.1f} mm", flush=True)
                self._iid_dbg_prev = out[0, 0, :3].clone()
                self._iid_dbg_n = n + 1
        return out

    TaskStudentEnv._resample_goal_error = resample
    TaskStudentEnv._apply_rel_error = apply
    _APPLIED = True
    print(f"[iid_inject] per-step i.i.d. goal error: pool of M={M} draws per episode expanded at reset "
          f"(same anchor and marginal as the frozen arm), one drawn per control step", flush=True)
    return True
