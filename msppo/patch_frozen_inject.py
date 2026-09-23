"""Training-time goal-error injection with DEPLOYMENT semantics: one fixed goal per episode.

WHY (measured 2026-08-28, <private-repo>/experiments/20260828_review/m3_*). `_apply_rel_error`
(task_student_env.py:476-491) reads the object's LIVE position on every control step and
expands the sampled relbank error in the current object->goal frame:
    goal_inj(t) = obj(t) + frac * (goal - obj(t)) + lateral(t).
As the object approaches the goal the injected error shrinks toward the lateral part and
the goal keeps "correcting itself". A planner goal at deployment is composed once and
never moves. Same student (`mt4_ramp_s0`), same relbank, same a=1, only the goal frozen at
the first step of each episode: PickCube 0.871 -> 0.148, StackCube 0.563 -> 0.086. The
apparent noise tolerance of every `--goal-err-rel` student is therefore a property of the
moving target, not of the policy.

WHAT. The injected goal samples [B,K,7] are computed at the first observation after each
env's reset and HELD until that env resets again. `_resample_goal_error(idx)` -- called by
`reset_masks(idx)` on every (partial) reset -- marks the envs whose cached goal must be
recomputed. Everything downstream (canonical-frame Kabsch, K-mean, sd/k slots, sig) is
unchanged, so the observation layout is identical and run.json needs no new field; the
driver writes `runs_rl/<tag>/patches.json` as the sidecar record.

This is the correct replacement for `--goal-delta-train`, which applies a bank's WORLD
frame (dR, dt) to a different scene (multi_distill.py:458-477 never reads `c_src`).

USE
    MSPPO_FROZEN_INJECT=1 python - <<'PY'
    from msppo.patch_frozen_inject import maybe_patch; assert maybe_patch()
    import runpy, sys; sys.argv = ["multi_distill", ...]; runpy.run_module("msppo.multi_distill", run_name="__main__")
    PY
Nothing in msppo/ is edited.
"""
from __future__ import annotations

import os

_APPLIED = False


def maybe_patch() -> bool:
    global _APPLIED
    if _APPLIED or os.environ.get("MSPPO_FROZEN_INJECT", "0") in ("0", ""):
        return _APPLIED
    import torch
    from msppo.task_student_env import TaskStudentEnv

    orig_apply = TaskStudentEnv._apply_rel_error
    orig_resample = TaskStudentEnv._resample_goal_error

    def resample(self, idx=None):
        orig_resample(self, idx)
        cache = getattr(self, "_frz_gs", None)
        if cache is None or idx is None:
            self._frz_gs = None                    # full reset: recompute everything
            return
        self._frz_stale = self._frz_stale | idx.to(self._frz_stale.device)

    def apply(self, gp, gq):
        cache = getattr(self, "_frz_gs", None)
        if cache is None:
            fresh = orig_apply(self, gp, gq)
            self._frz_gs = fresh.clone()
            self._frz_stale = torch.zeros(fresh.shape[0], dtype=torch.bool, device=fresh.device)
            return self._frz_gs
        st = self._frz_stale
        if bool(st.any()):
            fresh = orig_apply(self, gp, gq)
            self._frz_gs = torch.where(st[:, None, None], fresh, self._frz_gs)
            st.zero_()
        return self._frz_gs

    TaskStudentEnv._resample_goal_error = resample
    TaskStudentEnv._apply_rel_error = apply
    _APPLIED = True
    print("[frozen_inject] injected goal computed once per episode and held (deployment "
          "semantics); recomputed only for envs that reset", flush=True)
    return True
