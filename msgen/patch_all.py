"""One entry point for every parent-process patch, plus a guard against silent drops.

WHY THIS EXISTS -- A MEASURED FAILURE, NOT A TIDINESS ARGUMENT.

`msgen/run_eval.py` builds the model by running TraceGen's `test_benchmark.py` under runpy
and never called `maybe_patch()`. So the vispos chain's stage 1 ran with `MSGEN_VISPOS=1` in
the environment and the patch NOT applied. Evidence, from that run's own log
(`results/metrics/peg_n50_vispos.log`):

    unexpected_keys:
      _orig_mod.vis_pos_scale

`load_state_dict(..., strict=False)` (`trainer.py:1230`) dropped the trained parameter in
silence, so the reported trace metrics (`trajectory_mse 0.000164`, `endpoint_mse 0.000324`)
belong to neither arm: they are the co-adapted decoder weights evaluated WITHOUT the term
they were trained with. The chain's own header warned about exactly this and it happened
anyway, because the warning lived in a comment and the check did not exist.

`msgen/predict.py` and `msgen/run_train.py` did call it, so the misclassification rate,
goal error and the other numbers of that run are sound. Only the run_eval-produced
Layer-1 row was affected.

Two things fix the class of bug rather than the instance:

  1. every entry point calls `apply_all()`, so a new script cannot forget one patch
  2. a dropped patch key becomes a RuntimeError instead of a line of log nobody reads

    from msgen.patch_all import apply_all
    apply_all()          # before anything builds the model
"""
from __future__ import annotations

# Parameters introduced by our patches. A checkpoint carrying one of these that the running
# model does not define means the patch is off and the metrics would be meaningless.
SENTINELS = ("vis_pos_scale", "q_feat_scale", "move_head", "cfg_null")

_GUARDED = False


def apply_all(guard: bool = True) -> dict:
    """Apply every env-flagged patch. Returns {name: bool} for the log."""
    from msgen.patch_cfg import maybe_patch as cfg
    from msgen.patch_depthnorm import maybe_patch as depthnorm
    from msgen.patch_depthscale import maybe_patch as depthscale
    from msgen.patch_depthtower import maybe_patch as depthtower
    from msgen.patch_divergence import maybe_patch as divergence
    from msgen.patch_lrgroups import maybe_patch as lrgroups
    from msgen.patch_wall import maybe_patch as wall
    from msgen.patch_movehead import maybe_patch as movehead
    from msgen.patch_odestats import maybe_patch as odestats
    from msgen.patch_qfeat import maybe_patch as qfeat
    from msgen.patch_t2k_couple import maybe_patch as t2k_couple
    from msgen.patch_rotloss import maybe_patch as rotloss
    from msgen.patch_seed import maybe_patch as seed
    from msgen.patch_steps import maybe_patch as steps
    from msgen.patch_vispos import maybe_patch as vispos
    from msgen.patch_t2k import maybe_patch as t2k
    from msgen.patch_depthaug import maybe_patch as depthaug
    from msgen.patch_coloraug import maybe_patch as coloraug

    # `steps` last: it wraps whatever `predict_trajectory` is current, so movehead's
    # wrapper stays on the inside and still runs.
    active = {"vispos": vispos(), "qfeat": qfeat(), "movehead": movehead(), "t2k": t2k(), "t2k_couple": t2k_couple(), "depthaug": depthaug(), "coloraug": coloraug(),
              "cfg": cfg(), "seed": seed(), "rotloss": rotloss(),
              # depthnorm touches the DATA transform, not the model, so it adds no
              # sentinel parameter -- a run trained with it and inferred without it
              # (or the reverse) is silently wrong and the load guard cannot catch
              # it. run.json records the flag; check it before comparing arms.
              "depthnorm": depthnorm(), "lrgroups": lrgroups(), "wall": wall(),
              # depth arms: target-channel rescale (dataset) and depth-tower unfreeze
              # (model __init__); both are data/model-side, no sentinel parameter -> the run tag
              # and the predict driver's env carry the flag (see each docstring).
              "depthscale": depthscale(), "depthtower": depthtower(), "divergence": divergence(),
              # odestats wraps predict_trajectory too; installed before `steps` so the
              # steps shim stays outermost and the recorder sees the patched dt.
              "odestats": odestats(), "steps": steps()}
    if guard:
        install_load_guard()
    on = [k for k, v in active.items() if v]
    print(f"[patch_all] active: {', '.join(on) if on else 'none (stock TraceGen)'}")
    return active


def install_load_guard():
    """Turn a silently-dropped patch parameter into a crash.

    Wraps the trainer's loader and spies on the `load_state_dict` result, because
    `load_checkpoint_singlegpu` prints the key lists but does not return them
    (`trainer.py:1230-1238`).
    """
    global _GUARDED
    if _GUARDED:
        return
    from trainer.trainer import TrajectoryDiffusionTrainer

    orig = TrajectoryDiffusionTrainer.load_checkpoint_singlegpu

    def load_checkpoint_singlegpu(self, checkpoint_path):
        captured = {}
        real = self.model.load_state_dict

        def spy(state_dict, strict=False, **kw):
            r = real(state_dict, strict=strict, **kw)
            captured["r"] = r
            return r

        self.model.load_state_dict = spy
        try:
            out = orig(self, checkpoint_path)
        finally:
            try:
                del self.model.load_state_dict
            except AttributeError:
                pass

        r = captured.get("r")
        if r is not None:
            dropped = [k for k in r.unexpected_keys if any(s in k for s in SENTINELS)]
            if dropped:
                raise RuntimeError(
                    f"checkpoint {checkpoint_path} carries {dropped}, but the running model "
                    f"does not define it, so load_state_dict(strict=False) DROPPED it. The "
                    f"metrics from this process would belong to neither arm. Set the "
                    f"matching env flag (MSGEN_VISPOS / MSGEN_QFEAT) and re-run."
                )
            absent = [k for k in r.missing_keys if any(s in k for s in SENTINELS)]
            if absent:
                print(f"[patch_all] NOTE: {absent} not in the checkpoint -> staying at zero "
                      f"init. Correct when fine-tuning from a checkpoint that predates the "
                      f"patch; WRONG if you meant to evaluate a patch-trained checkpoint.")
        return out

    TrajectoryDiffusionTrainer.load_checkpoint_singlegpu = load_checkpoint_singlegpu
    _GUARDED = True


def assert_gates_trained(model, names=SENTINELS):
    """For eval of an arm-B checkpoint: a gate still at exactly zero means arm B == arm A."""
    import torch

    found = {}
    for n, p in model.named_parameters():
        if any(s in n for s in names):
            found[n] = float(torch.linalg.vector_norm(p.detach().float()))
    if not found:
        raise RuntimeError(f"no gate parameter matching {names} exists on this model")
    zero = [k for k, v in found.items() if v == 0.0]
    if zero:
        raise RuntimeError(f"gate(s) {zero} have L2 exactly 0 -> this checkpoint is "
                           f"indistinguishable from stock TraceGen")
    return found
