"""Dump per-sample trace predictions for visualization.

Builds the trainer in-process (rather than through test_benchmark.py) so the
predicted tensors can be captured, then calls the model's own
`predict_trajectory` -- the exact call the metric path uses (trainer.py:965).

Predictions are per-step displacements in normalized space; this script cumsums
them back to absolute positions the same way the metric does (trainer.py:791),
prepending the GT step 0, and converts to pixels + metres.

Run in the trace_gen env:
  python -m msgen.predict --dataset data/ds/peg_test --ckpt <path> --tag peg_ft
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from msgen.paths import GENERALIST, TRACEGEN_DIR, add_tracegen_to_path, patch_torch_load
from msgen.run_eval import patch_split_all_to_val
from msgen.tasks import IMAGE_SIZE


def build_trainer(dataset_dir, ckpt, batch_size, num_workers):
    add_tracegen_to_path()
    patch_torch_load()
    patch_split_all_to_val()
    from msgen.patch_all import apply_all
    apply_all()
    from trainer.trainer import TrajectoryDiffusionTrainer

    cfg = OmegaConf.load(f"{TRACEGEN_DIR}/cfg/eval.yaml")
    cfg.data.dataset_dirs = [os.path.abspath(dataset_dir)]
    cfg.data.val_split = 1.0
    cfg.data.num_workers = num_workers
    cfg.data.cache_dir = os.path.abspath("data/cache")
    cfg.train.batch_size = batch_size
    cfg.train.visualize_during_validation = True
    cfg.hardware.compile_model = True
    cfg.logging.use_wandb = False
    # A raw DictConfig makes the decoder receive an illegal `_metadata` kwarg.
    cfg = OmegaConf.to_container(cfg, resolve=True)

    trainer = TrajectoryDiffusionTrainer(cfg, rank=0, world_size=1, local_rank=0)
    trainer.load_checkpoint_singlegpu(os.path.abspath(ckpt))
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    # Not registered buffers, so they must be re-set after loading or
    # normalization silently breaks (cogvideox_flow.py:227).
    model.diffusion_decoder.set_data_act_statistics(trainer.action_max, trainer.action_min)
    return trainer, model


def to_absolute(pred, target):
    """Deltas -> absolute positions, mirroring trainer.py:791-800.

    Step 0 is taken from the GT for both, so the prediction is scored purely on
    the motion it adds. Returns pixels for x,y and metres for z.
    """
    pred_abs = torch.cat([target[:, :, :1, :], pred], dim=2).cumsum(dim=2)
    gt_abs = target.cumsum(dim=2)
    for t in (pred_abs, gt_abs):
        t[..., 0] *= IMAGE_SIZE
        t[..., 1] *= IMAGE_SIZE
    return pred_abs, gt_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", default=GENERALIST)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/preds")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    trainer, model = build_trainer(args.dataset, args.ckpt, args.batch_size, args.num_workers)

    recs = []
    # Seed HERE, not at build time: worker seeds are derived from the main process RNG when the
    # iterator is created, and the dataset picks one of three instructions with the global
    # `random` inside worker processes (`dataio/datasets.py:559,578`). Seeding at this point
    # pins that choice and makes it identical across arms. No-op unless MSGEN_SEED is set.
    from msgen.patch_seed import seed_from_env
    seed_from_env()
    with torch.no_grad():
        for batch in trainer.val_loader:
            images = batch["image"].to(trainer.device)
            depth = batch["depth"].to(trainer.device)
            _m = getattr(model, "_orig_mod", model)
            if hasattr(_m, "t2k_head"):
                _m._t2k_batch = batch          # the head needs depth_m / K / a fallback mask (patch_t2k)
            pred = model.predict_trajectory(
                images=images,
                texts=batch["text"],
                depth=depth,
                is_depth_valid=batch["is_depth_valid"],
                first_keypoint=batch["gt_xy"].to(trainer.device),
                noise_scheduler=trainer.criterion.noise_scheduler,
                guidance_scale=1.0,
            )
            target = batch["trajectory"].to(trainer.device)
            pred_abs, gt_abs = to_absolute(pred.float(), target.float())
            # MSGEN_DEPTH_SCALE (msgen/patch_depthscale.py): the dataset multiplied the depth
            # channel by s for training; undo it here so the saved preds/gt are in metres.
            from msgen.patch_depthscale import scale as _dscale
            if _dscale() != 1.0:
                pred_abs[..., 2] /= _dscale(); gt_abs[..., 2] /= _dscale()
            mask = batch["trajectory_mask"].cpu().numpy()

            # The movement head (msgen/patch_movehead.py:115) computes a per-query "does this
            # point move" logit during predict_trajectory and stashes it on the model. Previously
            # nothing read it back, so a confidence the model was trained to produce was
            # discarded at the only place it could be used. Saved when present; absent for an
            # unpatched checkpoint, and consumers must treat the key as optional.
            logit = getattr(model, "_move_logit", None)
            logit = (logit.float().cpu().numpy().astype(np.float32)
                     if logit is not None else None)
            # ODE path statistics (msgen/patch_odestats.py), per sample, when MSGEN_ODESTATS=1.
            ode = getattr(model, "_ode_stats", None)
            ode = ({k: v.float().cpu().numpy().astype(np.float32) for k, v in ode.items()}
                   if ode else None)
            # divergence / posterior-covariance statistics (msgen/patch_divergence.py), per sample
            dv = getattr(model, "_div_stats", None)
            if dv:
                ode = dict(ode or {}); ode.update({k: v.float().cpu().numpy().astype(np.float32)
                                                   for k, v in dv.items() if k != "div_t"})
                div_t = dv["div_t"].float().cpu().numpy().astype(np.float32)
            for i in range(pred_abs.shape[0]):
                r = dict(
                    episode_id=str(batch["episode_id"][i]),
                    frame_id=str(batch["frame_id"][i]),
                    pred=pred_abs[i].cpu().numpy().astype(np.float32),
                    gt=gt_abs[i].cpu().numpy().astype(np.float32),
                    mask=mask[i],
                )
                if logit is not None and i < logit.shape[0]:
                    r["move_logit"] = logit[i]
                t2k = getattr(getattr(model, "_orig_mod", model), "_t2k_out", None)
                if t2k is not None:
                    for k, v in t2k.items():
                        if i < len(v): r[f"t2k_{k}"] = np.asarray(v[i], dtype=np.float32)
                if ode is not None:
                    for k, v in ode.items():
                        r[k] = v[i]
                recs.append(r)
            print(f"  {len(recs)} samples", flush=True)

    out = f"{args.out}/{args.tag}.npz"
    # PROVENANCE. Previously the file recorded nothing about WHICH
    # checkpoint produced it, so answering "which planner made these goals" meant
    # reading the driver script. Additive only: existing keys and values unchanged,
    # older files still load.
    import os as _os
    _prov = dict(
        ckpt=_os.path.abspath(args.ckpt),
        dataset=_os.path.abspath(args.dataset),
        depth_scale=str(_os.environ.get("MSGEN_DEPTH_SCALE", "")),
        wall=str(_os.environ.get("MSGEN_WALL", "")),
        scene_gate=str(_os.environ.get("MSGEN_SCENE_GATE", "")),
        seed_env=str(_os.environ.get("MSGEN_SEED", "")),
    )
    np.savez_compressed(
        out,
        **{f"prov_{k}": np.array(v) for k, v in _prov.items()},
        pred=np.stack([r["pred"] for r in recs]),
        gt=np.stack([r["gt"] for r in recs]),
        mask=np.stack([r["mask"] for r in recs]),
        **{k: np.stack([r[k] for r in recs]) for k in recs[0] if k.startswith("t2k_") and all(k in r for r in recs)},
        episode_id=np.array([r["episode_id"] for r in recs]),
        frame_id=np.array([r["frame_id"] for r in recs]),
        **({"move_logit": np.stack([r["move_logit"] for r in recs])}
           if all("move_logit" in r for r in recs) else {}),
        **{k: np.stack([r[k] for r in recs]) for k in
           ("ode_pathlen", "ode_chord", "ode_curv", "ode_vnorm_cv", "ode_tailturn", "ode_vnorm", "ode_steps", "ode_effdt",
            "div_trace", "div_U", "div_diag")
           if all(k in r for r in recs)},
        **({"div_t": div_t} if "div_t" in dir() else {}),
    )
    print(f"wrote {len(recs)} predictions -> {out}")


if __name__ == "__main__":
    main()
