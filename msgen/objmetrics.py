"""Interpretable trace error on the TARGET OBJECT, in pixels and millimetres.

TraceGen's own metric is a single scalar mixing normalized x,y (image fractions)
with z (metres), so it is not directly readable, and it averages the whole
400-point grid -- ~75% of which is static background the model gets right for
free. This module reports the quantity the task is actually about: how far the
predicted trace of the cube/peg ends up from the truth.

Run:  python -m msgen.objmetrics --task peg
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from msgen.labels import grid_pixels, unproject
from msgen.tasks import IMAGE_SIZE, get_task

MOVING_PATH_PX = 40.0


def horizon(mask):
    """(last usable step index, per-point validity) for ONE sample's mask [400, 33, 3].

    Every scorer in this repo used to hardcode `mask[:, 1:, :2].all(axis=(1,2))`, i.e. "a point
    counts only if all 33 steps are valid, and the endpoint is step 32". That is correct for
    labels that always fill 32 future steps, and STRUCTURALLY BROKEN for any label convention
    that emits fewer steps and pads the rest with -inf -- TraceForge's does exactly that, and
    the padded samples then contribute zero scorable points:

        peg_tf 46% of samples padded, stack_tf 68%, liftpeg_tf 57%, pusht_tf 49%,
        pickcube_tf 100% -> "NO truly-moving object points found"

    So the horizon is read from the data instead: the last step that any point still has, and
    validity is required only up to there. For a full-length sample this returns (32, the old
    mask), which is why every number recorded before the fix is unaffected -- asserted by
    the regression in `<private-repo>/experiments/20260817_tfconv2/f0_verify.py`.
    """
    step_ok = mask[:, 1:, :2].all(axis=2)             # [400, 32] per point per step
    any_pt = step_ok.any(axis=0)                      # [32] does ANY point have this step
    n = int(np.argmax(~any_pt)) if (~any_pt).any() else int(len(any_pt))
    if n < 1:
        return 0, np.zeros(mask.shape[0], dtype=bool)
    return n, step_ok[:, :n].all(axis=1)


def target_id_for(clip_meta, task):
    """The segmentation id of the task's target body IN THIS CLIP, or None.

    Two resolution paths, in order:

      1. `clip_meta["target_id"]`, written by the clip generator, which held the env handle and
         did not have to guess. REQUIRED for PickSingleYCB: its object actor is named per scene
         (`061_foam_brick-0`, `063-a_marbles-0`, `072-a_toy_airplane-0`), so no constant name and
         no single regex works -- a `^\\d{3}_` pattern misses every hyphenated name, measured at
         7 of 16 seeds in `<private-repo>/experiments/20260816_ycb/y0_discover.json`.
      2. the constant `target_body` name from `msgen/tasks.py`, the original behaviour, kept so
         that every manifest written before the fix resolves exactly as it did.
    """
    tid = clip_meta.get("target_id")
    if tid is not None:
        return int(tid)
    names = clip_meta["body_names"]
    ids = np.asarray(clip_meta["body_ids"])
    target = get_task(task).get("target_body")
    if target is None or target not in names:
        return None
    return int(ids[names.index(target)])


def object_mask_for(raw, clip_meta, task, t0, keypoints=None):
    """Which of the queries land on the target at frame t0.

    `keypoints` MUST be passed for any dataset built with object-aware sampling
    (`msgen.labels --n-obj > 0`): there the query pixels are chosen PER FRAME from the object
    mask, so recomputing `grid_pixels()` would score a different set of points than the ones
    the model was actually asked about. They are stored in every sample npz. Omitting it keeps
    the old behaviour, which is correct only for uniform-grid datasets.
    """
    tid = target_id_for(clip_meta, task)
    if tid is None:
        return np.zeros(400, dtype=bool)
    px = grid_pixels() if keypoints is None else np.asarray(keypoints, dtype=np.float64)
    ix = np.clip(px[:, 0].astype(int), 0, IMAGE_SIZE - 1)
    iy = np.clip(px[:, 1].astype(int), 0, IMAGE_SIZE - 1)
    return raw["seg"][t0][iy, ix] == tid


def to_world(tr, K, ext):
    N, T, _ = tr.shape
    flat = tr.reshape(-1, 3)
    return unproject(flat[:, :2], flat[:, 2], K, ext).reshape(N, T, 3)


def evaluate(task, arm, pred_dir="results/preds", raw_dir=None):
    raw_dir = raw_dir or f"data/raw/{task}_test"
    d = np.load(f"{pred_dir}/{task}_{arm}.npz", allow_pickle=True)
    pred, gt, mask = d["pred"], d["gt"], d["mask"]
    eps, frames = d["episode_id"], d["frame_id"]
    manifest = {c["clip"]: c for c in json.load(open(f"{raw_dir}/manifest.json"))["clips"]}

    raws = {}
    px_err, mm_err, n_obj = [], [], []
    for i in range(len(pred)):
        clip, stem = str(eps[i]), str(frames[i])
        if clip not in manifest:
            continue
        if clip not in raws:
            raws[clip] = dict(np.load(f"{raw_dir}/{clip}.npz"))
        raw = raws[clip]
        t0 = int(stem)

        valid = mask[i][:, 1:, :2].all(axis=(1, 2))
        on_obj = object_mask_for(raw, manifest[clip], task, t0) & valid
        # only count query frames where the object actually moves
        path = np.abs(np.diff(gt[i][:, :, :2], axis=1)).sum(axis=(1, 2))
        on_obj = on_obj & (path > MOVING_PATH_PX)
        if on_obj.sum() == 0:
            continue

        px_err.append(float(np.linalg.norm(
            pred[i][on_obj, -1, :2] - gt[i][on_obj, -1, :2], axis=-1).mean()))
        pw = to_world(pred[i][on_obj], raw["K"], raw["extrinsic_cv"])
        gw = to_world(gt[i][on_obj], raw["K"], raw["extrinsic_cv"])
        mm_err.append(float(np.linalg.norm(pw[:, -1] - gw[:, -1], axis=-1).mean() * 1000.0))
        n_obj.append(int(on_obj.sum()))

    if not px_err:
        return None
    return dict(task=task, arm=arm, n_samples=len(px_err),
                mean_obj_grid_points=float(np.mean(n_obj)),
                endpoint_px_mean=float(np.mean(px_err)),
                endpoint_px_median=float(np.median(px_err)),
                endpoint_mm_mean=float(np.mean(mm_err)),
                endpoint_mm_median=float(np.median(mm_err)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, choices=["stack", "peg"])
    ap.add_argument("--out", default="results/metrics")
    args = ap.parse_args()

    tasks = [args.task] if args.task else ["stack", "peg"]
    recs = []
    for t in tasks:
        for arm in ("zeroshot", "finetuned"):
            if not os.path.exists(f"results/preds/{t}_{arm}.npz"):
                continue
            r = evaluate(t, arm)
            if r:
                recs.append(r)
                print(json.dumps(r), flush=True)
    if recs:
        os.makedirs(args.out, exist_ok=True)
        json.dump(recs, open(f"{args.out}/object_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
