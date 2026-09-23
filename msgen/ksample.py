"""K samples of the same planner on the same scenes -> ONE averaged preds file.

The averaged file has exactly the schema `msgen.predict` writes, so
`msppo.task_tgbank solve` consumes it unchanged: the ensemble is a planner-side
operation and nothing downstream learns that K > 1 happened.

Two refusals, both measured failure modes:
  * identical samples: `patch_seed` seeds from MSGEN_SEED; if the env var did
    not reach the process every "sample" is the same draw and the mean is a
    single sample wearing an ensemble's name.
  * misaligned episode_id: the val loader is unshuffled, but a bank re-render or
    a different --dataset would silently pair clip_003 of one run with clip_007
    of another.

    $PG -m msgen.ksample --out results/preds/pickcube_visobj_kmean.npz \
        results/preds/pickcube_visobj_s1234.npz results/preds/pickcube_visobj_s1235.npz ...
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def mean_preds(paths, out):
    if len(paths) < 2:
        raise SystemExit("need at least two sample files")
    zs = [np.load(p, allow_pickle=True) for p in paths]
    eps = [np.asarray(z["episode_id"]).astype(str) for z in zs]
    for k in range(1, len(zs)):
        if eps[k].shape != eps[0].shape or not np.array_equal(eps[k], eps[0]):
            raise SystemExit(f"episode_id of {paths[k]} does not match {paths[0]}")
    preds = np.stack([z["pred"].astype(np.float64) for z in zs])          # [K,N,400,T,3]
    diffs = [float(np.nanmax(np.abs(preds[i] - preds[j])))
             for i in range(len(zs)) for j in range(i + 1, len(zs))]
    if min(diffs) == 0.0:
        raise SystemExit("two sample files are bit-identical: MSGEN_SEED did not "
                         "vary between predict runs (or the same file was given twice)")
    with np.errstate(all="ignore"):
        mean = np.nanmean(preds, axis=0).astype(np.float32)
    mask = np.logical_and.reduce([z["mask"] for z in zs])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, pred=mean, gt=zs[0]["gt"], mask=mask,
                        episode_id=zs[0]["episode_id"], frame_id=zs[0]["frame_id"],
                        seed_list=np.array(",".join(os.path.basename(p) for p in paths)))
    return dict(n=int(mean.shape[0]), K=len(paths), min_pairwise_diff=min(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("samples", nargs="+")
    a = ap.parse_args()
    r = mean_preds(a.samples, a.out)
    print(f"K={r['K']} N={r['n']} min pairwise max|diff| {r['min_pairwise_diff']:.4g} -> {a.out}")


if __name__ == "__main__":
    main()
