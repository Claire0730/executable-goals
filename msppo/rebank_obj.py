"""Turn a fixed-grid scene bank into an OBJECT-AWARE one, in place of a re-render.

The bank's images, depth, segmentation and configs.json all stay as they are --
only the query pixels change, from `msgen.labels.grid_pixels()` (a 20x20 lattice
over the whole frame) to `msgen.labels.object_aware_pixels()` (n_obj of them
placed on the object, the rest as before).

WHY IT MATTERS. Under the lattice only ~5 of 400 points land on a 2 cm cube or a
thin peg, so the planner sees almost no moving queries and learns to predict
the object as static; placing 16 object-aware queries (`--n-obj`) on the object
raises the moving-query share so that the object's motion is represented.

    python -m msppo.rebank_obj --src data/bank/task_stack --dst data/bank/task_stack_obj
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n-obj", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from msgen.labels import object_aware_pixels

    meta = json.load(open(f"{a.src}/configs.json"))
    os.makedirs(a.dst, exist_ok=True)
    n_on = []
    for c in meta["configs"]:
        clip = c["clip"]
        d = os.path.join(a.dst, clip)
        os.makedirs(d, exist_ok=True)
        for sub in ("images", "depth"):
            link = os.path.join(d, sub)
            if not os.path.exists(link):
                os.symlink(os.path.realpath(os.path.join(a.src, clip, sub)), link)
        for f in ("seg.npz", "three_instructions.json"):
            t = os.path.join(d, f)
            if not os.path.exists(t):
                shutil.copy(os.path.join(a.src, clip, f), t)

        seg = np.load(os.path.join(a.src, clip, "seg.npz"))["seg"]
        px = object_aware_pixels(seg, int(c["obj_seg_id"]), a.n_obj, seed=a.seed)
        z = np.load(os.path.join(a.src, clip, "samples/00000.npz"))
        T = z["traj"].shape[1]
        os.makedirs(os.path.join(d, "samples"), exist_ok=True)

        H, W = seg.shape[-2:]
        ix = np.clip(px[:, 0].astype(int), 0, W - 1)
        iy = np.clip(px[:, 1].astype(int), 0, H - 1)
        depth = np.load(os.path.join(a.src, clip, "depth/00000_raw.npz"))["depth"]
        d0 = depth[iy, ix]
        # A dummy target, exactly as the peg and pickcube banks do: the loader
        # DROPS any sample whose movement is zero, so a static placeholder would
        # be filtered out and the scene would silently get no prediction.
        # `predict_trajectory` never sees the target, so its content cannot leak.
        traj = np.full((len(px), T, 3), -np.inf, np.float32)
        valid = (d0 > 0.05) & (d0 < 3.0)
        ramp = np.linspace(0, 60.0, T)[None, :]
        traj[valid, :, 0] = px[valid, 0:1] + ramp
        traj[valid, :, 1] = px[valid, 1:2]
        traj[valid, :, 2] = d0[valid, None]
        np.savez(os.path.join(d, "samples/00000.npz"),
                 keypoints=px.astype(np.float32), traj=traj,
                 valid_steps=np.ones(T, bool))
        n_on.append(int((seg[iy, ix] == int(c["obj_seg_id"])).sum()))

    shutil.copy(f"{a.src}/configs.json", a.dst)
    n = np.array(n_on)
    print(f"rebanked {len(meta['configs'])} scenes -> {a.dst}")
    print(f"  query points ON the object: mean {n.mean():.1f} min {n.min()} max {n.max()}")
    print(f"  (the source lattice put ~{np.mean([c.get('n_grid_on_obj', 0) for c in meta['configs']]):.1f} there)")


if __name__ == "__main__":
    main()
