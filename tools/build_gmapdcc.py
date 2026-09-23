"""build_gmapdcc.py <seed> -- StackCube gmapdcc goal bank for one seed: goal-map-weighted xy depth centroid,
z from the last valid keyframe, 0.12 m gate against the keyframe proposal, fallback = the seed's _dcc goal.
Rotation / per-step dR from the seed's _t2kpos bank."""
import json
import os
import sys

import numpy as np
TAG_MIX4, TAG_HEAD, TAG_GMAP = (os.environ.get("TAG_MIX4", "mix4_realcam_n2400"), os.environ.get("TAG_HEAD", "mix5_t2k_n3000"), os.environ.get("TAG_GMAP", "mix5_t2k_gmap"))

from msppo.goal_depth_check import unproject_full

sd = sys.argv[1]
R = os.environ.get("EG_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GTAG, PTAG = TAG_GMAP, TAG_HEAD
PATCH, GRID, GATE = 16, 24, 0.12

B = f"{R}/data/bank/realcam_stack_{sd}_obj"
cfgs = {c["clip"]: c for c in json.load(open(f"{B}/configs.json"))["configs"]}
d = np.load(f"{R}/results/preds/stack_{GTAG}_{sd}_s1234.npz", allow_pickle=True)
idx = {str(c): i for i, c in enumerate(d["episode_id"])}
g = np.load(f"{R}/results/stack_goals_{PTAG}_{sd}_dcc.npz", allow_pickle=True)
dR, dt, gp = g["dR"].copy(), g["dt"].copy(), g["goal_p"].copy()
n_ap, errs = 0, []
for clip, c in cfgs.items():
    i = idx[clip]; j = c["env_idx"]
    kv, kt = d["t2k_kf_valid"][i], d["t2k_kf_t"][i]
    vs = np.where(kv > 0)[0]
    if not len(vs):
        continue
    ext = np.eye(4); ext[:3] = np.array(c["extrinsic_cv"])
    g0 = (np.linalg.inv(ext) @ np.r_[kt[vs[-1]], 1.0])[:3]
    K = np.array(c["K"])
    dep = np.load(f"{B}/{clip}/depth/00000_raw.npz")["depth"].astype(np.float64)
    gm = np.exp(d["t2k_gmap"][i].astype(np.float64) - d["t2k_gmap"][i].max()); gm /= gm.sum()
    gm = gm.reshape(GRID, GRID)
    val = (dep > 0.05) & (dep < 3.0)
    W3 = unproject_full(dep, K, ext).reshape(dep.shape[0], dep.shape[1], 3)
    w = np.kron(gm, np.ones((PATCH, PATCH)))[:dep.shape[0], :dep.shape[1]] * val
    s = w.sum()
    if s < 1e-9:
        continue
    cxy = (W3[..., :2] * w[..., None]).sum((0, 1)) / s
    gnew = np.array([cxy[0], cxy[1], g0[2]])
    if np.linalg.norm(gnew - g0) > GATE:
        continue
    op = np.array(c["obj"][:3]); tg = np.array(c["goal"][:3])
    dt[j, -1] = gnew - dR[j, -1] @ op; gp[j, -1] = gnew
    n_ap += 1; errs.append(np.linalg.norm(gnew - tg) * 1000)
out = f"{R}/results/stack_goals_{GTAG}_{sd}_gmapdcc.npz"
np.savez_compressed(out, **{k: g[k] for k in g.files if k not in ("dR", "dt", "goal_p")},
                    dR=dR, dt=dt, goal_p=gp, prov_gmap=f"centroid readout, fallback=dcc, seed {sd}")
e = np.array(errs)
print(f"stack {sd}: applied {n_ap}/{len(cfgs)}  gmap goal err med {np.median(e):.1f}mm -> {out}")
