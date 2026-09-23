"""build_pc_src.py <seed> -- pickcube COLOUR-FREE goal banks for one seed, both variants:
  gmappeakNC : goal-map peak + ray from the gmap head (proposal / fallback = last valid keyframe, world),
               0.12 m gate, abstain -> head proposal; no goal-marker segmentation is used.
  islandNC   : depth-island mean-shift (3x5cm ball, median) anchored at the gmappeakNC goal,
               compactness gate (rms<=3.5cm, n in [10,1500], shift<=12cm), abstain -> anchor.
Base dR/dt: the seed's _t2kpos bank (solve rotation). Writes both npz + prints med/P."""
import json
import os
import sys

import numpy as np
TAG_MIX4, TAG_HEAD, TAG_GMAP = (os.environ.get("TAG_MIX4", "mix4_realcam_n2400"), os.environ.get("TAG_HEAD", "mix5_t2k_n3000"), os.environ.get("TAG_GMAP", "mix5_t2k_gmap"))

from msppo.goal_depth_check import unproject_full

sd = sys.argv[1]
R = os.environ.get("EG_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATCH, GRID, GATE = 16, 24, 0.12
B = f"{R}/data/bank/realcam_pickcube_{sd}_obj"
cfgs = {c["clip"]: c for c in json.load(open(f"{B}/configs.json"))["configs"]}
d = np.load(f"{R}/results/preds/pickcube_{TAG_GMAP}_{sd}_s1234.npz", allow_pickle=True)
idx = {str(c): i for i, c in enumerate(d["episode_id"])}
g = np.load(f"{R}/results/pickcube_goals_{TAG_HEAD}_{sd}_t2kpos.npz", allow_pickle=True)
out = {}
for var in ("gmappeakNC", "islandNC"):
    out[var] = (g["dR"].copy(), g["dt"].copy(), g["goal_p"].copy(), [])
for clip, c in cfgs.items():
    i = idx[clip]; j = c["env_idx"]
    tg = np.array(c["goal"][:3]); op = np.array(c["obj"][:3])
    ext = np.eye(4); ext[:3] = np.array(c["extrinsic_cv"]); K = np.array(c["K"])
    kv, kt = d["t2k_kf_valid"][i], d["t2k_kf_t"][i]
    vs = np.where(kv > 0)[0]
    if not len(vs):
        for var in out: out[var][3].append(np.linalg.norm(out[var][2][j, -1] - tg) * 1000)
        continue
    g0 = (np.linalg.inv(ext) @ np.r_[kt[vs[-1]], 1.0])[:3]
    dep = np.load(f"{B}/{clip}/depth/00000_raw.npz")["depth"].astype(np.float64)
    val = (dep > 0.05) & (dep < 3.0)
    x = d["t2k_gmap"][i].astype(np.float64); gm = np.exp(x - x.max()); gm /= gm.sum()
    gm = gm.reshape(GRID, GRID)
    # R2 peak+ray
    gpk = g0
    p = int(np.argmax(gm)); r0, c0 = (p // GRID) * PATCH, (p % GRID) * PATCH
    blk = dep[r0:r0 + PATCH, c0:c0 + PATCH]; bval = val[r0:r0 + PATCH, c0:c0 + PATCH]
    if bval.sum() >= 5:
        z = float(np.median(blk[bval])); u, v = c0 + PATCH / 2, r0 + PATCH / 2
        pc = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        cand = ext[:3, :3].T @ (pc - ext[:3, 3])
        if np.linalg.norm(cand - g0) <= GATE:
            gpk = cand
    # island mean-shift anchored at gpk
    gis = gpk
    W3 = unproject_full(dep, K, ext).reshape(*dep.shape, 3)
    P = W3[val & (W3[..., 2] > 0.05)]
    x = gpk.copy(); ok = True
    for _ in range(3):
        m = np.linalg.norm(P - x, axis=-1) < 0.05
        if m.sum() < 10:
            ok = False; break
        x = np.median(P[m], 0)
    if ok:
        m = np.linalg.norm(P - x, axis=-1) < 0.05
        C = P[m]; rms = float(np.sqrt(((C - C.mean(0)) ** 2).sum(1).mean()))
        if 10 <= len(C) <= 1500 and rms <= 0.035 and np.linalg.norm(x - gpk) <= GATE:
            gis = x
    for var, gsel in (("gmappeakNC", gpk), ("islandNC", gis)):
        dR, dt, gp, E = out[var]
        dt[j, -1] = gsel - dR[j, -1] @ op; gp[j, -1] = gsel
        E.append(np.linalg.norm(gsel - tg) * 1000)
for var, (dR, dt, gp, E) in out.items():
    f = f"{R}/results/pickcube_goals_{TAG_GMAP}_{sd}_{var}.npz"
    np.savez_compressed(f, **{k: g[k] for k in g.files if k not in ("dR", "dt", "goal_p")},
                        dR=dR, dt=dt, goal_p=gp, prov_nc=f"{var}: colour-free, abstain->head/anchor, seed {sd}")
    e = np.array(E)
    print(f"pickcube {sd} {var:11s}: med {np.median(e):5.1f} mm  P(<25) {(e < 25).mean():.3f} -> {f.split('/')[-1]}")
