"""mk_fitgoals.py <task> <seed> [repo] -- build the DEPLOYED row4 goal bank.

Paths are parameterised; no cwd assumption.

The deployed goal is a SPLICE, and both halves are pinned:
  rotation  <- the mix4_realcam_n2400 planner's traces, K-mean + RANSAC Kabsch
               (results/<task>_goals_mix4_realcam_n2400_<seed>_kmean_ransac.npz)
  position  <- the mix5_t2k_n3000 T2K head (results/<task>_goals_..._t2k.npz, from
               tools/t2k_decode.py), written into the last step only.
Writes _t2kpos.npz; for stack additionally _dcc.npz (depth cross-check snap via
msppo.goal_depth_check.cross_check). PegInsert keeps the full head pose -- the caller
copies _t2k.npz instead (see scripts/30_goals.sh).
"""
import json
import os
import sys

import numpy as np

task, sd = sys.argv[1], sys.argv[2]
REPO = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("EG_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# msppo resolves via scripts/config.sh (PYTHONPATH=$EG_REPO:$TRACEGEN_DIR)
TAG = os.environ.get("PLANNER_TAG", "mix5_t2k_n3000")
# rotation-source tag: mix4 for the original four tasks; pushcube's rotation comes from
# the mix5_t2k solve (pushcube has no mix4 prediction family; config.sh ROT_TAG passes TAG_HEAD) -- caller passes ROT_TAG.
ROT = os.environ.get("ROT_TAG", "mix4_realcam_n2400")
RES, BANK = f"{REPO}/results", f"{REPO}/data/bank/realcam_{task}_{sd}_obj"

h = np.load(f"{RES}/{task}_goals_{TAG}_{sd}_t2k.npz", allow_pickle=True)
p = np.load(f"{RES}/{task}_goals_{ROT}_{sd}_kmean_ransac.npz", allow_pickle=True)
dR, dt, gp, gq = p["dR"].copy(), p["dt"].copy(), p["goal_p"].copy(), p["goal_q"].copy()
gp[:, -1] = h["goal_p"][:, -1]
cfgs = json.load(open(f"{BANK}/configs.json"))["configs"]
obj_p = np.stack([np.array(c["obj"][:3]) for c in sorted(cfgs, key=lambda c: c["env_idx"])])
dt[:, -1] = gp[:, -1] - np.einsum("nij,nj->ni", dR[:, -1], obj_p)
np.savez_compressed(f"{RES}/{task}_goals_{TAG}_{sd}_t2kpos.npz",
                    **{k: p[k] for k in p.files if k not in ("dR", "dt", "goal_p", "goal_q")},
                    dR=dR, dt=dt, goal_p=gp, goal_q=gq,
                    prov_t2kpos="head position, solver rotation (release builder)")

if task == "stack":
    from msppo.goal_depth_check import cross_check
    z = np.load(f"{RES}/preds/{task}_{TAG}_{sd}_s1234.npz", allow_pickle=True)
    idx = {str(c): i for i, c in enumerate(z["episode_id"])}
    g2 = np.load(f"{RES}/{task}_goals_{TAG}_{sd}_t2kpos.npz", allow_pickle=True)
    dR, dt, gp = g2["dR"].copy(), g2["dt"].copy(), g2["goal_p"].copy()
    n_snap = 0
    for c in cfgs:
        j = c["env_idx"]; op = np.array(c["obj"][:3])
        pt = dR[j, -1] @ op + dt[j, -1]
        if not np.isfinite(pt).all():
            continue
        d = np.load(f"{BANK}/{c['clip']}/depth/00000_raw.npz")["depth"]
        gnew, info = cross_check(pt, op, gp[j, :, 2], z["t2k_contact_mode"][idx[str(c["clip"])]],
                                 d, np.array(c["K"]),
                                 np.vstack([np.array(c["extrinsic_cv"]), [0, 0, 0, 1]]))
        if info["applied"]:
            n_snap += 1; dt[j, -1] = gnew - dR[j, -1] @ op; gp[j, -1] = gnew
    np.savez_compressed(f"{RES}/{task}_goals_{TAG}_{sd}_dcc.npz",
                        **{k: g2[k] for k in g2.files if k not in ("dR", "dt", "goal_p")},
                        dR=dR, dt=dt, goal_p=gp)
    print(f"stack {sd}: dcc snapped {n_snap}")

gsrc = f"{RES}/{task}_goals_{TAG}_{sd}_" + ("dcc" if task == "stack" else "t2kpos") + ".npz"
g = np.load(gsrc, allow_pickle=True)
E = []
for c in cfgs:
    j = c["env_idx"]; tg = np.array(c["goal"][:3]); op = np.array(c["obj"][:3])
    pt = g["dR"][j, -1] @ op + g["dt"][j, -1]
    if np.isfinite(pt).all():
        E.append(np.linalg.norm(pt - tg) * 1000)
print(f"{task} {sd}: n={len(E)} med {np.median(E):.1f} mm")
