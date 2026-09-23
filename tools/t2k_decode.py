"""t2k_decode.py <task> <bank_dir> <pred_npz>[,<pred_npz>...] <production_goals_npz> <out_goals_npz>

Turn the Trace-to-Keyframe head outputs saved by msgen.predict (camera frame) into a goal bank in the production schema:
    goal (world) = ext^-1 o (struct_T_cam o rel_T)
K prediction files of the same bank (the K=4 seeds) are averaged on translations (struct_t, rel_t); rotations come from the
first file. Everything else in the output npz is copied from the production goals file (per-step dR/dt for steps < 32 stay
the RANSAC solve; only the terminal step is replaced), so multi_eval / multi_video consume it unchanged.
Prints terminal error vs the true goal (mm / deg, P(<tol)) for: production, head (goal = struct o rel), head-keyframe (the
last keyframe pose) -- the last one is the head's own absolute estimate without the structure decomposition."""
import json, os, sys, numpy as np
# import resolution comes from scripts/config.sh (PYTHONPATH=$EG_REPO:$TRACEGEN_DIR)
from msgen.labels import quat_to_R
task, bank, preds, prod, out = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4], sys.argv[5]
TOL = {"pickcube": 25, "stack": 25, "peginsert": 20, "peg": 20, "liftpeg": 95, "pushcube": 100,
       "placesphere": 25}[task]  # zero-shot probe; its own predicate is xy 5 mm, 25 keeps family comparability
cfg = {str(c["clip"]): c for c in json.load(open(f"{bank}/configs.json"))["configs"]}
Z = [np.load(p, allow_pickle=True) for p in preds]
for z in Z[1:]:
    assert list(z["episode_id"]) == list(Z[0]["episode_id"])
if "t2k_struct_t" not in Z[0].files:
    raise SystemExit(f"{preds[0]} carries no t2k_* outputs (run msgen.predict with MSGEN_T2K=1 on a T2K checkpoint)")
g = np.load(prod, allow_pickle=True); dR, dt, goal_p, goal_q = g["dR"].copy(), g["dt"].copy(), g["goal_p"].copy(), g["goal_q"].copy()
def R2q(R):
    t = np.trace(R)
    if t > 0: s = np.sqrt(t + 1) * 2; return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3; s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2; q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s; q[i + 1] = 0.25 * s; q[j + 1] = (R[j, i] + R[i, j]) / s; q[k + 1] = (R[k, i] + R[i, k]) / s; return q
E = {"production": [], "head": [], "head-kf": []}; A = {k: [] for k in E}; n_dec = 0
for i, clip in enumerate(Z[0]["episode_id"]):
    c = cfg[str(clip)]; j = int(c["env_idx"]); ext = np.eye(4); ext[:3] = np.array(c["extrinsic_cv"]); ext_inv = np.linalg.inv(ext)
    tg = np.array(c["goal"]); gR = quat_to_R(tg[3:7]); op = np.array(c["obj"][:3]); oR = quat_to_R(np.array(c["obj"][3:7]))
    _zk = "t2k_struct_t_zoom" if (os.environ.get("MSGEN_DECODE_ZOOM") == "1" and "t2k_struct_t_zoom" in Z[0].files) else "t2k_struct_t"
    st_t = np.mean([z[_zk][i] for z in Z], 0); rel_t = np.mean([z["t2k_rel_t"][i] for z in Z], 0)
    st_R = Z[0]["t2k_struct_R"][i]; rel_R = Z[0]["t2k_rel_R"][i]
    S = np.eye(4); S[:3, :3] = st_R; S[:3, 3] = st_t; Rl = np.eye(4); Rl[:3, :3] = rel_R; Rl[:3, 3] = rel_t
    Gw = ext_inv @ S @ Rl                                            # goal pose (world)
    kv = Z[0]["t2k_kf_valid"][i]; kk = int(np.where(kv > 0)[0][-1]) if (kv > 0).any() else Z[0]["t2k_kf_t"].shape[1] - 1
    Kf = np.eye(4); Kf[:3, :3] = Z[0]["t2k_kf_R"][i, kk]; Kf[:3, 3] = np.mean([z["t2k_kf_t"][i, kk] for z in Z], 0); Kw = ext_inv @ Kf
    def err(T):
        ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3] @ gR.T) - 1) / 2, -1, 1))); return np.linalg.norm(T[:3, 3] - tg[:3]) * 1000, ang
    if g["solved"][j] and np.isfinite(goal_p[j, -1]).all():
        Tp = np.eye(4); Tp[:3, :3] = dR[j, -1] @ oR; Tp[:3, 3] = goal_p[j, -1]; e, a = err(Tp); E["production"].append(e); A["production"].append(a)
    e, a = err(Gw); E["head"].append(e); A["head"].append(a); e, a = err(Kw); E["head-kf"].append(e); A["head-kf"].append(a)
    # write the head goal into the terminal step (schema: goal = dR @ obj_p + dt, goal_R = dR @ obj_R)
    dR[j, -1] = Gw[:3, :3] @ oR.T; dt[j, -1] = Gw[:3, 3] - dR[j, -1] @ op; goal_p[j, -1] = Gw[:3, 3]; goal_q[j, -1] = R2q(Gw[:3, :3]); n_dec += 1
np.savez_compressed(out, **{k: g[k] for k in g.files if k not in ("dR", "dt", "goal_p", "goal_q")}, dR=dR, dt=dt, goal_p=goal_p, goal_q=goal_q,
                    prov_t2k=f"terminal step from the T2K head ({len(preds)} pred files averaged): goal = ext^-1 o struct o rel")
for k in E:
    e, a = np.array(E[k]), np.array(A[k])
    print(f"{task} {k:11s} n={len(e):3d}  pos med {np.median(e):6.1f} mm  p90 {np.percentile(e, 90):6.1f}  P(<{TOL}) {np.mean(e < TOL):.2f}  rot med {np.median(a):5.1f} deg")
print(f"decoded {n_dec} scenes -> {out}")
