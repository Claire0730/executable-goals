"""psi_bank.py <task> <seed> [pred_npz] -- the deployment-side psi = (k1, h) per scene from the PLANNER trace alone
(msgen.trace_seg: rule-based, no learning, no task identity), rows in env order (configs.json env_idx), for
`multi_eval --psi-bank task=...npz`. k1 = unit displacement of the arm points over the 4 steps before the grasp step (world);
h = carry height = max object-centroid z minus its initial z. Scenes where the extraction fails fall back to the task prior
(k1 = cone axis, h = mid of the teacher's range) and are flagged ok=False. Output: results/<task>_psi_<tag>_<seed>.npz"""
import json, os, sys, numpy as np
# import resolution comes from scripts/config.sh (PYTHONPATH=$EG_REPO:$TRACEGEN_DIR)
from msgen.trace_seg import segment, world_profiles
task, seed = sys.argv[1], sys.argv[2]
tag = os.environ.get("TAG_MIX4", "mix4_realcam_n2400")
pred = sys.argv[3] if len(sys.argv) > 3 else f"results/preds/{task}_{tag}_{seed}_kmean.npz"
PRIOR = {"pickcube": ([0, 0, -1], 0.10), "liftpeg": ([0, 0, -1], 0.10), "peginsert": ([-0.28, -0.56, -0.78], 0.09), "stack": ([0, 0, -1], 0.095)}
HR = {"pickcube": (0.06, 0.14), "liftpeg": (0.06, 0.14), "peginsert": (0.06, 0.12), "stack": (0.06, 0.13)}
bank_task = {"peginsert": "peg"}.get(task, task)
B = f"data/bank/realcam_{bank_task}_{seed}_obj"
if not os.path.isdir(B): B = f"data/bank/realcam_{task}_{seed}_obj"
cfgs = json.load(open(f"{B}/configs.json"))["configs"]; byclip = {str(c["clip"]): c for c in cfgs}; N = len(cfgs)
z = np.load(pred, allow_pickle=True)
a0 = np.array(PRIOR[task][0], float); a0 /= np.linalg.norm(a0); h0 = PRIOR[task][1]; lo, hi = HR[task]
k1 = np.tile(a0, (N, 1)); h = np.full(N, h0); ok = np.zeros(N, bool); angs, hs = [], []
for i, clip in enumerate(z["episode_id"]):
    c = byclip[str(clip)]; e = int(c["env_idx"]); K, E = np.array(c["K"]), np.array(c["extrinsic_cv"])
    sf = segment(z["pred"][i])
    if sf["obj_mask"].sum() < 3 or sf["grasp_step"] < 0: continue
    w = world_profiles(z["pred"][i], sf, K, E)
    if w["approach"] is None or not np.isfinite(w["carry_mm"]): continue
    k1[e] = w["approach"]; h[e] = float(np.clip(w["carry_mm"] / 1000.0, lo, hi)); ok[e] = True
    angs.append(np.degrees(np.arccos(np.clip(w["approach"] @ a0, -1, 1)))); hs.append(w["carry_mm"])
# TRUST REGION: the teachers were trained inside a 25-degree cone around the task axis; a command outside it is out of the
# teacher's prior, so the deployed k1 is the raw trace direction pulled back onto the cone boundary (raw kept as k1_raw).
k1_raw = k1.copy(); TH = np.radians(25.0)
for e in range(N):
    c = float(np.clip(k1[e] @ a0, -1, 1)); ang = np.arccos(c)
    if ang > TH:
        perp = k1[e] - c * a0; n = np.linalg.norm(perp)
        k1[e] = np.cos(TH) * a0 + (np.sin(TH) * perp / n if n > 1e-9 else 0.0)
    k1[e] /= np.linalg.norm(k1[e])
out = f"results/{task}_psi_{tag}_{seed}.npz"
np.savez_compressed(out, k1=k1.astype(np.float32), k1_raw=k1_raw.astype(np.float32), h=h.astype(np.float32), ok=ok, prov=f"{pred} via msgen.trace_seg (k1 = pre-grasp arm displacement, h = carry height)")
print(f"{task} seed {seed}: {ok.sum()}/{N} scenes extracted | k1 angle to prior axis med {np.median(angs):.0f} deg (p90 {np.percentile(angs, 90):.0f}) | carry med {np.median(hs):.0f} mm -> h clipped to [{lo},{hi}] | {out}")
