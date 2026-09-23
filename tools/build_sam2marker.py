"""build_sam2marker.py <seed> -- pickcube goal bank from the SAM2 marker locator (zero colour rules).
Base bank = gmappeakNC (rotation, non-terminal steps, and the fallback goal for abstained scenes);
terminal position replaced by op_marker_sam2 where it applies. Reports application rate, error vs the
true goal on applied scenes, the colour-era `_marker` bank as a reference (per-scene difference), and a
test-only "capped" diagnostic (abstentions whose floating footprint near the prompt exceeds the area cap)."""
import json, os, sys
import numpy as np
TAG_MIX4, TAG_HEAD, TAG_GMAP = (os.environ.get("TAG_MIX4", "mix4_realcam_n2400"), os.environ.get("TAG_HEAD", "mix5_t2k_n3000"), os.environ.get("TAG_GMAP", "mix5_t2k_gmap"))
import imageio.v2 as iio
from msppo import goal_depth_check as G
R = os.environ.get("EG_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sd = sys.argv[1]
TASK = os.environ.get("SAM2MK_TASK", "pickcube")   # pickcube (sphere marker) | pushcube (flat 10 cm disc on the table)
OPK = dict(pickcube=dict(), pushcube=dict(radius_m=0.0, area=(200, 12000), aspect_max=3.0))[TASK]
TOLMM = dict(pickcube=25, pushcube=100)[TASK]
B = f"{R}/data/bank/realcam_{TASK}_{sd}_obj"
cfgs = {c["clip"]: c for c in json.load(open(f"{B}/configs.json"))["configs"]}
d = np.load(f"{R}/results/preds/{TASK}_{TAG_GMAP}_{sd}_s1234.npz", allow_pickle=True)
idx = {str(e): i for i, e in enumerate(d["episode_id"])}
BASE = os.environ.get("SAM2MK_BASE", "gmappeakNC")   # base bank supplying dR/dt/non-terminal steps and the abstain fallback
if TASK == "pushcube" and BASE == "gmappeakNC": BASE = "t2kpos"          # pushcube production base = raw T2K head position
BASE_PATH = os.environ.get("SAM2MK_BASE_PATH") or (f"{R}/results/{TASK}_goals_{TAG_GMAP}_{sd}_gmappeakNC.npz" if BASE == "gmappeakNC" else f"{R}/results/{TASK}_goals_{TAG_HEAD}_{sd}_{BASE}.npz")
g = np.load(BASE_PATH, allow_pickle=True)
mk_path = f"{R}/results/{TASK}_goals_{TAG_HEAD}_{sd}_marker.npz"   # colour-era reference bank (not shipped; optional)
mk = np.load(mk_path, allow_pickle=True) if os.path.exists(mk_path) else None


def floating_footprint(depth, uv, win=24):
    """test-only diagnostic: pixels within a (2*win)^2 window around uv that float >= SAM2_FLOAT_M in front of their 6-px ring."""
    from scipy.ndimage import grey_dilation
    u, v = int(uv[0]), int(uv[1]); H, W = depth.shape
    r0, r1, c0, c1 = max(0, v - win), min(H, v + win), max(0, u - win), min(W, u + win)
    blk = depth[r0:r1, c0:c1]; val = blk > 0.05
    far = grey_dilation(np.where(val, blk, 0.0), size=(2 * G.SAM2_RING_PX + 1, 2 * G.SAM2_RING_PX + 1))   # local max depth (ring background)
    return int((((far - blk) >= 0.03) & val).sum())   # diagnostic keeps the original 30 mm floating definition


dR, dt, gp = g["dR"].copy(), g["dt"].copy(), g["goal_p"].copy()
N = len(dR); applied = np.zeros(N, bool); reason = np.array([""] * N, dtype=object); errs, diffs, infos = [], [], {}; capped = 0
for clip, c in cfgs.items():
    i, j = idx[clip], int(c["env_idx"])
    rgb = iio.imread(f"{B}/{clip}/images/00000.png")[:, :, :3]
    dep = np.load(f"{B}/{clip}/depth/00000_raw.npz")["depth"].astype(np.float32)
    K, E = np.array(c["K"]), np.array(c["extrinsic_cv"])
    pw, info = G.op_marker_sam2(gp[j, -1], rgb, dep, K, E, d["t2k_gmap"][i], **OPK)
    reason[j] = info.get("reason", ""); infos[clip] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info.items()}
    if info["applied"]:
        applied[j] = True; op = np.array(c["obj"][:3]); gp[j, -1] = pw; dt[j, -1] = pw - dR[j, -1] @ op
        errs.append(np.linalg.norm(pw - np.array(c["goal"][:3])) * 1000)
        if mk is not None and np.isfinite(mk["goal_p"][j, -1]).all():
            diffs.append(np.linalg.norm(pw - mk["goal_p"][j, -1]) * 1000)
    elif info.get("reason") == "no_mask":
        fp = floating_footprint(dep, info["prompt_uv"][0]); infos[clip]["floating_fp"] = fp
        capped += int(fp > G.SAM2_MASK_AREA[1])
TAG = os.environ.get("SAM2MK_TAG", "sam2mk6")   # the released PickCube bank suffix
out = f"{R}/results/{TASK}_goals_{TAG_GMAP}_{sd}_{TAG}.npz"
np.savez_compressed(out, **{k: g[k] for k in g.files if k not in ("dR", "dt", "goal_p")}, dR=dR, dt=dt, goal_p=gp,
                    applied=applied, reason=np.array(reason.astype(str)),
                    prov_sam2marker=f"terminal position = SAM2 marker locator (gmap-peak prompt); abstained scenes keep the {BASE} base goal")
e = np.array(errs); rs = {r: int((reason == r).sum()) for r in sorted(set(reason.tolist()))}
summ = dict(seed=sd, applied=int(applied.sum()), N=N, err_med=float(np.median(e)) if len(e) else None,
            err_p90=float(np.percentile(e, 90)) if len(e) else None, P10=float((e < 10).mean()) if len(e) else None,
            P25=float((e < TOLMM).mean()) if len(e) else None, diff_vs_colour_med=float(np.median(diffs)) if diffs else None,
            reasons=rs, capped=capped)
json.dump(dict(summary=summ, per_scene=infos), open(f"{R}/results/sam2marker_offline_{TASK}_{TAG}_{sd}.json", "w"), indent=1)
f = lambda x: "None" if x is None else f"{x:.2f}"
print(f"sam2marker {TASK} {sd}: applied {applied.sum()}/{N} ({applied.mean():.2f}) | err med {f(summ['err_med'])} p90 {f(summ['err_p90'])} "
      f"P<10 {f(summ['P10'])} P<tol {f(summ['P25'])} | vs colour med {f(summ['diff_vs_colour_med'])} | reasons {rs} capped~{capped} -> {out}")
