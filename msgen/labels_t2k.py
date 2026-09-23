"""labels_t2k.py -- Trace-to-Keyframe labels, derived OFFLINE from the raw replays already on disk (no re-rendering).

    $PG -m msgen.labels_t2k --ds data/ds/realcam_stack_v0 [--clips N] [--overwrite]

The raw clip (`<source_raw>/<clip>.npz`, msgen/replay.py) stores the pose of EVERY rigid body per kept frame, robot links
included (panda_hand_tcp, both fingers), so every label below is a pure function of what is on disk:

  object SE(3) per frame        -> keyframes (twist change points), per-segment twist (screw axis / type)
  tcp + finger poses            -> contact descriptor at motion onset (contact point + normal in the OBJECT frame,
                                   mode grasp/push, precision flag from the approach speed)
  structure actor pose at t0    -> structure frame; terminal pose of the object RELATIVE to it
  segmentation at t0            -> object membership of the 400 grid queries (entity-token supervision)

Per sample (query frame t0, same stems as samples/<stem>.npz) we write samples/<stem>_t2k.npz. Keyframes are mapped
onto the sample's 33-step trace axis with the SAME arclength time map labels.py used, so the head's targets and the
dense trace share one clock. Keyframes before t0 are dropped (the sample starts mid-episode); the terminal pose is the
object's pose at the last frame (it rests there).

Keyframe rule (object-centric, task-agnostic): onset = first frame the object has moved > 5 mm; end = last frame it
moves > 1 mm/frame; interior = change points of the (smoothed) 6-D twist direction (angle > 35 deg, min gap 4 frames),
capped so K <= 4. Twist type per segment from the screw decomposition of the relative transform: `line` (|w| < 5 deg),
`revolute` (rotation with the translation explained by the axis), `screw` (both).
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from msgen.labels import (IMAGE_SIZE, TRAJ_STEPS, arclen_indices, body_transforms, quat_to_R, unproject,
                          _positions_all_frames)
from msgen.tasks import get_task

STRUCT = {"stack": "cubeB", "peg": "box_with_hole_0", "pickcube": "goal_site", "liftpeg": "table-workspace",
          "pushcube": "goal_region", "pusht": "table-workspace", "pokecube": "goal_region"}
ONSET_MM, END_MM_PER_FRAME, TURN_DEG, MIN_GAP, K_MAX, L_ROT = 5.0, 1.0, 35.0, 4, 4, 0.05
TYPES = {"line": 0, "revolute": 1, "screw": 2, "still": 3}

# gmap label (v1): per-sample 24x24 patch soft
# mask = fraction of the sample's t0 depth pixels whose WORLD point lies within GMAP_R of
# the object's TERMINAL position. Task-agnostic (no structure semantics: cubeB top, bin,
# table target all produce their own mask); all-zero when the destination is occluded at
# t0 -- that IS the abstain label.
GMAP_PATCH, GMAP_GRID, GMAP_R = 16, 24, 0.06
# v2: Gaussian weighting (sigma 2cm, cutoff 3*sigma=6cm, same footprint as the v1 ball but weighted) replaces the hard ball --
# concentrates the label at the terminal, suppressing the homogeneous-neighbourhood dilution
# that drowned pushcube (entropy 2.78); and a 48x48 (8px) variant for sub-16px readouts
# (peg's quantisation floor). Both written: `gmap` (24x24) and `gmap48`.
GMAP_SIG = 0.02
_GMAP_UV = None


def amap_label(K, ext, p_term, r_m=0.03):
    """appearance-target label: 2D projected disc around the terminal position (no depth
    requirement -- covers RGB-defined and aerial destinations: peg hole face, painted
    region, lifted peg). Radius = r_m metres scaled to pixels at the terminal's range."""
    T = np.eye(4); T[:3] = ext if ext.shape == (3, 4) else ext[:3]
    pc = T[:3, :3] @ p_term + T[:3, 3]
    if pc[2] <= 0.05:
        return np.zeros((GMAP_GRID, GMAP_GRID), np.float16)
    u = pc[0] / pc[2] * K[0, 0] + K[0, 2]; v = pc[1] / pc[2] * K[1, 1] + K[1, 2]
    r_px = max(6.0, r_m / pc[2] * K[0, 0])
    H = W = GMAP_PATCH * GMAP_GRID
    vv, uu = np.mgrid[0:H, 0:W]
    w = np.exp(-0.5 * (((uu - u) ** 2 + (vv - v) ** 2) / r_px ** 2))
    w[w < 0.01] = 0.0
    return w.reshape(GMAP_GRID, GMAP_PATCH, GMAP_GRID, GMAP_PATCH).mean((1, 3)).astype(np.float16)


def seg_patch_masks(seg, obj_id, st_id):
    """patch-level entity masks straight from the t0 segmentation: fraction of each 16px
    patch covered by the object / the structure actor. Supervises the entity heads that
    decouple pooling and readout exclusion from the trace-motion rule mask."""
    H = W = GMAP_PATCH * GMAP_GRID
    sg = seg[:H, :W]
    om = (sg == obj_id).reshape(GMAP_GRID, GMAP_PATCH, GMAP_GRID, GMAP_PATCH).mean((1, 3))
    sm = (sg == st_id).reshape(GMAP_GRID, GMAP_PATCH, GMAP_GRID, GMAP_PATCH).mean((1, 3))
    return om.astype(np.float16), sm.astype(np.float16)


def gmap_label(depth_m, K, ext, p_term):
    """-> (gmap24 [24,24] f16, gmap48 [48,48] f16), Gaussian-weighted destination maps."""
    global _GMAP_UV
    H = W = GMAP_PATCH * GMAP_GRID
    if _GMAP_UV is None:
        vv, uu = np.mgrid[0:H, 0:W]
        _GMAP_UV = (uu.ravel().astype(np.float64), vv.ravel().astype(np.float64))
    uu, vv = _GMAP_UV
    z = depth_m[:H, :W].ravel().astype(np.float64)
    x = (uu - K[0, 2]) * z / K[0, 0]; y = (vv - K[1, 2]) * z / K[1, 1]
    cam = np.stack([x, y, z], -1)
    R, t = ext[:3, :3], ext[:3, 3]
    world = (cam - t) @ R
    d = np.linalg.norm(world - p_term, axis=-1)
    w = np.exp(-0.5 * (d / GMAP_SIG) ** 2)
    w[(d > 3 * GMAP_SIG) | (z <= 0.05) | (z >= 5.0)] = 0.0
    w = w.reshape(H, W)
    g24 = w.reshape(GMAP_GRID, GMAP_PATCH, GMAP_GRID, GMAP_PATCH).mean((1, 3)).astype(np.float16)
    g48 = w.reshape(2 * GMAP_GRID, GMAP_PATCH // 2, 2 * GMAP_GRID, GMAP_PATCH // 2).mean((1, 3)).astype(np.float16)
    return g24, g48


def T_of(pose7):
    T = np.eye(4); T[:3, :3] = quat_to_R(pose7[3:7]); T[:3, 3] = pose7[:3]; return T


def log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-8: return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def screw(T_rel):
    """relative transform -> (type, axis unit (world), theta rad, d metres along axis)."""
    w = log_so3(T_rel[:3, :3]); th = np.linalg.norm(w); t = T_rel[:3, 3]
    if th < np.radians(5):
        n = np.linalg.norm(t)
        if n < 1e-3: return "still", np.zeros(3), 0.0, 0.0
        return "line", t / n, 0.0, float(n)
    u = w / th; d = float(t @ u); perp = t - d * u
    return ("revolute" if abs(d) < 0.25 * np.linalg.norm(perp) + 5e-3 else "screw"), u, float(th), d


def keyframes(T_obj):
    """T_obj [F,4,4] -> sorted frame indices of the object's keyframes (onset, turns, end); empty if it never moves."""
    F = len(T_obj); p = T_obj[:, :3, 3]
    disp0 = np.linalg.norm(p - p[0], axis=-1) * 1000
    on = np.where(disp0 > ONSET_MM)[0]
    if len(on) == 0: return []
    f_on = int(max(on[0] - 1, 0))
    speed = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=-1) * 1000]
    mv = np.where(speed > END_MM_PER_FRAME)[0]; f_end = int(mv[-1]) if len(mv) else F - 1
    # 6-D twist direction between consecutive frames, rotation scaled to a length
    xi = []
    for f in range(F - 1):
        rel = T_obj[f + 1] @ np.linalg.inv(T_obj[f]); xi.append(np.r_[log_so3(rel[:3, :3]) * L_ROT, rel[:3, 3]])
    xi = np.array(xi); n = np.linalg.norm(xi, axis=-1)
    live = n > 1.5e-3
    u = np.where(live[:, None], xi / np.maximum(n, 1e-9)[:, None], 0.0)
    k = 3
    us = np.array([u[max(0, i - k):i + k + 1].mean(0) for i in range(len(u))]); us /= np.maximum(np.linalg.norm(us, axis=-1), 1e-9)[:, None]
    turns = []
    for f in range(f_on + MIN_GAP, f_end - MIN_GAP):
        if not (live[f - 1] and live[f + 1]): continue
        a, b = us[f - MIN_GAP], us[f + MIN_GAP]
        ang = np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
        if ang > TURN_DEG and (not turns or f - turns[-1][0] >= MIN_GAP): turns.append((f, ang))
        elif ang > TURN_DEG and ang > turns[-1][1]: turns[-1] = (f, ang)
    turns = sorted(turns, key=lambda x: -x[1])[:K_MAX - 2]
    kf = sorted({f_on, f_end} | {f for f, _ in turns})
    return kf


def contact_descriptor(T_obj, T_tcp, p_lf, p_rf, f_on):
    """contact point / normal in the object frame at onset, mode, precision flag."""
    Ti = np.linalg.inv(T_obj[f_on]); c = (Ti @ np.r_[T_tcp[f_on, :3, 3], 1.0])[:3]
    n = Ti[:3, :3] @ T_tcp[f_on, :3, :3][:, 2]                      # tcp z-axis in the object frame
    gap = np.linalg.norm(p_lf[min(f_on + 2, len(p_lf) - 1)] - p_rf[min(f_on + 2, len(p_rf) - 1)])
    f1 = min(f_on + 6, len(T_obj) - 1)
    rel0 = np.linalg.inv(T_tcp[f_on]) @ T_obj[f_on]; rel1 = np.linalg.inv(T_tcp[f1]) @ T_obj[f1]
    comove = np.linalg.norm(rel0[:3, 3] - rel1[:3, 3]) < 0.01
    # mode from the finger gap alone: a grasp holds the fingers OPEN at the
    # object's width (PickCube/Stack 36-42 mm, LiftPeg the peg diameter); a closed gripper (gap < 12 mm) that moves the
    # object is a push (PushCube: 198/200 had been mislabelled grasp by the tcp-near rule, LiftPeg 200/200 push by co-move)
    moved = np.linalg.norm(T_obj[f1, :3, 3] - T_obj[f_on, :3, 3]) > 0.003
    mode = 1 if (0.012 < gap < 0.075) else (2 if moved else 0)         # 0 none, 1 grasp, 2 push
    # precision flag: the tcp's speed over the 4 frames before onset RELATIVE to its median speed in the episode
    v_all = np.linalg.norm(np.diff(T_tcp[:, :3, 3], axis=0), axis=-1); v_med = max(np.median(v_all[v_all > 1e-4]) if (v_all > 1e-4).any() else 1e-3, 1e-3)
    a0 = max(f_on - 4, 0); sp = v_all[a0:max(f_on, a0 + 1)]
    precise = int(sp.mean() < 0.5 * v_med) if len(sp) else 0
    return c, n / max(np.linalg.norm(n), 1e-9), mode, precise, float(gap)


def build_clip(ds, clip, task, source_raw, overwrite):
    raw = np.load(f"{source_raw}/{clip}.npz"); man = json.load(open(f"{source_raw}/manifest.json"))
    names = next(m for m in man["clips"] if m["clip"] == clip)["body_names"]
    cfg = get_task(task); obj_name = cfg["target_body"]; st_name = STRUCT[task]
    row = {n: i for i, n in enumerate(names)}
    suf = lambda s: next(i for n, i in row.items() if n.endswith(s))
    r_obj, r_st, r_tcp, r_lf, r_rf = row[obj_name], row[st_name], suf("/panda_hand_tcp"), suf("/panda_leftfinger"), suf("/panda_rightfinger")
    P = raw["poses"]; F = len(P)
    T_obj = np.stack([T_of(P[f, r_obj]) for f in range(F)]); T_tcp = np.stack([T_of(P[f, r_tcp]) for f in range(F)])
    T_st0 = T_of(P[0, r_st]); p_lf, p_rf = P[:, r_lf, :3], P[:, r_rf, :3]
    if st_name == "table-workspace":
        # support-surface frame: table rotation, translation at the object's terminal xy on the table top (z = 0)
        T_st0 = np.eye(4); T_st0[:3, 3] = [P[F - 1, r_obj, 0], P[F - 1, r_obj, 1], 0.0]
    kf = keyframes(T_obj)
    if not kf: return dict(clip=clip, skipped="object never moves")
    f_on, f_end = kf[0], kf[-1]
    c, n, mode, precise, gap = contact_descriptor(T_obj, T_tcp, p_lf, p_rf, f_on)
    segs = [screw(T_obj[b] @ np.linalg.inv(T_obj[a])) for a, b in zip(kf[:-1], kf[1:])]
    rel_end = np.linalg.inv(T_st0) @ T_obj[F - 1]
    ext = np.eye(4); ext[:3] = raw["extrinsic_cv"]                    # world -> camera (OpenCV), fixed per clip
    cam = lambda T: ext @ T                                           # pose expressed in the camera frame
    body_ids = raw["body_ids"]; obj_id = int(body_ids[r_obj]); st_id = int(body_ids[r_st])
    out_dir = f"{ds}/{clip}/samples"; n_written = 0; stats = []
    for sp in sorted(glob.glob(f"{out_dir}/*.npz")):
        stem = os.path.basename(sp)[:-4]
        if stem.endswith("_t2k"): continue
        out = f"{out_dir}/{stem}_t2k.npz"
        if os.path.exists(out) and not overwrite: continue
        t0 = int(stem); z = np.load(sp); px = z["keypoints"].astype(np.float64)
        # the same time map labels.py used for this sample (arclength over the remaining frames)
        ix = np.clip(px[:, 0].astype(int), 0, IMAGE_SIZE - 1); iy = np.clip(px[:, 1].astype(int), 0, IMAGE_SIZE - 1)
        d0 = raw["depth_mm"][t0].astype(np.float64)[iy, ix] / 1000.0; sid = raw["seg"][t0][iy, ix]
        id_to_row = {int(b): r for r, b in enumerate(body_ids)}; rw = np.array([id_to_row.get(int(s), -1) for s in sid])
        valid = (rw >= 0) & (d0 > 0.05) & (d0 < 5.0)
        p0 = unproject(px, d0, raw["K"], raw["extrinsic_cv"])
        pos = _positions_all_frames(raw, t0, px, valid, rw, p0)
        fidx = arclen_indices(pos[:, :, :2], valid)                     # raw frame offsets (from t0) at steps 1..32
        f_of_step = np.r_[0.0, fidx]                                    # step -> frame offset
        step_of = lambda f: float(np.interp(f - t0, f_of_step, np.arange(TRAJ_STEPS)))
        kf_vis = [f for f in kf if f >= t0]
        kf_step = np.array([step_of(f) for f in kf_vis]); kf_T = np.stack([T_obj[f] for f in kf_vis]) if kf_vis else np.zeros((0, 4, 4))
        kf_kind = np.array([0 if f == f_on else (2 if f == f_end else 1) for f in kf_vis])   # 0 onset, 1 turn, 2 end
        seg_vis = [(a, b, s) for (a, b), s in zip(zip(kf[:-1], kf[1:]), segs) if b >= t0]
        tw_type = np.array([TYPES[s[0]] for _, _, s in seg_vis]); tw_axis = np.array([s[1] for _, _, s in seg_vis]).reshape(-1, 3)
        tw_mag = np.array([[s[2], s[3]] for _, _, s in seg_vis]).reshape(-1, 2)
        obj_mask = (sid == obj_id) & valid; st_mask = (sid == st_id) & valid
        gmap, gmap48 = gmap_label(raw["depth_mm"][t0].astype(np.float64) / 1000.0, raw["K"], ext, T_obj[F - 1][:3, 3])
        amap = amap_label(raw["K"], raw["extrinsic_cv"], T_obj[F - 1][:3, 3])
        omask24, smask24 = seg_patch_masks(raw["seg"][t0], obj_id, st_id)
        np.savez_compressed(out, gmap=gmap, gmap48=gmap48, amap=amap, omask24=omask24, smask24=smask24, kf_step=kf_step.astype(np.float32), kf_T=kf_T.astype(np.float32), kf_kind=kf_kind.astype(np.int8),
                            tw_type=tw_type.astype(np.int8), tw_axis=tw_axis.astype(np.float32), tw_mag=tw_mag.astype(np.float32),
                            contact_c=c.astype(np.float32), contact_n=n.astype(np.float32), contact_mode=np.int8(mode), contact_precise=np.int8(precise),
                            contact_step=np.float32(step_of(f_on)) if f_on >= t0 else np.float32(-1),
                            struct_T=T_st0.astype(np.float32), rel_end_T=rel_end.astype(np.float32), obj_T_t0=T_obj[t0].astype(np.float32),
                            kf_T_cam=np.stack([cam(T_obj[f]) for f in kf_vis]).astype(np.float32) if kf_vis else np.zeros((0, 4, 4), np.float32),
                            struct_T_cam=cam(T_st0).astype(np.float32), obj_T_t0_cam=cam(T_obj[t0]).astype(np.float32), obj_T_end_cam=cam(T_obj[F - 1]).astype(np.float32),
                            tw_axis_cam=(ext[:3, :3] @ tw_axis.T).T.astype(np.float32) if len(tw_axis) else np.zeros((0, 3), np.float32),
                            contact_n_cam=n.astype(np.float32), K=raw["K"].astype(np.float32), extrinsic_cv=raw["extrinsic_cv"].astype(np.float32),
                            obj_T_end=T_obj[F - 1].astype(np.float32), obj_mask=obj_mask, struct_mask=st_mask, valid=valid,
                            f_on=f_on, f_end=f_end, t0=t0, n_frames=F, obj_name=obj_name, struct_name=st_name)
        n_written += 1; stats.append((len(kf_vis), int(obj_mask.sum())))
    return dict(clip=clip, K=len(kf), kf=kf, types=[s[0] for s in segs], mode=mode, precise=precise, gap_mm=gap * 1000,
                rel_end_mm=np.linalg.norm(rel_end[:3, 3]) * 1000, samples=n_written, mean_obj_pts=float(np.mean([s[1] for s in stats])) if stats else 0.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--clips", type=int, default=None); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(); meta = json.load(open(f"{a.ds}/dataset.json")); task = meta["task"]; src = meta["source_raw"]
    clips = [c["clip"] for c in meta["clips"]][: a.clips]; res = []
    for c in clips:
        r = build_clip(a.ds, c, task, src, a.overwrite); res.append(r); print(json.dumps(r, default=float), flush=True)
    ok = [r for r in res if "K" in r]
    import collections
    summ = dict(task=task, ds=a.ds, clips=len(res), skipped=len(res) - len(ok),
                K=dict(collections.Counter(r["K"] for r in ok)), twist_types=dict(collections.Counter(t for r in ok for t in r["types"])),
                mode=dict(collections.Counter(r["mode"] for r in ok)), precise=float(np.mean([r["precise"] for r in ok])) if ok else None,
                gap_mm_med=float(np.median([r["gap_mm"] for r in ok])) if ok else None, rel_end_mm_med=float(np.median([r["rel_end_mm"] for r in ok])) if ok else None,
                samples=sum(r["samples"] for r in ok))
    json.dump(summ, open(f"{a.ds}/t2k_summary.txt", "w"), indent=1); print("SUMMARY", json.dumps(summ))


if __name__ == "__main__":
    main()
